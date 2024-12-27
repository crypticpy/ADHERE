"""
Ticket Processing Pipeline Module

This module implements a sophisticated concurrent processing pipeline for IT service tickets
with the following key features:

1. Adaptive Concurrency (with CPU-based scaling + random jitter)
2. Safety Mechanisms:
   - Circuit Breaker
   - Token Bucket
   - Error State Tracking
3. Data Flow:
   Raw Tickets → PII Redaction → LLM Processing → Validation → Output Files
4. Performance Features:
   - Batch Processing
   - Thread Pool
   - Graceful Shutdown

The pipeline is designed to handle large volumes of tickets while maintaining
system stability and respecting API rate limits.
"""

import json
import logging
import re
import time
import random  # NEW: For jitter in scaling intervals
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from multiprocessing import Queue as MPQueue
import psutil
from typing import Tuple, Optional, List

import threading

from config import Config
from models import TicketSchema, TicketParsedSchema
from llm_api import retry_call_llm_parse, parse_response
from utils import redact_pii, fill_missing_fields
from circuit_breaker import CircuitBreaker
from token_bucket import TokenBucket
from io_utils import writer_process
from shared_state import error_state, good_tickets_queue, bad_tickets_queue
from openai import APIError, RateLimitError

from main import shutdown_requested


def _process_ticket(
    ticket_json: str,
    index: int,
    cfg: Config,
    bucket: TokenBucket,
    circuit_breaker: CircuitBreaker,
) -> Tuple[Optional[TicketSchema], Optional[str]]:
    """
    Processes a single ticket through the complete transformation pipeline.

    1. JSON Parsing
    2. Field Normalization
    3. PII Redaction
    4. Rate Limiting (Token Bucket)
    5. Circuit Breaking
    6. LLM Processing
    7. Validation

    Args:
        ticket_json: Raw ticket data as JSON string
        index: Worker thread identifier
        cfg: System configuration
        bucket: Rate limiting token bucket
        circuit_breaker: Circuit breaker for failure protection

    Returns:
        (TicketSchema, None) for successful processing
        (None, error_message) for failed processing
    """
    # Additional debug log for incoming ticket
    logging.debug("Worker %d received ticket: %s", index, ticket_json[:200])

    try:
        data = json.loads(ticket_json)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"

    filled = fill_missing_fields(data)
    redacted = redact_pii(filled)

    # Consume a token from the token bucket
    while not bucket.consume(1):
        time.sleep(1)

    if not circuit_breaker.allow_request():
        return None, "Circuit breaker open (requests not allowed)"

    raw_str = json.dumps(redacted)
    try:
        response = retry_call_llm_parse(raw_str, cfg)
        parsed_resp = parse_response(response)
        if not parsed_resp:
            return None, "Failed to parse LLM response"
    except (RateLimitError, APIError) as exc:
        circuit_breaker.record_failure()
        error_state.increment()
        return None, f"API Error (final): {exc}"
    except Exception as exc:
        circuit_breaker.record_failure()
        error_state.increment()
        return None, f"Worker {index} unexpected final fail: {exc}"

    if parsed_resp.status.lower() == "bad":
        return None, parsed_resp.data

    if not isinstance(parsed_resp.data, dict):
        return None, "No parsed data in response"

    try:
        ticket = TicketSchema(**parsed_resp.data)
    except Exception:
        return None, "TicketSchema validation error"

    # Check minimal word counts
    instr_words = re.findall(r"\b\w+\b", ticket.instruction or "")
    outcome_words = re.findall(r"\b\w+\b", ticket.outcome or "")
    if len(instr_words) < cfg.min_word_count or len(outcome_words) < cfg.min_word_count:
        return None, "Instruction or outcome too short"

    return ticket, None


def worker(
    task_q: Queue,
    cfg: Config,
    index: int,
    bucket: TokenBucket,
    circuit_breaker: CircuitBreaker,
    concurrency_sema: threading.Semaphore,
) -> None:
    """
    Worker thread function that processes tickets from a shared queue.

    1. Acquires concurrency permit
    2. Fetches ticket from queue
    3. Processes ticket
    4. Routes result to good or bad queue
    5. Releases concurrency permit

    Thread Safety:
    - Uses semaphore for concurrency control
    - Queue for thread-safe task distribution
    - Thread-safe result queues for output

    Args:
        task_q: Queue containing tickets to process
        cfg: System configuration
        index: Worker identifier
        bucket: Rate limiting token bucket
        circuit_breaker: Circuit breaker for failure protection
        concurrency_sema: Semaphore for controlling active workers

    Returns:
        None. Results are pushed into good_tickets_queue or bad_tickets_queue.
    """
    logging.info("Worker %d started.", index)
    while True:
        concurrency_sema.acquire()
        try:
            try:
                ticket_data = task_q.get(timeout=1)
            except Empty:
                break
            if ticket_data is None:
                task_q.task_done()
                break

            result, err = _process_ticket(
                ticket_data, index, cfg, bucket, circuit_breaker
            )
            if result:
                good_tickets_queue.put(result)
            elif err:
                bad_tickets_queue.put(err)
            task_q.task_done()
        finally:
            concurrency_sema.release()


def maybe_warmup(
    num_tickets: int, cfg: Config, concurrency_sema: threading.Semaphore
) -> None:
    """
    Implements adaptive warm-up strategy based on workload size.

    Warm-up Strategies:
    1. Simple Warm-up (< 10,000 tickets): sets concurrency to cfg.initial_workers
    2. Advanced Warm-up (>= 10,000 tickets):
       - Start with 2 workers
       - Gradually ramp to cfg.initial_workers over ~120s

    Args:
        num_tickets: Total number of tickets to process
        cfg: System configuration
        concurrency_sema: Semaphore controlling worker count
    """
    if num_tickets < 10000:
        needed = cfg.initial_workers - concurrency_sema._value
        if needed > 0:
            concurrency_sema.release(needed)
        logging.info("Simple warmup set concurrency to %d", cfg.initial_workers)
    else:
        current = concurrency_sema._value
        if current > 2:
            for _ in range(current - 2):
                concurrency_sema.acquire()
        else:
            concurrency_sema.release(max(0, 2 - current))

        steps = cfg.initial_workers - 2
        if steps > 0:
            interval = 120 / steps
            logging.info("Advanced warmup from 2 to %d over ~120s", cfg.initial_workers)
            for _ in range(steps):
                time.sleep(interval)
                concurrency_sema.release(1)


def process_tickets_with_scaling(
    tickets: List[str],
    cfg: Config,
    file_q: MPQueue,
    circuit_breaker: CircuitBreaker,
    bucket: TokenBucket,
) -> None:
    """
    Main orchestrator for the ticket processing pipeline with adaptive scaling and random jitter.

    System Architecture:
        Input Queue → Workers → Result Queue → File Writer
                       |          |
                (Rate Limit) (Circuit Breaker)

    Features:
    1. Concurrency Management with CPU-based scaling + random jitter
    2. Safety Mechanisms (circuit breaker, token bucket, error tracking)
    3. I/O Efficiency (batch processing of results)
    4. Monitoring (progress bar, logging)

    Args:
        tickets: List of raw ticket JSON strings
        cfg: System configuration
        file_q: Queue for file writing operations
        circuit_breaker: Circuit breaker for failure protection
        bucket: Token bucket for rate limiting
    """
    logging.info("Starting ticket processing with adaptive concurrency.")
    task_q = Queue()
    for tk in tickets:
        task_q.put(tk)

    executor = ThreadPoolExecutor(max_workers=cfg.max_workers)
    concurrency_sema = threading.Semaphore(1)  # Start small, ramp up in warmup

    for i in range(cfg.max_workers):
        executor.submit(
            worker, task_q, cfg, i, bucket, circuit_breaker, concurrency_sema
        )

    total = len(tickets)
    processed_count = 0

    # Initial warmup
    maybe_warmup(total, cfg, concurrency_sema)

    from tqdm import tqdm

    pbar = tqdm(total=total, desc="Processing Tickets")

    last_scale_time = time.monotonic()
    cooldown_active = False
    current_workers = cfg.initial_workers

    # Ensure concurrency matches initial_workers after warmup
    delta = concurrency_sema._value - current_workers
    if delta > 0:
        for _ in range(delta):
            concurrency_sema.acquire()
    elif delta < 0:
        concurrency_sema.release(-delta)

    last_log_time = time.monotonic()

    up_threshold = cfg.cpu_scale_up_threshold
    down_threshold = cfg.cpu_scale_down_threshold

    while True:
        if shutdown_requested:
            logging.warning("Graceful shutdown triggered.")
            break

        # Drain good tickets
        if good_tickets_queue.qsize() >= cfg.batch_size:
            batch = []
            while not good_tickets_queue.empty() and len(batch) < cfg.batch_size:
                batch.append(good_tickets_queue.get())
            file_q.put((batch, cfg.good_tickets_file))
            processed_count += len(batch)
            pbar.update(len(batch))

        # Drain bad tickets
        if bad_tickets_queue.qsize() >= cfg.batch_size:
            batch = []
            while not bad_tickets_queue.empty() and len(batch) < cfg.batch_size:
                batch.append(bad_tickets_queue.get())
            file_q.put((batch, cfg.bad_tickets_file))
            processed_count += len(batch)
            pbar.update(len(batch))

        # Check error threshold => cooldown
        if error_state.error_count >= cfg.error_threshold and not cooldown_active:
            cooldown_active = True
            logging.warning("Too many errors. Cooling down to 1 worker.")
            if current_workers > 1:
                for _ in range(current_workers - 1):
                    concurrency_sema.acquire()
                current_workers = 1

            error_state.reset()
            time.sleep(cfg.cool_down_time)
            circuit_breaker.reset()
            cooldown_active = False

            concurrency_sema.release(cfg.initial_workers - current_workers)
            current_workers = cfg.initial_workers
            last_scale_time = time.monotonic()

        # Periodic adaptive scaling with hysteresis + random jitter
        now = time.monotonic()
        if not cooldown_active and (now - last_scale_time) >= cfg.scale_interval:
            cpu_usage = psutil.cpu_percent() if cfg.enable_resource_monitoring else 0

            # Introduce a small random jitter (0 - 5 seconds) to avoid thrashing
            time.sleep(random.uniform(0, 5))

            if cpu_usage < up_threshold and current_workers < cfg.max_workers:
                new_count = min(current_workers + cfg.scale_step, cfg.max_workers)
                concurrency_sema.release(new_count - current_workers)
                logging.info(
                    "Scaling up from %d to %d (cpu=%.1f%%)",
                    current_workers,
                    new_count,
                    cpu_usage,
                )
                current_workers = new_count
            elif cpu_usage > down_threshold and current_workers > 2:
                to_reduce = min(cfg.scale_step, current_workers - 2)
                for _ in range(to_reduce):
                    concurrency_sema.acquire()
                new_count = current_workers - to_reduce
                logging.info(
                    "Scaling down from %d to %d (cpu=%.1f%%)",
                    current_workers,
                    new_count,
                    cpu_usage,
                )
                current_workers = new_count

            last_scale_time = time.monotonic()

        # Periodically log progress
        if (time.monotonic() - last_log_time) > 30:
            logging.info(
                "Progress: %d/%d processed. Current concurrency=%d",
                processed_count,
                total,
                current_workers,
            )
            last_log_time = time.monotonic()

        # Check if queue is empty and no tasks remain
        if task_q.empty():
            # Attempt final flush
            if good_tickets_queue.qsize() == 0 and bad_tickets_queue.qsize() == 0:
                time.sleep(1)
                if (
                    task_q.empty()
                    and good_tickets_queue.empty()
                    and bad_tickets_queue.empty()
                ):
                    break

        time.sleep(0.2)

    pbar.close()
    executor.shutdown(wait=True)

    # Final flush
    final_good = []
    while not good_tickets_queue.empty():
        final_good.append(good_tickets_queue.get())
    if final_good:
        file_q.put((final_good, cfg.good_tickets_file))

    final_bad = []
    while not bad_tickets_queue.empty():
        final_bad.append(bad_tickets_queue.get())
    if final_bad:
        file_q.put((final_bad, cfg.bad_tickets_file))

    file_q.put(None)  # Signal writer to stop
