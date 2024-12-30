"""
Ticket Processing Pipeline Module (Updated)

Changes in this version:
1. We removed references to redaction and PII handling (previously used `redact_pii`). 
2. We relocated the GPT-based sanity check logic to a new file, sanity_check.py.
3. We import `run_sanity_check` from `sanity_check.py`.
4. The rest of the concurrency, worker draining, and scaling logic remains the same.
"""

import json
import logging
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from multiprocessing import Queue as MPQueue
import psutil
from typing import Tuple, Optional, List
import threading
import sys
from collections import deque

# Project imports
from config import Config
from models import TicketSchema, TicketParsedSchema
from llm_api import retry_call_llm_parse, parse_response
from utils import fill_missing_fields
from circuit_breaker import CircuitBreaker
from token_bucket import TokenBucket
from io_utils import writer_process
from shared_state import (
    error_state,
    good_tickets_queue,
    bad_tickets_queue,
    is_shutdown_requested,
    request_shutdown,
    is_pause_event_set,
)
from openai import APIError, RateLimitError

# Import the new sanity checker
from sanity_check import run_sanity_check

################################################################################
# EMA CONFIGURATION & BAD TICKET THRESHOLDS
################################################################################
EMA_ALPHA = 0.05         # Smoothing factor for exponential moving average of bad ratio
BAD_RATIO_WARNING = 0.90 # If bad_ema >= 50%, log a warning
BAD_RATIO_STOP = 99.0    # If bad_ema >= 80%, we pause pipeline and run a sanity check

################################################################################
# SENTINELS FOR WORKER CONTROL
################################################################################
STOP_SENTINEL = None  # Hard stop sentinel

################################################################################
# RING BUFFER FOR RECENT BAD TICKETS
################################################################################
RECENT_BAD_MAX = 20
recent_bad_buffer = deque(maxlen=RECENT_BAD_MAX)
recent_bad_lock = threading.Lock()

################################################################################
# TICKET TIMEOUT
################################################################################
TICKET_TIMEOUT_SECS = 180  # max time to process one ticket (in seconds)


def _process_ticket(
    ticket_json: str,
    index: int,
    cfg: Config,
    bucket: TokenBucket,
    circuit_breaker: CircuitBreaker,
    file_q: MPQueue
) -> Tuple[Optional[TicketSchema], Optional[dict]]:
    """
    Process a single ticket through the transformation pipeline.

    Steps:
      1. Parse the incoming JSON ticket data.
      2. fill_missing_fields to ensure the major fields exist.
      3. Rate limit check via token bucket.
      4. Circuit breaker check.
      5. LLM call for structured parsing, with raw request/response logged.
      6. Validate final TicketSchema or return a "bad" dict if invalid/bad.
    
    Returns:
      (TicketSchema, None) if "good"
      (None, dict) if "bad", where 'dict' contains raw_ticket plus error info.
    """

    # Step 1: Parse the JSON
    try:
        data = json.loads(ticket_json)
    except json.JSONDecodeError as exc:
        return None, {
            "raw_ticket": ticket_json,
            "error": f"Invalid JSON: {exc}"
        }

    # Step 2: Fill missing fields (no PII redaction anymore)
    filled = fill_missing_fields(data)

    # Step 3: Rate limiting
    while not bucket.consume(1):
        time.sleep(1)
        if is_shutdown_requested():
            return None, {
                "raw_ticket": ticket_json,
                "error": "Shutdown requested mid-rate-limit"
            }

    # Step 4: Circuit breaker check
    if not circuit_breaker.allow_request():
        return None, {
            "raw_ticket": ticket_json,
            "error": "Circuit breaker open (requests not allowed)"
        }

    # Step 5: Convert to JSON string for LLM call
    raw_str = json.dumps(filled, ensure_ascii=False, indent=None, separators=(",", ":"))

    try:
        # LLM parse with retry
        response = retry_call_llm_parse(raw_str, cfg)

        # Log raw LLM transaction
        transaction_data = {
            "ticket_json": ticket_json,
            "request_payload": raw_str,
            "raw_response": str(response),
        }
        file_q.put(([transaction_data], cfg.raw_llm_transactions_file))

        parsed_resp = parse_response(response)
        if not parsed_resp:
            return None, {
                "raw_ticket": ticket_json,
                "error": "Failed to parse LLM response"
            }

    except (RateLimitError, APIError) as exc:
        circuit_breaker.record_failure()
        error_state.increment()
        return None, {
            "raw_ticket": ticket_json,
            "error": f"API Error (final): {exc}"
        }
    except Exception as exc:
        # Catch-all for unexpected errors
        circuit_breaker.record_failure()
        error_state.increment()
        return None, {
            "raw_ticket": ticket_json,
            "error": f"Worker {index} unexpected final fail: {exc}"
        }

    # If LLM explicitly says it's "bad"
    if parsed_resp.status.lower() == "bad":
        return None, {
            "raw_ticket": ticket_json,
            "error": parsed_resp.data if parsed_resp.data else "LLM says BAD with no reason"
        }

    # Otherwise "good." Validate TicketSchema
    if isinstance(parsed_resp.data, TicketSchema):
        ticket = parsed_resp.data
    elif isinstance(parsed_resp.data, dict):
        try:
            ticket = TicketSchema(**parsed_resp.data)
        except Exception:
            return None, {
                "raw_ticket": ticket_json,
                "error": "TicketSchema validation error"
            }
    else:
        return None, {
            "raw_ticket": ticket_json,
            "error": "No parsed data in LLM response"
        }

    # Check minimal word counts
    instr_words = re.findall(r"\b\w+\b", ticket.instruction or "")
    resp_words = re.findall(r"\b\w+\b", ticket.response or "")
    if len(instr_words) < cfg.min_word_count or len(resp_words) < cfg.min_word_count:
        return None, {
            "raw_ticket": ticket_json,
            "error": "Instruction or response too short"
        }

    return ticket, None


def worker_task(
    ticket_data: str,
    index: int,
    cfg: Config,
    bucket: TokenBucket,
    circuit_breaker: CircuitBreaker,
    file_q: MPQueue
) -> Tuple[Optional[TicketSchema], Optional[dict]]:
    """
    Wraps _process_ticket with a time-limited approach. 
    If the ticket can't be processed within TICKET_TIMEOUT_SECS, 
    we treat it as "bad" to avoid indefinite blocking.
    """
    start_time = time.time()
    result, err_obj = _process_ticket(
        ticket_json=ticket_data,
        index=index,
        cfg=cfg,
        bucket=bucket,
        circuit_breaker=circuit_breaker,
        file_q=file_q
    )
    duration = time.time() - start_time
    if duration > TICKET_TIMEOUT_SECS:
        return None, {
            "raw_ticket": ticket_data,
            "error": f"Ticket timed out after {int(duration)}s"
        }
    return result, err_obj


def worker(
    task_q: Queue,
    cfg: Config,
    index: int,
    bucket: TokenBucket,
    circuit_breaker: CircuitBreaker,
    concurrency_sema: threading.Semaphore,
    file_q: MPQueue
) -> None:
    """
    Worker thread function that processes tickets from a shared queue.

    - Respects a global pause_event. If paused, the worker waits until unpaused.
    - Checks for global shutdown or STOP_SENTINEL.
    - concurrency_sema controls the actual concurrency (slots available).

    We call worker_task(...) to process each ticket with a time limit.
    """
    logging.info("Worker %d started.", index)

    while True:
        if is_shutdown_requested():
            logging.info("Worker %d sees shutdown_requested => exiting immediately.", index)
            break

        # If paused, wait in small increments until unpaused or shutdown
        while is_pause_event_set():
            if is_shutdown_requested():
                break
            time.sleep(0.2)
        if is_shutdown_requested():
            break

        # Acquire concurrency
        acquired = concurrency_sema.acquire(timeout=0.5)
        if not acquired:
            # Could not acquire in 0.5s, re-check shutdown
            if is_shutdown_requested():
                break
            continue

        try:
            # Try to get a ticket
            try:
                ticket_data = task_q.get(timeout=0.2)
            except Empty:
                concurrency_sema.release()
                continue

            if ticket_data is STOP_SENTINEL:
                task_q.task_done()
                concurrency_sema.release()
                logging.info("Worker %d got STOP_SENTINEL => exiting", index)
                break

            try:
                result, err_obj = worker_task(
                    ticket_data,
                    index,
                    cfg,
                    bucket,
                    circuit_breaker,
                    file_q
                )
                if result:
                    good_tickets_queue.put(result)
                elif err_obj:
                    bad_tickets_queue.put(err_obj)
                    with recent_bad_lock:
                        recent_bad_buffer.append(err_obj)
            finally:
                task_q.task_done()
                concurrency_sema.release()

        except Exception as e:
            logging.error("Worker %d encountered unexpected error: %s", index, e, exc_info=True)
            break

    logging.info("Worker %d exiting.", index)


def wait_for_workers_to_drain(
    task_q: Queue,
    concurrency_sema: threading.Semaphore,
    active_workers: int
) -> None:
    """
    Blocks until:
      1) The task queue is empty
      2) All concurrency slots are free (i.e., no tickets in flight).

    :param task_q: The queue of tasks.
    :param concurrency_sema: The semaphore controlling concurrency.
    :param active_workers: Current concurrency setting, 
                          so we know how many slots should be free for "idle".
    """
    while True:
        if task_q.empty():
            if concurrency_sema._value == active_workers:
                break
        time.sleep(0.2)


def process_tickets_with_scaling(
    tickets: List[str],
    cfg: Config,
    file_q: MPQueue,
    circuit_breaker: CircuitBreaker,
    bucket: TokenBucket,
) -> None:
    """
    Main orchestrator for ticket processing, including:
      - concurrency scaling
      - circuit breaker & error threshold
      - exponential moving average for bad tickets
      - GPT-based sanity check if too many tickets are flagged "bad"
      - pause/resume logic
      - graceful stop

    :param tickets: List of raw JSON strings representing the tickets.
    :param cfg: Configuration object.
    :param file_q: Queue for writer process.
    :param circuit_breaker: Circuit breaker instance.
    :param bucket: TokenBucket instance.
    """
    logging.info("Starting ticket processing with adaptive concurrency for %d tickets.", len(tickets))

    task_q = Queue()
    for tk in tickets:
        task_q.put(tk)

    max_possible_workers = cfg.max_workers
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=max_possible_workers)

    concurrency_sema = threading.Semaphore(0)  # start concurrency at 0, ramp up

    # Launch the worker threads
    for i in range(max_possible_workers):
        executor.submit(
            worker,
            task_q,
            cfg,
            i,
            bucket,
            circuit_breaker,
            concurrency_sema,
            file_q
        )

    total = len(tickets)
    processed_count = 0
    good_count = 0
    bad_count = 0
    bad_ema = 0.0  # Exponential moving average for "bad" ratio

    # Start concurrency at cfg.initial_workers
    current_workers = 0

    def warmup_workers():
        nonlocal current_workers
        needed = cfg.initial_workers - current_workers
        if needed > 0:
            for _ in range(needed):
                concurrency_sema.release()
            current_workers += needed
            time.sleep(0.5)

    warmup_workers()

    from tqdm import tqdm
    pbar = tqdm(total=total, desc="Processing Tickets")
    last_scale_time = time.monotonic()
    last_log_time = time.monotonic()
    force_stop = False

    while True:
        if is_shutdown_requested():
            logging.warning("Main loop sees global shutdown => force_stop.")
            force_stop = True

        # Drain good tickets
        while not good_tickets_queue.empty():
            good_batch = []
            while not good_tickets_queue.empty() and len(good_batch) < cfg.batch_size:
                good_batch.append(good_tickets_queue.get())
            if good_batch:
                processed_count += len(good_batch)
                good_count += len(good_batch)
                pbar.update(len(good_batch))
                file_q.put((good_batch, cfg.good_tickets_file))
                # bad_ema update with 0.0 for each good
                for _ in good_batch:
                    bad_ema = EMA_ALPHA * 0.0 + (1 - EMA_ALPHA) * bad_ema

        # Drain bad tickets
        while not bad_tickets_queue.empty():
            bad_batch = []
            while not bad_tickets_queue.empty() and len(bad_batch) < cfg.batch_size:
                bad_batch.append(bad_tickets_queue.get())
            if bad_batch:
                processed_count += len(bad_batch)
                bad_count += len(bad_batch)
                pbar.update(len(bad_batch))
                file_q.put((bad_batch, cfg.bad_tickets_file))
                # bad_ema update with 1.0 for each bad
                for _ in bad_batch:
                    bad_ema = EMA_ALPHA * 1.0 + (1 - EMA_ALPHA) * bad_ema

        # Check thresholds
        if bad_ema >= BAD_RATIO_WARNING and not force_stop:
            logging.warning(
                "High bad ticket EMA=%.2f. Good=%d, Bad=%d so far.",
                bad_ema,
                good_count,
                bad_count
            )

        if bad_ema >= BAD_RATIO_STOP and not force_stop:
            logging.error(
                "Bad ticket EMA=%.2f >= %.2f, pausing to run sanity check.",
                bad_ema,
                BAD_RATIO_STOP
            )
            logging.warning("Waiting for workers to drain all tasks before sanity check...")
            wait_for_workers_to_drain(task_q, concurrency_sema, current_workers)

            # Gather a copy of recent bad tickets for sanity check
            with recent_bad_lock:
                samples = list(recent_bad_buffer)

            decision = run_sanity_check(cfg, samples)
            if decision == "stop":
                logging.error("GPT-4 sanity check says STOP. Forcing stop.")
                force_stop = True
            else:
                logging.warning("GPT-4 sanity check says CONTINUE. Resetting thresholds.")
                bad_ema = 0.0
                error_state.reset()
                circuit_breaker.reset()
                # Optionally reduce concurrency
                new_count = max(1, current_workers // 2)
                to_reduce = current_workers - new_count
                for _ in range(to_reduce):
                    concurrency_sema.acquire()
                current_workers = new_count
                # Then re-warm
                warmup_workers()
                logging.warning("Resumed pipeline after sanity check with concurrency=%d", current_workers)

        if force_stop:
            logging.warning("Forcing stop. Inserting STOP_SENTINEL for all workers.")
            for _ in range(max_possible_workers):
                task_q.put(STOP_SENTINEL)
            break

        # Adaptive scaling (CPU-based)
        now = time.monotonic()
        if not force_stop and (now - last_scale_time) >= cfg.scale_interval:
            cpu_usage = psutil.cpu_percent() if cfg.enable_resource_monitoring else 0
            time.sleep(random.uniform(0, 2))
            if cpu_usage < cfg.cpu_scale_up_threshold and current_workers < cfg.max_workers:
                new_count = min(current_workers + cfg.scale_step, cfg.max_workers)
                for _ in range(new_count - current_workers):
                    concurrency_sema.release()
                logging.info(
                    "Scaling up from %d to %d (cpu=%.1f%%)",
                    current_workers,
                    new_count,
                    cpu_usage
                )
                current_workers = new_count
            elif cpu_usage > cfg.cpu_scale_down_threshold and current_workers > 2:
                to_reduce = min(cfg.scale_step, current_workers - 2)
                for _ in range(to_reduce):
                    concurrency_sema.acquire()
                new_count = current_workers - to_reduce
                logging.info(
                    "Scaling down from %d to %d (cpu=%.1f%%)",
                    current_workers,
                    new_count,
                    cpu_usage
                )
                current_workers = new_count

            last_scale_time = time.monotonic()

        # Periodic progress logging
        if (time.monotonic() - last_log_time) > 30:
            logging.info(
                "Progress: %d/%d processed. Good=%d, Bad=%d, Concurrency=%d, BadEMA=%.2f",
                processed_count,
                total,
                good_count,
                bad_count,
                current_workers,
                bad_ema
            )
            last_log_time = time.monotonic()

        # Are we done?
        if task_q.empty() and good_tickets_queue.empty() and bad_tickets_queue.empty():
            if concurrency_sema._value == current_workers:
                break
        time.sleep(0.2)

    pbar.close()
    executor.shutdown(wait=True, cancel_futures=True)

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

    logging.info(
        "Done processing. Good=%d, Bad=%d, total processed=%d, final badEMA=%.2f",
        good_count,
        bad_count,
        processed_count,
        bad_ema
    )
