"""
Ticket Processing Application Entry Point (Updated)

Major changes in this version:
1. Advanced logging setup with a separate logger level for openai/urllib3 to reduce noisy POST lines.
2. Maintaining a 'setup_logging()' function that configures logging, including console/file handlers.
3. The rest of the logic remains similar, except we've improved docstrings.
4. No more direct reference to concurrency_sema._value in this file; concurrency logic is in pipeline.
"""

import argparse
import json
import logging
import psutil
import sys
import time
import threading
from multiprocessing import Process, Queue as MPQueue
from typing import List, Dict, Any, Generator

import openai
import urllib3

from config import Config
from circuit_breaker import CircuitBreaker
from token_bucket import TokenBucket
from io_utils import writer_process
from pipeline import process_tickets_with_scaling
from shared_state import (
    request_shutdown,
    is_shutdown_requested,
    toggle_pause_event,
)


def setup_logging() -> None:
    """
    Sets up global logging for the application. Includes:
      - StreamHandler for console output
      - FileHandler for 'app.log'
      - Lower log levels for 'openai' and 'urllib3' to suppress excessive request logs.
    """

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", mode="a", encoding="utf-8"),
        ]
    )

    # Reduce log noise from openai & urllib3
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _keyboard_listener():
    """
    Background thread that listens for space-bar presses to toggle the global pause event.
    It runs indefinitely until the main program exits.
    """
    # Reading from stdin in a blocking manner. If space is pressed, we toggle.
    while True:
        ch = sys.stdin.read(1)  # read one character
        if ch == ' ':
            logging.info("Space bar detected. Toggling pause state...")
            toggle_pause_event()
        # Could also watch for other keys if we want expansions (like 'q' for immediate quit).


def read_json_in_chunks(
    file_path: str,
    chunk_size: int = 130000,
    memory_limit_mb: int = 28000,
) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Memory-efficient JSON file reader that processes one JSON object per line (JSONL).
    Yields lists (chunks) of items. This helps handle large files with minimal memory usage.

    :param file_path: Path to the JSONL file
    :param chunk_size: Number of lines to accumulate before yielding
    :param memory_limit_mb: Soft memory usage limit in MB; if exceeded, yield early
    :return: Generator yielding lists of dicts
    """
    import psutil

    chunk = []
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.error("Failed to decode JSON line: %s", exc)
                continue

            chunk.append(item)
            if len(chunk) >= chunk_size:
                mem = psutil.virtual_memory()
                if mem.used / (1024**10) > memory_limit_mb:
                    logging.debug("Memory limit reached, yielding chunk early.")
                yield chunk
                chunk = []

    if chunk:
        yield chunk


def main():
    """
    Main function that sets up the environment, logging, configuration, and keyboard listener.
    Then processes the input file in chunks, sending them to 'process_tickets_with_scaling()'.

    Once all chunks are processed (or a shutdown signal is requested), it gracefully shuts down
    the writer process. Ctrl-C also triggers a graceful shutdown.
    """
    setup_logging()
    
    logging.info("Starting the Ticket Processing Application...")

    # Start a background thread for keyboard input (space bar) to toggle pause
    listener_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    listener_thread.start()

    parser = argparse.ArgumentParser(description="Ticket Processing App with GPT-based sanity checks")
    parser.add_argument("--input-file", help="Path to the JSON input file of tickets")
    parser.add_argument("--chunk-size", type=int, default=130000, help="How many tickets per chunk")
    parser.add_argument("--memory-limit-mb", type=int, default=20000, help="Memory usage threshold in MB")
    args = parser.parse_args()

    cfg = Config()
    if args.input_file:
        cfg.input_file = args.input_file

    # Initialize the circuit breaker & token bucket
    breaker = CircuitBreaker(cfg.error_threshold, cfg.cool_down_time)
    bucket = TokenBucket(cfg.token_bucket_capacity, cfg.token_refill_rate)

    # Setup the dedicated writer process (for writing good/bad/transaction logs)
    file_q = MPQueue()
    writer_p = Process(target=writer_process, args=(file_q,), daemon=True)
    writer_p.start()

    total_processed = 0

    try:
        for chunk_data in read_json_in_chunks(
            file_path=str(cfg.input_file),
            chunk_size=args.chunk_size,
            memory_limit_mb=args.memory_limit_mb,
        ):
            if is_shutdown_requested():
                logging.warning("Shutdown requested. Stopping further ingestion.")
                break

            raw_tickets = [json.dumps(item) for item in chunk_data]

            process_tickets_with_scaling(raw_tickets, cfg, file_q, breaker, bucket)
            total_processed += len(raw_tickets)

            logging.info(
                "Processed a chunk of %d tickets, total processed so far: %d",
                len(raw_tickets),
                total_processed,
            )

    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received. Requesting shutdown.")
        request_shutdown()

    finally:
        logging.info("Exiting main loop. Sending shutdown signal to writer process...")
        file_q.put(None)  # The only place we send None to kill the writer
        writer_p.join()
        logging.info("All chunks processed. Total tickets processed: %d", total_processed)


if __name__ == "__main__":
    main()