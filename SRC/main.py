"""
Ticket Processing Application Entry Point

This module serves as the main entry point for the ticket processing application.
It handles:
1. Command-line argument parsing
2. Memory-efficient data loading
3. Process management and graceful shutdown
4. Chunk-based processing for large datasets

Usage Examples:
    # Process tickets using OpenAI
    python main.py --input-file tickets.json --provider openai

    # Process large file with custom chunk size
    python main.py --input-file large_tickets.json --chunk-size 1000

    # Process with memory limit
    python main.py --input-file tickets.json --memory-limit-mb 2048

The application supports graceful shutdown via SIGTERM/SIGINT (Ctrl+C),
ensuring that in-progress work is completed before exit.
"""

import argparse
import json
import logging
import signal
import psutil
from multiprocessing import Process, Queue as MPQueue
from typing import List, Dict, Any, Generator

from config import Config
from circuit_breaker import CircuitBreaker
from token_bucket import TokenBucket
from io_utils import writer_process
from pipeline import process_tickets_with_scaling

# Global flag for coordinating graceful shutdown
shutdown_requested = False


def handle_signal(signum, frame):
    """
    Signal handler for graceful shutdown coordination.

    This handler:
    1. Sets global shutdown flag
    2. Allows current processing to complete
    3. Prevents new chunk processing

    Handles:
        - SIGTERM: Termination request
        - SIGINT: Keyboard interrupt (Ctrl+C)
    """
    global shutdown_requested
    logging.warning("Shutdown requested (signal: %s)", signum)
    shutdown_requested = True


def read_json_in_chunks(
    file_path: str, chunk_size: int = 5000, memory_limit_mb: int = 4096
) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Memory-efficient JSON file reader that yields data in manageable chunks.

    This generator function implements smart chunking that considers both:
    1. Maximum chunk size (for processing efficiency)
    2. Memory usage limits (to prevent OOM errors)

    Args:
        file_path: Path to JSON file containing ticket data
        chunk_size: Maximum number of items per chunk
        memory_limit_mb: Memory usage threshold in MB

    Yields:
        List[Dict[str, Any]]: Chunks of parsed JSON data
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            logging.error("Failed to decode JSON file: %s", exc)
            return

    chunk = []
    for item in data:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            mem = psutil.virtual_memory()
            if mem.used / (1024**2) > memory_limit_mb:
                # Debug log to indicate forced yield due to memory constraints
                logging.debug("Memory limit reached, yielding chunk early.")
                yield chunk
                chunk = []
            else:
                yield chunk
                chunk = []

    if chunk:
        yield chunk


def main():
    """
    Main application entry point and orchestrator.

    This function:
    1. Sets up logging and signal handlers
    2. Parses command-line arguments
    3. Initializes system components:
       - Configuration
       - Circuit Breaker
       - Token Bucket
       - File Writer Process
    4. Manages the processing pipeline:
       - Chunk-based file reading
       - Ticket processing
       - Progress tracking

    Note:
        The application can be safely interrupted with Ctrl+C,
        which triggers a graceful shutdown sequence.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    parser = argparse.ArgumentParser(description="Ticket Processing App")
    parser.add_argument("--input-file", help="Path to the JSON input file of tickets")
    parser.add_argument("--provider", help="openai or azure")
    parser.add_argument(
        "--chunk-size", type=int, default=5000, help="How many tickets per chunk"
    )
    parser.add_argument(
        "--memory-limit-mb", type=int, default=4096, help="Memory usage threshold in MB"
    )
    args = parser.parse_args()

    cfg = Config()
    if args.input_file:
        cfg.input_file = args.input_file
    if args.provider:
        cfg.provider = args.provider

    # Extra debug message for developer awareness
    logging.debug("Configuration loaded: %s", cfg)

    breaker = CircuitBreaker(cfg.error_threshold, cfg.cool_down_time)
    bucket = TokenBucket(cfg.token_bucket_capacity, cfg.token_refill_rate)

    # A separate process for writing batches to disk
    file_q = MPQueue()
    writer_p = Process(target=writer_process, args=(file_q,), daemon=True)
    writer_p.start()

    total_processed = 0

    for chunk_data in read_json_in_chunks(
        file_path=str(cfg.input_file),
        chunk_size=args.chunk_size,
        memory_limit_mb=args.memory_limit_mb,
    ):
        if shutdown_requested:
            logging.warning("Shutdown requested. Stopping further ingestion.")
            break

        # Convert each item to a JSON string
        raw_tickets = [json.dumps(item) for item in chunk_data]

        process_tickets_with_scaling(raw_tickets, cfg, file_q, breaker, bucket)
        total_processed += len(raw_tickets)
        logging.info(
            "Processed a chunk of %d tickets, total processed so far: %d",
            len(raw_tickets),
            total_processed,
        )

    writer_p.join()
    logging.info("All chunks processed. Total tickets processed: %d", total_processed)


if __name__ == "__main__":
    main()
