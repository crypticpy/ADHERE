"""
I/O Utilities Module

This module provides thread-safe file I/O operations for the ticket processing pipeline.
It handles atomic file writes and implements a dedicated writer process for handling
concurrent file operations safely.

Key features:
- Atomic file writing using temporary files
- JSONL format for efficient streaming of records
- Multiprocessing queue for thread-safe file operations
- Graceful handling of various data types (TicketSchema, dict, str)
- Error handling and logging for I/O operations
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple, Union
from multiprocessing import Queue as MPQueue

from models import TicketSchema


def write_batch_to_file(batch: List[Any], file_path: Path) -> None:
    """
    Writes a batch of items to a file in JSONL (JSON Lines) format using atomic operations.

    This function implements safe file writing by:
    1. Writing to a temporary file first
    2. Using atomic rename operation to replace the target file
    3. Cleaning up temporary files in case of errors

    Args:
        batch: List of items to write. Each item can be:
            - TicketSchema: Converted to dict using model_dump()
            - dict: Written directly
            - str: Wrapped in {"error": str} format
        file_path: Path where the JSONL file should be written

    Example:
        tickets = [
            TicketSchema(instruction="Reset password", ...),
            {"custom_field": "value"},
            "Error processing ticket"
        ]
        write_batch_to_file(tickets, Path("output.jsonl"))

        # Creates file with content like:
        # {"instruction": "Reset password", ...}
        # {"custom_field": "value"}
        # {"error": "Error processing ticket"}

    Note:
        - Empty batches are skipped
        - Uses UTF-8 encoding for universal character support
        - Implements atomic write operations for data safety
        - Handles cleanup of temporary files in error cases
    """
    if not batch:
        return
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            for item in batch:
                if isinstance(item, TicketSchema):
                    item = item.model_dump()
                elif isinstance(item, dict):
                    item = item
                elif isinstance(item, str):
                    item = {"error": item}
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(tmp_path, file_path)  # atomic on most POSIX systems
        logging.info("Wrote batch of %d items to %s", len(batch), file_path)
    except OSError as exc:
        logging.error("Error writing batch to %s: %s", file_path, exc, exc_info=True)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def writer_process(file_queue: MPQueue) -> None:
    """
    Dedicated process for handling file writing operations from a multiprocessing queue.

    This function runs in a separate process and:
    1. Continuously monitors a queue for write requests
    2. Processes batches of data as they arrive
    3. Handles graceful shutdown when receiving None

    Args:
        file_queue: Multiprocessing queue that receives tuples of (batch, path)
                   or None for shutdown

    Queue Input Format:
        - Normal operation: Tuple[List[Any], Path] for batch writing
        - Shutdown signal: None to stop the process

    Example:
        from multiprocessing import Process, Queue

        # Create queue and start writer process
        file_queue = Queue()
        writer = Process(target=writer_process, args=(file_queue,))
        writer.start()

        # Send data to be written
        batch = [{"field": "value"}]
        file_queue.put((batch, Path("output.jsonl")))

        # Shutdown writer process
        file_queue.put(None)
        writer.join()

    Note:
        - Thread-safe due to queue-based communication
        - Handles one batch at a time to maintain order
        - Implements graceful shutdown protocol
    """
    while True:
        data = file_queue.get()
        if data is None:
            break
        batch, path = data
        write_batch_to_file(batch, path)
