"""
I/O Utilities Module

This module provides thread-safe file I/O operations for the ticket processing pipeline.
It handles atomic file writes and implements a dedicated writer process for handling
concurrent file operations safely.
"""

import json
import logging
from pathlib import Path
from typing import Any, List
from multiprocessing import Queue as MPQueue

from models import TicketSchema


def write_batch_to_file(batch: List[Any], file_path: Path) -> None:
    """
    Writes a batch of items to a file in JSONL (JSON Lines) format, appending instead of overwriting.
    Each item is written as one JSON object per line.
    """
    if not batch:
        return

    try:
        with open(file_path, "a", encoding="utf-8") as fh:
            for item in batch:
                if isinstance(item, TicketSchema):
                    # If we have an actual TicketSchema, convert to dict
                    item = item.model_dump()
                elif isinstance(item, dict):
                    item = item
                elif isinstance(item, str):
                    # If there's a raw string error, store it as {'error': "..."}
                    item = {"error": item}

                line = json.dumps(item, ensure_ascii=False)
                fh.write(line + "\n")

        logging.info("Appended %d items to %s", len(batch), file_path)
    except OSError as exc:
        logging.error("Error writing batch to %s: %s", file_path, exc, exc_info=True)


def writer_process(file_queue: MPQueue) -> None:
    """
    Dedicated process for handling file writing operations from a multiprocessing queue.
    This ensures safe parallel writes by serializing them in a single process.
    """
    while True:
        data = file_queue.get()
        if data is None:
            break
        batch, path = data
        write_batch_to_file(batch, path)