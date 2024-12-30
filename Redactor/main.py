"""
main.py

A high-performance, production-ready ticket anonymization system using the Presidio framework,
with adjusted logging levels, smaller chunks, and 10 workers for concurrency.
"""

import os
import sys
import json
import time
import signal
import logging
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import queue

from tqdm.auto import tqdm
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Raise Presidio's log level to WARNING to reduce spam
logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)
logging.getLogger("presidio-anonymizer").setLevel(logging.WARNING)

from recognizers import get_custom_recognizers
from anonymize import create_anonymizer_operators, anonymize_text
from io_utils import read_ticket_chunks, write_tickets, count_tickets

# Configure logging for our own app
logging.basicConfig(
    level=os.getenv('ANONYMIZER_LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anonymizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Adjust concurrency and chunk settings here
CONFIG = {
    'chunk_size': 10000,
    'max_workers': 10,  
    'monitor_interval': 1,
    'write_buffer_size': 1000,
    'memory_threshold': 85,
    'cpu_threshold': 90,
}

class PerformanceMetrics:
    def __init__(self):
        self.start_time = datetime.now()
        self.processed_count = 0
        self.error_count = 0
        self.processing_times: List[float] = []
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []

    def add_processing_time(self, seconds: float):
        self.processing_times.append(seconds)

    def add_resource_sample(self, cpu_percent: float, memory_percent: float):
        self.cpu_samples.append(cpu_percent)
        self.memory_samples.append(memory_percent)

    @property
    def average_processing_time(self) -> float:
        return (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times
            else 0
        )

    @property
    def tickets_per_second(self) -> float:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.processed_count / elapsed if elapsed > 0 else 0

    def get_summary(self) -> Dict[str, float]:
        return {
            'total_processed': self.processed_count,
            'error_rate': (self.error_count / self.processed_count)
            if self.processed_count
            else 0,
            'avg_processing_time': self.average_processing_time,
            'tickets_per_second': self.tickets_per_second,
            'avg_cpu_usage': (sum(self.cpu_samples) / len(self.cpu_samples))
            if self.cpu_samples
            else 0,
            'avg_memory_usage': (sum(self.memory_samples) / len(self.memory_samples))
            if self.memory_samples
            else 0,
            'peak_memory_usage': max(self.memory_samples) if self.memory_samples else 0,
            'peak_cpu_usage': max(self.cpu_samples) if self.cpu_samples else 0,
        }

class ResourceMonitor:
    def __init__(self, interval: int = 1):
        self.interval = interval
        self._stop = mp.Event()
        self.metrics = PerformanceMetrics()

    def start(self) -> None:
        self.process = mp.Process(target=self._monitor)
        self.process.start()
        logger.info("Resource monitor started")

    def stop(self) -> None:
        self._stop.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
        logger.info("Resource monitor stopped")

    def _monitor(self) -> None:
        while not self._stop.is_set():
            try:
                cpu_percent = psutil.cpu_percent(interval=self.interval)
                memory = psutil.virtual_memory()
                self.metrics.add_resource_sample(cpu_percent, memory.percent)
                logger.info(
                    f"System Status - CPU: {cpu_percent}% | "
                    f"Memory: {memory.percent}% | "
                    f"Available Memory: {memory.available/1024/1024:.0f}MB"
                )
                if memory.percent > CONFIG['memory_threshold']:
                    logger.warning(
                        f"High memory usage detected: {memory.percent}%. "
                        "Consider reducing chunk size."
                    )
                if cpu_percent > CONFIG['cpu_threshold']:
                    logger.warning(
                        f"High CPU usage detected: {cpu_percent}%. "
                        "Consider reducing worker count."
                    )
            except Exception as e:
                logger.error(f"Error in resource monitor: {str(e)}", exc_info=True)
                time.sleep(self.interval)

def initialize_presidio_engines() -> Tuple[AnalyzerEngine, AnonymizerEngine, Dict]:
    try:
        logger.info("Initializing Presidio engines in worker...")

        # Initialize AnalyzerEngine with default settings
        analyzer = AnalyzerEngine(supported_languages=["en"])
        anonymizer = AnonymizerEngine()

        # No need to remove recognizers; use Presidio as-is

        # Create anonymizer operators
        operators = create_anonymizer_operators()

        logger.info("Presidio engines initialized successfully (default recognizers)")
        return analyzer, anonymizer, operators

    except Exception as e:
        logger.error("Failed to initialize Presidio engines", exc_info=True)
        raise RuntimeError("Engine initialization failed") from e
    

def process_chunk_worker(chunk: List[Dict]) -> List[Dict]:
    """
    A top-level worker function that re-initializes Presidio engines for each worker.
    This avoids pickling issues with the engines.
    """
    analyzer, anonymizer, operators = initialize_presidio_engines()

    processed_tickets = []
    for ticket in chunk:
        if not isinstance(ticket, dict):
            # Skip or raise exception if line is not a dict.
            continue

        anonymized_ticket = {}
        for key, value in ticket.items():
            if isinstance(value, str):
                try:
                    redacted_value = anonymize_text(
                        text=value,
                        analyzer=analyzer,
                        anonymizer=anonymizer,
                        operators=operators
                    )
                    anonymized_ticket[key] = redacted_value
                except Exception as e:
                    # If anonymization fails for one field,
                    # you can skip the field or the entire ticket.
                    anonymized_ticket[key] = f"<REDACTION_ERROR: {e}>"
            else:
                anonymized_ticket[key] = value

        processed_tickets.append(anonymized_ticket)

    return processed_tickets

class AnonymizationPipeline:
    """
    Main pipeline in the parent process. Uses top-level worker function for concurrency.
    """

    def __init__(
        self,
        input_file: str,
        output_file: str,
        chunk_size: Optional[int] = None,
        max_workers: Optional[int] = None
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.chunk_size = chunk_size or CONFIG['chunk_size']
        self.max_workers = max_workers or CONFIG['max_workers']
        self.metrics = PerformanceMetrics()
        self.monitor = ResourceMonitor(CONFIG['monitor_interval'])

        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        if self.output_file.exists():
            logger.warning(f"Output file {self.output_file} will be overwritten")

    def optimize_chunk_size(self) -> int:
        available_memory = psutil.virtual_memory().available
        total_memory = psutil.virtual_memory().total
        memory_ratio = available_memory / total_memory
        if memory_ratio < 0.2:
            return max(10, self.chunk_size // 2)
        elif memory_ratio > 0.6:
            return min(5000, self.chunk_size * 2)
        return self.chunk_size

    def run(self) -> None:
        start_time = datetime.now()
        logger.info(f"Starting anonymization pipeline at {start_time}")

        pbar = None
        try:
            self.monitor.start()

            total_tickets = count_tickets(self.input_file)
            logger.info(f"Found {total_tickets} tickets to process")

            if total_tickets == 0:
                logger.warning("No tickets found in input file")
                return

            pbar = tqdm(
                total=total_tickets,
                desc='Processing Tickets',
                unit='ticket',
                dynamic_ncols=True
            )

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                write_buffer = []

                for chunk_num, chunk in enumerate(
                    read_ticket_chunks(
                        self.input_file,
                        chunk_size=self.optimize_chunk_size(),
                        show_progress=False
                    )
                ):
                    chunk_start = time.time()

                    # Submit chunk to the top-level worker function.
                    future = executor.submit(process_chunk_worker, chunk)
                    futures[future] = chunk_num

                    done_futures = []
                    for completed in as_completed(futures):
                        done_futures.append(completed)
                        try:
                            processed_chunk = completed.result()
                            completed_chunk_num = futures[completed]
                            del futures[completed]

                            # Buffer results before writing
                            write_buffer.extend(processed_chunk)
                            if len(write_buffer) >= CONFIG['write_buffer_size']:
                                write_tickets(self.output_file, write_buffer)
                                write_buffer.clear()

                            pbar.update(len(processed_chunk))
                            self.metrics.processed_count += len(processed_chunk)

                            chunk_time = time.time() - chunk_start
                            logger.info(
                                f"Chunk {completed_chunk_num} completed: "
                                f"{len(processed_chunk)} tickets in {chunk_time:.2f}s "
                                f"({len(processed_chunk)/chunk_time:.1f} tickets/s)"
                            )

                        except Exception as e:
                            self.metrics.error_count += 1
                            logger.error(
                                f"Error processing chunk {futures[completed]}: {str(e)}",
                                exc_info=True
                            )

                    # Update progress bar
                    pbar.set_postfix({
                        'Speed': f"{self.metrics.tickets_per_second:.1f} t/s",
                        'CPU': f"{psutil.cpu_percent()}%",
                        'Mem': f"{psutil.virtual_memory().percent}%",
                        'Errors': self.metrics.error_count
                    })

                # Write any remaining data in the buffer
                if write_buffer:
                    write_tickets(self.output_file, write_buffer)

        except Exception as e:
            logger.error("Pipeline execution failed", exc_info=True)
            raise RuntimeError("Pipeline execution failed") from e
        finally:
            if pbar:
                pbar.close()
            self.monitor.stop()

            elapsed_time = datetime.now() - start_time
            metrics_summary = self.metrics.get_summary()

            logger.info(
                f"\nProcessing completed in {elapsed_time}\n"
                f"Total processed: {metrics_summary['total_processed']}\n"
                f"Processing rate: {metrics_summary['tickets_per_second']:.1f} tickets/s\n"
                f"Error rate: {metrics_summary['error_rate']*100:.2f}%\n"
                f"Average processing time: {metrics_summary['avg_processing_time']:.3f}s/ticket\n"
                f"Peak CPU usage: {metrics_summary['peak_cpu_usage']:.1f}%\n"
                f"Peak memory usage: {metrics_summary['peak_memory_usage']:.1f}%"
            )

def main():
    try:
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        pipeline = AnonymizationPipeline(
            input_file='reformatted_tickets.jsonl',
            output_file='anonymized_tickets.jsonl'
        )
        pipeline.run()

    except Exception as e:
        logger.error("Fatal error in main process", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
