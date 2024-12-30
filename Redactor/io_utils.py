"""
io_utils.py

High-performance I/O utilities for processing JSONL files in a memory-efficient manner.
Provides robust chunk-based reading and buffered writing capabilities with comprehensive
error handling and performance monitoring.

Key Features:
    - Memory-efficient chunk-based file reading
    - Buffered writing for optimal I/O performance
    - Comprehensive error handling and recovery
    - Detailed performance metrics and logging
    - Auto-detection of file encoding
    - Progress tracking for long operations
    - File integrity verification

Performance Characteristics:
    - Optimal chunk sizes automatically calculated based on available memory
    - Buffered writing with configurable buffer sizes
    - Automatic handling of compressed files (.gz, .bz2, .xz)
    - UTF-8 encoding with fallback options

Usage:
    from io_utils import read_ticket_chunks, write_tickets, count_tickets
    
    # Reading in chunks
    for chunk in read_ticket_chunks('input.jsonl', chunk_size=1000):
        process_chunk(chunk)
    
    # Writing tickets
    write_tickets('output.jsonl', processed_tickets)
    
    # Counting tickets
    total_tickets = count_tickets('input.jsonl')

Dependencies:
    - python >= 3.8
    - tqdm >= 4.65.0
    - chardet >= 4.0.0

Author: [Your Name]
Last Modified: [Date]
"""

import os
import json
import logging
import gzip
import bz2
import lzma
import mmap
import chardet
from pathlib import Path
from typing import List, Dict, Generator, Union, Optional, BinaryIO
from datetime import datetime
from contextlib import contextmanager
import io
from tqdm import tqdm
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BUFFER_SIZE = 8192  # 8KB buffer size for reading
MAX_SAMPLE_SIZE = 1024 * 1024  # 1MB maximum sample size for encoding detection
DEFAULT_CHUNK_SIZE = 1000
COMPRESSION_EXTENSIONS = {
    '.gz': gzip.open,
    '.bz2': bz2.open,
    '.xz': lzma.open
}

class IOPerformanceMetrics:
    """
    Tracks I/O performance metrics for optimization and monitoring.
    
    Attributes:
        start_time (datetime): Operation start time
        bytes_read (int): Total bytes read
        bytes_written (int): Total bytes written
        read_operations (int): Number of read operations
        write_operations (int): Number of write operations
        errors (int): Number of I/O errors encountered
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.bytes_read = 0
        self.bytes_written = 0
        self.read_operations = 0
        self.write_operations = 0
        self.errors = 0
        
    def get_summary(self) -> Dict:
        """Returns a summary of I/O performance metrics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'duration_seconds': elapsed,
            'bytes_read': self.bytes_read,
            'bytes_written': self.bytes_written,
            'read_operations': self.read_operations,
            'write_operations': self.write_operations,
            'read_speed_mbps': (self.bytes_read / 1024 / 1024) / elapsed if elapsed > 0 else 0,
            'write_speed_mbps': (self.bytes_written / 1024 / 1024) / elapsed if elapsed > 0 else 0,
            'error_count': self.errors
        }

class FileHandler:
    """
    Handles file operations with automatic compression detection and encoding management.
    
    Features:
        - Automatic compression detection and handling
        - Encoding detection and validation
        - Buffered I/O operations
        - Error handling and recovery
    """
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.metrics = IOPerformanceMetrics()
        self._detect_compression()
        self._detect_encoding()
        
    def _detect_compression(self) -> None:
        """Detects file compression based on extension."""
        self.compression_opener = COMPRESSION_EXTENSIONS.get(
            self.file_path.suffix.lower(),
            open
        )
        
    def _detect_encoding(self) -> None:
        """Detects file encoding using chardet."""
        try:
            with open(self.file_path, 'rb') as f:
                raw_data = f.read(MAX_SAMPLE_SIZE)
                result = chardet.detect(raw_data)
                self.encoding = result['encoding'] or 'utf-8'
                self.encoding_confidence = result['confidence']
                
                if self.encoding_confidence < 0.9:
                    logger.warning(
                        f"Low confidence ({self.encoding_confidence:.2f}) in detected "
                        f"encoding: {self.encoding}. Falling back to UTF-8."
                    )
                    self.encoding = 'utf-8'
                    
        except Exception as e:
            logger.error(f"Error detecting file encoding: {str(e)}")
            self.encoding = 'utf-8'
            self.encoding_confidence = 0.0

@contextmanager
def smart_open(file_path: Union[str, Path], mode: str = 'r') -> Generator[BinaryIO, None, None]:
    """
    Context manager for file handling with automatic compression detection.
    
    Args:
        file_path: Path to the file
        mode: File opening mode ('r', 'w', 'a', etc.)
        
    Yields:
        File object suitable for the detected format
    """
    handler = FileHandler(file_path)
    
    try:
        with handler.compression_opener(file_path, mode=mode, encoding=handler.encoding) as f:
            yield f
    except Exception as e:
        logger.error(f"Error handling file {file_path}: {str(e)}")
        raise

def calculate_optimal_chunk_size() -> int:
    """
    Calculates optimal chunk size based on available system memory.
    
    Returns:
        int: Recommended chunk size for reading operations
    """
    available_memory = psutil.virtual_memory().available
    # Use 10% of available memory as maximum chunk size guide
    max_chunk_memory = available_memory * 0.1
    # Estimate memory per ticket (adjust based on your data)
    estimated_ticket_size = 1024  # 1KB per ticket estimate
    
    optimal_size = int(max_chunk_memory / estimated_ticket_size)
    return max(100, min(optimal_size, 10000))  # Keep between 100 and 10000

def read_ticket_chunks(
    file_path: Union[str, Path],
    chunk_size: Optional[int] = None,
    show_progress: bool = True
) -> Generator[List[Dict], None, None]:
    """
    Reads tickets from a JSONL file in memory-efficient chunks.
    
    Features:
        - Automatic chunk size optimization
        - Progress tracking
        - Comprehensive error handling
        - Performance monitoring
        - Memory usage optimization
    
    Args:
        file_path: Path to the input JSONL file
        chunk_size: Number of tickets per chunk (None for automatic optimization)
        show_progress: Whether to show progress bar
    
    Yields:
        List[Dict]: Chunks of ticket dictionaries
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        JSONDecodeError: If JSON parsing fails
        IOError: For general I/O errors
    """
    metrics = IOPerformanceMetrics()
    chunk_size = chunk_size or calculate_optimal_chunk_size()
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    logger.info(f"Starting to read {file_path} in chunks of {chunk_size}")
    chunk: List[Dict] = []
    total_size = file_path.stat().st_size
    
    try:
        with smart_open(file_path, 'r') as infile:
            pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                disable=not show_progress,
                desc="Reading tickets"
            )
            
            for line_num, line in enumerate(infile, 1):
                try:
                    line_size = len(line.encode('utf-8'))
                    metrics.bytes_read += line_size
                    metrics.read_operations += 1
                    pbar.update(line_size)
                    
                    if not line.strip():
                        continue
                        
                    ticket = json.loads(line)
                    chunk.append(ticket)
                    
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                        
                except json.JSONDecodeError as e:
                    metrics.errors += 1
                    logger.error(
                        f"Invalid JSON at line {line_num}: {str(e)}. "
                        f"Content: {line[:100]}..."
                    )
                    continue
                    
            if chunk:  # Yield remaining tickets
                yield chunk
                
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise
        
    finally:
        if show_progress:
            pbar.close()
            
        # Log performance metrics
        summary = metrics.get_summary()
        logger.info(
            f"File reading completed:\n"
            f"- Read {summary['bytes_read']/1024/1024:.2f}MB\n"
            f"- Average speed: {summary['read_speed_mbps']:.2f}MB/s\n"
            f"- Errors: {summary['error_count']}"
        )

def write_tickets(
    file_path: Union[str, Path],
    tickets: List[Dict],
    mode: str = 'a',
    buffer_size: int = BUFFER_SIZE
) -> None:
    """
    Writes tickets to a JSONL file with buffered I/O and progress tracking.
    
    Features:
        - Buffered writing for performance
        - Progress tracking
        - Error handling and recovery
        - Performance monitoring
    
    Args:
        file_path: Path to output file
        tickets: List of ticket dictionaries to write
        mode: File opening mode ('a' for append, 'w' for write/overwrite)
        buffer_size: Size of the write buffer in bytes
    
    Raises:
        IOError: If writing fails
        OSError: If file system errors occur
    """
    metrics = IOPerformanceMetrics()
    file_path = Path(file_path)
    
    try:
        with smart_open(file_path, mode) as outfile:
            buffer = io.StringIO()
            
            for ticket in tqdm(
                tickets,
                desc="Writing tickets",
                unit="ticket"
            ):
                try:
                    json_line = json.dumps(ticket) + '\n'
                    buffer.write(json_line)
                    metrics.write_operations += 1
                    
                    if buffer.tell() >= buffer_size:
                        buffer_content = buffer.getvalue()
                        outfile.write(buffer_content)
                        metrics.bytes_written += len(buffer_content.encode('utf-8'))
                        buffer = io.StringIO()
                        
                except Exception as e:
                    metrics.errors += 1
                    logger.error(f"Error writing ticket: {str(e)}")
                    continue
                    
            # Write remaining buffer content
            remaining = buffer.getvalue()
            if remaining:
                outfile.write(remaining)
                metrics.bytes_written += len(remaining.encode('utf-8'))
                
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {str(e)}")
        raise
        
    finally:
        # Log performance metrics
        summary = metrics.get_summary()
        logger.info(
            f"File writing completed:\n"
            f"- Wrote {summary['bytes_written']/1024/1024:.2f}MB\n"
            f"- Average speed: {summary['write_speed_mbps']:.2f}MB/s\n"
            f"- Errors: {summary['error_count']}"
        )

def count_tickets(file_path: Union[str, Path]) -> int:
    """
    Counts the total number of valid tickets in a JSONL file.
    
    Features:
        - Memory-efficient counting
        - Progress tracking
        - Invalid line detection
    
    Args:
        file_path: Path to the JSONL file
    
    Returns:
        int: Number of valid ticket entries
    
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    total = 0
    invalid = 0
    file_size = file_path.stat().st_size
    
    try:
        with smart_open(file_path, 'r') as infile:
            pbar = tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
                desc="Counting tickets"
            )
            
            for line in infile:
                line_size = len(line.encode('utf-8'))
                pbar.update(line_size)
                
                if line.strip():
                    try:
                        json.loads(line)
                        total += 1
                    except json.JSONDecodeError:
                        invalid += 1
                        
    except Exception as e:
        logger.error(f"Error counting tickets in {file_path}: {str(e)}")
        raise
        
    finally:
        pbar.close()
        
    logger.info(
        f"Ticket counting completed:\n"
        f"- Valid tickets: {total}\n"
        f"- Invalid lines: {invalid}"
    )
    
    return total

# Optional: Add file integrity verification
def verify_file_integrity(file_path: Union[str, Path]) -> bool:
    """
    Verifies the integrity of a JSONL file by checking all entries.
    
    Args:
        file_path: Path to the JSONL file
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        with smart_open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON at line {line_num}: {str(e)}")
                        return False
        return True
    except Exception as e:
        logger.error(f"Error verifying file {file_path}: {str(e)}")
        return False
    