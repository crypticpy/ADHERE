"""
Shared State Management Module

This module manages shared state across multiple processes and threads in the ticket
processing pipeline. It provides thread-safe counters and queues for:
1. Error tracking and cooldown management
2. Ticket processing results routing

The shared state pattern is crucial for maintaining consistency in a concurrent
processing environment, ensuring that all workers have access to the same
state information without race conditions.
"""

import threading
from dataclasses import dataclass, field
from queue import Queue


@dataclass
class ErrorState:
    """
    Thread-safe error counter for managing cooldown periods in the processing pipeline.

    This class tracks the number of errors that have occurred during ticket processing,
    helping to implement cooldown logic when too many errors occur. It uses a lock to
    ensure thread-safe operations in a concurrent environment.

    Attributes:
        error_count (int): Number of errors that have occurred
        lock (threading.Lock): Thread lock for safe concurrent access

    Example:
        error_tracker = ErrorState()

        # When an error occurs
        error_tracker.increment()

        # After successful recovery
        error_tracker.reset()

        # Check current error count (thread-safe)
        with error_tracker.lock:
            current_errors = error_tracker.error_count
    """

    error_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self) -> None:
        """
        Safely increments the error counter by 1.

        This method is thread-safe and should be called whenever an error
        occurs during ticket processing. The lock ensures that concurrent
        increments don't result in lost updates.
        """
        with self.lock:
            self.error_count += 1

    def reset(self) -> None:
        """
        Safely resets the error counter to 0.

        This method is thread-safe and should be called when the system
        has recovered from errors and normal operation can resume.
        """
        with self.lock:
            self.error_count = 0


# Global shared state instances
error_state = ErrorState()  # Tracks error state across all workers

# Thread-safe queues for routing processed tickets
good_tickets_queue = Queue()  # Successfully processed tickets
bad_tickets_queue = Queue()  # Failed or invalid tickets
