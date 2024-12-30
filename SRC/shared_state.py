"""
Shared State Management Module (Updated)

Previously, we used a boolean _pause_requested with a threading.Lock. 
Now we use a threading.Event (pause_event) for more efficient synchronization.

Still maintains global queues for good/bad tickets, error counts, and 
shutdown signals.
"""

import threading
from dataclasses import dataclass, field
from queue import Queue


@dataclass
class ErrorState:
    """
    Thread-safe error counter for managing cooldown periods in the processing pipeline.
    """
    error_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self) -> None:
        with self.lock:
            self.error_count += 1

    def reset(self) -> None:
        with self.lock:
            self.error_count = 0


# Global shared state instances
error_state = ErrorState()  # Tracks error state across all workers

# Thread-safe queues for routing processed tickets
good_tickets_queue = Queue()  # Successfully processed tickets
bad_tickets_queue = Queue()   # Failed or invalid tickets

# Global shutdown flag
_shutdown_requested = False
_shutdown_lock = threading.Lock()

def is_shutdown_requested() -> bool:
    """
    Returns True if a shutdown has been requested.
    Used by workers to exit gracefully.
    """
    with _shutdown_lock:
        return _shutdown_requested

def request_shutdown() -> None:
    """
    Sets the global shutdown flag to True, instructing all workers to exit.
    """
    global _shutdown_requested
    with _shutdown_lock:
        _shutdown_requested = True


# Global pause event
pause_event = threading.Event()
pause_event.clear()  # By default, we start unpaused

def toggle_pause_event() -> None:
    """
    Toggles the global pause_event. If it's set, we clear it (unpause).
    If it's clear, we set it (pause).
    """
    if pause_event.is_set():
        pause_event.clear()
    else:
        pause_event.set()

def is_pause_event_set() -> bool:
    """
    Returns True if the system is currently paused (pause_event is set).
    """
    return pause_event.is_set()