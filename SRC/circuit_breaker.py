"""
Circuit Breaker Pattern Implementation

This module implements the Circuit Breaker design pattern, which is used to detect failures and
encapsulates the logic of preventing a failure from constantly recurring during maintenance,
temporary external system failure, or unexpected system difficulties.

The circuit breaker has three states:
- CLOSED: Normal operation, requests are allowed through
- OPEN: Failure threshold exceeded, requests are blocked
- HALF-OPEN: Testing if the system has recovered

Key attributes/methods:
  - record_failure(): increments the failure_count and opens the circuit if threshold exceeded
  - allow_request(): checks if calls are permitted based on state and time elapsed
  - reset(): resets circuit to CLOSED state
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
import threading


@dataclass
class CircuitBreaker:
    """
    A circuit breaker implementation that helps prevent repeated calls to failing endpoints.

    The circuit breaker monitors for failures and automatically "trips" (opens) when a threshold
    is exceeded, preventing further calls that are likely to fail. After a timeout period,
    it allows a test request through to check if the system has recovered.

    Attributes:
        failure_threshold (int): Number of consecutive failures before the circuit opens
        recovery_timeout (int): Seconds to wait before attempting recovery
        failure_count (int): Current number of consecutive failures
        state (str): Current circuit state ("CLOSED", "OPEN", or "HALF-OPEN")
        last_failure_time (datetime): Timestamp of the last recorded failure
        lock (threading.Lock): Thread lock for thread-safe operations
    """

    failure_threshold: int = 5
    recovery_timeout: int = 60
    failure_count: int = 0
    state: str = "CLOSED"
    last_failure_time: datetime = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_failure(self) -> None:
        """
        Records a failure and updates the circuit breaker state.
        If the number of failures meets or exceeds the threshold,
        moves the circuit to OPEN.
        """
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logging.warning("Circuit breaker transition to OPEN")

    def allow_request(self) -> bool:
        """
        Determines if a request should be allowed through the circuit breaker.

        Returns:
            bool: True if request is allowed, False if circuit is OPEN and not yet ready to HALF-OPEN
        """
        with self.lock:
            if self.state == "OPEN":
                # Check if enough time has passed since the last failure
                if (
                    self.last_failure_time
                    and (datetime.now() - self.last_failure_time).total_seconds()
                    > self.recovery_timeout
                ):
                    self.state = "HALF_OPEN"
                    logging.warning("Circuit breaker transition to HALF_OPEN")
                    return True
                return False
            return True

    def reset(self) -> None:
        """
        Resets the circuit breaker to its initial CLOSED state.
        This is typically called after a successful "test" request or a sanity check.
        """
        with self.lock:
            self.state = "CLOSED"
            self.failure_count = 0
            logging.info("Circuit breaker reset to CLOSED")