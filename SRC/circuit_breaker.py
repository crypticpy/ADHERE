"""
Circuit Breaker Pattern Implementation

This module implements the Circuit Breaker design pattern, which is used to detect failures and
encapsulates the logic of preventing a failure from constantly recurring during maintenance,
temporary external system failure, or unexpected system difficulties.

The circuit breaker has three states:
- CLOSED: Normal operation, requests are allowed through
- OPEN: Failure threshold exceeded, requests are blocked
- HALF-OPEN: Testing if the system has recovered

Think of it like an electrical circuit breaker in your home:
- CLOSED: Everything works normally, electricity flows
- OPEN: Too many problems detected, power is cut off to prevent damage
- HALF-OPEN: Testing if it's safe to restore power
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
        failure_threshold (int): Number of failures before the circuit opens (default: 5)
        recovery_timeout (int): Seconds to wait before attempting recovery (default: 60)
        failure_count (int): Current number of consecutive failures
        state (str): Current circuit state ("CLOSED", "OPEN", or "HALF-OPEN")
        last_failure_time (datetime): Timestamp of the last recorded failure
        lock (threading.Lock): Thread lock for thread-safe operations

    Example:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

        if breaker.allow_request():
            try:
                # Make your API call or other operation here
                response = make_api_call()
                breaker.reset()  # Success! Reset the breaker
            except Exception:
                breaker.record_failure()  # Record the failure
        else:
            # Circuit is open, handle accordingly
            logging.warning("Circuit breaker is open, skipping request")
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

        When the number of failures reaches the threshold, the circuit
        transitions to the OPEN state, preventing further requests.

        Thread-safe method that:
        1. Increments the failure counter
        2. Updates the last failure timestamp
        3. Changes state to OPEN if threshold is reached
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
            bool: True if the request should be allowed, False if it should be blocked

        This method implements the core circuit breaker logic:
        - If CLOSED: Always allows requests
        - If OPEN: Blocks requests until recovery_timeout has elapsed
        - If recovery_timeout has elapsed: Transitions to HALF-OPEN and allows one test request

        Thread-safe method that ensures consistent state across multiple threads.
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
        Resets the circuit breaker to its initial closed state.

        This method should be called when a request succeeds, particularly
        in the HALF-OPEN state, indicating that the system has recovered.

        Thread-safe method that:
        1. Changes state back to CLOSED
        2. Resets the failure counter to 0
        3. Logs the successful reset
        """
        with self.lock:
            self.state = "CLOSED"
            self.failure_count = 0
            logging.info("Circuit breaker reset to CLOSED")
