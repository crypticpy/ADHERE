"""
Token Bucket Rate Limiting Implementation

This module implements the Token Bucket algorithm, a widely used rate limiting mechanism.
"""

from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class TokenBucket:
    """
    A thread-safe implementation of the Token Bucket rate limiting algorithm.

    Attributes:
        capacity (int): Maximum number of tokens the bucket can hold
        refill_rate (int): Number of tokens added per second
        tokens (float): Current token count
        last_refill (datetime): When the bucket was last refilled
    """

    capacity: int
    refill_rate: int
    tokens: float = field(init=False)
    last_refill: datetime = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = datetime.now()

    def _refill(self) -> None:
        """
        Calculates how many tokens to add based on elapsed time, 
        ensuring we never exceed capacity.
        """
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        added = elapsed * self.refill_rate
        with self.lock:
            if self.tokens < self.capacity:
                self.tokens = min(self.capacity, self.tokens + added)
            self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempts to consume 'tokens' from the bucket.
        If enough tokens exist, deduct and return True.
        Otherwise, return False.

        This method refills the bucket before checking availability.
        """
        self._refill()
        with self.lock:
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
