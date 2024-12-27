"""
Token Bucket Rate Limiting Implementation

This module implements the Token Bucket algorithm, a widely used rate limiting mechanism
that allows for controlled bursts of activity while maintaining a long-term rate limit.

Think of it like a bucket that:
- Has a maximum capacity (e.g., 100 tokens)
- Continuously fills at a steady rate (e.g., 10 tokens per second)
- Each operation consumes one or more tokens
- If the bucket is empty, operations must wait

Visual representation:
    ┌──────────────┐
    │  Tokens (○)  │  ← Tokens refill at refill_rate
    │    ○○○○○    │
    │    ○○○○     │  ← Tokens are consumed by operations
    └──────────────┘

This helps prevent overwhelming external services while still allowing
short bursts of higher activity when the bucket is full.
"""

from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class TokenBucket:
    """
    A thread-safe implementation of the Token Bucket rate limiting algorithm.

    The token bucket allows for rate limiting with controlled bursts. It's particularly
    useful for API calls where you want to maintain a steady average rate while still
    allowing occasional bursts of activity.

    Attributes:
        capacity (int): Maximum number of tokens the bucket can hold
        refill_rate (int): Number of tokens added per second
        tokens (float): Current number of tokens in the bucket
        last_refill (datetime): Timestamp of the last token refill
        lock (threading.Lock): Thread lock for thread-safe operations

    Example:
        # Create a bucket that allows 100 requests with 10 new tokens per second
        bucket = TokenBucket(capacity=100, refill_rate=10)

        # Try to make a request
        if bucket.consume():
            # Make your API call here
            make_api_call()
        else:
            # Rate limit exceeded, handle accordingly
            handle_rate_limit()
    """

    capacity: int
    refill_rate: int
    tokens: float = field(init=False)
    last_refill: datetime = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        """
        Initializes the token bucket after instance creation.

        Sets up the initial state:
        1. Fills the bucket to capacity with tokens
        2. Sets the initial refill timestamp
        """
        self.tokens = self.capacity
        self.last_refill = datetime.now()

    def _refill(self) -> None:
        """
        Internal method to refill tokens based on elapsed time.

        This method:
        1. Calculates time elapsed since last refill
        2. Adds new tokens based on refill_rate * elapsed_time
        3. Ensures token count doesn't exceed capacity
        4. Updates the last refill timestamp

        The refill is time-based, so if 0.5 seconds have passed with a
        refill_rate of 10/sec, 5 tokens would be added.
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
        Attempts to consume tokens from the bucket.

        Args:
            tokens (int): Number of tokens to consume (default: 1)

        Returns:
            bool: True if tokens were consumed, False if insufficient tokens

        Example:
            # Consume 5 tokens for a more expensive operation
            if bucket.consume(tokens=5):
                # Perform expensive operation
                perform_expensive_operation()
            else:
                # Handle rate limit, maybe wait and retry
                wait_and_retry()
        """
        self._refill()
        with self.lock:
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
