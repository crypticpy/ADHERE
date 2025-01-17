"""
Configuration Management Module

This module defines the configuration settings for the ticket processing pipeline,
handling everything from file paths to performance tuning parameters. It uses
environment variables, `.env` (optional), and sensible defaults to configure the system.

It also includes validation logic to ensure the configuration is in a valid state.
"""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Config:
    """
    Primary configuration for the ticket-processing pipeline.

    This class manages all configuration settings, organized into several logical groups:

    1. API Configuration:
        - api_key: OpenAI API key

    2. File Handling:
        - input_file: Source of ticket data
        - good_tickets_file: Output for successfully processed tickets
        - bad_tickets_file: Output for failed ticket processing
        - raw_llm_transactions_file: Output for raw LLM transactions

    3. Processing Parameters:
        - batch_size: Number of tickets to process in one batch
        - min_word_count: Minimum words required for valid ticket
        - model_name: LLM model to use
        - max_tokens: Maximum tokens per request
        - token_margin: Safety margin for token calculations

    4. Concurrency Settings:
        - num_workers: Default number of worker threads (dynamic up/down)
        - initial_workers: Starting concurrency
        - scale_step: Worker count adjustment step
        - scale_interval: Time between scaling decisions (seconds)
        - max_workers: Maximum allowed worker threads
        - throttle_delay: Delay (seconds) between requests

    5. Error Handling:
        - error_threshold: Failures before circuit breaker trips
        - cool_down_time: Recovery wait time after errors

    6. Rate Limiting:
        - token_bucket_capacity: Maximum burst capacity
        - token_refill_rate: Rate of token replenishment

    7. Resource Monitoring:
        - enable_resource_monitoring: Toggle CPU-based scaling
        - cpu_scale_up_threshold: CPU usage threshold for scaling up
        - cpu_scale_down_threshold: CPU usage threshold for scaling down
    """

    api_key: str = os.getenv("OPENAI_API_KEY", "")
    input_file: Path = Path("anonymized_tickets.jsonl")
    good_tickets_file: Path = Path("good_tickets.jsonl")
    bad_tickets_file: Path = Path("bad_tickets.jsonl")
    raw_llm_transactions_file: Path = Path("raw_llm_transactions.jsonl")

    num_workers: int = max(1, (os.cpu_count() or 1) * 2)
    batch_size: int = 500
    min_word_count: int = 5
    model_name: str = "gpt-4o-mini-2024-07-18"
    max_tokens: int = 4000
    token_margin: int = 50

    initial_workers: int = 20
    scale_step: int = 5
    scale_interval: int = 10
    max_workers: int = 50
    throttle_delay: float = 0.1

    error_threshold: int = 10
    cool_down_time: int = 60

    token_bucket_capacity: int = 200
    token_refill_rate: int = 40

    enable_resource_monitoring: bool = True
    cpu_scale_up_threshold: float = 75.0
    cpu_scale_down_threshold: float = 90.0

    def __post_init__(self) -> None:
        if self.max_workers < self.initial_workers:
            raise ValueError("max_workers cannot be smaller than initial_workers.")
        if self.scale_step <= 0:
            raise ValueError("scale_step must be > 0.")