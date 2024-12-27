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
from typing import Optional

# Optional .env support for local development
try:
    from dotenv import load_dotenv

    load_dotenv()  # Loads variables from a .env file if present
except ImportError:
    pass  # If python-dotenv isn't installed, we skip loading .env without error


@dataclass
class Config:
    """
    Primary configuration for the ticket-processing pipeline.

    This class manages all configuration settings, organized into several logical groups:

    1. API Configuration:
        - api_key: OpenAI API key
        - provider: AI provider selection ('openai' or 'azure')
        - Azure-specific settings (when using Azure)

    2. File Handling:
        - input_file: Source of ticket data
        - good_tickets_file: Output for successfully processed tickets
        - bad_tickets_file: Output for failed ticket processing

    3. Processing Parameters:
        - batch_size: Number of tickets to process in one batch
        - min_word_count: Minimum words required for valid ticket
        - model_name: LLM model to use
        - max_tokens: Maximum tokens per request
        - token_margin: Safety margin for token calculations

    4. Concurrency Settings:
        - num_workers: Total worker processes
        - initial_workers: Starting number of workers
        - scale_step: Worker count adjustment step
        - scale_interval: Time between scaling decisions (seconds)
        - max_workers: Maximum allowed workers
        - throttle_delay: Delay (seconds) between requests

    5. Error Handling:
        - error_threshold: Failures before circuit breaker trips
        - cool_down_time: Recovery wait time after errors

    6. Rate Limiting:
        - token_bucket_capacity: Maximum burst capacity
        - token_refill_rate: Rate of token replenishment

    7. Resource Management:
        - enable_resource_monitoring: Toggle CPU monitoring
        - cpu_scale_up_threshold: CPU usage threshold for scaling up
        - cpu_scale_down_threshold: CPU usage threshold for scaling down

    Example:
        config = Config()
        # All environment-dependent settings will be loaded or use defaults
    """

    api_key: str = os.getenv("OPENAI_API_KEY", "")
    input_file: Path = Path("cleaned_ticket_data.json")
    good_tickets_file: Path = Path("good_tickets.jsonl")
    bad_tickets_file: Path = Path("bad_tickets.jsonl")
    num_workers: int = max(1, (os.cpu_count() or 1) * 2)
    batch_size: int = 500
    min_word_count: int = 5
    model_name: str = "gpt-4o-2024-12-17"
    max_tokens: int = 8192
    token_margin: int = 50

    initial_workers: int = 10
    scale_step: int = 5
    scale_interval: int = 30
    max_workers: int = 50
    throttle_delay: float = 0.2

    error_threshold: int = 5
    cool_down_time: int = 60

    token_bucket_capacity: int = 100
    token_refill_rate: int = 10

    provider: str = "openai"  # or "azure"
    azure_api_base: str = ""
    azure_api_version: str = ""
    azure_deployment_name: str = ""

    enable_resource_monitoring: bool = True

    cpu_scale_up_threshold: float = 75.0
    cpu_scale_down_threshold: float = 90.0

    def __post_init__(self) -> None:
        """
        Performs validation after initialization to ensure configuration consistency.

        1. Worker count consistency:
           - Ensures max_workers >= initial_workers
           - Validates scale_step is positive

        2. Azure provider requirements:
           - If using Azure, validates all required Azure settings are provided
           - Checks for azure_api_base, azure_api_version, azure_deployment_name

        Raises:
            ValueError: If any validation checks fail, with a descriptive error message
        """
        if self.max_workers < self.initial_workers:
            raise ValueError("max_workers cannot be smaller than initial_workers.")
        if self.scale_step <= 0:
            raise ValueError("scale_step must be > 0.")
        if self.provider.lower() == "azure":
            if (
                not self.azure_api_base
                or not self.azure_api_version
                or not self.azure_deployment_name
            ):
                raise ValueError(
                    "Must provide Azure base, version, and deployment name if provider='azure'."
                )
