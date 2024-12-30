"""
LLM API Integration Module

This module handles all interactions with the Large Language Model (LLM) API
for ticket processing. It manages:
1. API configuration and calls
2. Prompt engineering and response formatting
3. Error handling and retries
4. Response validation and parsing

Now uses the `openai.beta.chat.completions.parse` method for structured outputs.
"""

import logging
import openai
from openai import APIError, RateLimitError

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Config
from models import TicketParsedSchema

PROMPT_INSTRUCTIONS = """
You are preparing IT service tickets for training an LLM (PHI-3). 
Your prime goal is to make the ticket into a 'good' format for LLM training.
Each ticket has these raw fields: 
  'Short Description', 'Description', 'Close Notes', 'Category', 'Subcategory', 'Assignment Group'.
The raw data is provided to you in JSON format.

Your tasks:
1. Convert the ticket into a 'good' format:
   - Merge 'Short Description' + 'Description' into an 'instruction' 
     (remove personal info like phone #s, disclaimers, or signatures NOTE Addresses are ok).
   - Summarize 'Close Notes' (or the action taken) into 'response'.
   - Keep 'Category', 'Subcategory', 'Assignment Group' as-is (or 'unknown' if missing).
2. If the ticket data is corrupted, mark it as 'bad' with a short explanation.
3. Remove disclaimers, signature blocks, or obvious extraneous text.
4. Output a JSON response with exactly two fields: 
  "status" (either "good" or "bad"), and 
  "data" (dictionary or string). 
No extra text, no disclaimers, no markdown code fences.
"""

def call_llm_parse(raw_ticket_str: str, config: Config):
    """
    Uses openai.beta.chat.completions.parse() to request a strictly validated JSON output
    conforming to TicketParsedSchema. This ensures the LLM returns well-formed data.
    """
    openai.api_key = config.api_key

    completion = openai.beta.chat.completions.parse(
        model=config.model_name,
        messages=[
            {"role": "system", "content": PROMPT_INSTRUCTIONS},
            {"role": "user", "content": raw_ticket_str},
        ],
        response_format=TicketParsedSchema,
        max_tokens=1000,
        temperature=0.0,
    )
    return completion


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=(retry_if_exception_type(RateLimitError) | retry_if_exception_type(APIError)),
)
def retry_call_llm_parse(raw_ticket_str: str, config: Config):
    """
    Wrapper around call_llm_parse that implements retry logic for rate limits and 500 errors.

    Retries up to 5 times with exponential backoff. If it still fails, an exception is raised.
    """
    return call_llm_parse(raw_ticket_str, config)


def parse_response(response):
    """
    Extracts the structured data from the completion response.
    If the model refused, returns None.
    Otherwise returns the parsed data as a TicketParsedSchema instance.
    """
    choice = response.choices[0].message

    if choice.refusal:
        logging.error(f"Model refused to answer: {choice.refusal}")
        return None

    parsed = choice.parsed
    return parsed
