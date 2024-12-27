"""
LLM API Integration Module

This module handles all interactions with the Large Language Model (LLM) API,
specifically OpenAI's GPT models, for ticket processing. It manages:
1. API configuration and calls
2. Prompt engineering and response formatting
3. Error handling and retries
4. Response validation and parsing

The module supports both OpenAI and Azure OpenAI Services, with automatic
retry logic for rate limits and API errors.
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
from pydantic import BaseModel, ValidationError

from config import Config
from models import TicketParsedSchema

# System prompt that instructs the LLM how to process tickets
PROMPT_INSTRUCTIONS = """
You are preparing IT service tickets for training an LLM (PHI-3).
Each ticket has these raw fields: 'Short Description', 'Description', 'Close Notes', 'Category', 'Subcategory', 
and 'Assignment Group'. The raw data is provided to you in JSON format. Please remember as you process these tickets 
that you must remove peoples names, phone numbers, email addresses, badge numbers, and any other sensitive 
information from the text using the proper tags.

Your tasks:
1. Decide if the ticket is GOOD or BAD for training:
   - GOOD if it has a meaningful user problem & some resolution (Close Notes) or resolution somewhere in the ticket text, 
   that would be valid for LLM Training.
   - BAD if it's incomplete/spam/no real resolution or meaningful outcome for training a service desk LLM.
2. If GOOD:
   - Merge 'Short Description' + 'Description' as 'instruction' (remove PII like phone #s, disclaimers).
   - Summarize 'Close Notes' as 'response'.
   - Use 'Category', 'Subcategory', 'Assignment Group' as is, or 'unknown' if missing.
3. If BAD:
   - Return an explanation in your refusal property or a short reason.
4. Output MUST comply with the JSON schema.
5. If the raw text is short but valid, try to paraphrase to get at least 5 words each for 'instruction' & 'response', but do NOT make up facts.
6. PRIORITY: You absolutely MUST remove phone numbers, replace people names with <NAME>, replace personal emails with <EMAIL_ADDRESS>, 
badge numbers with <BADGE_NUMBER>, replace any sensitive info that slips through replace exactly with <PII>, remove disclaimers, email 
signature blocks, and any other text that might be irrelevant or pure noise from final text. Your output must conform to the JSON schema for TicketParsedSchema
"""


def get_json_schema() -> dict:
    """
    Retrieves the JSON schema that defines the expected LLM response format.

    This function uses Pydantic's schema generation to ensure the LLM's output
    will match our TicketParsedSchema model structure.

    Returns:
        dict: JSON schema defining the required response format

    Example:
        schema = get_json_schema()
        # Schema will include fields like:
        # {
        #     "type": "object",
        #     "properties": {
        #         "status": {"type": "string"},
        #         "data": {...}
        #     },
        #     "required": ["status", "data"]
        # }
    """
    return TicketParsedSchema.schema()


def call_llm_parse(raw_ticket_str: str, config: Config):
    """
    Makes a direct call to the OpenAI API to process a ticket.

    This function:
    1. Configures the API client based on provider (OpenAI/Azure)
    2. Sends the ticket data with system instructions
    3. Requests a structured JSON response

    Args:
        raw_ticket_str: JSON string containing the raw ticket data
        config: Configuration object with API settings

    Returns:
        OpenAI API response object

    Raises:
        Various OpenAI exceptions (handled by retry decorator)

    Example:
        config = Config(api_key="...", model_name="gpt-4")
        ticket = '{"Short_Description": "Reset password", ...}'
        response = call_llm_parse(ticket, config)
    """
    if config.provider.lower() == "azure":
        openai.api_type = "azure"
        openai.api_base = config.azure_api_base
        openai.api_version = config.azure_api_version
    else:
        openai.api_type = "openai"

    openai.api_key = config.api_key

    model_or_deployment = (
        {"deployment_id": config.azure_deployment_name}
        if config.provider.lower() == "azure"
        else {"model": config.model_name}
    )

    try:
        response = openai.ChatCompletion.create(
            **model_or_deployment,
            messages=[
                {"role": "system", "content": PROMPT_INSTRUCTIONS},
                {"role": "user", "content": raw_ticket_str},
            ],
            max_tokens=1000,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": get_json_schema(),
                "strict": True,
            },
        )
        return response
    except Exception as exc:
        logging.error("API call failed: %s", exc, exc_info=True)
        raise exc


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=(retry_if_exception_type(RateLimitError) | retry_if_exception_type(APIError)),
)
def retry_call_llm_parse(raw_ticket_str: str, config: Config):
    """
    Wrapper around call_llm_parse that implements retry logic.

    This function adds automatic retries with exponential backoff for handling
    rate limits and temporary API failures. It will:
    1. Retry up to 5 times
    2. Wait between 1-20 seconds between retries (exponential backoff)
    3. Only retry on rate limits and API errors

    Args:
        raw_ticket_str: JSON string containing the raw ticket data
        config: Configuration object with API settings

    Returns:
        OpenAI API response object

    Example:
        config = Config(api_key="...", model_name="gpt-4")
        ticket = '{"Short_Description": "Reset password", ...}'
        try:
            response = retry_call_llm_parse(ticket, config)
        except Exception as e:
            # Handle case where all retries failed
            pass
    """
    return call_llm_parse(raw_ticket_str, config)


def parse_response(response):
    """
    Parses and validates the LLM's response into a TicketParsedSchema object.

    This function:
    1. Extracts the content from the API response
    2. Validates it against our schema
    3. Handles any parsing or validation errors

    Args:
        response: Raw OpenAI API response object

    Returns:
        TicketParsedSchema: Validated response object, or None if parsing fails

    Example:
        response = retry_call_llm_parse(ticket, config)
        parsed = parse_response(response)
        if parsed:
            if parsed.status == "good":
                # Process successful ticket parsing
                ticket_data = parsed.data
            else:
                # Handle rejected ticket
                rejection_reason = parsed.data
        else:
            # Handle parsing failure
            pass
    """
    try:
        content_str = response.choices[0].message.content
        parsed_resp = TicketParsedSchema.parse_raw(content_str)
        return parsed_resp
    except ValidationError as ve:
        logging.error("Validation error in LLM response JSON: %s", ve)
        return None
    except Exception as exc:
        logging.error("Unexpected error while parsing LLM response: %s", exc)
        return None
