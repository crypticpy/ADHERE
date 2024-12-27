"""
Utility Functions Module

This module provides utility functions for data processing and PII (Personally
Identifiable Information) redaction in the ticket processing pipeline. It uses
regular expressions for pattern matching and spaCy for named entity recognition
to identify and redact sensitive information.

Key features:
- PII redaction (phone numbers, emails, SSNs, etc.) with centralized regex patterns
- Named entity recognition for identifying person names
- Data validation and field normalization
"""

import re
import spacy
from typing import Any, Dict, Union
from models import RawTicket

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Centralized list of PII-related regex patterns
# New approach: consolidated global patterns for maintainability
PII_PATTERNS = [
    # North American phone numbers
    (r"\+?1?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "<PHONE_NUMBER>"),
    # Local phone number format
    (r"\b\d{3}[-.\s]\d{4}\b", "<PHONE_NUMBER>"),
    # Email addresses
    (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "<EMAIL_ADDRESS>"),
    # Social Security Numbers
    (r"\b\d{3}-\d{2}-\d{4}\b", "<SSN>"),
    # Credit card numbers (simple 16-digit)
    (r"\b\d{16}\b", "<CARD_NUMBER>"),
    # IP addresses
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<IP_ADDRESS>"),
    # Badge numbers (APD, AFD, etc.)
    (r"\b[Aa][PpDd][Dd]?\d+\b", "<BADGE_NUMBER>"),
    # Simple street addresses
    (r"\b\d{1,5}\s\w+\s\w+", "<ADDRESS>"),
    # Incident IDs
    (r"\bINC\d+\b", "<INCIDENT_ID>"),
]


def redact_pii(
    content: Union[str, Dict[str, Any], None]
) -> Union[str, Dict[str, Any], None]:
    """
    Redacts Personally Identifiable Information (PII) from text or dictionary content.

    This function identifies and replaces sensitive information with placeholder tags.
    It handles both string and dictionary inputs, recursively processing dictionary
    values. For strings, it uses both our consolidated regex patterns and spaCy’s
    named entity recognition to identify PII.

    Args:
        content: Input text or dictionary to redact.
            - str: Text containing potential PII
            - Dict[str, Any]: Dictionary with values to redact
            - None: Returns None unchanged

    Returns:
        Redacted version of the input with PII replaced by placeholder tags

    Example:
        # Redacting a string
        text = "Please contact John Doe at john.doe@email.com"
        redacted = redact_pii(text)
        # "Please contact <NAME> at <EMAIL_ADDRESS>"
    """
    if content is None:
        return content

    if isinstance(content, dict):
        return {k: redact_pii(v) for k, v in content.items()}

    if not isinstance(content, str):
        return content

    text = content
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Use spaCy for named entity recognition of PERSON
    doc = nlp(text)
    redacted = text
    for ent in reversed(doc.ents):
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            redacted = f"{redacted[:start]}<NAME>{redacted[end:]}"

    return redacted


def fill_missing_fields(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes ticket data by ensuring all required fields are present.

    This function takes a raw ticket dictionary and ensures it has all required
    fields with proper default values. It uses the RawTicket model for validation
    and field normalization.

    Args:
        ticket: Dictionary containing raw ticket data, potentially missing fields

    Returns:
        Dictionary with all required fields, using empty strings or "unknown" for
        missing values
    """
    validated = RawTicket(**ticket)
    return {
        "Short_Description": validated.Short_Description or "",
        "Description": validated.Description or "",
        "Close_Notes": validated.Close_Notes or "",
        "Category": (
            validated.Category if isinstance(validated.Category, str) else "unknown"
        ),
        "Subcategory": (
            validated.Subcategory
            if isinstance(validated.Subcategory, str)
            else "unknown"
        ),
        "Assignment_Group": validated.Assignment_Group or "unknown",
    }
