"""
Utility Functions Module

All references to PII detection and redaction have been removed. 
We no longer need spaCy or regular expressions for PII handling.
Now, this module only provides `fill_missing_fields` to ensure 
tickets have the expected keys.

If you decide to reintroduce advanced PII redaction, 
you can integrate an external tool (e.g., Presidio) outside this pipeline.
"""

from typing import Any, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator


class RawTicket(BaseModel):
    model_config = ConfigDict(extra="allow")
    Short_Description: str = "unknown"
    Description: str = "unknown"
    Close_Notes: str = "unknown"
    Category: str = "unknown"
    Subcategory: str = "unknown"
    Assignment_Group: str = "unknown"

    @model_validator(mode="before")
    @classmethod
    def replace_nones_with_unknown(cls, values: Any) -> Any:
        """
        For any field missing or explicitly None, replace it with 'unknown'.
        This is called in 'before' mode, which means it runs before field-level validation.
        """
        # If the input isn't a dictionary, abort or transform it
        if not isinstance(values, dict):
            return values

        # For every field defined in this model, if the field is missing or is None,
        # set it to "unknown".
        for field_name in cls.__fields__:
            if field_name not in values or values[field_name] is None:
                values[field_name] = "unknown"
        return values


def fill_missing_fields(ticket: Dict[str, Any]) -> Dict[str, Any]:
    # Convert None to "unknown" so Pydantic won't reject it
    safe_ticket = dict(ticket)
    for field_name in ["Category", "Subcategory", "Assignment_Group"]:
        if field_name not in safe_ticket or safe_ticket[field_name] is None:
            safe_ticket[field_name] = "unknown"

    validated = RawTicket(**safe_ticket)
    return {
        "Short_Description": validated.Short_Description or "",
        "Description": validated.Description or "",
        "Close_Notes": validated.Close_Notes or "",
        "Category": validated.Category,
        "Subcategory": validated.Subcategory,
        "Assignment_Group": validated.Assignment_Group,
    }

