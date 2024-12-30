"""
Data Models Module

Defines the data structures used for ticket processing, using strict Pydantic models
so we can leverage OpenAI structured outputs for validated returns.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Union, Literal

###############################################################################
# TicketSchema: the "good" ticket object
###############################################################################
class TicketSchema(BaseModel):
    """
    Normalized representation of a successfully processed ticket.

    Fields:
        instruction: The merged short description + description (with redactions)
        response: Summarized close notes or resolution
        category: Category
        subcategory: Subcategory
        assignment_group: Group assigned to handle the ticket
    """

    instruction: str
    response: str
    category: str
    subcategory: str
    assignment_group: str

    model_config = ConfigDict(extra="forbid")  # No extra keys beyond these


###############################################################################
# TicketParsedSchema: top-level structure returned by the LLM
###############################################################################
class TicketParsedSchema(BaseModel):
    """
    This model enforces a strict contract:
      status = "good" or "bad"
      data = TicketSchema if "good", or a string with explanation if "bad"
    """

    status: Literal["good", "bad"] = Field(..., description="LLM outcome: 'good' or 'bad'")
    data: Union[TicketSchema, str] = Field(..., description="Either a parsed ticket or short explanation")

    model_config = ConfigDict(extra="forbid")  # Keep it strict
