"""
Data Models Module

This module defines the data structures used throughout the ticket processing pipeline.
It uses Pydantic models to ensure type safety and data validation at runtime.

The module implements a progression of ticket representations:
1. RawTicket: Initial ticket data with optional fields
2. TicketSchema: Normalized, validated ticket data
3. ProcessingResult: Container for processing outcomes
4. TicketParsedSchema: LLM processing results

Each model serves a specific purpose in the data transformation pipeline,
ensuring data consistency and proper error handling.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Union, Any


class TicketSchema(BaseModel):
    """
    Normalized representation of a successfully processed ticket.

    This model represents the standardized format of a ticket after processing.
    It contains all required fields with proper validation and no optional fields,
    ensuring consistency in the processed data.

    Attributes:
        instruction (str): Clear description of what needs to be done
        outcome (str): Description of how the ticket was resolved
        category (str): Primary classification of the ticket
        subcategory (str): Secondary classification for more detail
        assignment_group (str): Team or group responsible for the ticket

    Example:
        good_ticket = TicketSchema(
            instruction="Reset user password",
            outcome="Password reset and communicated to user",
            category="Account Management",
            subcategory="Password Reset",
            assignment_group="Help Desk"
        )
    """

    instruction: str
    outcome: str
    category: str
    subcategory: str
    assignment_group: str


class RawTicket(BaseModel):
    """
    Initial ticket data model that accepts raw input with flexible validation.

    This model handles incoming ticket data which may be incomplete or inconsistent.
    It uses optional fields with defaults to ensure we can always create a valid
    instance, even with missing data.

    Attributes:
        Short_Description (Optional[str]): Brief ticket summary
        Description (Optional[str]): Detailed ticket description
        Close_Notes (Optional[str]): Notes added when closing the ticket
        Category (Optional[str]): Primary classification (defaults to "unknown")
        Subcategory (Optional[str]): Secondary classification (defaults to "unknown")
        Assignment_Group (Optional[str]): Assigned team (defaults to "unknown")

    Example:
        raw_ticket = RawTicket(
            Short_Description="User cannot log in",
            Description="User reports being unable to access their account",
            Category="Access Issues"
        )
    """

    model_config = ConfigDict(extra="allow")  # Allows additional fields in raw data
    Short_Description: Optional[str] = Field(
        default="", description="Brief summary of the ticket"
    )
    Description: Optional[str] = Field(
        default="", description="Detailed description of the issue"
    )
    Close_Notes: Optional[str] = Field(
        default="", description="Notes added when resolving the ticket"
    )
    Category: Optional[str] = Field(
        default="unknown", description="Primary ticket classification"
    )
    Subcategory: Optional[str] = Field(
        default="unknown", description="Secondary ticket classification"
    )
    Assignment_Group: Optional[str] = Field(
        default="unknown", description="Team assigned to the ticket"
    )


class ProcessingResult(BaseModel):
    """
    Container for processing results that handles both successful and failed outcomes.

    This model acts as a wrapper that can contain either a successfully processed
    ticket (TicketSchema) or an error message explaining what went wrong.

    Attributes:
        status (str): Processing status ("success" or "error")
        data (Union[TicketSchema, str]): Either a processed ticket or error message

    Example:
        # Successful processing
        success_result = ProcessingResult(
            status="success",
            data=TicketSchema(...)
        )

        # Failed processing
        error_result = ProcessingResult(
            status="error",
            data="Invalid ticket format: missing required fields"
        )
    """

    status: str = Field(..., description="Processing outcome status")
    data: Union[TicketSchema, str] = Field(
        ..., description="Processed ticket or error message"
    )


class TicketParsedSchema(BaseModel):
    """
    Schema for the Large Language Model (LLM) processing output.

    This model represents the structured output from the LLM processing step.
    It includes a status indicating whether the ticket was successfully parsed
    and the resulting data (either a structured ticket or rejection reason).

    Attributes:
        status (str): LLM processing status ("good" or "bad")
        data (Any): Flexible field that contains either:
            - A dictionary matching TicketSchema for "good" status
            - A string explaining rejection reason for "bad" status

    Example:
        # Successful LLM parsing
        good_parse = TicketParsedSchema(
            status="good",
            data={
                "instruction": "Reset user password",
                "outcome": "Password reset completed",
                "category": "Account Management",
                "subcategory": "Password Reset",
                "assignment_group": "Help Desk"
            }
        )

        # Failed LLM parsing
        bad_parse = TicketParsedSchema(
            status="bad",
            data="Ticket description too vague to determine clear action items"
        )
    """

    status: str = Field(..., description="LLM processing outcome ('good' or 'bad')")
    data: Any = Field(..., description="Parsed ticket data or rejection reason")
