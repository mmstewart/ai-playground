from typing import Literal

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    task: str = Field(description="What needs to be done, as one short sentence.")
    owner: str = Field(description="The person or team responsible.")


class DepartmentRisks(BaseModel):
    department: str = Field(description="The department the risks belong to.")
    risks: list[str] = Field(
        description="Blockers or risks raised for this department."
    )


class MeetingNotes(BaseModel):
    title: str = Field(max_length=100, description="A concise title for the meeting.")
    decisions: list[str] = Field(description="The top decisions made in the meeting.")
    risks: list[DepartmentRisks] = Field(
        description="Key blockers or risks, grouped by department."
    )
    action_items: list[ActionItem] = Field(
        description="Action items, each with a clear owner."
    )


class Summary(BaseModel):
    title: str = Field(max_length=100, description="A concise title for the summary.")
    key_points: list[str] = Field(
        max_length=5, description="A list of up to 5 key points from the text."
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="The overall sentiment of the text."
    )
