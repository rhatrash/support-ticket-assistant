"""Pydantic schemas: validate the final output and any structured output coming from an LLM."""
from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class SimilarCase(BaseModel):
    id: int
    text: str
    category: str
    score: float


class TicketAnalysis(BaseModel):
    text: str
    language: Literal["en", "ar"]
    intent: str
    category: str
    priority: Literal["high", "medium", "low"]
    confidence: Dict[str, float]
    similar_cases: List[SimilarCase]
    draft_reply: str
    reply_source: Literal["llm", "template"]
    grounded_on: List[int] = Field(default_factory=list)


class ReplyDraft(BaseModel):
    """Expected JSON from the LLM."""

    reply: str = Field(min_length=10)
    used_case_ids: List[int] = Field(default_factory=list)
