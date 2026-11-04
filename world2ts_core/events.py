"""Retrieved event candidates used by Greenfield discovery."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


EvidenceQuality = Literal["web_snippet", "structured_record"]


class DiscoveredEvent(BaseModel):
    """A source result before Agent selection and canonicalization."""

    event_id: str
    source_name: str
    source_kind: Literal["news_search", "web_search", "calendar"]

    title: str
    description: str
    evidence_quality: EvidenceQuality
    structured_payload: dict[str, Any] = Field(default_factory=dict)

    timestamp: datetime
    timestamp_precision: Literal["exact", "hour", "day", "week", "month"] = "day"

    geo_scope: Literal["local", "regional", "national", "global"] = "global"
    geo_tags: list[str] = Field(default_factory=list)

    url: str = ""
    query_intent_id: str = ""
    window_id: str = ""
    event_summary_id: str = ""
    retrieval_rank: int = 0
    retrieval_score: float = 0.0
