"""Canonical series record returned by remote collection adapters."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SeriesRecord(BaseModel):
    """One source series before strict Arrow materialization."""

    series_id: str
    domain: str
    entity: str
    geo: str = ""
    freq: str
    start: datetime
    end: datetime
    values: list[float]
    timestamps: list[datetime]
    columns: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
