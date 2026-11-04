"""Event source and content extractor base classes."""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from world2ts_core.events import DiscoveredEvent


class EventSource:
    """Base interface for event/news search APIs."""
    def search(self, query: str, start: datetime, end: datetime, geo: str = "") -> List[DiscoveredEvent]:
        raise NotImplementedError


class ContentExtractor:
    """Base interface for full-text article extraction."""
    def extract(self, url: str, event_id: str, timestamp: datetime, source: str = "") -> Optional[DiscoveredEvent]:
        raise NotImplementedError
