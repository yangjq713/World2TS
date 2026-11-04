"""Series source protocol."""

from __future__ import annotations

from datetime import datetime
from typing import List

from world2ts_core.series.model import SeriesRecord


class SeriesSource:
    """Base interface for time series data sources."""

    def fetch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        domain: str,
    ) -> List[SeriesRecord]:
        raise NotImplementedError
