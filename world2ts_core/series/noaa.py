"""NOAA CDO API series source.

Fetches daily climate observations from the Global Historical Climatology
Network Daily (GHCND) dataset via NOAA's Climate Data Online (CDO) v2 API.

API docs: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
Token: https://www.ncei.noaa.gov/cdo-web/token
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from world2ts_core.series.model import SeriesRecord
from world2ts_core.config import get as config_get
from world2ts_core.registry import SeriesSourceCapability, register_series

_CDO_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
_RATE_LIMIT_DELAY = 0.25  # CDO allows 5 req/sec; 0.25s gap is safe


def _api_get(url: str, params: Dict[str, Any], token: str) -> Dict[str, Any]:
    """Make a single CDO API GET request via system curl.

    Uses /usr/bin/curl instead of httpx to avoid Anaconda OpenSSL issues
    with proxy configurations (same pattern as other adapters).
    """
    query = urlencode(params)
    full_url = f"{url}?{query}"
    cmd = [
        "/usr/bin/curl", "-f", "-sS", "-g", "--max-time", "30",
        "-H", f"token: {token}",
    ]
    proxy = config_get("http_proxy", default="")
    if proxy:
        cmd += ["-x", proxy]
    cmd.append(full_url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConnectionError(f"NOAA CDO curl execution failed: {exc}") from exc
    if result.returncode != 0:
        raise ConnectionError(f"curl failed (code {result.returncode}): {result.stderr[:200]}")
    if not result.stdout.strip():
        raise ConnectionError("NOAA CDO returned an empty response body")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("NOAA CDO response must be a JSON object")
    return payload


def _fetch_cdo_pages(
    params: Dict[str, Any],
    token: str,
) -> List[Dict[str, Any]]:
    """Fetch and validate every page reported by one CDO query."""
    requested_offset = 1
    expected_count: Optional[int] = None
    results: List[Dict[str, Any]] = []

    while True:
        page_params = dict(params, offset=requested_offset)
        response = _api_get(_CDO_BASE, page_params, token)
        time.sleep(_RATE_LIMIT_DELAY)

        metadata = response.get("metadata")
        resultset = metadata.get("resultset") if isinstance(metadata, dict) else None
        if not isinstance(resultset, dict):
            raise ValueError("NOAA CDO response is missing metadata.resultset")

        count = resultset.get("count")
        response_offset = resultset.get("offset")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("NOAA CDO resultset count must be a non-negative integer")
        if response_offset != requested_offset:
            raise ValueError(
                "NOAA CDO returned unexpected page offset "
                f"{response_offset!r}; expected {requested_offset}"
            )
        if expected_count is None:
            expected_count = count
        elif count != expected_count:
            raise ValueError(
                f"NOAA CDO result count changed from {expected_count} to {count} during pagination"
            )

        page_results = response.get("results", [])
        if not isinstance(page_results, list):
            raise ValueError("NOAA CDO results must be a list")
        if count == 0:
            if page_results:
                raise ValueError("NOAA CDO returned rows while reporting zero results")
            return []
        if not page_results:
            raise ValueError(
                f"NOAA CDO returned an empty page at offset {requested_offset} "
                f"before all {count} results were fetched"
            )
        if any(not isinstance(row, dict) for row in page_results):
            raise ValueError("NOAA CDO results must contain JSON objects")

        results.extend(page_results)
        if len(results) > count:
            raise ValueError(
                f"NOAA CDO returned {len(results)} rows while reporting only {count}"
            )
        if len(results) == count:
            return results
        requested_offset += len(page_results)


@register_series
class NoaaSource:
    """Fetch daily climate data from NOAA CDO v2 API (GHCND dataset).

    Symbols are NOAA station IDs, e.g. ``USW00094728`` (Central Park, NY).
    They are automatically prefixed with ``GHCND:`` when sent to the API.

    Supported datatypes: ``TMAX`` (default), ``TMIN``, ``PRCP``. Requests use
    CDO's ``metric`` unit mode, so returned temperatures are already degrees
    Celsius and precipitation is already millimeters; values are not rescaled.
    """

    capability = SeriesSourceCapability(
        name="noaa",
        domains=["weather"],
        entity_types=["weather_station"],
        frequencies=["1d"],
        geo=["US"],
        history_days=3650,
        rate_per_min=20,
        auth_required=True,
        reliability="high",
        description="NOAA CDO — daily climate observations (TMAX, TMIN, PRCP) from US GHCND stations",
        primary_target_fields=["tmax", "tmin", "prcp"],
        target_option="datatype",
        adapter_options={"datatype": ["TMAX", "TMIN", "PRCP"]},
        metadata_dimensions=[
            "station_id",
            "ghcnd_id",
            "datatype",
            "dataset",
            "calendar_symbol",
        ],
        timestamp_semantics="station civil dates interpreted in plan timezone",
        range_semantics="[start, end)",
    )

    def __init__(
        self,
        api_token: Optional[str] = None,
        datatype: str = "TMAX",
    ):
        self._api_token = api_token
        self.datatype = datatype

    # Known station-to-state mapping for calendar cross-reference
    _STATION_STATES: Dict[str, str] = {
        "USW00094728": "NY", "USW00023174": "CA", "USW00012960": "FL",
        "USW00003927": "IL", "USW00014739": "MI", "USW00023234": "CA",
        "USW00013874": "GA", "USW00024233": "WA", "USW00093738": "TX",
        "USW00014922": "MN",
    }

    def _resolve_state(self, station_id: str) -> str:
        """Map station ID to US state abbreviation for storm calendar."""
        return self._STATION_STATES.get(station_id, "")

    def _get_token(self) -> str:
        token = (
            self._api_token
            or config_get("noaa_api_token")
            or os.environ.get("WORLD2TS_NOAA_API_TOKEN", "")
        )
        if not token:
            raise ValueError(
                "NOAA API token not set. "
                "Set noaa_api_token in world2ts.toml or WORLD2TS_NOAA_API_TOKEN env var. "
                "Get a free token at https://www.ncei.noaa.gov/cdo-web/token"
            )
        return token

    def fetch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        domain: str,
    ) -> List[SeriesRecord]:
        """Fetch GHCND daily data for each station in *symbols*.

        Returns one ``SeriesRecord`` per (station, datatype) combination that
        has data in the requested window.
        """
        token = self._get_token()
        records: List[SeriesRecord] = []

        for station_id in symbols:
            ghcnd_id = f"GHCND:{station_id}"

            # CDO API limits queries to 1 year; chunk by year.
            results: List[Dict[str, Any]] = []
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(end, chunk_start + timedelta(days=364))
                params = {
                    "datasetid": "GHCND",
                    "datatypeid": self.datatype,
                    "stationid": ghcnd_id,
                    "startdate": chunk_start.strftime("%Y-%m-%d"),
                    "enddate": chunk_end.strftime("%Y-%m-%d"),
                    "units": "metric",
                    "limit": 1000,
                }
                try:
                    results.extend(_fetch_cdo_pages(params, token))
                except Exception as exc:
                    raise RuntimeError(
                        f"NOAA CDO fetch failed for station {station_id}, "
                        f"chunk [{chunk_start.date()}, {chunk_end.date()}]"
                    ) from exc
                chunk_start = chunk_end + timedelta(days=1)

            if not results:
                continue

            # Filter to the requested datatype and the canonical half-open range.
            rows = []
            for result in results:
                if result.get("datatype") != self.datatype:
                    continue
                try:
                    timestamp = datetime.fromisoformat(
                        result["date"].replace("T00:00:00", "").split("T")[0]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"NOAA CDO returned an invalid date for station {station_id}: {result!r}"
                    ) from exc
                if start <= timestamp < end:
                    rows.append((timestamp, result))
            if not rows:
                continue

            # Sort by date to ensure chronological order
            rows.sort(key=lambda item: item[0])

            timestamps = [timestamp for timestamp, _ in rows]
            if any(
                earlier >= later
                for earlier, later in zip(timestamps, timestamps[1:])
            ):
                raise ValueError(
                    f"NOAA CDO returned duplicate or unordered dates for station {station_id}"
                )

            values = []
            for _, result in rows:
                try:
                    value = float(result["value"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"NOAA CDO returned an invalid value for station {station_id}: {result!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"NOAA CDO returned a non-finite value for station {station_id}"
                    )
                values.append(value)

            series_id = station_id
            # Try to resolve state from station metadata for calendar_symbol
            state = self._resolve_state(station_id)
            meta = {
                "source": "noaa_ghcnd",
                "station_id": station_id,
                "ghcnd_id": ghcnd_id,
                "datatype": self.datatype,
                "dataset": "GHCND",
                "api_units": "metric",
            }
            if state:
                meta["calendar_symbol"] = state

            records.append(SeriesRecord(
                series_id=series_id,
                domain=domain,
                entity=f"Station {station_id}",
                geo=state or "",
                freq="1d",
                start=timestamps[0],
                end=timestamps[-1],
                values=values,
                timestamps=timestamps,
                metadata=meta,
            ))
        return records
