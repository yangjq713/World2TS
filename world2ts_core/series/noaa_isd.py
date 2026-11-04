"""NOAA Integrated Surface Data (ISD) hourly weather series source.

Downloads hourly weather observations from NOAA's ISD archive.
No authentication required.

Dataset: https://www.ncei.noaa.gov/data/global-hourly/access/{YEAR}/{station}.csv
Key columns: STATION, DATE, TMP (temperature), DEW (dewpoint), WND (wind)

TMP format: "+0125,1" → value is int("+0125") / 10.0 = 12.5°C
"+9999" means missing — skip those rows.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd

from world2ts_core.series.model import SeriesRecord
from world2ts_core.registry import SeriesSourceCapability, register_series


_ISD_BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"


def _fetch_isd_csv(station: str, year: int) -> pd.DataFrame:
    """Download a single station/year CSV from NOAA ISD via system curl.

    Uses /usr/bin/curl (LibreSSL) to avoid Anaconda OpenSSL issues.
    Separated for easy mocking in tests.
    """
    from world2ts_core.config import get as config_get

    url = f"{_ISD_BASE_URL}/{year}/{station}.csv"
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()

    cmd = ["/usr/bin/curl", "-f", "-sS", "--max-time", "60", "-o", tmp.name]
    proxy = config_get("http_proxy", default="")
    if proxy:
        cmd += ["-x", proxy]
    cmd.append(url)

    try:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=65)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ConnectionError(
                f"NOAA ISD curl execution failed for station {station}, year {year}: {exc}"
            ) from exc
        if result.returncode != 0:
            raise ConnectionError(
                f"NOAA ISD curl failed for station {station}, year {year} "
                f"(code {result.returncode}): {result.stderr[:200]}"
            )
        if os.path.getsize(tmp.name) == 0:
            raise ConnectionError(
                f"NOAA ISD returned an empty response for station {station}, year {year}"
            )
        try:
            return pd.read_csv(tmp.name, low_memory=False)
        except Exception as exc:
            raise ValueError(
                f"NOAA ISD returned invalid CSV for station {station}, year {year}: {exc}"
            ) from exc
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass


def _parse_tmp(raw: str) -> float | None:
    """Parse ISD TMP field (e.g. '+0125,1') into degrees Celsius.

    Returns None for missing data (value part is '+9999').
    """
    if not isinstance(raw, str):
        return None
    value_part = raw.split(",")[0]
    if value_part.strip() == "+9999":
        return None
    try:
        return int(value_part) / 10.0
    except (ValueError, TypeError):
        return None


@register_series
class NoaaIsdSource:
    """Fetch hourly weather data from NOAA Integrated Surface Data (ISD).

    Symbols are ISD station identifiers, e.g. ``72505394728``
    (JFK International Airport, NY).

    The adapter downloads one CSV per (station, year) from the NOAA ISD
    archive, iterating over all years in the requested date range, then
    concatenates the results.

    Default field: temperature (``TMP``).
    """

    capability = SeriesSourceCapability(
        name="noaa_isd",
        domains=["weather"],
        entity_types=["weather_station"],
        frequencies=["1h"],
        geo=["global"],
        history_days=3650,
        rate_per_min=30,
        auth_required=False,
        reliability="high",
        description="NOAA ISD — hourly weather observations (temperature, dewpoint, wind) from global stations",
        primary_target_fields=["tmp", "dew"],
        target_option="field",
        adapter_options={"field": ["TMP", "DEW"]},
        metadata_dimensions=["station_id", "field", "dataset_url"],
        timestamp_semantics="UTC",
        range_semantics="[start, end)",
    )

    def __init__(self, field: str = "TMP"):
        self.field = field

    def fetch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        domain: str,
    ) -> List[SeriesRecord]:
        """Return one ``SeriesRecord`` per station that has data."""
        records: List[SeriesRecord] = []

        start_utc = (
            start.replace(tzinfo=timezone.utc)
            if start.tzinfo is None
            else start.astimezone(timezone.utc)
        )
        end_utc = (
            end.replace(tzinfo=timezone.utc)
            if end.tzinfo is None
            else end.astimezone(timezone.utc)
        )
        if start_utc >= end_utc:
            raise ValueError("NOAA ISD requires start to be earlier than end")
        start_year = start_utc.year
        end_year = (end_utc - timedelta(microseconds=1)).year

        for station in symbols:
            frames: List[pd.DataFrame] = []
            for year in range(start_year, end_year + 1):
                try:
                    df_year = _fetch_isd_csv(station, year)
                except Exception as exc:
                    raise RuntimeError(
                        f"NOAA ISD fetch failed for station {station}, year {year}"
                    ) from exc
                missing_columns = sorted({"DATE", self.field} - set(df_year.columns))
                if missing_columns:
                    raise ValueError(
                        f"NOAA ISD file for station {station}, year {year} "
                        f"is missing required columns: {', '.join(missing_columns)}"
                    )
                frames.append(df_year)

            df = pd.concat(frames, ignore_index=True)

            parsed_dates = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
            if parsed_dates.isna().any():
                raise ValueError(
                    f"NOAA ISD file for station {station} contains invalid DATE values"
                )
            df["DATE"] = parsed_dates

            complete_start = pd.Timestamp(start_utc).ceil("h")
            complete_end = pd.Timestamp(end_utc).floor("h")
            mask = (df["DATE"] >= complete_start) & (df["DATE"] < complete_end)
            df = df[mask].sort_values("DATE").reset_index(drop=True)

            if df.empty:
                continue

            df["_HOUR"] = df["DATE"].dt.floor("h")
            df["_TARGET"] = df[self.field].apply(_parse_tmp)
            hourly_target = (
                df.dropna(subset=["_TARGET"])
                .groupby("_HOUR", sort=True)["_TARGET"]
                .mean()
            )
            hourly_target = hourly_target[
                (hourly_target.index >= complete_start)
                & (hourly_target.index < complete_end)
            ]

            if hourly_target.empty:
                continue

            values = [float(value) for value in hourly_target.tolist()]
            timestamps = [timestamp.to_pydatetime() for timestamp in hourly_target.index]
            if any(
                earlier >= later
                for earlier, later in zip(timestamps, timestamps[1:])
            ):
                raise RuntimeError(
                    f"NOAA ISD hourly aggregation produced a non-increasing axis for {station}"
                )

            # Aggregate auxiliary variables independently onto the target hour axis.
            columns = {self.field.lower(): values}
            for extra_field in ["TMP", "DEW", "WND"]:
                if extra_field == self.field:
                    continue
                if extra_field in df.columns:
                    extra_values = df[extra_field].apply(_parse_tmp)
                    extra_hourly = (
                        pd.DataFrame({"hour": df["_HOUR"], "value": extra_values})
                        .dropna(subset=["value"])
                        .groupby("hour", sort=True)["value"]
                        .mean()
                        .reindex(hourly_target.index)
                    )
                    if extra_hourly.notna().any():
                        columns[extra_field.lower()] = extra_hourly.tolist()

            meta = {
                "source": "noaa_isd",
                "station_id": station,
                "field": self.field,
                "dataset_url": _ISD_BASE_URL,
                "hourly_aggregation": "mean_of_finite_observations",
            }

            records.append(SeriesRecord(
                series_id=station,
                domain=domain,
                entity=f"Station {station}",
                geo="",
                freq="1h",
                start=timestamps[0],
                end=timestamps[-1],
                values=values,
                timestamps=timestamps,
                columns=columns,
                metadata=meta,
            ))

        return records
