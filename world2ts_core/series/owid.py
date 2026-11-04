"""OWID (Our World in Data) COVID-19 series source.

Downloads the OWID COVID-19 dataset CSV once and serves slices by ISO country
code. No authentication required.

Dataset: https://covid.ourworldindata.org/data/owid-covid-data.csv
Columns include: iso_code, location, date, new_cases, new_deaths,
                 new_vaccinations, total_cases, total_deaths, …
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd

from world2ts_core.series.model import SeriesRecord
from world2ts_core.registry import SeriesSourceCapability, register_series


_OWID_CSV_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
_REQUIRED_CSV_COLUMNS = {
    "iso_code",
    "location",
    "date",
    "new_cases",
    "new_deaths",
    "new_vaccinations",
}

# Module-level cache — downloaded at most once per process
_cached_df: Optional[pd.DataFrame] = None


def _fetch_owid_csv() -> pd.DataFrame:
    """Download OWID COVID CSV via system curl and return as DataFrame.

    Uses /usr/bin/curl (LibreSSL) to avoid Anaconda OpenSSL issues.
    Separated for easy mocking in tests.
    """
    import os
    import subprocess
    import tempfile

    from world2ts_core.config import get as config_get

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()

    proxy = config_get("http_proxy", default="")

    cmd = [
        "/usr/bin/curl",
        "-f",
        "-sS",
        "-L",
        "--max-time",
        "300",
        "-o",
        tmp.name,
    ]
    if proxy:
        cmd += ["-x", proxy]
    cmd.append(_OWID_CSV_URL)

    try:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=310)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ConnectionError(f"OWID CSV download failed: {exc}") from exc
        if result.returncode != 0:
            raise ConnectionError(
                f"OWID CSV download failed (code {result.returncode}): "
                f"{result.stderr[:200]}"
            )
        if os.path.getsize(tmp.name) == 0:
            raise ConnectionError("OWID CSV download returned an empty response")
        try:
            dataframe = pd.read_csv(tmp.name, low_memory=False)
        except Exception as exc:
            raise ValueError(f"OWID CSV response is not valid CSV: {exc}") from exc
        missing_columns = sorted(_REQUIRED_CSV_COLUMNS - set(dataframe.columns))
        if missing_columns:
            raise ValueError(
                "OWID CSV is missing required columns: " + ", ".join(missing_columns)
            )
        return dataframe
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass


def _get_df() -> pd.DataFrame:
    """Return cached DataFrame, downloading it on first call."""
    global _cached_df
    if _cached_df is None:
        _cached_df = _fetch_owid_csv()
    return _cached_df


_RESAMPLE_TO_FREQ = {
    "W": "1w",
    "W-SUN": "1w",
    "M": "1mo",
    "ME": "1mo",
    "Q": "1q",
    "QE": "1q",
}

_RESAMPLE_ALIASES = {
    "M": "ME",
    "Q": "QE",
}

_HEALTH_METRICS = [
    "new_cases",
    "new_deaths",
    "new_vaccinations",
    "total_cases",
    "total_deaths",
    "total_vaccinations",
    "icu_patients",
    "hosp_patients",
    "reproduction_rate",
]


def _complete_weekly_buckets(
    daily: pd.DataFrame,
    *,
    columns: list[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Aggregate only complete Monday-through-Sunday daily buckets."""

    range_start = pd.Timestamp(start)
    range_end = pd.Timestamp(end)
    indexed = daily.set_index("date")[columns]
    labels: list[pd.Timestamp] = []
    rows: list[pd.Series] = []

    for raw_label, bucket in indexed.groupby(pd.Grouper(freq="W-SUN")):
        label = pd.Timestamp(raw_label)
        expected_dates = pd.date_range(end=label, periods=7, freq="D")
        observed_dates = pd.DatetimeIndex(bucket.index).normalize()
        if (
            len(bucket) != 7
            or not observed_dates.equals(expected_dates)
            or expected_dates[0] < range_start
            or label >= range_end
        ):
            continue
        labels.append(label)
        rows.append(bucket.sum(min_count=1))

    if not rows:
        return pd.DataFrame(
            columns=columns,
            index=pd.DatetimeIndex([], name="date"),
        )
    return pd.DataFrame(rows, index=pd.DatetimeIndex(labels, name="date"))


@register_series
class OwidSource:
    """Fetch COVID-19 time series from Our World in Data.

    Symbols are ISO 3166-1 alpha-3 country codes (e.g. ``USA``, ``GBR``,
    ``IND``).

    Supported metrics (default ``new_cases``):
    - ``new_cases``
    - ``new_deaths``
    - ``new_vaccinations``

    Parameters
    ----------
    metric:
        Column name to extract from the OWID CSV.
    resample:
        Optional pandas offset alias for resampling (e.g. ``"W"`` for weekly,
        ``"M"`` for monthly).  When empty string (the default) no resampling is
        applied and the series is returned at daily frequency.
    """

    capability = SeriesSourceCapability(
        name="owid",
        domains=["health"],
        entity_types=["country"],
        frequencies=["1d", "1w", "1mo"],
        geo=["global"],
        history_days=1095,
        rate_per_min=10,
        auth_required=False,
        reliability="medium",
        description="OWID COVID-19 — daily epidemiological data (cases, deaths, vaccinations) by country",
        primary_target_fields=["new_cases", "new_deaths", "new_vaccinations"],
        target_option="metric",
        adapter_options={
            "metric": ["new_cases", "new_deaths", "new_vaccinations"],
        },
        metadata_dimensions=["iso_code", "location", "metric", "dataset_url"],
        timestamp_semantics="UTC civil dates",
        range_semantics="[start, end)",
    )

    def __init__(self, metric: str = "new_cases", resample: str = ""):
        self.metric = metric
        self._resample = resample

    def _normalized_resample(self) -> str:
        """Normalize legacy pandas offset aliases for newer pandas versions."""
        return _RESAMPLE_ALIASES.get(self._resample, self._resample)

    def fetch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        domain: str,
    ) -> List[SeriesRecord]:
        """Return one ``SeriesRecord`` per ISO code that has data."""
        df = _get_df()

        # Ensure date column is datetime for filtering
        df["date"] = pd.to_datetime(df["date"])

        records: List[SeriesRecord] = []

        for iso_code in symbols:
            country_df = df[df["iso_code"] == iso_code].copy()
            if country_df.empty:
                continue

            # Filter to requested date range
            mask = (country_df["date"] >= pd.Timestamp(start)) & (
                country_df["date"] < pd.Timestamp(end)
            )
            country_df = country_df[mask].sort_values("date")

            if country_df.empty:
                continue

            if self.metric not in country_df.columns:
                continue

            metric_series = country_df[self.metric]
            if metric_series.isna().all():
                continue

            # Drop NaN rows for this metric
            valid = country_df.dropna(subset=[self.metric])
            if valid.empty:
                continue

            # Grab location name before any resampling
            location = str(valid["location"].iloc[0]) if "location" in valid.columns else iso_code

            health_metrics = [col for col in _HEALTH_METRICS if col in valid.columns]

            if self._resample:
                resample_rule = self._normalized_resample()
                aggregate_columns = list(dict.fromkeys([self.metric, *health_metrics]))
                if _RESAMPLE_TO_FREQ.get(self._resample) == "1w":
                    resampled = _complete_weekly_buckets(
                        valid,
                        columns=aggregate_columns,
                        start=start,
                        end=end,
                    )
                else:
                    resampled = (
                        valid.set_index("date")[aggregate_columns]
                        .resample(resample_rule)
                        .sum(min_count=1)
                        .dropna(subset=[self.metric])
                    )
                if resampled.empty:
                    continue
                values = resampled[self.metric].tolist()
                timestamps = [ts.to_pydatetime() for ts in resampled.index]
                freq = _RESAMPLE_TO_FREQ.get(self._resample, self._resample)
            else:
                values = valid[self.metric].tolist()
                timestamps = [ts.to_pydatetime() for ts in valid["date"]]
                freq = "1d"

            series_id = iso_code

            columns = {}
            for col in health_metrics:
                col_data = (
                    resampled[col].tolist()
                    if self._resample
                    else valid[col].tolist()
                )
                if col_data and len(col_data) == len(values):
                    columns[col] = col_data

            meta = {
                "source": "owid_covid",
                "iso_code": iso_code,
                "location": location,
                "metric": self.metric,
                "dataset_url": _OWID_CSV_URL,
            }

            records.append(SeriesRecord(
                series_id=series_id,
                domain=domain,
                entity=location,
                geo=iso_code,
                freq=freq,
                start=timestamps[0],
                end=timestamps[-1],
                values=values,
                timestamps=timestamps,
                columns=columns,
                metadata=meta,
            ))

        return records
