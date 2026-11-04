"""Structured holiday candidates supported by local time-series movement."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median, pstdev
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from world2ts_core.offline_discover.holiday_calendar import country_holiday_dates
from world2ts_core.events import DiscoveredEvent

from .io import load_arrow_examples
from .models import (
    DataReadinessReport,
    HolidayCalendarSpec,
    HolidayImpactRecord,
    HolidaySeriesImpact,
    TimeWindowRecord,
)
from .readiness import frequency_seconds


_BASELINE_WEEKS = 8


def analyze_holidays(
    run_dir: Path,
    window: TimeWindowRecord,
    calendars: list[HolidayCalendarSpec],
) -> tuple[list[HolidayImpactRecord], list[DiscoveredEvent]]:
    """Evaluate official holidays and emit only data-supported candidates."""
    readiness = DataReadinessReport.model_validate(
        json.loads(
            (run_dir / "data_readiness_report.json").read_text(encoding="utf-8")
        )
    )
    examples = {
        example.example_id: example
        for example in load_arrow_examples(
            [Path(path) for path in readiness.arrow_files],
            preserve_physical_order=True,
        )
    }
    impacts: list[HolidayImpactRecord] = []
    events: list[DiscoveredEvent] = []

    for calendar in calendars:
        calendar_timezone = _calendar_timezone(calendar)
        target_series_ids = [
            series_id
            for series_id in calendar.target_series_ids
            if series_id in window.active_series_ids
        ]
        if not target_series_ids:
            continue
        missing_series_ids = [
            series_id for series_id in target_series_ids if series_id not in examples
        ]
        if missing_series_ids:
            raise ValueError(
                "Holiday calendar targets active series absent from Arrow data: "
                f"{missing_series_ids}"
            )
        active_examples = [examples[series_id] for series_id in target_series_ids]
        explicit_timestamps = _explicit_timestamp_mode(active_examples)
        window_start, window_end = holiday_window_bounds(
            window,
            calendar.timezone,
            explicit_timestamps=explicit_timestamps,
        )
        expanded_start = window_start.date() - timedelta(weeks=_BASELINE_WEEKS)
        expanded_end = window_end.date() + timedelta(weeks=_BASELINE_WEEKS)
        calendar_dates = country_holiday_dates(
            calendar.country_code,
            calendar.subdivision_code,
            expanded_start,
            expanded_end,
        )
        excluded_dates = {holiday_date for holiday_date, _ in calendar_dates}
        for holiday_date, holiday_name in calendar_dates:
            if not window_start.date() <= holiday_date <= window_end.date():
                continue
            series_impacts = [
                _series_impact(
                    series_id,
                    examples[series_id],
                    holiday_date,
                    excluded_dates,
                    calendar_timezone,
                    event_window=(window_start, window_end)
                    if explicit_timestamps
                    else None,
                )
                for series_id in target_series_ids
            ]
            affected = [
                impact.series_id for impact in series_impacts if impact.supported
            ]
            impact_id = holiday_impact_id(
                window.window_id,
                calendar.country_code,
                calendar.subdivision_code,
                calendar.timezone,
                calendar.target_series_ids,
                holiday_date,
                holiday_name,
            )
            impact = HolidayImpactRecord(
                impact_id=impact_id,
                window_id=window.window_id,
                country_code=calendar.country_code,
                subdivision_code=calendar.subdivision_code,
                timezone=calendar.timezone,
                holiday_date=holiday_date,
                holiday_name=holiday_name,
                accepted=bool(affected),
                affected_series_ids=affected,
                series_impacts=series_impacts,
            )
            impacts.append(impact)
            if impact.accepted:
                events.append(
                    _holiday_event(
                        impact,
                        calendar_timezone,
                        explicit_timestamps=explicit_timestamps,
                    )
                )

    return impacts, events


def _series_impact(
    series_id: str,
    example,
    holiday_date: date,
    excluded_dates: set[date],
    calendar_timezone: ZoneInfo,
    event_window: tuple[datetime, datetime] | None,
) -> HolidaySeriesImpact:
    timestamps = _observation_timestamps(example, calendar_timezone)
    if len(timestamps) != len(example.target):
        raise ValueError(
            f"Series {series_id!r} timestamp count {len(timestamps)} does not "
            f"match target length {len(example.target)}"
        )

    values_by_local_slot: dict[tuple[date, time], list[float]] = defaultdict(list)
    event_values_by_local_time: dict[time, list[float]] = defaultdict(list)
    for timestamp, raw_value in zip(timestamps, example.target):
        value = float(raw_value)
        if not math.isfinite(value):
            continue
        local_time = _wall_clock_time(timestamp)
        values_by_local_slot[(timestamp.date(), local_time)].append(value)
        if timestamp.date() != holiday_date:
            continue
        if event_window is not None and not (
            event_window[0] <= timestamp <= event_window[1]
        ):
            continue
        event_values_by_local_time[local_time].append(value)

    event_slots = list(event_values_by_local_time.items())
    event_values = [value for _, values in event_slots for value in values]
    baseline_values: list[float] = []
    for local_time, values in event_slots:
        for week in range(1, _BASELINE_WEEKS + 1):
            for direction in (-1, 1):
                candidate_date = holiday_date + direction * timedelta(weeks=week)
                if candidate_date in excluded_dates:
                    continue
                candidates = values_by_local_slot.get((candidate_date, local_time), [])
                for _ in values:
                    baseline_values.extend(candidates)

    minimum_baseline = max(4, len(event_values) * 4)
    if not event_values or len(baseline_values) < minimum_baseline:
        return HolidaySeriesImpact(
            series_id=series_id,
            event_count=len(event_values),
            baseline_count=len(baseline_values),
            supported=False,
        )

    event_mean = sum(event_values) / len(event_values)
    baseline_median = median(baseline_values)
    absolute_deviations = [
        abs(value - baseline_median) for value in baseline_values
    ]
    robust_scale = 1.4826 * median(absolute_deviations)
    if robust_scale == 0:
        robust_scale = pstdev(baseline_values)
    if robust_scale == 0:
        robust_scale = max(abs(baseline_median) * 0.01, 1e-12)
    difference = event_mean - baseline_median
    robust_z = difference / robust_scale
    pct_difference = (
        difference / abs(baseline_median) * 100
        if abs(baseline_median) > 1e-12
        else None
    )
    supported = abs(robust_z) >= 2.0 or (
        pct_difference is not None
        and abs(pct_difference) >= 20.0
        and abs(robust_z) >= 1.2
    )
    return HolidaySeriesImpact(
        series_id=series_id,
        event_count=len(event_values),
        baseline_count=len(baseline_values),
        event_mean=round(event_mean, 6),
        baseline_median=round(baseline_median, 6),
        pct_difference=(
            round(pct_difference, 6) if pct_difference is not None else None
        ),
        robust_z=round(robust_z, 6),
        supported=supported,
    )


def _calendar_timezone(calendar: HolidayCalendarSpec) -> ZoneInfo:
    try:
        return ZoneInfo(calendar.timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Holiday calendar has unknown IANA timezone: {calendar.timezone!r}"
        ) from exc


def _explicit_timestamp_mode(examples: list) -> bool:
    explicit_modes = {example.timestamps is not None for example in examples}
    if len(explicit_modes) > 1:
        raise ValueError(
            "Holiday calendar target series mix explicit UTC timestamps with "
            "legacy inferred timelines in one window"
        )
    return explicit_modes == {True}


def holiday_window_bounds(
    window: TimeWindowRecord,
    timezone_name: str,
    *,
    explicit_timestamps: bool | None = None,
) -> tuple[datetime, datetime]:
    """Return exact window instants in one approved calendar timezone."""
    try:
        calendar_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Holiday calendar has unknown IANA timezone: {timezone_name!r}"
        ) from exc
    if explicit_timestamps is None:
        awareness = {
            timestamp.tzinfo is not None and timestamp.utcoffset() is not None
            for timestamp in (window.start, window.end)
        }
        if len(awareness) != 1:
            raise ValueError("Holiday window bounds mix aware and naive timestamps")
        explicit_timestamps = awareness == {True}
    return (
        _localize_timestamp(
            window.start,
            calendar_timezone,
            naive_is_utc=explicit_timestamps,
        ),
        _localize_timestamp(
            window.end,
            calendar_timezone,
            naive_is_utc=explicit_timestamps,
        ),
    )


def _observation_timestamps(example, calendar_timezone: ZoneInfo) -> list[datetime]:
    if example.timestamps is not None:
        return [
            _localize_timestamp(
                timestamp,
                calendar_timezone,
                naive_is_utc=True,
            )
            for timestamp in example.timestamps
        ]

    step = timedelta(seconds=frequency_seconds(example.freq))
    return [
        _localize_timestamp(
            example.start + index * step,
            calendar_timezone,
            naive_is_utc=False,
        )
        for index in range(len(example.target))
    ]


def _localize_timestamp(
    timestamp: datetime,
    calendar_timezone: ZoneInfo,
    *,
    naive_is_utc: bool,
) -> datetime:
    if timestamp.tzinfo is None:
        source_timezone = timezone.utc if naive_is_utc else calendar_timezone
        timestamp = timestamp.replace(tzinfo=source_timezone)
    return timestamp.astimezone(calendar_timezone)


def _wall_clock_time(timestamp: datetime) -> time:
    return time(
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
        timestamp.microsecond,
    )


def _holiday_event(
    impact: HolidayImpactRecord,
    calendar_timezone: ZoneInfo,
    *,
    explicit_timestamps: bool,
) -> DiscoveredEvent:
    place = "/".join(
        part
        for part in (impact.country_code, impact.subdivision_code)
        if part
    )
    title = f"{place} holiday: {impact.holiday_name}"
    description = (
        f"{impact.holiday_name} fell on {impact.holiday_date.isoformat()} in "
        f"{place}; local analysis found marked movement in "
        f"{len(impact.affected_series_ids)} active series."
    )
    event_timestamp = datetime.combine(impact.holiday_date, time.min)
    if explicit_timestamps:
        event_timestamp = event_timestamp.replace(
            tzinfo=calendar_timezone
        ).astimezone(timezone.utc)
    return DiscoveredEvent(
        event_id="holiday_" + impact.impact_id.removeprefix("holiday_"),
        source_name="holidays",
        source_kind="calendar",
        title=title,
        description=description,
        evidence_quality="structured_record",
        timestamp=event_timestamp,
        timestamp_precision="day",
        geo_scope="regional" if impact.subdivision_code else "national",
        geo_tags=[
            part
            for part in (impact.subdivision_code, impact.country_code)
            if part
        ],
        structured_payload={
            "holiday_impact_id": impact.impact_id,
            "country_code": impact.country_code,
            "subdivision_code": impact.subdivision_code,
            "timezone": impact.timezone,
            "holiday_name": impact.holiday_name,
            "affected_series_ids": impact.affected_series_ids,
            "series_impacts": [
                item.model_dump(mode="json") for item in impact.series_impacts
            ],
            "support_rule": (
                "abs(pct_difference)>=20 and abs(robust_z)>=1.2, "
                "or abs(robust_z)>=2.0"
            ),
        },
    )


def holiday_impact_id(
    window_id: str,
    country_code: str,
    subdivision_code: str,
    timezone_name: str,
    target_series_ids: list[str],
    holiday_date: date,
    holiday_name: str,
) -> str:
    identity = "|".join(
        (
            window_id,
            country_code,
            subdivision_code,
            timezone_name,
            ",".join(sorted(target_series_ids)),
            holiday_date.isoformat(),
            holiday_name,
        )
    )
    return "holiday_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
