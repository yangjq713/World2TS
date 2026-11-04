"""Time-window construction for offline discovery."""

from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from .models import (
    CurationStrategy,
    DataReadinessReport,
    SeriesSemanticDimensionsRecord,
    TimeWindowPeriod,
    TimeWindowRecord,
)


_FIXED_PERIOD_DAYS = {"weekly": 7, "biweekly": 14}
_CALENDAR_PERIOD_MONTHS = {
    "monthly": 1,
    "quarterly": 3,
    "semiannual": 6,
    "yearly": 12,
}
_PERIODS = set(_FIXED_PERIOD_DAYS) | set(_CALENDAR_PERIOD_MONTHS)


def build_time_windows(
    data_readiness: DataReadinessReport,
    series_semantics: list[SeriesSemanticDimensionsRecord],
    strategy: CurationStrategy,
) -> list[TimeWindowRecord]:
    """Build all strategy-requested periods over verified series coverage."""
    if data_readiness.status != "ready":
        raise ValueError("Time windows require a ready data_readiness_report")
    if strategy.dataset_id != data_readiness.dataset_id:
        raise ValueError("Strategy and data readiness dataset IDs must match")

    semantics_by_id = {item.series_id: item for item in series_semantics}
    if list(semantics_by_id) != data_readiness.series_ids:
        raise ValueError("Series semantics must follow the verified series ID order")

    outlines_by_period: dict[TimeWindowPeriod, list[str]] = defaultdict(list)
    for outline in strategy.search_intent_outline:
        outlines_by_period[outline.time_window_period].append(outline.outline_id)

    profiles = data_readiness.series_profiles
    if not profiles:
        return []
    dataset_start = min(profile.start for profile in profiles)
    dataset_end = max(profile.end for profile in profiles)

    windows: list[TimeWindowRecord] = []
    for period, outline_ids in outlines_by_period.items():
        for window_id, start, end in window_bounds(dataset_start, dataset_end, period):
            active_profiles = [
                profile
                for profile in profiles
                if _overlaps(start, end, profile.start, profile.end)
            ]
            if not active_profiles:
                continue
            active_ids = [profile.series_id for profile in active_profiles]
            windows.append(
                TimeWindowRecord(
                    window_id=window_id,
                    period=period,
                    start=start,
                    end=end,
                    active_series_ids=active_ids,
                    active_dimension_values=_active_dimensions(active_ids, semantics_by_id),
                    outline_ids=outline_ids,
                    priority=float(len(active_ids)),
                )
            )
    return sorted(windows, key=lambda item: (item.start, item.end, item.period, item.window_id))


def window_bounds(
    dataset_start: datetime,
    dataset_end: datetime,
    period: TimeWindowPeriod | str,
) -> list[tuple[str, datetime, datetime]]:
    """Return complete, non-overlapping bounds clipped to dataset coverage."""
    normalized_period = _validated_period(period)
    if dataset_end < dataset_start:
        raise ValueError("Dataset end must not be before dataset start")
    if normalized_period in _CALENDAR_PERIOD_MONTHS:
        bounds = _calendar_bounds(dataset_start, dataset_end, normalized_period)
    else:
        step = timedelta(days=_FIXED_PERIOD_DAYS[normalized_period])
        bounds = []
        start = dataset_start
        while start <= dataset_end:
            end = min(start + step - timedelta(microseconds=1), dataset_end)
            bounds.append(
                (
                    f"time_{normalized_period}_{start.strftime('%Y%m%d')}",
                    start,
                    end,
                )
            )
            start += step
    if len(bounds) > 1 and bounds[-1][1] == bounds[-1][2]:
        _, _, trailing_end = bounds.pop()
        window_id, start, _ = bounds[-1]
        bounds[-1] = (window_id, start, trailing_end)
    return bounds


def _active_dimensions(
    active_ids: list[str],
    semantics_by_id: dict[str, SeriesSemanticDimensionsRecord],
) -> dict[str, Any]:
    values: dict[str, list[Any]] = defaultdict(list)
    for series_id in active_ids:
        for dimension, value in semantics_by_id[series_id].dimensions.items():
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if candidate not in values[dimension]:
                    values[dimension].append(candidate)
    return dict(sorted(values.items()))


def _calendar_bounds(
    dataset_start: datetime,
    dataset_end: datetime,
    period: str,
) -> list[tuple[str, datetime, datetime]]:
    month_step = _CALENDAR_PERIOD_MONTHS[period]
    bounds = []
    current = _calendar_period_start(dataset_start, period)
    while current <= dataset_end:
        end_month = current.month + month_step - 1
        end_year = current.year + (end_month - 1) // 12
        end_month = (end_month - 1) % 12 + 1
        period_end = datetime(
            end_year,
            end_month,
            calendar.monthrange(end_year, end_month)[1],
            23,
            59,
            59,
            999999,
            tzinfo=current.tzinfo,
        )
        start = max(current, dataset_start)
        end = min(period_end, dataset_end)
        bounds.append((_window_label(current, period), start, end))
        next_month = current.month + month_step
        current = datetime(
            current.year + (next_month - 1) // 12,
            (next_month - 1) % 12 + 1,
            1,
            tzinfo=current.tzinfo,
        )
    return bounds


def _calendar_period_start(reference: datetime, period: str) -> datetime:
    if period == "monthly":
        month = reference.month
    elif period == "quarterly":
        month = ((reference.month - 1) // 3) * 3 + 1
    elif period == "semiannual":
        month = 1 if reference.month <= 6 else 7
    else:
        month = 1
    return datetime(reference.year, month, 1, tzinfo=reference.tzinfo)


def _window_label(start: datetime, period: str) -> str:
    if period == "monthly":
        suffix = start.strftime("%Y%m")
    elif period == "quarterly":
        suffix = f"{start.year}Q{(start.month - 1) // 3 + 1}"
    elif period == "semiannual":
        suffix = f"{start.year}H{1 if start.month <= 6 else 2}"
    else:
        suffix = str(start.year)
    return f"time_{period}_{suffix}"


def _validated_period(period: TimeWindowPeriod | str) -> str:
    value = str(period).strip().lower()
    if value not in _PERIODS:
        raise ValueError(
            "time_window_period must be one of weekly, biweekly, monthly, "
            f"quarterly, semiannual, yearly; got {period!r}"
        )
    return value


def _overlaps(
    first_start: datetime,
    first_end: datetime,
    second_start: datetime,
    second_end: datetime,
) -> bool:
    return first_start <= second_end and second_start <= first_end
