from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from world2ts_core.offline_discover.holiday_impact import analyze_holidays
from world2ts_core.offline_discover.models import (
    DataReadinessReport,
    HolidayCalendarSpec,
    SeriesReadinessProfile,
    TimeWindowRecord,
)
from tests.test_offline_discover_readiness import _write_arrow


def _write_readiness(
    tmp_path,
    arrow,
    *,
    series_ids: list[str],
    start: datetime,
    end: datetime,
    freq: str,
    length: int,
    has_explicit_timestamps: bool = False,
) -> None:
    readiness = DataReadinessReport(
        dataset_id="holiday_fixture",
        status="ready",
        arrow_files=[str(arrow)],
        series_count=len(series_ids),
        series_ids=series_ids,
        series_id_order_sha256="a" * 64,
        series_profiles=[
            SeriesReadinessProfile(
                series_id=series_id,
                start=start,
                end=end,
                freq=freq,
                length=length,
                missing_rate=0.0,
                has_explicit_timestamps=has_explicit_timestamps,
            )
            for series_id in series_ids
        ],
        dataset_semantics_coverage=1.0,
        series_semantics_coverage=1.0,
    )
    (tmp_path / "data_readiness_report.json").write_text(
        json.dumps(readiness.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )


def test_holiday_source_keeps_only_series_supported_events(tmp_path) -> None:
    start = datetime(2023, 11, 1)
    end = datetime(2024, 2, 29, 23)
    length = int((end - start).total_seconds() // 3600) + 1
    signal = [1.0] * length
    control = [1.0] * length
    holiday_start = int(
        (datetime(2024, 1, 1) - start).total_seconds() // 3600
    )
    signal[holiday_start : holiday_start + 24] = [3.0] * 24
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        [
            {
                "item_id": "signal",
                "start": start,
                "freq": "H",
                "target": signal,
            },
            {
                "item_id": "control",
                "start": start,
                "freq": "H",
                "target": control,
            },
        ],
    )
    _write_readiness(
        tmp_path,
        arrow,
        series_ids=["signal", "control"],
        start=start,
        end=end,
        freq="H",
        length=length,
    )
    window = TimeWindowRecord(
        window_id="time_monthly_202401",
        period="monthly",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 31, 23, 59, 59),
        active_series_ids=["signal", "control"],
        outline_ids=["weather"],
    )

    impacts, events = analyze_holidays(
        tmp_path,
        window,
        [
            HolidayCalendarSpec(
                country_code="US",
                subdivision_code="MA",
                timezone="America/New_York",
                target_series_ids=["signal", "control"],
            )
        ],
    )

    new_year = next(
        impact
        for impact in impacts
        if impact.holiday_date.isoformat() == "2024-01-01"
    )
    assert new_year.accepted is True
    assert new_year.affected_series_ids == ["signal"]
    event = next(
        event
        for event in events
        if event.structured_payload["holiday_impact_id"] == new_year.impact_id
    )
    assert event.source_name == "holidays"
    assert event.structured_payload["affected_series_ids"] == ["signal"]


def test_holiday_source_uses_explicit_sparse_timestamps_and_calendar_scope(
    tmp_path,
) -> None:
    holiday = datetime(2024, 1, 1, 17, tzinfo=timezone.utc)
    timestamps = sorted(
        holiday + direction * timedelta(weeks=week)
        for week in range(1, 9)
        for direction in (-1, 1)
    )
    timestamps.insert(8, holiday)
    signal = [1.0] * len(timestamps)
    signal[8] = 3.0
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        [
            {
                "item_id": "signal",
                "start": timestamps[0],
                "freq": "D",
                "target": signal,
                "timestamps": timestamps,
            },
            {
                "item_id": "out_of_scope",
                "start": timestamps[0],
                "freq": "D",
                "target": signal,
                "timestamps": timestamps,
            },
        ],
    )
    _write_readiness(
        tmp_path,
        arrow,
        series_ids=["signal", "out_of_scope"],
        start=timestamps[0],
        end=timestamps[-1],
        freq="D",
        length=len(timestamps),
        has_explicit_timestamps=True,
    )
    window = TimeWindowRecord(
        window_id="time_monthly_202401",
        period="monthly",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc),
        active_series_ids=["signal", "out_of_scope"],
        outline_ids=["finance"],
    )

    impacts, events = analyze_holidays(
        tmp_path,
        window,
        [
            HolidayCalendarSpec(
                country_code="US",
                subdivision_code="MA",
                timezone="America/New_York",
                target_series_ids=["signal"],
            ),
            HolidayCalendarSpec(
                country_code="US",
                subdivision_code="MA",
                timezone="America/Chicago",
                target_series_ids=["out_of_scope"],
            ),
        ],
    )

    new_year_impacts = [
        impact
        for impact in impacts
        if impact.holiday_date.isoformat() == "2024-01-01"
    ]
    assert len(new_year_impacts) == 2
    assert len({impact.impact_id for impact in new_year_impacts}) == 2
    new_year = next(
        impact
        for impact in new_year_impacts
        if impact.timezone == "America/New_York"
    )
    assert [item.series_id for item in new_year.series_impacts] == ["signal"]
    assert new_year.series_impacts[0].event_count == 1
    assert new_year.series_impacts[0].baseline_count == 13
    assert new_year.affected_series_ids == ["signal"]
    assert new_year.timezone == "America/New_York"
    chicago_new_year = next(
        impact
        for impact in new_year_impacts
        if impact.timezone == "America/Chicago"
    )
    assert [item.series_id for item in chicago_new_year.series_impacts] == [
        "out_of_scope"
    ]
    assert len({event.event_id for event in events}) == len(events)


def test_holiday_source_converts_explicit_utc_timestamp_to_local_date(tmp_path) -> None:
    local_holiday_utc = datetime(2024, 1, 2, 4, tzinfo=timezone.utc)
    timestamps = sorted(
        local_holiday_utc + direction * timedelta(weeks=week)
        for week in range(1, 9)
        for direction in (-1, 1)
    )
    timestamps.insert(8, local_holiday_utc)
    values = [1.0] * len(timestamps)
    values[8] = 3.0
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        [
            {
                "item_id": "late_session",
                "start": timestamps[0],
                "freq": "D",
                "target": values,
                "timestamps": timestamps,
            }
        ],
    )
    _write_readiness(
        tmp_path,
        arrow,
        series_ids=["late_session"],
        start=timestamps[0],
        end=timestamps[-1],
        freq="D",
        length=len(timestamps),
        has_explicit_timestamps=True,
    )
    window = TimeWindowRecord(
        window_id="time_monthly_202401",
        period="monthly",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc),
        active_series_ids=["late_session"],
        outline_ids=["finance"],
    )

    impacts, events = analyze_holidays(
        tmp_path,
        window,
        [
            HolidayCalendarSpec(
                country_code="US",
                timezone="America/New_York",
                target_series_ids=["late_session"],
            )
        ],
    )

    new_year = next(
        impact for impact in impacts if impact.holiday_date.isoformat() == "2024-01-01"
    )
    assert new_year.accepted is True
    assert new_year.series_impacts[0].event_count == 1
    event = next(
        event
        for event in events
        if event.structured_payload["holiday_impact_id"] == new_year.impact_id
    )
    assert event.timestamp == datetime(2024, 1, 1, 5, tzinfo=timezone.utc)
    assert event.structured_payload["timezone"] == "America/New_York"


def test_holiday_source_preserves_legacy_inferred_local_timeline(tmp_path) -> None:
    start = datetime(2023, 11, 1)
    end = datetime(2024, 2, 29, 23)
    length = int((end - start).total_seconds() // 3600) + 1
    values = [1.0] * length
    holiday_start = int(
        (datetime(2024, 1, 1) - start).total_seconds() // 3600
    )
    values[holiday_start : holiday_start + 24] = [3.0] * 24
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        [{"item_id": "legacy", "start": start, "freq": "H", "target": values}],
    )
    _write_readiness(
        tmp_path,
        arrow,
        series_ids=["legacy"],
        start=start,
        end=end,
        freq="H",
        length=length,
    )
    window = TimeWindowRecord(
        window_id="time_monthly_202401",
        period="monthly",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 31, 23, 59, 59),
        active_series_ids=["legacy"],
        outline_ids=["transport"],
    )

    impacts, events = analyze_holidays(
        tmp_path,
        window,
        [
            HolidayCalendarSpec(
                country_code="US",
                timezone="America/Los_Angeles",
                target_series_ids=["legacy"],
            )
        ],
    )

    new_year = next(
        impact for impact in impacts if impact.holiday_date.isoformat() == "2024-01-01"
    )
    assert new_year.accepted is True
    assert new_year.series_impacts[0].event_count == 24
    event = next(
        event
        for event in events
        if event.structured_payload["holiday_impact_id"] == new_year.impact_id
    )
    assert event.timestamp == datetime(2024, 1, 1)


def test_explicit_utc_adjacent_windows_do_not_share_holiday_observations(
    tmp_path,
) -> None:
    event_timestamps = [
        datetime(2024, 1, 1, 23, 30, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 0, 30, tzinfo=timezone.utc),
    ]
    timestamps = sorted(
        [
            timestamp + direction * timedelta(weeks=week)
            for timestamp in event_timestamps
            for week in range(1, 9)
            for direction in (-1, 1)
        ]
        + event_timestamps
    )
    values = [3.0 if timestamp in event_timestamps else 1.0 for timestamp in timestamps]
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        [
            {
                "item_id": "boundary_signal",
                "start": timestamps[0],
                "freq": "H",
                "target": values,
                "timestamps": timestamps,
            }
        ],
    )
    _write_readiness(
        tmp_path,
        arrow,
        series_ids=["boundary_signal"],
        start=timestamps[0],
        end=timestamps[-1],
        freq="H",
        length=len(timestamps),
        has_explicit_timestamps=True,
    )
    calendar = HolidayCalendarSpec(
        country_code="US",
        timezone="America/New_York",
        target_series_ids=["boundary_signal"],
    )
    windows = [
        TimeWindowRecord(
            window_id="before_midnight_utc",
            period="weekly",
            start=datetime(2024, 1, 1, 23, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 23, 59, 59, 999999, tzinfo=timezone.utc),
            active_series_ids=["boundary_signal"],
            outline_ids=["finance"],
        ),
        TimeWindowRecord(
            window_id="after_midnight_utc",
            period="weekly",
            start=datetime(2024, 1, 2, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, 1, tzinfo=timezone.utc),
            active_series_ids=["boundary_signal"],
            outline_ids=["finance"],
        ),
    ]

    event_counts = []
    for window in windows:
        impacts, _ = analyze_holidays(tmp_path, window, [calendar])
        new_year = next(
            impact
            for impact in impacts
            if impact.holiday_date.isoformat() == "2024-01-01"
        )
        event_counts.append(new_year.series_impacts[0].event_count)

    assert event_counts == [1, 1]
