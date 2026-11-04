from __future__ import annotations

from datetime import datetime, timedelta

from world2ts_core.offline_discover.models import (
    CurationStrategy,
    DataReadinessReport,
    SeriesReadinessProfile,
    SeriesSemanticDimensionsRecord,
)
from world2ts_core.offline_discover.time_windows import (
    build_time_windows,
    window_bounds,
)


def _readiness() -> DataReadinessReport:
    profiles = [
        SeriesReadinessProfile(
            series_id="north",
            start=datetime(2024, 1, 15),
            end=datetime(2024, 3, 10),
            freq="1d",
            length=56,
            missing_rate=0.0,
            has_explicit_timestamps=False,
        ),
        SeriesReadinessProfile(
            series_id="south",
            start=datetime(2024, 2, 10),
            end=datetime(2024, 3, 20),
            freq="1d",
            length=40,
            missing_rate=0.0,
            has_explicit_timestamps=False,
        ),
    ]
    return DataReadinessReport(
        dataset_id="energy",
        status="ready",
        arrow_files=["energy.arrow"],
        series_count=2,
        series_ids=["north", "south"],
        series_id_order_sha256="a" * 64,
        series_profiles=profiles,
        dataset_semantics_coverage=1.0,
        series_semantics_coverage=1.0,
    )


def _semantics() -> list[SeriesSemanticDimensionsRecord]:
    return [
        SeriesSemanticDimensionsRecord(
            series_id="north",
            confidence=1.0,
            dimensions={"region": "north", "market": "power"},
        ),
        SeriesSemanticDimensionsRecord(
            series_id="south",
            confidence=1.0,
            dimensions={"region": "south", "market": "power"},
        ),
    ]


def _strategy() -> CurationStrategy:
    common = {
        "event_family": "weather",
        "priority": "high",
        "why_relevant": "Weather affects energy operations.",
        "context_pathways": ["weather -> operations"],
        "target_dimensions": ["region"],
        "initial_query_roles": ["broad"],
        "in_scope_examples": ["storms"],
        "out_of_scope_examples": ["sports"],
        "completion_evidence": "Each active region and period was reviewed.",
    }
    return CurationStrategy.model_validate(
        {
            "schema_version": "2.0",
            "dataset_id": "energy",
            "greenfield": True,
            "dataset_understanding": "Daily regional energy series.",
            "event_policy": {
                "allow_weak_context": True,
                "allow_retrospective_context": True,
                "allow_forecasts": False,
                "exclude_target_only_milestones": True,
                "canonical_date_preference": ["actual_start"],
                "duplicate_policy": "merge_same_episode",
                "compound_record_policy": "split_distinct_episodes",
            },
            "search_policy": {
                "default_sources": ["tavily"],
                "top_k": 10,
                "initial_queries_per_outline": 1,
                "max_refinement_queries_per_window": 2,
                "time_window_period": "monthly",
                "stop_policy": "outline coverage",
                "holiday_calendars": [
                    {
                        "country_code": "US",
                        "timezone": "America/New_York",
                        "target_series_ids": ["north", "south"],
                    }
                ],
            },
            "search_intent_outline": [
                {
                    **common,
                    "outline_id": "monthly_weather",
                    "time_window_period": "monthly",
                },
                {
                    **common,
                    "outline_id": "quarterly_weather",
                    "time_window_period": "quarterly",
                },
            ],
            "projection_policy": {
                "allowed_dimensions": ["region"],
                "allow_dataset_wide": True,
                "ambiguous_scope_action": "hold_for_agent_review",
                "link_time_policy": "event_must_overlap_series_coverage",
            },
            "quality_policy": {
                "zero_event_series_allowed": True,
                "holiday_share_is_diagnostic_only": True,
                "require_user_final_approval": True,
            },
        }
    )


def test_mixed_outline_periods_create_distinct_window_sets() -> None:
    windows = build_time_windows(_readiness(), _semantics(), _strategy())

    monthly = [window for window in windows if window.period == "monthly"]
    quarterly = [window for window in windows if window.period == "quarterly"]
    assert [window.outline_ids for window in monthly] == [
        ["monthly_weather"],
        ["monthly_weather"],
        ["monthly_weather"],
    ]
    assert [window.outline_ids for window in quarterly] == [["quarterly_weather"]]


def test_windows_cover_exact_dataset_boundaries_without_gaps() -> None:
    bounds = window_bounds(
        datetime(2024, 1, 15),
        datetime(2024, 3, 20),
        "monthly",
    )

    assert bounds[0][1] == datetime(2024, 1, 15)
    assert bounds[-1][2] == datetime(2024, 3, 20)
    for previous, current in zip(bounds, bounds[1:]):
        assert previous[2] + timedelta(microseconds=1) == current[1]


def test_boundary_only_tail_is_merged_into_previous_window() -> None:
    bounds = window_bounds(
        datetime(2014, 1, 1, 1),
        datetime(2017, 1, 1),
        "semiannual",
    )

    assert len(bounds) == 6
    assert bounds[-1][0] == "time_semiannual_2016H2"
    assert bounds[-1][2] == datetime(2017, 1, 1)


def test_active_series_and_dimensions_follow_coverage_overlap() -> None:
    monthly = [
        window
        for window in build_time_windows(_readiness(), _semantics(), _strategy())
        if window.period == "monthly"
    ]

    assert monthly[0].active_series_ids == ["north"]
    assert monthly[0].active_dimension_values == {
        "market": ["power"],
        "region": ["north"],
    }
    assert monthly[1].active_series_ids == ["north", "south"]
    assert monthly[1].active_dimension_values["region"] == ["north", "south"]
    assert monthly[2].active_series_ids == ["north", "south"]
