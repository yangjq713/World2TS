from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from world2ts_core.events import DiscoveredEvent
from world2ts_core.offline_discover import search as search_module
from world2ts_core.offline_discover.models import (
    AgentEventSelectionRecord,
    AgentPriorEventHypothesis,
    AgentQueryIntentRecord,
    AgentWindowSearchPlanRecord,
    CurationStrategy,
    HolidayImpactRecord,
    HolidaySeriesImpact,
    SeriesSemanticDimensionsRecord,
    StrategyReviewDecision,
    TimeWindowRecord,
)
from world2ts_core.offline_discover.query_intents import (
    build_agent_query_briefing,
    validate_agent_query_intents,
)
from world2ts_core.offline_discover.search import run_search_iteration
from world2ts_core.offline_discover.strategy import strategy_sha256


NOW = datetime(2024, 1, 1)
END = datetime(2024, 1, 31, 23, 59, 59)


def _strategy(
    *,
    refinements: int = 1,
    initial_queries_per_outline: int = 1,
) -> CurationStrategy:
    return CurationStrategy.model_validate(
        {
            "schema_version": "2.0",
            "dataset_id": "energy",
            "greenfield": True,
            "dataset_understanding": "Daily regional energy demand.",
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
                "default_sources": ["tavily", "anysearch"],
                "top_k": 10,
                "initial_queries_per_outline": initial_queries_per_outline,
                "max_refinement_queries_per_window": refinements,
                "time_window_period": "monthly",
                "stop_policy": "window evidence reviewed",
                "holiday_calendars": [
                    {
                        "country_code": "US",
                        "subdivision_code": "MA",
                        "timezone": "America/New_York",
                        "target_series_ids": ["north"],
                    }
                ],
            },
            "search_intent_outline": [
                {
                    "outline_id": "weather",
                    "event_family": "weather",
                    "priority": "high",
                    "why_relevant": "Weather affects demand.",
                    "context_pathways": ["weather -> demand"],
                    "target_dimensions": ["region"],
                    "time_window_period": "monthly",
                    "initial_query_roles": ["broad"],
                    "in_scope_examples": ["storm"],
                    "out_of_scope_examples": ["sports"],
                    "completion_evidence": "Weather episodes reviewed.",
                },
                {
                    "outline_id": "policy",
                    "event_family": "policy",
                    "priority": "medium",
                    "why_relevant": "Policy affects operations.",
                    "context_pathways": ["policy -> operations"],
                    "target_dimensions": ["region"],
                    "time_window_period": "monthly",
                    "initial_query_roles": ["official"],
                    "in_scope_examples": ["rule"],
                    "out_of_scope_examples": ["commentary"],
                    "completion_evidence": "Policy actions reviewed.",
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


def _windows() -> list[TimeWindowRecord]:
    return [
        TimeWindowRecord(
            window_id="time_monthly_202401",
            period="monthly",
            start=NOW,
            end=END,
            active_series_ids=["north"],
            active_dimension_values={"region": ["north"]},
            outline_ids=["weather", "policy"],
            priority=1.0,
        )
    ]


def _plan() -> AgentWindowSearchPlanRecord:
    return AgentWindowSearchPlanRecord(
        window_id="time_monthly_202401",
        prior_event_hypotheses=[
            AgentPriorEventHypothesis(
                hypothesis_id="prior_grid_emergency",
                event_name="Possible regional grid emergency",
                date_hint="January 2024",
                relevance_pathway="grid emergency -> operations -> demand",
                reasoning="Regional emergencies can alter electricity demand.",
                confidence=0.5,
            )
        ],
        generated_by="agent_active_review",
    )


def _initial(outline: str) -> AgentQueryIntentRecord:
    return AgentQueryIntentRecord(
        query_intent_id=f"{outline}_initial",
        window_id="time_monthly_202401",
        stage="initial",
        outline_id=outline,
        query_text=f"north {outline} January 2024",
        search_start=NOW,
        search_end=END,
        context_hypothesis=f"{outline} may affect demand.",
        relevance_pathway=f"{outline} -> demand",
        generated_by="agent_active_review",
    )


def _refinement() -> AgentQueryIntentRecord:
    return AgentQueryIntentRecord(
        query_intent_id="window_refinement_1",
        window_id="time_monthly_202401",
        stage="window_refinement",
        query_text="north regional grid emergency January 2024",
        search_start=NOW,
        search_end=END,
        context_hypothesis="A cross-topic grid emergency may explain the gap.",
        relevance_pathway="emergency -> operations -> demand",
        feedback_query_intent_ids=["weather_initial", "policy_initial"],
        prior_event_hypothesis_ids=["prior_grid_emergency"],
        refinement_reason="Weather and policy searches missed a known pathway.",
        failed_query_feedback=["No geographically specific episode was accepted."],
        generated_by="agent_active_review",
    )


def _event(
    source: str,
    *,
    title: str = "Regional storm",
    url: str = "https://example.com/storm",
) -> DiscoveredEvent:
    return DiscoveredEvent(
        event_id=f"{source}_{title}",
        source_name=source,
        source_kind="calendar" if source == "holidays" else "web_search",
        title=title,
        description="A dated regional event affected demand.",
        evidence_quality=(
            "structured_record" if source == "holidays" else "web_snippet"
        ),
        timestamp=datetime(2024, 1, 10),
        url=url,
    )


def _write_model(path: Path, model) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, models: list) -> None:
    path.write_text(
        "".join(model.model_dump_json() + "\n" for model in models),
        encoding="utf-8",
    )


def _approved_run(
    tmp_path: Path,
    *,
    strategy: CurationStrategy | None = None,
    intents: list[AgentQueryIntentRecord] | None = None,
) -> CurationStrategy:
    strategy = strategy or _strategy()
    _write_model(tmp_path / "curation_strategy.draft.json", strategy)
    _write_model(tmp_path / "curation_strategy.approved.json", strategy)
    _write_model(
        tmp_path / "strategy_review_decision.json",
        StrategyReviewDecision(
            decision="approve",
            reviewed_strategy_sha256=strategy_sha256(strategy),
            reviewed_at=NOW,
        ),
    )
    _write_jsonl(tmp_path / "time_windows.json", _windows())
    _write_jsonl(tmp_path / "agent_window_search_plans.jsonl", [_plan()])
    if intents is not None:
        _write_jsonl(tmp_path / "agent_query_intents.jsonl", intents)
    return strategy


def _state(result: dict) -> dict:
    return result["search_windows"][0]


def _no_holidays(*args):
    return [], []


def test_briefing_contains_windows_outlines_and_prior_event_contract() -> None:
    briefing = build_agent_query_briefing(
        _strategy(),
        _windows(),
        [
            SeriesSemanticDimensionsRecord(
                series_id="north",
                confidence=1.0,
                dimensions={"region": "north"},
            )
        ],
    )

    assert [window["window_id"] for window in briefing["windows"]] == [
        "time_monthly_202401"
    ]
    assert [
        outline["outline_id"] for outline in briefing["windows"][0]["outlines"]
    ] == ["weather", "policy"]
    assert "agent_window_search_plans.jsonl" in briefing["agent_outputs"]
    assert all("query_text" not in window for window in briefing["windows"])


def test_intent_validation_allows_cross_outline_window_refinement() -> None:
    intents = [_initial("weather"), _initial("policy"), _refinement()]
    validate_agent_query_intents(
        intents,
        strategy=_strategy(),
        time_windows=_windows(),
        window_plans=[_plan()],
    )

    duplicate = _refinement().model_copy(
        update={"query_intent_id": "duplicate", "query_text": " NORTH regional grid emergency, January 2024 "}
    )
    with pytest.raises(ValueError, match="Equivalent query repeated"):
        validate_agent_query_intents(
            [*intents, duplicate],
            strategy=_strategy(),
            time_windows=_windows(),
            window_plans=[_plan()],
        )


def test_all_outlines_run_before_window_refinement(tmp_path) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )
    calls = []

    result = run_search_iteration(
        tmp_path,
        search_fn=lambda source, query, *args: calls.append((source, query)) or [],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )

    assert len(calls) == 4
    assert set(_state(result)["attempted_initial_query_intent_ids"]) == {
        "weather_initial",
        "policy_initial",
    }
    assert _state(result)["status"] == "awaiting_refinement"
    assert result["next_agent_action"] == "write_window_refined_query"


def test_pre_generated_refinement_is_rejected(tmp_path) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy"), _refinement()],
    )

    with pytest.raises(ValueError, match="Do not pre-generate"):
        run_search_iteration(
            tmp_path,
            search_fn=lambda *args: [],
            holiday_fn=_no_holidays,
            source_available_fn=lambda source: True,
        )


def test_multiple_pending_refinements_are_rejected(tmp_path) -> None:
    _approved_run(
        tmp_path,
        strategy=_strategy(refinements=2),
        intents=[_initial("weather"), _initial("policy")],
    )
    run_search_iteration(
        tmp_path,
        search_fn=lambda *args: [],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )
    second_refinement = _refinement().model_copy(
        update={
            "query_intent_id": "window_refinement_2",
            "query_text": "north emergency electricity curtailment January 2024",
        }
    )
    _write_jsonl(
        tmp_path / "agent_query_intents.jsonl",
        [
            _initial("weather"),
            _initial("policy"),
            _refinement(),
            second_refinement,
        ],
    )

    with pytest.raises(ValueError, match="exactly one window refinement"):
        run_search_iteration(
            tmp_path,
            search_fn=lambda *args: [],
            holiday_fn=_no_holidays,
            source_available_fn=lambda source: True,
        )


def test_multi_source_results_are_unioned_and_deduplicated(tmp_path) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )

    run_search_iteration(
        tmp_path,
        search_fn=lambda source, query, *args: [
            _event(source, title=query, url=f"https://example.com/{query}")
        ],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "discovered_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(events) == 2
    assert all(
        event["structured_payload"]["source_hits"] == ["tavily", "anysearch"]
        for event in events
    )


def test_holiday_candidate_does_not_mask_failed_web_search(tmp_path) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )
    impact = HolidayImpactRecord(
        impact_id="holiday_1",
        window_id="time_monthly_202401",
        country_code="US",
        subdivision_code="MA",
        timezone="America/New_York",
        holiday_date=date(2024, 1, 1),
        holiday_name="New Year's Day",
        accepted=True,
        affected_series_ids=["north"],
        series_impacts=[
            HolidaySeriesImpact(
                series_id="north",
                event_count=24,
                baseline_count=192,
                event_mean=2.0,
                baseline_median=1.0,
                pct_difference=100.0,
                robust_z=3.0,
                supported=True,
            )
        ],
    )

    def failed_web_search(*args):
        raise RuntimeError("web unavailable")

    result = run_search_iteration(
        tmp_path,
        search_fn=failed_web_search,
        holiday_fn=lambda *args: (
            [impact],
            [_event("holidays", title="New Year's Day", url="")],
        ),
        source_available_fn=lambda source: True,
    )

    assert _state(result)["status"] == "failed"
    assert result["next_agent_action"] == "retry_failed_search_window"
    assert len(_state(result)["candidate_event_summary_ids"]) == 1
    stored = json.loads(
        (tmp_path / "holiday_impacts.jsonl").read_text(encoding="utf-8")
    )
    assert stored["accepted"] is True


def test_holiday_candidates_with_distinct_scope_ids_do_not_merge(tmp_path) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )
    holiday = _event("holidays", title="Shared Holiday", url="")

    result = run_search_iteration(
        tmp_path,
        search_fn=lambda *args: [],
        holiday_fn=lambda *args: (
            [],
            [
                holiday.model_copy(update={"event_id": "holiday_scope_a"}),
                holiday.model_copy(update={"event_id": "holiday_scope_b"}),
            ],
        ),
        source_available_fn=lambda source: True,
    )

    assert _state(result)["status"] == "awaiting_selection"
    assert len(_state(result)["candidate_event_summary_ids"]) == 2


def test_other_queries_do_not_mask_fully_failed_intent(tmp_path) -> None:
    strategy = _strategy(initial_queries_per_outline=2)
    intents = [
        _initial(outline).model_copy(
            update={
                "query_intent_id": f"{outline}_initial_{index}",
                "query_text": f"north {outline} January 2024 query {index}",
            }
        )
        for outline in ("weather", "policy")
        for index in range(2)
    ]
    _approved_run(
        tmp_path,
        strategy=strategy,
        intents=intents,
    )
    failed_query = "north weather January 2024 query 0"

    def search(source, query, *args):
        if query == failed_query:
            raise RuntimeError(f"{source} weather search unavailable")
        return [_event(source, title=query, url=f"https://example.com/{query}")]

    failed = run_search_iteration(
        tmp_path,
        search_fn=search,
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )
    failed_candidates = _state(failed)["candidate_event_summary_ids"]

    assert _state(failed)["status"] == "failed"
    assert len(failed_candidates) == 3
    assert len(_state(failed)["failure_reasons"]) == 2

    retry_calls = []
    retry = run_search_iteration(
        tmp_path,
        search_fn=lambda source, query, *args: retry_calls.append(
            (source, query)
        ) or [],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: source == "tavily",
        retry_failed=True,
    )

    assert _state(retry)["status"] == "awaiting_selection"
    assert retry_calls == [("tavily", failed_query)]
    assert _state(retry)["candidate_event_summary_ids"] == failed_candidates
    assert len(_state(retry)["attempted_initial_query_intent_ids"]) == 4
    assert len(_state(retry)["normalized_queries"]) == 4
    traces = [
        json.loads(line)
        for line in (tmp_path / "search_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(traces) == 9
    assert len({trace["trace_id"] for trace in traces}) == 9


def test_unavailable_requested_source_is_skipped(tmp_path) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )
    calls = []

    result = run_search_iteration(
        tmp_path,
        search_fn=lambda *args: calls.append(args) or [],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: source == "tavily",
    )

    assert len(calls) == 2
    assert {call[0] for call in calls} == {"tavily"}
    assert _state(result)["status"] == "awaiting_refinement"


def test_all_requested_sources_unavailable_fails_loudly(tmp_path) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )

    with pytest.raises(
        ValueError,
        match="No configured Tavily/AnySearch source is available",
    ):
        run_search_iteration(
            tmp_path,
            search_fn=lambda *args: [],
            holiday_fn=_no_holidays,
            source_available_fn=lambda source: False,
        )


def test_failed_retry_requires_one_available_source(tmp_path) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )

    run_search_iteration(
        tmp_path,
        search_fn=lambda *args: (_ for _ in ()).throw(
            RuntimeError("temporary search failure")
        ),
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )
    trace_path = tmp_path / "search_trace.jsonl"
    traces_before = trace_path.read_text(encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="No configured Tavily/AnySearch source is available",
    ):
        run_search_iteration(
            tmp_path,
            search_fn=lambda *args: [],
            holiday_fn=_no_holidays,
            source_available_fn=lambda source: False,
            retry_failed=True,
        )

    assert trace_path.read_text(encoding="utf-8") == traces_before


def test_rejected_combined_results_enable_topic_unrestricted_refinement(
    tmp_path,
) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )
    first = run_search_iteration(
        tmp_path,
        search_fn=lambda source, query, *args: [
            _event(source, title=query, url=f"https://example.com/{query}")
        ],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )
    candidates = _state(first)["candidate_event_summary_ids"]
    selected = candidates[:1]
    rejected = candidates[1:]
    _write_jsonl(
        tmp_path / "agent_event_selections.jsonl",
        [
            AgentEventSelectionRecord(
                window_id="time_monthly_202401",
                decision="continue_search",
                selected_event_summary_ids=selected,
                rejected_event_summary_ids=rejected,
                rejected_rationales={
                    candidate: "Insufficient geographic specificity."
                    for candidate in rejected
                },
                selected_by="agent_active_review",
            )
        ],
    )

    blocked = run_search_iteration(
        tmp_path,
        search_fn=lambda *args: [],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )
    assert blocked["blocked_reason"] == "agent_window_refined_query_required"

    _write_jsonl(
        tmp_path / "agent_query_intents.jsonl",
        [_initial("weather"), _initial("policy"), _refinement()],
    )
    exhausted = run_search_iteration(
        tmp_path,
        search_fn=lambda *args: [],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )
    assert _state(exhausted)["status"] == "exhausted"
    assert selected[0] in _state(exhausted)["reviewed_candidate_event_summary_ids"]
    traces = [
        json.loads(line)
        for line in (tmp_path / "search_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    refinement_traces = [
        trace for trace in traces if trace["stage"] == "window_refinement"
    ]
    assert {tuple(trace["feedback_query_intent_ids"]) for trace in refinement_traces} == {
        ("weather_initial", "policy_initial")
    }


def test_refinement_returning_only_reviewed_candidate_does_not_require_selection(
    tmp_path,
) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )
    first = run_search_iteration(
        tmp_path,
        search_fn=lambda source, query, *args: [
            _event(source, title=query, url=f"https://example.com/{query}")
        ],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )
    candidates = _state(first)["candidate_event_summary_ids"]
    _write_jsonl(
        tmp_path / "agent_event_selections.jsonl",
        [
            AgentEventSelectionRecord(
                window_id="time_monthly_202401",
                decision="continue_search",
                rejected_event_summary_ids=candidates,
                selected_by="agent_active_review",
            )
        ],
    )
    _write_jsonl(
        tmp_path / "agent_query_intents.jsonl",
        [_initial("weather"), _initial("policy"), _refinement()],
    )

    result = run_search_iteration(
        tmp_path,
        search_fn=lambda source, *args: [
            _event(
                source,
                title="same reviewed event",
                url="https://example.com/north weather January 2024",
            )
        ],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )

    assert _state(result)["status"] == "exhausted"
    assert _state(result)["candidate_event_summary_ids"] == candidates
    assert set(_state(result)["reviewed_candidate_event_summary_ids"]) == set(
        candidates
    )


def test_window_search_concurrency_is_bounded(tmp_path, monkeypatch) -> None:
    strategy = _strategy(initial_queries_per_outline=3)
    intents = [
        _initial(outline).model_copy(
            update={
                "query_intent_id": f"{outline}_initial_{index}",
                "query_text": f"north {outline} January 2024 query {index}",
            }
        )
        for outline in ("weather", "policy")
        for index in range(3)
    ]
    _approved_run(tmp_path, strategy=strategy, intents=intents)
    executor_class = search_module.ThreadPoolExecutor
    worker_limits = []

    def recording_executor(*args, **kwargs):
        worker_limits.append(kwargs.get("max_workers", args[0] if args else None))
        return executor_class(*args, **kwargs)

    monkeypatch.setattr(search_module, "ThreadPoolExecutor", recording_executor)
    calls = []
    run_search_iteration(
        tmp_path,
        search_fn=lambda source, query, *args: calls.append((source, query)) or [],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )

    assert len(calls) == 12
    assert sorted(worker_limits) == [2, 4]
    worker_limits_before_empty_search = list(worker_limits)
    assert (
        search_module._search_intents(
            [],
            strategy,
            lambda source: True,
            lambda *args: [],
        )
        == []
    )
    assert worker_limits == worker_limits_before_empty_search


def test_old_selection_cannot_skip_review_of_refinement_results(tmp_path) -> None:
    _approved_run(
        tmp_path,
        intents=[_initial("weather"), _initial("policy")],
    )
    first = run_search_iteration(
        tmp_path,
        search_fn=lambda source, query, *args: [
            _event(source, title=query, url=f"https://example.com/{query}")
        ],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )
    candidates = _state(first)["candidate_event_summary_ids"]
    _write_jsonl(
        tmp_path / "agent_event_selections.jsonl",
        [
            AgentEventSelectionRecord(
                window_id="time_monthly_202401",
                decision="continue_search",
                rejected_event_summary_ids=candidates,
                selected_by="agent_active_review",
            )
        ],
    )
    _write_jsonl(
        tmp_path / "agent_query_intents.jsonl",
        [_initial("weather"), _initial("policy"), _refinement()],
    )

    refined = run_search_iteration(
        tmp_path,
        search_fn=lambda source, query, *args: [
            _event(source, title="refined result", url="https://example.com/refined")
        ],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )
    waiting = run_search_iteration(
        tmp_path,
        search_fn=lambda *args: [],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )

    assert _state(refined)["status"] == "awaiting_selection"
    assert _state(waiting)["status"] == "awaiting_selection"
    assert waiting["next_agent_action"] == "write_agent_event_selection"

    new_candidates = [
        candidate
        for candidate in _state(refined)["candidate_event_summary_ids"]
        if candidate not in candidates
    ]
    _write_jsonl(
        tmp_path / "agent_event_selections.jsonl",
        [
            AgentEventSelectionRecord(
                window_id="time_monthly_202401",
                decision="continue_search",
                selected_event_summary_ids=candidates[:1],
                rejected_event_summary_ids=candidates[1:],
                selected_by="agent_active_review",
            ),
            AgentEventSelectionRecord(
                window_id="time_monthly_202401",
                decision="complete",
                rejected_event_summary_ids=new_candidates,
                selected_by="agent_active_review",
            ),
        ],
    )
    completed = run_search_iteration(
        tmp_path,
        search_fn=lambda *args: [],
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )

    assert _state(completed)["status"] == "covered"


def test_search_adapter_is_not_called_before_strategy_approval(tmp_path) -> None:
    calls = 0

    def search(*args):
        nonlocal calls
        calls += 1
        return []

    result = run_search_iteration(
        tmp_path,
        search_fn=search,
        holiday_fn=_no_holidays,
        source_available_fn=lambda source: True,
    )

    assert calls == 0
    assert result["blocked_reason"] == "user_strategy_review_required"
