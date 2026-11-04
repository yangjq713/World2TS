"""Window-level feedback search with structured holiday evidence."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from world2ts_core.events import DiscoveredEvent

from . import source_runtime
from .holiday_impact import analyze_holidays
from .models import (
    AgentEventSelectionRecord,
    AgentQueryIntentRecord,
    CurationStrategy,
    HolidayCalendarSpec,
    HolidayImpactRecord,
    SearchTraceRecord,
    SearchWindowStateRecord,
    TimeWindowRecord,
)
from .query_intents import (
    QUERY_INTENTS_FILENAME,
    WINDOW_PLAN_FILENAME,
    load_agent_query_intents,
    load_agent_window_search_plans,
    normalize_query,
)
from .strategy import APPROVED_FILENAME, strategy_search_status


SearchFn = Callable[[str, str, datetime, datetime], list[DiscoveredEvent]]
HolidayFn = Callable[
    [Path, TimeWindowRecord, list[HolidayCalendarSpec]],
    tuple[list[HolidayImpactRecord], list[DiscoveredEvent]],
]
SourceAvailabilityFn = Callable[[str], bool]

_STATE_FILENAME = "search_state.json"
_TRACE_FILENAME = "search_trace.jsonl"
_HOLIDAY_IMPACTS_FILENAME = "holiday_impacts.jsonl"
_EVENTS_FILENAME = "discovered_events.jsonl"
_SELECTIONS_FILENAME = "agent_event_selections.jsonl"
_MAX_SEARCH_WORKERS = 4
_TERMINAL_STATUSES = {
    "covered",
    "exhausted",
    "no_event_found",
    "failed",
    "stopped",
}


def run_search_iteration(
    run_dir: Path,
    *,
    search_fn: SearchFn = source_runtime._search_source,
    holiday_fn: HolidayFn = analyze_holidays,
    source_available_fn: SourceAvailabilityFn | None = None,
    retry_failed: bool = False,
) -> dict:
    """Advance one complete time window until the next Agent action."""
    gate = strategy_search_status(run_dir)
    if not gate["can_search"]:
        return {
            "blocked_reason": gate["blocked_reason"],
            "next_agent_action": gate["next_agent_action"],
            "search_windows": [],
        }

    strategy = _read_model(run_dir / APPROVED_FILENAME, CurationStrategy)
    windows = _read_models(run_dir / "time_windows.json", TimeWindowRecord)
    states = _load_or_initialize_states(run_dir, windows)
    traces = _read_optional_models(run_dir / _TRACE_FILENAME, SearchTraceRecord)
    impacts = _read_optional_models(
        run_dir / _HOLIDAY_IMPACTS_FILENAME,
        HolidayImpactRecord,
    )
    events = _read_optional_models(run_dir / _EVENTS_FILENAME, DiscoveredEvent)

    plans_path = run_dir / WINDOW_PLAN_FILENAME
    if not plans_path.exists():
        active = next(
            (state for state in states if state.status not in _TERMINAL_STATUSES),
            None,
        )
        if active:
            active.next_agent_action = "write_agent_window_search_plans"
        _save_search_artifacts(run_dir, states, traces, impacts, events)
        return _status_payload(
            states,
            active,
            blocked_reason="agent_window_search_plans_required",
        )
    plans = load_agent_window_search_plans(plans_path, time_windows=windows)

    intents_path = run_dir / QUERY_INTENTS_FILENAME
    intents = (
        load_agent_query_intents(
            intents_path,
            strategy=strategy,
            time_windows=windows,
            window_plans=plans,
        )
        if intents_path.exists()
        else []
    )
    selections = _read_optional_models(
        run_dir / _SELECTIONS_FILENAME, AgentEventSelectionRecord
    )

    _apply_selections(
        states,
        selections,
        traces,
        max_refinements=strategy.search_policy.max_refinement_queries_per_window,
    )
    retry_intent_ids_by_window: dict[str, set[str]] = {}
    if retry_failed:
        intent_by_id = {intent.query_intent_id: intent for intent in intents}
        failed_state = next(
            (state for state in states if state.status == "failed"),
            None,
        )
        if failed_state is not None:
            retry_intent_ids = _failed_intent_ids(
                failed_state,
                traces,
                intent_by_id,
                strategy,
            )
            if not retry_intent_ids:
                raise ValueError(
                    "No fully failed query intent can be retried for "
                    f"{failed_state.window_id!r}"
                )
            retry_intent_ids_by_window[failed_state.window_id] = retry_intent_ids
            failed_state.status = "awaiting_refinement"
            failed_state.next_agent_action = "retry_failed_search_window"
    state = next(
        (item for item in states if item.status not in _TERMINAL_STATUSES),
        None,
    )
    if state is None:
        _save_search_artifacts(run_dir, states, traces, impacts, events)
        return _status_payload(states)

    window = next(item for item in windows if item.window_id == state.window_id)
    window_intents = [
        intent for intent in intents if intent.window_id == state.window_id
    ]
    retry_intent_ids = retry_intent_ids_by_window.get(state.window_id, set())
    attempted_ids = {
        *state.attempted_initial_query_intent_ids,
        *state.attempted_refinement_query_intent_ids,
    }
    if state.status == "awaiting_selection":
        state.next_agent_action = "write_agent_event_selection"
        _save_search_artifacts(run_dir, states, traces, impacts, events)
        return _status_payload(states, state)

    if retry_intent_ids:
        to_execute = [
            intent
            for intent in window_intents
            if intent.query_intent_id in retry_intent_ids
        ]
        if {intent.query_intent_id for intent in to_execute} != retry_intent_ids:
            raise ValueError(
                f"Failed query intents are missing for retry in {state.window_id!r}"
            )
        run_holidays = False
    elif state.status == "pending":
        if any(intent.stage == "window_refinement" for intent in window_intents):
            raise ValueError(
                "Do not pre-generate window refinements before all initial "
                f"queries run for {state.window_id!r}"
            )
        to_execute = _initial_intents(
            window,
            window_intents,
            strategy.search_policy.initial_queries_per_outline,
        )
        if not to_execute:
            state.next_agent_action = "write_initial_query_intents"
            _save_search_artifacts(run_dir, states, traces, impacts, events)
            return _status_payload(
                states,
                state,
                blocked_reason="agent_query_intents_required",
            )
        run_holidays = not state.holiday_analysis_complete
    else:
        pending_refinements = [
            intent
            for intent in window_intents
            if intent.stage == "window_refinement"
            and intent.query_intent_id not in attempted_ids
        ]
        if not pending_refinements:
            state.next_agent_action = "write_window_refined_query"
            _save_search_artifacts(run_dir, states, traces, impacts, events)
            return _status_payload(
                states,
                state,
                blocked_reason="agent_window_refined_query_required",
            )
        if len(pending_refinements) != 1:
            raise ValueError(
                "Write exactly one window refinement after reviewing the latest "
                f"search feedback for {state.window_id!r}"
            )
        to_execute = pending_refinements
        attempted = {
            *state.attempted_initial_query_intent_ids,
            *state.attempted_refinement_query_intent_ids,
        }
        missing_feedback = sorted(
            set(to_execute[0].feedback_query_intent_ids) - attempted
        )
        if missing_feedback:
            raise ValueError(
                "Window refinement references unattempted query intents: "
                + ", ".join(missing_feedback)
            )
        run_holidays = False

    available = source_available_fn or _source_available
    with ThreadPoolExecutor(max_workers=2 if run_holidays else 1) as executor:
        search_future = executor.submit(
            _search_intents,
            to_execute,
            strategy,
            available,
            search_fn,
        )
        holiday_future = (
            executor.submit(
                holiday_fn,
                run_dir,
                window,
                strategy.search_policy.holiday_calendars,
            )
            if run_holidays
            else None
        )
        source_results = search_future.result()
        new_impacts, holiday_events = (
            holiday_future.result() if holiday_future else ([], [])
        )

    new_candidate_ids: list[str] = []
    events_by_id = {event.event_summary_id: event for event in events}
    failures_by_intent: dict[str, int] = {}
    sources_by_intent: dict[str, int] = {}
    trace_occurrences: dict[tuple[str, str], int] = {}
    for trace in traces:
        key = (trace.query_intent_id, trace.source)
        trace_occurrences[key] = trace_occurrences.get(key, 0) + 1
    for intent, source, returned, failure, started_at, elapsed_ms in source_results:
        sources_by_intent[intent.query_intent_id] = (
            sources_by_intent.get(intent.query_intent_id, 0) + 1
        )
        if failure:
            failures_by_intent[intent.query_intent_id] = (
                failures_by_intent.get(intent.query_intent_id, 0) + 1
            )
            state.failure_reasons.append(f"{source}: {failure}")
        candidate_ids = []
        for event in returned:
            _set_web_provenance(event, intent, source)
            candidate_ids.append(event.event_summary_id)
            new_candidate_ids.append(event.event_summary_id)
            _merge_event(events_by_id, event)
        trace_key = (intent.query_intent_id, source)
        trace_occurrence = trace_occurrences.get(trace_key, 0)
        traces.append(
            SearchTraceRecord(
                trace_id=_trace_id(intent, source, trace_occurrence),
                window_id=intent.window_id,
                query_intent_id=intent.query_intent_id,
                stage=intent.stage,
                outline_id=intent.outline_id,
                query_text=intent.query_text,
                source=source,
                started_at=started_at,
                elapsed_ms=elapsed_ms,
                returned_count=len(returned),
                failure=failure,
                event_summary_ids=candidate_ids,
                feedback_query_intent_ids=intent.feedback_query_intent_ids,
            )
        )
        trace_occurrences[trace_key] = trace_occurrence + 1

    for event in holiday_events:
        _set_holiday_provenance(event, window.window_id)
        new_candidate_ids.append(event.event_summary_id)
        _merge_event(events_by_id, event)
    impacts.extend(new_impacts)
    state.holiday_analysis_complete = (
        state.holiday_analysis_complete or run_holidays
    )

    for intent in to_execute:
        if intent.query_intent_id in retry_intent_ids:
            continue
        normalized = normalize_query(intent.query_text)
        if normalized in state.normalized_queries:
            raise ValueError(
                f"Equivalent query was already attempted in {state.window_id!r}"
            )
        state.normalized_queries.append(normalized)
        if intent.stage == "initial":
            state.attempted_initial_query_intent_ids.append(
                intent.query_intent_id
            )
        else:
            state.attempted_refinement_query_intent_ids.append(
                intent.query_intent_id
            )

    events = sorted(
        events_by_id.values(),
        key=lambda item: (item.timestamp, item.event_summary_id),
    )
    state.candidate_event_summary_ids = list(
        dict.fromkeys([*state.candidate_event_summary_ids, *new_candidate_ids])
    )
    fully_failed_intent_ids = {
        query_intent_id
        for query_intent_id, source_count in sources_by_intent.items()
        if failures_by_intent.get(query_intent_id, 0) == source_count
    }
    pending_candidate_ids = set(state.candidate_event_summary_ids) - set(
        state.reviewed_candidate_event_summary_ids
    )
    if fully_failed_intent_ids:
        state.status = "failed"
        state.next_agent_action = "retry_failed_search_window"
    elif pending_candidate_ids:
        state.status = "awaiting_selection"
        state.next_agent_action = "write_agent_event_selection"
    elif (
        len(state.attempted_refinement_query_intent_ids)
        >= strategy.search_policy.max_refinement_queries_per_window
    ):
        state.status = (
            "exhausted" if state.candidate_event_summary_ids else "no_event_found"
        )
        state.next_agent_action = ""
    else:
        state.status = "awaiting_refinement"
        state.next_agent_action = "write_window_refined_query"

    _save_search_artifacts(run_dir, states, traces, impacts, events)
    return _status_payload(states, state)


def _initial_intents(
    window: TimeWindowRecord,
    intents: list[AgentQueryIntentRecord],
    per_outline: int,
) -> list[AgentQueryIntentRecord]:
    initial = [intent for intent in intents if intent.stage == "initial"]
    counts = {
        outline_id: sum(intent.outline_id == outline_id for intent in initial)
        for outline_id in window.outline_ids
    }
    if any(counts[outline_id] != per_outline for outline_id in window.outline_ids):
        return []
    if len(initial) != per_outline * len(window.outline_ids):
        raise ValueError(
            f"Initial query intents for {window.window_id!r} exceed the "
            "approved per-outline budget"
        )
    return initial


def _load_or_initialize_states(
    run_dir: Path,
    windows: list[TimeWindowRecord],
) -> list[SearchWindowStateRecord]:
    path = run_dir / _STATE_FILENAME
    states = (
        _read_models(path, SearchWindowStateRecord)
        if path.exists()
        else [
            SearchWindowStateRecord(window_id=window.window_id)
            for window in windows
        ]
    )
    if [state.window_id for state in states] != [
        window.window_id for window in windows
    ]:
        raise ValueError("search_state.json does not match current time windows")
    return states


def _failed_intent_ids(
    state: SearchWindowStateRecord,
    traces: list[SearchTraceRecord],
    intent_by_id: dict[str, AgentQueryIntentRecord],
    strategy: CurationStrategy,
) -> set[str]:
    failed_intent_ids: set[str] = set()
    attempted_ids = [
        *state.attempted_initial_query_intent_ids,
        *state.attempted_refinement_query_intent_ids,
    ]
    for query_intent_id in attempted_ids:
        intent = intent_by_id.get(query_intent_id)
        if intent is None:
            raise ValueError(
                f"Attempted query intent {query_intent_id!r} is missing for retry"
            )
        requested_sources = set(
            intent.source_names or strategy.search_policy.default_sources
        )
        latest_by_source = {
            trace.source: trace
            for trace in traces
            if trace.window_id == state.window_id
            and trace.query_intent_id == query_intent_id
            and trace.source in requested_sources
        }
        if latest_by_source and all(
            trace.failure for trace in latest_by_source.values()
        ):
            failed_intent_ids.add(query_intent_id)
    return failed_intent_ids


def _apply_selections(
    states: list[SearchWindowStateRecord],
    selections: list[AgentEventSelectionRecord],
    traces: list[SearchTraceRecord],
    *,
    max_refinements: int,
) -> None:
    state_by_id = {state.window_id: state for state in states}
    windows_with_selected_candidates = {
        selection.window_id
        for selection in selections
        if selection.selected_event_summary_ids
    }
    for selection in selections:
        state = state_by_id.get(selection.window_id)
        if state is None:
            raise ValueError(
                f"Selection references unknown time window {selection.window_id!r}"
            )
        if state.status == "failed":
            continue
        known = set(state.candidate_event_summary_ids)
        referenced = set(selection.selected_event_summary_ids) | set(
            selection.rejected_event_summary_ids
        )
        if referenced - known:
            raise ValueError("Selection references candidates outside its time window")
        reviewed = set(state.reviewed_candidate_event_summary_ids)
        if referenced <= reviewed:
            continue
        pending = known - reviewed
        if referenced != pending:
            raise ValueError(
                "Selection must review every new candidate in its time window"
            )
        state.reviewed_candidate_event_summary_ids.extend(
            candidate_id
            for candidate_id in state.candidate_event_summary_ids
            if candidate_id in referenced
        )
        if selection.decision == "accept":
            state.status = "covered"
            state.next_agent_action = ""
        elif selection.decision == "complete":
            state.status = (
                "covered"
                if state.window_id in windows_with_selected_candidates
                else "no_event_found"
            )
            state.next_agent_action = ""
        elif len(state.attempted_refinement_query_intent_ids) >= max_refinements:
            state.status = "exhausted"
            state.next_agent_action = ""
        else:
            state.status = "awaiting_refinement"
            state.next_agent_action = "write_window_refined_query"

        for trace in traces:
            if trace.window_id == state.window_id and not trace.selection_decision:
                trace.selection_decision = selection.decision
                trace.rejected_rationales = dict(selection.rejected_rationales)


def _search_intents(
    intents: list[AgentQueryIntentRecord],
    strategy: CurationStrategy,
    available: SourceAvailabilityFn,
    search_fn: SearchFn,
) -> list[
    tuple[
        AgentQueryIntentRecord,
        str,
        list[DiscoveredEvent],
        str,
        datetime,
        int,
    ]
]:
    jobs = [
        (intent, source)
        for intent in intents
        for source in _sources_for_intent(intent, strategy, available)
    ]
    if not jobs:
        return []

    def search_one(job):
        intent, source = job
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            events = list(
                search_fn(
                    source,
                    intent.query_text,
                    intent.search_start,
                    intent.search_end,
                )
            )
            failure = ""
        except Exception as exc:
            events = []
            failure = str(exc)
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return intent, source, events, failure, started_at, elapsed_ms

    with ThreadPoolExecutor(
        max_workers=min(len(jobs), _MAX_SEARCH_WORKERS)
    ) as executor:
        return list(executor.map(search_one, jobs))


def _sources_for_intent(
    intent: AgentQueryIntentRecord,
    strategy: CurationStrategy,
    available: SourceAvailabilityFn,
) -> list[str]:
    requested = list(
        dict.fromkeys(
            intent.source_names or strategy.search_policy.default_sources
        )
    )
    sources = [source for source in requested if available(source)]
    if not sources:
        raise ValueError(
            "No configured Tavily/AnySearch source is available for "
            f"{intent.query_intent_id!r}"
        )
    return sources


def _set_web_provenance(
    event: DiscoveredEvent,
    intent: AgentQueryIntentRecord,
    source: str,
) -> None:
    event.query_intent_id = intent.query_intent_id
    event.window_id = intent.window_id
    event.event_summary_id = _event_summary_id(event)
    event.structured_payload = {
        **(event.structured_payload or {}),
        "source_hits": [source],
        "query_intent_ids": [intent.query_intent_id],
        "window_ids": [intent.window_id],
        "query_stages": [intent.stage],
        "outline_ids": [intent.outline_id] if intent.outline_id else [],
    }


def _set_holiday_provenance(event: DiscoveredEvent, window_id: str) -> None:
    event.window_id = window_id
    event.event_summary_id = _holiday_event_summary_id(event)
    event.structured_payload = {
        **(event.structured_payload or {}),
        "source_hits": ["holidays"],
        "query_intent_ids": [],
        "window_ids": [window_id],
        "query_stages": [],
        "outline_ids": [],
    }


def _merge_event(
    events_by_id: dict[str, DiscoveredEvent],
    incoming: DiscoveredEvent,
) -> None:
    existing = events_by_id.get(incoming.event_summary_id)
    if existing is None:
        events_by_id[incoming.event_summary_id] = incoming
        return
    for key in (
        "source_hits",
        "query_intent_ids",
        "window_ids",
        "query_stages",
        "outline_ids",
    ):
        existing.structured_payload[key] = list(
            dict.fromkeys(
                [
                    *(existing.structured_payload.get(key) or []),
                    *(incoming.structured_payload.get(key) or []),
                ]
            )
        )
    existing.retrieval_score = max(
        existing.retrieval_score, incoming.retrieval_score
    )


def _event_summary_id(event: DiscoveredEvent) -> str:
    identity = (
        f"url:{event.url.strip().casefold()}"
        if event.url.strip()
        else "|".join(
            [
                event.timestamp.isoformat(),
                event.title.strip().casefold(),
            ]
        )
    )
    return "summary_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _holiday_event_summary_id(event: DiscoveredEvent) -> str:
    identity = f"holiday_event_id:{event.event_id.strip().casefold()}"
    return "summary_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _trace_id(
    intent: AgentQueryIntentRecord,
    source: str,
    occurrence: int = 0,
) -> str:
    raw = f"{intent.query_intent_id}|{source}"
    if occurrence:
        raw += f"|retry:{occurrence}"
    return "trace_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _source_available(source: str) -> bool:
    return source_runtime.source_available(source)


def _status_payload(
    states: list[SearchWindowStateRecord],
    active: SearchWindowStateRecord | None = None,
    *,
    blocked_reason: str = "",
) -> dict:
    return {
        "blocked_reason": blocked_reason,
        "next_agent_action": active.next_agent_action if active else "",
        "active_window_id": active.window_id if active else "",
        "pacing_hint": (
            "Run one complete window at a time. Refine only after reviewing "
            "all outline searches, holiday impacts, and prior hypotheses."
        ),
        "search_windows": [
            state.model_dump(mode="json") for state in states
        ],
    }


def _save_search_artifacts(
    run_dir: Path,
    states: list[SearchWindowStateRecord],
    traces: list[SearchTraceRecord],
    impacts: list[HolidayImpactRecord],
    events: list[DiscoveredEvent],
) -> None:
    _write_jsonl(run_dir / _TRACE_FILENAME, traces)
    _write_jsonl(run_dir / _HOLIDAY_IMPACTS_FILENAME, impacts)
    _write_jsonl(run_dir / _EVENTS_FILENAME, events)
    _write_json(run_dir / _STATE_FILENAME, states)


def _read_model(path: Path, model):
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _read_models(path: Path, model):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    payloads = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    return [model.model_validate(item) for item in payloads]


def _read_optional_models(path: Path, model):
    return _read_models(path, model) if path.exists() else []


def _write_json(path: Path, models: list) -> None:
    path.write_text(
        json.dumps(
            [model.model_dump(mode="json") for model in models],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, models: list) -> None:
    path.write_text(
        "".join(
            json.dumps(model.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for model in models
        ),
        encoding="utf-8",
    )
