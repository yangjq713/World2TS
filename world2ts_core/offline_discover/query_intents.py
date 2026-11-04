"""Agent-authored window plans, query intents, and feedback lineage."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import (
    AgentQueryIntentRecord,
    AgentWindowSearchPlanRecord,
    CurationStrategy,
    SeriesSemanticDimensionsRecord,
    TimeWindowRecord,
)


WINDOW_PLAN_FILENAME = "agent_window_search_plans.jsonl"
QUERY_INTENTS_FILENAME = "agent_query_intents.jsonl"


def build_agent_query_briefing(
    strategy: CurationStrategy,
    time_windows: list[TimeWindowRecord],
    series_semantics: list[SeriesSemanticDimensionsRecord],
) -> dict[str, Any]:
    """Describe complete windows without inventing queries or prior events."""
    outlines = {
        outline.outline_id: outline for outline in strategy.search_intent_outline
    }
    semantics_by_id = {item.series_id: item for item in series_semantics}
    windows = []
    for window in time_windows:
        windows.append(
            {
                "window_id": window.window_id,
                "period": window.period,
                "search_start": window.start.isoformat(),
                "search_end": window.end.isoformat(),
                "active_series_ids": list(window.active_series_ids),
                "active_dimension_values": window.active_dimension_values,
                "series_context": [
                    semantics_by_id[series_id].model_dump(mode="json")
                    for series_id in window.active_series_ids
                    if series_id in semantics_by_id
                ],
                "outlines": [
                    {
                        "outline_id": outline_id,
                        "event_family": outlines[outline_id].event_family,
                        "why_relevant": outlines[outline_id].why_relevant,
                        "context_pathways": list(
                            outlines[outline_id].context_pathways
                        ),
                        "initial_query_roles": list(
                            outlines[outline_id].initial_query_roles
                        ),
                        "completion_evidence": outlines[
                            outline_id
                        ].completion_evidence,
                    }
                    for outline_id in window.outline_ids
                ],
            }
        )
    return {
        "dataset_id": strategy.dataset_id,
        "query_policy": {
            "initial_queries_per_outline": (
                strategy.search_policy.initial_queries_per_outline
            ),
            "max_refinement_queries_per_window": (
                strategy.search_policy.max_refinement_queries_per_window
            ),
            "default_sources": list(strategy.search_policy.default_sources),
            "holiday_calendars": [
                item.model_dump(mode="json")
                for item in strategy.search_policy.holiday_calendars
            ],
            "rule": (
                "Write all initial outline queries and one Agent window plan first. "
                "After every outline in a window has run, review the combined Web "
                "and holiday evidence before writing one topic-unrestricted "
                "window_refinement query."
            ),
        },
        "windows": windows,
        "agent_outputs": {
            WINDOW_PLAN_FILENAME: list(
                AgentWindowSearchPlanRecord.model_fields
            ),
            QUERY_INTENTS_FILENAME: list(AgentQueryIntentRecord.model_fields),
        },
    }


def load_agent_window_search_plans(
    path: Path,
    *,
    time_windows: list[TimeWindowRecord],
) -> list[AgentWindowSearchPlanRecord]:
    plans = [
        AgentWindowSearchPlanRecord.model_validate(item)
        for item in _read_records(path)
    ]
    expected = [window.window_id for window in time_windows]
    actual = [plan.window_id for plan in plans]
    if actual != expected:
        raise ValueError(
            "Agent window search plans must cover every time window in order"
        )
    hypothesis_ids = [
        hypothesis.hypothesis_id
        for plan in plans
        for hypothesis in plan.prior_event_hypotheses
    ]
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        raise ValueError("prior event hypothesis IDs must be unique across the run")
    return plans


def load_agent_query_intents(
    path: Path,
    *,
    strategy: CurationStrategy,
    time_windows: list[TimeWindowRecord],
    window_plans: list[AgentWindowSearchPlanRecord],
) -> list[AgentQueryIntentRecord]:
    intents = [
        AgentQueryIntentRecord.model_validate(item) for item in _read_records(path)
    ]
    validate_agent_query_intents(
        intents,
        strategy=strategy,
        time_windows=time_windows,
        window_plans=window_plans,
    )
    return intents


def validate_agent_query_intents(
    intents: list[AgentQueryIntentRecord],
    *,
    strategy: CurationStrategy,
    time_windows: list[TimeWindowRecord],
    window_plans: list[AgentWindowSearchPlanRecord],
) -> None:
    windows = {window.window_id: window for window in time_windows}
    outlines = {
        outline.outline_id: outline for outline in strategy.search_intent_outline
    }
    plans = {plan.window_id: plan for plan in window_plans}
    intents_by_id = {intent.query_intent_id: intent for intent in intents}
    if len(intents_by_id) != len(intents):
        raise ValueError("Duplicate query_intent_id")

    prior_window_by_id = {
        hypothesis.hypothesis_id: plan.window_id
        for plan in window_plans
        for hypothesis in plan.prior_event_hypotheses
    }
    seen_queries: set[tuple[str, str]] = set()
    for intent in intents:
        window = windows.get(intent.window_id)
        if window is None:
            raise ValueError(f"Unknown time window: {intent.window_id}")
        if intent.window_id not in plans:
            raise ValueError(
                f"Missing Agent window search plan: {intent.window_id}"
            )
        if intent.search_start < window.start or intent.search_end > window.end:
            raise ValueError(
                f"Intent {intent.query_intent_id!r} search bounds exceed its time window"
            )

        normalized = normalize_query(intent.query_text)
        query_key = (intent.window_id, normalized)
        if query_key in seen_queries:
            raise ValueError(
                f"Equivalent query repeated in window {intent.window_id!r}: "
                f"{intent.query_text!r}"
            )
        seen_queries.add(query_key)

        if intent.stage == "initial":
            if intent.outline_id not in outlines:
                raise ValueError(f"Unknown strategy outline: {intent.outline_id}")
            if intent.outline_id not in window.outline_ids:
                raise ValueError(
                    f"Outline {intent.outline_id!r} is not active in "
                    f"{intent.window_id!r}"
                )
            continue

        for query_id in intent.feedback_query_intent_ids:
            parent = intents_by_id.get(query_id)
            if parent is None:
                raise ValueError(
                    f"Window refinement {intent.query_intent_id!r} references "
                    f"unknown query intent {query_id!r}"
                )
            if parent.window_id != intent.window_id:
                raise ValueError(
                    "Window refinement feedback cannot cross time windows"
                )
        for hypothesis_id in intent.prior_event_hypothesis_ids:
            if prior_window_by_id.get(hypothesis_id) != intent.window_id:
                raise ValueError(
                    "Window refinement prior hypotheses must belong to its window"
                )


def normalize_query(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array or JSONL records")
        return [dict(item) for item in payload]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]
