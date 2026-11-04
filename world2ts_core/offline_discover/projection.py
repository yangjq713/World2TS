"""Project canonical events through approved series semantic dimensions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonicalization import load_canonical_events
from .models import (
    CanonicalEventRecord,
    CurationStrategy,
    DataReadinessReport,
    ExampleEventLink,
    SemanticProjectionFilter,
    SemanticProjectionRule,
    SeriesReadinessProfile,
    SeriesSemanticDimensionsRecord,
)
from .strategy import APPROVED_FILENAME

PROJECTION_RULES_FILENAME = "projection_rules.jsonl"
PROJECTED_LINKS_FILENAME = "projected_event_series_links.jsonl"


def project_events(run_dir: Path) -> dict[str, Any]:
    """Validate Agent rules and write stable event-series links."""

    events, rules, links = _build_projected_links(run_dir)
    _write_jsonl(run_dir / PROJECTED_LINKS_FILENAME, links)

    ruled_event_ids = {rule.event_id for rule in rules}
    return {
        "status": "projected",
        "projection_rule_count": len(rules),
        "link_count": len(links),
        "held_event_ids": sorted(
            {event.event_id for event in events} - ruled_event_ids
        ),
        "output_path": str(run_dir / PROJECTED_LINKS_FILENAME),
    }


def validate_projected_links(run_dir: Path) -> list[ExampleEventLink]:
    """Require stored links to equal the current strategy/rule projection."""

    _, _, expected = _build_projected_links(run_dir)
    stored = load_projected_links(run_dir)
    if stored != expected:
        raise ValueError(
            "projected_event_series_links.jsonl is stale or does not match "
            "the current canonical events and projection rules"
        )
    return stored


def _build_projected_links(
    run_dir: Path,
) -> tuple[
    list[CanonicalEventRecord],
    list[SemanticProjectionRule],
    list[ExampleEventLink],
]:
    strategy = _read_model(run_dir / APPROVED_FILENAME, CurationStrategy)
    events = load_canonical_events(run_dir)
    rules = _read_jsonl(
        run_dir / PROJECTION_RULES_FILENAME,
        SemanticProjectionRule,
    )
    semantics = _read_jsonl(
        run_dir / "series_semantic_dimensions.jsonl",
        SeriesSemanticDimensionsRecord,
    )
    readiness = _read_model(
        run_dir / "data_readiness_report.json",
        DataReadinessReport,
    )
    if strategy.dataset_id != readiness.dataset_id:
        raise ValueError("Approved strategy and readiness dataset IDs must match")

    semantics_by_id, profiles_by_id = _validated_series_inventory(
        semantics,
        readiness,
    )
    event_by_id = {event.event_id: event for event in events}
    if len(event_by_id) != len(events):
        raise ValueError("canonical event_id values must be unique")
    _validate_rules(
        rules,
        event_by_id,
        strategy,
        semantics,
        readiness,
    )

    links: list[ExampleEventLink] = []
    for rule in rules:
        event = event_by_id[rule.event_id]
        matching_ids = [
            series_id
            for series_id, series in semantics_by_id.items()
            if _matches_rule(series.dimensions, rule)
            and _event_overlaps_profile(event.event_start, event.event_end, profiles_by_id[series_id])
        ]
        if not matching_ids:
            raise ValueError(f"Projection rule {rule.rule_id!r} matches no series")
        links.extend(
            ExampleEventLink(
                event_id=event.event_id,
                series_id=series_id,
                projection_rule_id=rule.rule_id,
                confidence=min(event.confidence, rule.confidence),
            )
            for series_id in matching_ids
        )

    links.sort(
        key=lambda link: (
            link.series_id,
            link.event_id,
            link.projection_rule_id,
        )
    )
    identities = [(link.event_id, link.series_id) for link in links]
    if len(identities) != len(set(identities)):
        raise ValueError("Projection produced duplicate event-series links")
    return events, rules, links


def load_projection_rules(run_dir: Path) -> list[SemanticProjectionRule]:
    """Load strict Agent-authored projection rules."""

    return _read_jsonl(
        run_dir / PROJECTION_RULES_FILENAME,
        SemanticProjectionRule,
    )


def load_projected_links(run_dir: Path) -> list[ExampleEventLink]:
    """Load projected event-series links."""

    return _read_jsonl(
        run_dir / PROJECTED_LINKS_FILENAME,
        ExampleEventLink,
    )


def _validated_series_inventory(
    semantics: list[SeriesSemanticDimensionsRecord],
    readiness: DataReadinessReport,
) -> tuple[
    dict[str, SeriesSemanticDimensionsRecord],
    dict[str, SeriesReadinessProfile],
]:
    semantic_ids = [series.series_id for series in semantics]
    if semantic_ids != readiness.series_ids:
        raise ValueError(
            "Series semantic dimensions must match readiness series order"
        )
    if readiness.status != "ready":
        raise ValueError("Projection requires ready data readiness")
    return (
        {series.series_id: series for series in semantics},
        {profile.series_id: profile for profile in readiness.series_profiles},
    )


def _validate_rules(
    rules: list[SemanticProjectionRule],
    event_by_id: dict[str, Any],
    strategy: CurationStrategy,
    semantics: list[SeriesSemanticDimensionsRecord],
    readiness: DataReadinessReport,
) -> None:
    rule_ids = [rule.rule_id for rule in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("projection rule_id values must be unique")
    event_ids = [rule.event_id for rule in rules]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("each canonical event may have at most one projection rule")

    available_dimensions = {profile.dimension for profile in readiness.dimensions}
    approved_dimensions = set(strategy.projection_policy.allowed_dimensions)
    missing_approved = approved_dimensions - available_dimensions
    if missing_approved:
        raise ValueError(
            "Approved projection dimensions are absent from readiness: "
            + ", ".join(sorted(missing_approved))
        )

    values_by_dimension = _dimension_values(semantics)
    for rule in rules:
        if rule.event_id not in event_by_id:
            raise ValueError(
                f"Projection rule references unknown canonical event: {rule.event_id}"
            )
        if rule.scope == "dataset":
            if not strategy.projection_policy.allow_dataset_wide:
                raise ValueError("Dataset-wide projection is not strategy-approved")
            continue
        for projection_filter in rule.filters:
            if projection_filter.dimension not in approved_dimensions:
                raise ValueError(
                    "Projection rule uses a dimension not approved by the strategy: "
                    f"{projection_filter.dimension}"
                )
            _validate_filter_values(
                projection_filter,
                values_by_dimension.get(projection_filter.dimension, []),
            )


def _dimension_values(
    semantics: list[SeriesSemanticDimensionsRecord],
) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {}
    for series in semantics:
        for dimension, raw_value in series.dimensions.items():
            for value in _as_values(raw_value):
                if not any(_same_value(value, existing) for existing in values.setdefault(dimension, [])):
                    values[dimension].append(value)
    return values


def _validate_filter_values(
    projection_filter: SemanticProjectionFilter,
    known_values: list[Any],
) -> None:
    if not known_values:
        raise ValueError(
            f"Projection dimension has no known values: {projection_filter.dimension}"
        )
    if projection_filter.operator == "exists":
        return
    if projection_filter.operator == "between":
        if not any(_is_number(value) for value in known_values):
            raise ValueError(
                f"between requires a numeric dimension: {projection_filter.dimension}"
            )
        return

    requested = _as_values(projection_filter.value)
    unknown = [
        value
        for value in requested
        if not any(_same_value(value, known) for known in known_values)
    ]
    if unknown:
        raise ValueError(
            f"Projection rule uses unknown values for {projection_filter.dimension}: "
            f"{unknown}"
        )


def _matches_rule(
    dimensions: dict[str, Any],
    rule: SemanticProjectionRule,
) -> bool:
    if rule.scope == "dataset":
        return True
    matches = [
        _matches_filter(dimensions, projection_filter)
        for projection_filter in rule.filters
    ]
    return all(matches) if rule.combine == "all" else any(matches)


def _matches_filter(
    dimensions: dict[str, Any],
    projection_filter: SemanticProjectionFilter,
) -> bool:
    if projection_filter.dimension not in dimensions:
        return False
    if projection_filter.operator == "exists":
        return True

    actual = _as_values(dimensions[projection_filter.dimension])
    if projection_filter.operator == "eq":
        return any(
            _same_value(value, projection_filter.value)
            for value in actual
        )
    if projection_filter.operator == "in":
        expected = _as_values(projection_filter.value)
        return any(
            _same_value(value, candidate)
            for value in actual
            for candidate in expected
        )

    lower, upper = projection_filter.value
    return any(
        _is_number(value) and lower <= value <= upper
        for value in actual
    )


def _event_overlaps_profile(
    event_start,
    event_end,
    profile: SeriesReadinessProfile,
) -> bool:
    if not profile.has_explicit_timestamps:
        end = event_end or event_start
        awareness = {
            _is_aware(value)
            for value in (event_start, end, profile.start, profile.end)
        }
        if len(awareness) != 1:
            raise ValueError(
                f"Event time awareness does not match legacy inferred timeline "
                f"for series {profile.series_id!r}"
            )
        return event_start <= profile.end and profile.start <= end

    start = _normalized_datetime(event_start)
    end = _normalized_datetime(event_end or event_start)
    profile_start = _normalized_datetime(profile.start)
    profile_end = _normalized_datetime(profile.end)
    return start <= profile_end and profile_start <= end


def _normalized_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _as_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if _is_number(left) and _is_number(right):
        return float(left) == float(right)
    return type(left) is type(right) and left == right


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_model(path: Path, model):
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path, model):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        raise ValueError(f"{path.name} must be JSONL, not a JSON array")
    return [
        model.model_validate(json.loads(line))
        for line in text.splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, models: list) -> None:
    path.write_text(
        "".join(
            json.dumps(model.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for model in models
        ),
        encoding="utf-8",
    )
