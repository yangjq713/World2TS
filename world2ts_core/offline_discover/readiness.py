"""Fail-loud data and Agent-semantic preflight for offline discovery."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from world2ts_core.minimal_semantics import validate_agent_authored_semantics

from .io import load_arrow_examples
from .models import (
    DataReadinessReport,
    DimensionReadinessProfile,
    MetadataConsistencyCheck,
    SemanticReadinessReport,
    SeriesReadinessProfile,
    SeriesSemanticDimensionsRecord,
)
from .runtime import load_runtime_metadata


_FREQUENCY_RE = re.compile(r"^(?:(\d+)\s*)?([a-z]+)$", re.IGNORECASE)
_TARGET_CHANNEL_RE = re.compile(r"^(.+)__target_(\d+)$")
_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "seconds": 1,
    "t": 60,
    "min": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hour": 3600,
    "hours": 3600,
    "hourly": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
    "daily": 86400,
    "w": 604800,
    "week": 604800,
    "weeks": 604800,
    "weekly": 604800,
}


def frequency_seconds(value: str) -> int:
    match = _FREQUENCY_RE.fullmatch(value.strip())
    if match is None or match.group(2).casefold() not in _UNIT_SECONDS:
        raise ValueError(f"Unsupported fixed frequency: {value!r}")
    count = int(match.group(1) or "1")
    if count < 1:
        raise ValueError(f"Unsupported fixed frequency: {value!r}")
    return count * _UNIT_SECONDS[match.group(2).casefold()]


def _normalized_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _time_range_matches(
    profile: SeriesReadinessProfile,
    start: datetime,
    end: datetime,
) -> bool:
    if profile.has_explicit_timestamps:
        return (
            _normalized_datetime(start) == _normalized_datetime(profile.start)
            and _normalized_datetime(end) == _normalized_datetime(profile.end)
        )
    return start == profile.start and end == profile.end


def _series_profile(example) -> SeriesReadinessProfile:
    length = len(example.target)
    step_seconds = frequency_seconds(example.freq)
    end = (
        example.timestamps[-1]
        if example.timestamps
        else example.start + timedelta(seconds=step_seconds * max(length - 1, 0))
    )
    missing = sum(not math.isfinite(value) for value in example.target)
    return SeriesReadinessProfile(
        series_id=example.example_id,
        start=example.start,
        end=end,
        freq=example.freq,
        length=length,
        missing_rate=missing / length if length else 0.0,
        has_explicit_timestamps=example.timestamps is not None,
    )


def _dimension_type(value: Any) -> str:
    if isinstance(value, list):
        return f"{_dimension_type(value[0])}_list"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _dimension_profiles(
    rows: Sequence[SeriesSemanticDimensionsRecord],
) -> list[DimensionReadinessProfile]:
    values: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        for dimension, value in row.dimensions.items():
            values[dimension].append(value)

    profiles: list[DimensionReadinessProfile] = []
    for dimension in sorted(values):
        dimension_values = values[dimension]
        value_types = {_dimension_type(value) for value in dimension_values}
        counts = Counter(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for value in dimension_values
        )
        profiles.append(
            DimensionReadinessProfile(
                dimension=dimension,
                value_type=next(iter(value_types)) if len(value_types) == 1 else "mixed",
                populated_series_count=len(dimension_values),
                coverage_ratio=len(dimension_values) / len(rows),
                value_counts=dict(sorted(counts.items())),
            )
        )
    return profiles


def _target_channels_match(series_ids: Sequence[str], metadata) -> bool:
    if set(series_ids) == set(metadata.item_schema.item_semantics):
        return len(metadata.item_schema.target) == 1
    parsed = [_TARGET_CHANNEL_RE.fullmatch(series_id) for series_id in series_ids]
    if not all(parsed):
        return False

    actual: dict[str, set[int]] = defaultdict(set)
    for match in parsed:
        assert match is not None
        actual[match.group(1)].add(int(match.group(2)))
    expected_channels = set(range(len(metadata.item_schema.target)))
    return set(actual) == set(metadata.item_schema.item_semantics) and all(
        channels == expected_channels for channels in actual.values()
    )


def _series_dimensions(series_id: str, metadata) -> dict[str, Any]:
    if series_id in metadata.item_schema.item_semantics:
        return dict(metadata.item_schema.item_semantics[series_id].dimensions)

    match = _TARGET_CHANNEL_RE.fullmatch(series_id)
    if match is None:
        raise ValueError(f"Series ID has no matching metadata dimensions: {series_id}")
    base_id, channel_text = match.groups()
    target = metadata.item_schema.target[int(channel_text)]
    dimensions = dict(metadata.item_schema.item_semantics[base_id].dimensions)
    dimensions.update(
        {
            "base_item_id": base_id,
            "target_channel_index": int(channel_text),
            "target_name": target.name,
        }
    )
    if target.description:
        dimensions["target_description"] = target.description
    if target.unit:
        dimensions["target_unit"] = target.unit
    return dimensions


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def prepare_data_readiness(
    *,
    dataset_id: str,
    arrow_files: Sequence[Path],
    metadata_file: Path,
    output_dir: Path,
) -> DataReadinessReport:
    """Validate Arrow plus metadata, write readiness artifacts, and stop if blocked."""

    paths = [Path(path).expanduser() for path in arrow_files]
    examples = load_arrow_examples(paths, preserve_physical_order=True)
    if not examples:
        raise ValueError("Arrow input contains no series")

    series_ids = [example.example_id for example in examples]
    profiles = [_series_profile(example) for example in examples]
    runtime = load_runtime_metadata(metadata_file.expanduser(), example_ids=series_ids)
    metadata = runtime.metadata

    semantic_rows = [
        SeriesSemanticDimensionsRecord(
            series_id=series_id,
            entity_label=runtime.item_by_id[series_id]["entity_label"],
            description=runtime.item_by_id[series_id]["description"],
            location=runtime.item_by_id[series_id]["location"],
            confidence=runtime.item_by_id[series_id]["confidence"],
            dimensions=_series_dimensions(series_id, metadata),
        )
        for series_id in series_ids
    ]

    checks: list[MetadataConsistencyCheck] = []
    blocking_reasons: list[str] = []

    def check(check_id: str, passed: bool, evidence: str) -> None:
        checks.append(
            MetadataConsistencyCheck(
                check_id=check_id,
                status="pass" if passed else "fail",
                evidence=evidence,
            )
        )
        if not passed:
            blocking_reasons.append(f"{check_id}: {evidence}")

    declared_id = metadata.dataset_semantics.dataset_name
    check(
        "dataset_id",
        dataset_id == declared_id,
        f"requested={dataset_id!r} metadata={declared_id!r}",
    )
    check(
        "series_ids",
        set(series_ids) == set(runtime.item_by_id),
        f"Arrow and metadata resolve to {len(series_ids)} unique series IDs",
    )
    check(
        "series_count",
        metadata.dataset_semantics.num_series == len(metadata.item_schema.item_semantics),
        "declared base series count="
        f"{metadata.dataset_semantics.num_series} metadata items="
        f"{len(metadata.item_schema.item_semantics)} Arrow series={len(series_ids)}",
    )
    channels_match = _target_channels_match(series_ids, metadata)
    check(
        "target_channels",
        channels_match,
        f"metadata targets={len(metadata.item_schema.target)} match Arrow channel expansion={channels_match}",
    )

    declared_step = frequency_seconds(metadata.dataset_semantics.time_granularity)
    frequency_matches = all(frequency_seconds(profile.freq) == declared_step for profile in profiles)
    check(
        "frequency",
        frequency_matches,
        f"metadata={metadata.dataset_semantics.time_granularity!r} "
        f"Arrow={sorted({profile.freq for profile in profiles})}",
    )

    actual_start = min(_normalized_datetime(profile.start) for profile in profiles)
    actual_end = max(_normalized_datetime(profile.end) for profile in profiles)
    declared_range = metadata.dataset_semantics.time_range
    range_matches = (
        actual_start == _normalized_datetime(declared_range.start)
        and actual_end == _normalized_datetime(declared_range.end)
    )
    check(
        "time_range",
        range_matches,
        f"metadata=[{declared_range.start.isoformat()}, {declared_range.end.isoformat()}] "
        f"Arrow=[{actual_start.isoformat()}, {actual_end.isoformat()}]",
    )

    empty_series = [profile.series_id for profile in profiles if profile.length == 0]
    if empty_series:
        blocking_reasons.append(f"empty target: {empty_series}")

    recoverable_rows = [
        row
        for row in semantic_rows
        if row.entity_label or row.description or row.location or row.dimensions
    ]
    recoverable_ids = {row.series_id for row in recoverable_rows}
    unrecoverable = [row.series_id for row in semantic_rows if row.series_id not in recoverable_ids]
    if unrecoverable:
        blocking_reasons.append(f"unrecoverable series semantics: {unrecoverable}")

    dataset_fields = (
        declared_id,
        metadata.dataset_semantics.category,
        metadata.dataset_semantics.subcategory,
        metadata.dataset_semantics.geo,
        metadata.dataset_semantics.time_granularity,
        metadata.dataset_semantics.description_short,
        metadata.dataset_semantics.description_long,
    )
    report = DataReadinessReport(
        dataset_id=dataset_id,
        status="blocked" if blocking_reasons else "ready",
        arrow_files=[str(path) for path in paths],
        series_count=len(series_ids),
        series_ids=series_ids,
        series_id_order_sha256=hashlib.sha256("\n".join(series_ids).encode("utf-8")).hexdigest(),
        series_profiles=profiles,
        metadata_consistency=checks,
        dataset_semantics_coverage=sum(bool(value) for value in dataset_fields) / len(dataset_fields),
        series_semantics_coverage=len(recoverable_rows) / len(semantic_rows),
        dimensions=_dimension_profiles(semantic_rows),
        unrecoverable_series_ids=unrecoverable,
        blocking_reasons=blocking_reasons,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dimensions_path = output_dir / "series_semantic_dimensions.jsonl"
    report_path = output_dir / "data_readiness_report.json"
    _write_jsonl(dimensions_path, [row.model_dump(mode="json") for row in semantic_rows])
    _write_json(report_path, report.model_dump(mode="json"))
    if report.status == "blocked":
        raise ValueError("Data readiness blocked: " + "; ".join(report.blocking_reasons))

    _write_json(
        output_dir / "agent_semantic_briefing.json",
        {
            "dataset_id": dataset_id,
            "verified_dataset_semantics": metadata.dataset_semantics.model_dump(mode="json"),
            "verified_inputs": {
                "data_readiness_report": str(report_path),
                "series_semantic_dimensions": str(dimensions_path),
            },
            "agent_outputs": {
                "agent_dataset_semantics.json": [
                    "dataset_id",
                    "description",
                    "domain",
                    "frequency_phrases",
                ],
                "agent_series_semantics.jsonl": [
                    "series_id",
                    "description",
                    "time_range",
                    "frequency_phrase",
                ],
            },
            "requirements": [
                "Write descriptions from the verified inputs; do not use a generated template.",
                "Cover every series ID exactly once.",
                "State the verified frequency and exact Arrow time range in every series description.",
            ],
        },
    )
    return report


def frequency_phrase_matches(freq: str, phrase: str) -> bool:
    seconds = frequency_seconds(freq)
    normalized = " ".join(re.sub(r"[-_]", " ", phrase.casefold()).split())
    units = (
        (604800, "week", "weekly"),
        (86400, "day", "daily"),
        (3600, "hour", "hourly"),
        (60, "minute", "minutely"),
        (1, "second", "secondly"),
    )
    for unit_seconds, unit, adjective in units:
        if seconds % unit_seconds:
            continue
        count = seconds // unit_seconds
        aliases = {
            " ".join(re.sub(r"[-_]", " ", freq.casefold()).split()),
            f"{count}{unit[0]}",
            f"{count} {unit}",
            f"{count} {unit}s",
            f"every {unit}" if count == 1 else f"every {count} {unit}s",
        }
        if count == 1:
            aliases.update({unit, adjective, f"per {unit}"})
        if seconds == 1800:
            aliases.update({"half hour", "half hourly"})
        elif seconds == 300:
            aliases.update({"five minute", "five minutes"})
        return normalized in aliases
    return False


def validate_current_agent_semantics(
    dataset: dict[str, Any],
    series_by_id: dict[str, dict[str, Any]],
    readiness: DataReadinessReport,
) -> None:
    """Revalidate persisted Agent semantics against current Arrow readiness."""

    if readiness.status != "ready":
        raise ValueError("Agent semantics require ready data")
    if dataset["dataset_id"] != readiness.dataset_id:
        raise ValueError("Agent dataset semantics do not match readiness")
    if list(series_by_id) != readiness.series_ids:
        raise ValueError("Agent series semantics do not match readiness series order")
    profiles = {profile.series_id: profile for profile in readiness.series_profiles}
    for series_id, semantics in series_by_id.items():
        profile = profiles[series_id]
        if not frequency_phrase_matches(
            profile.freq,
            semantics["frequency_phrase"],
        ):
            raise ValueError(
                f"Agent series frequency does not match readiness: {series_id}"
            )
        start = datetime.fromisoformat(
            semantics["time_range"]["start"].replace("Z", "+00:00")
        )
        end = datetime.fromisoformat(
            semantics["time_range"]["end"].replace("Z", "+00:00")
        )
        if not _time_range_matches(profile, start, end):
            raise ValueError(
                f"Agent series time range does not match readiness: {series_id}"
            )
    if not all(
        any(
            frequency_phrase_matches(profile.freq, phrase)
            for phrase in dataset["frequency_phrases"]
        )
        for profile in readiness.series_profiles
    ):
        raise ValueError("Agent dataset frequency phrases do not match readiness")


def validate_semantic_readiness(
    *,
    data_readiness: DataReadinessReport,
    agent_dataset_semantics_file: Path,
    agent_series_semantics_file: Path,
    output_dir: Path,
) -> SemanticReadinessReport:
    """Verify Agent-authored semantics against Arrow facts before any search."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "semantic_readiness_report.json"
    expected_ids = data_readiness.series_ids
    checks: list[MetadataConsistencyCheck] = []
    blocking_reasons: list[str] = []

    if data_readiness.status != "ready":
        blocking_reasons.append("data readiness is not ready")

    try:
        dataset, series_by_id = validate_agent_authored_semantics(
            agent_dataset_semantics_file=agent_dataset_semantics_file,
            agent_series_semantics_file=agent_series_semantics_file,
        )
        checks.append(
            MetadataConsistencyCheck(
                check_id="agent_semantics_contract",
                status="pass",
                evidence="Agent dataset and series files satisfy the strict no-fallback contract",
            )
        )
    except (OSError, ValueError) as exc:
        report = SemanticReadinessReport(
            dataset_id=data_readiness.dataset_id,
            status="blocked",
            expected_series_ids=expected_ids,
            validated_series_ids=[],
            checks=[
                MetadataConsistencyCheck(
                    check_id="agent_semantics_contract",
                    status="fail",
                    evidence=str(exc),
                )
            ],
            blocking_reasons=blocking_reasons + [str(exc)],
        )
        _write_json(report_path, report.model_dump(mode="json"))
        raise ValueError("Semantic readiness blocked: " + str(exc)) from exc

    def check(check_id: str, passed: bool, evidence: str) -> None:
        checks.append(
            MetadataConsistencyCheck(
                check_id=check_id,
                status="pass" if passed else "fail",
                evidence=evidence,
            )
        )
        if not passed:
            blocking_reasons.append(f"{check_id}: {evidence}")

    check(
        "dataset_id",
        dataset["dataset_id"] == data_readiness.dataset_id,
        f"Agent={dataset['dataset_id']!r} expected={data_readiness.dataset_id!r}",
    )
    semantic_ids = list(series_by_id)
    exact_ids = set(semantic_ids) == set(expected_ids)
    check(
        "series_ids",
        exact_ids,
        f"missing={sorted(set(expected_ids) - set(semantic_ids))} "
        f"extra={sorted(set(semantic_ids) - set(expected_ids))}",
    )

    profiles = {profile.series_id: profile for profile in data_readiness.series_profiles}
    frequency_matches = exact_ids and all(
        frequency_phrase_matches(profiles[series_id].freq, semantics["frequency_phrase"])
        for series_id, semantics in series_by_id.items()
    )
    dataset_frequency_matches = all(
        any(frequency_phrase_matches(profile.freq, phrase) for phrase in dataset["frequency_phrases"])
        for profile in data_readiness.series_profiles
    )
    check(
        "frequency",
        frequency_matches and dataset_frequency_matches,
        "Agent frequency phrases must match every verified Arrow frequency",
    )

    time_ranges_match = exact_ids
    if exact_ids:
        for series_id, semantics in series_by_id.items():
            profile = profiles[series_id]
            time_range = semantics["time_range"]
            start = datetime.fromisoformat(time_range["start"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(time_range["end"].replace("Z", "+00:00"))
            if not _time_range_matches(profile, start, end):
                time_ranges_match = False
                break
    check(
        "time_range",
        time_ranges_match,
        "Agent series time ranges must exactly match the Arrow-derived ranges",
    )

    status = "blocked" if blocking_reasons else "ready"
    report = SemanticReadinessReport(
        dataset_id=data_readiness.dataset_id,
        status=status,
        expected_series_ids=expected_ids,
        validated_series_ids=expected_ids if exact_ids else semantic_ids,
        checks=checks,
        blocking_reasons=blocking_reasons,
    )
    _write_json(report_path, report.model_dump(mode="json"))
    if status == "blocked":
        raise ValueError("Semantic readiness blocked: " + "; ".join(blocking_reasons))
    return report
