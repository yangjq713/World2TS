"""LOTSA dataset metadata core logic.

Handles HuggingFace API queries, JSONL registry I/O, strict metadata
briefings, validation, optional card rendering, and coverage reporting.

No CLI dependencies — raises exceptions on failure, never calls
typer.Exit() or Output directly.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from world2ts_core.config import get as config_get
from world2ts_core.errors import SourceError, StoreError
from world2ts_core.offline_discover.io import load_arrow_examples
from world2ts_core.offline_discover.readiness import (
    frequency_seconds,
    prepare_data_readiness,
)
from world2ts_core.offline_discover.schema import OfflineMetadata, parse_offline_metadata


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _lotsa_dir() -> Path:
    """Resolve LOTSA directory from config store_root."""
    return Path(config_get("store_root", default="store")) / "LOTSA"


def _meta_dir() -> Path:
    return _lotsa_dir() / "meta_data_info"


def list_arrow_datasets() -> List[str]:
    """Return dataset directories that contain at least one Arrow shard."""
    data_dir = _lotsa_dir() / "data"
    if not data_dir.is_dir():
        return []
    return sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_dir() and any(path.glob("*.arrow"))
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET = "Salesforce/lotsa_data"
HF_FIRST_ROWS_URL = "https://datasets-server.huggingface.co/first-rows"

PHASE_MAP = {
    1: "phase_01_transport_mobility",
    2: "phase_02_energy_buildings",
    3: "phase_03_environment_climate",
    4: "phase_04_health_public",
    5: "phase_05_economics_demand",
    6: "phase_06_compute_misc",
}

_TARGET_CHANNEL_RE = re.compile(r"^(.+)__target_(\d+)$")


def validate_phase(phase: int) -> str:
    """Return phase label or raise ValueError."""
    if phase not in PHASE_MAP:
        raise ValueError(f"Invalid phase {phase}, must be 1-6")
    return PHASE_MAP[phase]


# ---------------------------------------------------------------------------
# HF API client
# ---------------------------------------------------------------------------

def _hf_api_get(url: str, timeout: int = 30, retries: int = 3) -> dict:
    """Fetch JSON from HF datasets-server via /usr/bin/curl.

    Retries with exponential backoff. Raises SourceError on exhaustion.
    """
    cmd = ["/usr/bin/curl", "-s", "--max-time", str(timeout)]
    proxy = config_get("http_proxy", default="")
    if proxy:
        cmd += ["-x", proxy]
    cmd.append(url)

    last_err = ""
    for attempt in range(retries):
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                last_err = f"JSON decode error: {exc}"
        else:
            last_err = f"curl exit {result.returncode}"
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))

    raise SourceError(
        f"HF API request failed after {retries} attempts: {url} ({last_err})"
    )


def _extract_dim_from_feature(feature_type: dict) -> Optional[int]:
    """Extract dimensionality from a nested Sequence feature type.

    HF API format for multivariate target or pfdr:
      {"_type": "Sequence", "length": N, "feature": {"_type": "Sequence", ...}}
    The dimension N is on the OUTER Sequence (length key).
    """
    if feature_type.get("_type") != "Sequence":
        return None
    # Check outer length first (pfdr: Sequence of N Sequences)
    length = feature_type.get("length")
    if length is not None:
        return length
    # Check if inner is also a Sequence (multivariate target without explicit length)
    inner = feature_type.get("feature", {})
    if inner.get("_type") == "Sequence":
        return inner.get("length")
    return None


def hf_fetch_metadata(dataset_name: str, timeout: int = 30) -> dict:
    """Fetch structural metadata for one LOTSA dataset from HF.

    Calls both first-rows API (for schema + preview) and info API
    (for num_examples and size_bytes).

    Returns dict with hf_* prefixed keys.
    Raises SourceError on API failure.
    """
    # --- first-rows API: schema + preview row ---
    url = (
        f"{HF_FIRST_ROWS_URL}"
        f"?dataset={HF_DATASET}&config={dataset_name}&split=train"
    )
    data = _hf_api_get(url, timeout=timeout)

    # Features are top-level dicts with "name" and "type" keys
    feature_names = []
    target_dim: Optional[int] = 1  # default univariate
    pfdr_dim: Optional[int] = None
    has_freq = False

    for feat_entry in data.get("features", []):
        name = feat_entry.get("name", "")
        feature_names.append(name)
        ftype = feat_entry.get("type", {})

        if name == "freq":
            has_freq = True
        elif name == "target":
            dim = _extract_dim_from_feature(ftype)
            if dim is not None:
                target_dim = dim
        elif name == "past_feat_dynamic_real":
            pfdr_dim = _extract_dim_from_feature(ftype)

    # Extract from first row
    rows = data.get("rows", [])
    first_row = rows[0].get("row", {}) if rows else {}

    preview_start = first_row.get("start", "")
    preview_freq = first_row.get("freq", "")
    preview_item_id = first_row.get("item_id", "")

    target_val = first_row.get("target", [])
    preview_target_length = len(target_val) if isinstance(target_val, list) else None

    # --- info API: num_examples + size_bytes ---
    info_url = (
        f"https://datasets-server.huggingface.co/info"
        f"?dataset={HF_DATASET}"
    )
    num_examples = None
    size_bytes = None
    try:
        info_data = _hf_api_get(info_url, timeout=timeout)
        ds_info = info_data.get("dataset_info", {}).get(dataset_name, {})
        train_split = ds_info.get("splits", {}).get("train", {})
        num_examples = train_split.get("num_examples")
        size_bytes = train_split.get("num_bytes")
    except SourceError:
        # Fall back to first-rows data if info API fails
        num_examples = data.get("num_rows_total") or len(rows) or None

    return {
        "hf_num_examples": num_examples,
        "hf_target_dim": target_dim,
        "hf_past_feat_dynamic_real_dim": pfdr_dim,
        "hf_has_freq_field": has_freq,
        "hf_preview_start": preview_start,
        "hf_preview_freq": preview_freq,
        "hf_preview_target_length": preview_target_length,
        "hf_preview_item_id": preview_item_id,
        "hf_preview_keys": feature_names,
        "hf_size_bytes": size_bytes,
    }


# ---------------------------------------------------------------------------
# JSONL registry I/O
# ---------------------------------------------------------------------------

def _phase_jsonl_path(phase: int) -> Path:
    """Return path to coarse phase JSONL."""
    label = validate_phase(phase)
    return _lotsa_dir() / f"{label}.jsonl"


def _enriched_jsonl_path(phase: int) -> Path:
    """Return path to enriched phase JSONL."""
    label = validate_phase(phase)
    return _lotsa_dir() / f"{label}_enriched.jsonl"


def load_phase_datasets(phase: int) -> List[Dict[str, Any]]:
    """Read coarse phase JSONL. Raises StoreError if file missing."""
    path = _phase_jsonl_path(phase)
    if not path.exists():
        raise StoreError(f"Phase registry not found: {path}")
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def load_enriched_datasets(phase: int) -> List[Dict[str, Any]]:
    """Read enriched JSONL if exists, else empty list."""
    path = _enriched_jsonl_path(phase)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def save_enriched(phase: int, entries: List[Dict[str, Any]]) -> Path:
    """Write enriched JSONL via temp-file-plus-rename. Returns path."""
    path = _enriched_jsonl_path(phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    # Write to temp file then rename for atomicity
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem,
    )
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def merge_hf_into_entry(existing: Dict[str, Any], hf_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay hf_* fields onto existing entry, preserving all other keys.

    Only promotes metadata_status if currently phase1_coarse_inventory.
    """
    merged = dict(existing)
    for k, v in hf_meta.items():
        if k.startswith("hf_"):
            merged[k] = v
    if merged.get("metadata_status") == "phase1_coarse_inventory":
        merged["metadata_status"] = "phase2_hf_enriched"
        merged["classification_confidence"] = "high"
        merged["classification_method"] = "HF API + source tracing"
    return merged


def find_dataset_phase(dataset_name: str) -> int:
    """Find which phase a dataset belongs to. Raises ValueError if not found."""
    for phase in PHASE_MAP:
        entries = load_enriched_datasets(phase)
        try:
            entries.extend(load_phase_datasets(phase))
        except StoreError:
            pass
        for entry in entries:
            if entry.get("dataset_name") == dataset_name:
                return phase
    raise ValueError(f"Dataset '{dataset_name}' not found in any phase")


# ---------------------------------------------------------------------------
# Strict metadata preparation and validation
# ---------------------------------------------------------------------------

def metadata_briefing_path(dataset_name: str) -> Path:
    return _meta_dir() / f"{dataset_name.upper()}.metadata_briefing.json"


def offline_metadata_path(dataset_name: str) -> Path:
    return _meta_dir() / f"{dataset_name.upper()}.offline_metadata.json"


def metadata_card_path(dataset_name: str) -> Path:
    return _meta_dir() / f"{dataset_name.upper()}.md"


def _dataset_context(dataset_name: str) -> Dict[str, Any]:
    for phase in PHASE_MAP:
        for entry in load_enriched_datasets(phase):
            if entry.get("dataset_name") == dataset_name:
                return entry
    for phase in PHASE_MAP:
        try:
            entries = load_phase_datasets(phase)
        except StoreError:
            continue
        for entry in entries:
            if entry.get("dataset_name") == dataset_name:
                return entry
    return {
        "dataset_name": dataset_name,
        "metadata_status": "arrow_only",
    }


def _base_series_inventory(example_ids: Sequence[str]) -> tuple[list[str], int]:
    parsed = [_TARGET_CHANNEL_RE.fullmatch(example_id) for example_id in example_ids]
    if not all(parsed):
        return list(example_ids), 1

    channels_by_base: dict[str, set[int]] = defaultdict(set)
    for match in parsed:
        assert match is not None
        channels_by_base[match.group(1)].add(int(match.group(2)))
    channel_count = max(max(channels) for channels in channels_by_base.values()) + 1
    if channel_count == 1:
        return list(example_ids), 1
    expected_channels = set(range(channel_count))
    if any(channels != expected_channels for channels in channels_by_base.values()):
        raise ValueError("Arrow target channels are not aligned across base series")
    return sorted(channels_by_base), channel_count


def build_metadata_briefing(
    dataset_name: str,
    arrow_files: Sequence[Path],
) -> Dict[str, Any]:
    """Build deterministic facts for one Agent-authored OfflineMetadata JSON."""

    if not arrow_files:
        raise ValueError("Metadata preparation requires at least one Arrow file")
    enriched = _dataset_context(dataset_name)
    paths = [Path(path).expanduser() for path in arrow_files]
    examples = load_arrow_examples(paths)
    if not examples:
        raise ValueError("Arrow input contains no series")

    runtime_ids = [example.example_id for example in examples]
    base_ids, target_channel_count = _base_series_inventory(runtime_ids)
    profiles = []
    starts: list[datetime] = []
    ends: list[datetime] = []
    for example in examples:
        end = (
            example.timestamps[-1]
            if example.timestamps
            else example.start
            + timedelta(
                seconds=frequency_seconds(example.freq) * max(len(example.target) - 1, 0)
            )
        )
        start = (
            example.start
            if example.start.tzinfo is None
            else example.start.astimezone(timezone.utc).replace(tzinfo=None)
        )
        end = (
            end
            if end.tzinfo is None
            else end.astimezone(timezone.utc).replace(tzinfo=None)
        )
        starts.append(start)
        ends.append(end)
        profiles.append(
            {
                "series_id": example.example_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "frequency": example.freq,
                "length": len(example.target),
                "has_explicit_timestamps": example.timestamps is not None,
            }
        )

    legacy_card = metadata_card_path(dataset_name)
    hf_structure = {
        key.removeprefix("hf_"): value
        for key, value in enriched.items()
        if key.startswith("hf_")
    }
    registry_context = {
        key: value for key, value in enriched.items() if not key.startswith("hf_")
    }
    return {
        "briefing_version": "1.0",
        "dataset_name": dataset_name,
        "dataset_id": dataset_name,
        "registry_context": registry_context,
        "huggingface_structure": hf_structure,
        "arrow_facts": {
            "files": [str(path) for path in paths],
            "base_series_count": len(base_ids),
            "base_series_ids": base_ids,
            "runtime_series_count": len(runtime_ids),
            "runtime_series_ids": runtime_ids,
            "target_channel_count": target_channel_count,
            "frequencies": sorted({example.freq for example in examples}),
            "time_range": {
                "start": min(starts).isoformat(),
                "end": max(ends).isoformat(),
            },
            "series_profiles": profiles,
        },
        "legacy_card": str(legacy_card) if legacy_card.exists() else None,
        "agent_output": {
            "path": str(offline_metadata_path(dataset_name)),
            "model": "OfflineMetadata",
            "schema_version": "2.0",
            "requirements": [
                "Research and write semantic fields directly into the strict JSON.",
                "Cover every base_series_id exactly once in item_schema.item_semantics.",
                "Match dataset ID, series count, target channels, frequency, and time range.",
                "Do not add event families, event hypotheses, or search queries; those belong to the discovery strategy.",
                "Leave query_hints empty for compatibility with schema version 2.0.",
                "Do not invent facts that cannot be verified.",
            ],
        },
    }


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Atomically write one JSON object."""

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        suffix=".tmp",
        prefix=path.stem,
    )
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def load_offline_metadata(path: Path) -> OfflineMetadata:
    path = Path(path).expanduser()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_offline_metadata(raw, path)


def validate_offline_metadata(
    metadata_file: Path,
    arrow_files: Sequence[Path],
) -> Dict[str, Any]:
    """Run the same strict Arrow/metadata readiness used by offline discovery."""

    if not arrow_files:
        raise ValueError("Metadata validation requires at least one Arrow file")
    metadata = load_offline_metadata(metadata_file)
    with tempfile.TemporaryDirectory(prefix="world2ts-metadata-validate-") as temporary:
        report = prepare_data_readiness(
            dataset_id=metadata.dataset_semantics.dataset_name,
            arrow_files=[Path(path).expanduser() for path in arrow_files],
            metadata_file=Path(metadata_file).expanduser(),
            output_dir=Path(temporary),
        )
    return {
        "status": report.status,
        "dataset_id": report.dataset_id,
        "schema_version": metadata.schema_version,
        "base_series_count": metadata.dataset_semantics.num_series,
        "runtime_series_count": report.series_count,
        "series_id_order_sha256": report.series_id_order_sha256,
    }


def render_metadata_card(metadata: OfflineMetadata, source_name: str) -> str:
    """Render optional human-readable documentation from strict metadata."""

    def cell(value: str) -> str:
        return value.replace("|", "\\|")

    dataset = metadata.dataset_semantics
    schema = metadata.item_schema
    dimensions = sorted(
        {
            dimension
            for item in schema.item_semantics.values()
            for dimension in item.dimensions
        }
    )
    lines = [
        f"# {dataset.dataset_name.upper()}",
        "",
        f"> Generated from `{source_name}`. The strict JSON is authoritative.",
        "",
        "## Dataset",
        "",
        f"- Dataset ID: `{dataset.dataset_name}`",
        f"- Category: `{dataset.category}`",
        f"- Subcategory: `{dataset.subcategory}`",
        f"- Geography: {dataset.geo}",
        f"- Frequency: `{dataset.time_granularity}`",
        f"- Time range: `{dataset.time_range.start.isoformat()}` to `{dataset.time_range.end.isoformat()}`",
        f"- Base series: `{dataset.num_series}`",
        f"- Providers: {', '.join(dataset.providers) or 'unspecified'}",
        "",
        "## Description",
        "",
        dataset.description_short,
        "",
        dataset.description_long,
        "",
        "## Series Schema",
        "",
        f"- Entity type: `{schema.entity_type}`",
        f"- Item ID pattern: `{schema.item_id_pattern}`",
        f"- Spatial dimension: `{schema.spatial_dimension}`",
        f"- Item semantics entries: `{len(schema.item_semantics)}`",
        f"- Semantic dimensions: {', '.join(f'`{item}`' for item in dimensions) or 'none'}",
        "",
        "## Target Variables",
        "",
        "| Name | Description | Unit |",
        "|---|---|---|",
    ]
    for field in schema.target:
        lines.append(
            f"| {cell(field.name)} | {cell(field.description)} | {cell(field.unit)} |"
        )
    lines.extend(
        [
            "",
            "## Dynamic Covariates",
            "",
            "| Name | Description | Unit |",
            "|---|---|---|",
        ]
    )
    if schema.past_feat_dynamic_real:
        for field in schema.past_feat_dynamic_real:
            lines.append(
                f"| {cell(field.name)} | {cell(field.description)} | {cell(field.unit)} |"
            )
    else:
        lines.append("| none |  |  |")

    lines.extend(["", "## Item Groups", ""])
    if metadata.item_groups:
        lines.extend(["| Group | Label | Series |", "|---|---|---:|"])
        for group in metadata.item_groups:
            lines.append(
                f"| {cell(group.group_id)} | {cell(group.label)} | {len(group.example_ids)} |"
            )
    else:
        lines.append("No item groups declared.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------

def _metadata_status_entry(
    name: str,
    *,
    dataset_id: str,
    phase: str,
    hf_enriched: bool,
) -> Dict[str, Any]:
    briefing_path = metadata_briefing_path(name)
    metadata_path = offline_metadata_path(name)
    card_path = metadata_card_path(name)
    schema_valid = False
    schema_error = None
    if metadata_path.exists():
        try:
            load_offline_metadata(metadata_path)
            schema_valid = True
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            schema_error = str(exc)
    return {
        "dataset_name": name,
        "dataset_id": dataset_id,
        "phase": phase,
        "hf_enriched": hf_enriched,
        "briefing_exists": briefing_path.exists(),
        "offline_metadata_exists": metadata_path.exists(),
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "card_exists": card_path.exists(),
    }


def get_metadata_status(phase: Optional[int] = None) -> List[Dict[str, Any]]:
    """Cross-reference registry, briefings, strict JSON, and optional cards."""
    phases = [phase] if phase is not None else list(PHASE_MAP.keys())
    results: List[Dict[str, Any]] = []
    registered_names: set[str] = set()

    for p in phases:
        coarse = []
        try:
            coarse = load_phase_datasets(p)
        except StoreError:
            pass

        enriched = load_enriched_datasets(p)
        enriched_names = {e["dataset_name"] for e in enriched}
        datasets_by_name = {
            entry.get("dataset_name", ""): entry
            for entry in [*coarse, *enriched]
            if entry.get("dataset_name")
        }

        for name, entry in datasets_by_name.items():
            registered_names.add(name)
            results.append(
                _metadata_status_entry(
                    name,
                    dataset_id=entry.get("dataset_id", ""),
                    phase=PHASE_MAP.get(p, ""),
                    hf_enriched=name in enriched_names,
                )
            )

    if phase is None:
        for name in list_arrow_datasets():
            if name in registered_names:
                continue
            results.append(
                _metadata_status_entry(
                    name,
                    dataset_id=name,
                    phase="arrow_only",
                    hf_enriched=False,
                )
            )

    results.sort(key=lambda item: item["dataset_name"])
    return results
