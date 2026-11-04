"""CLI commands for LOTSA dataset metadata management."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from world2ts_core.output import Output
from world2ts_core.errors import SourceError, StoreError
from world2ts_core import lotsa

metadata_app = typer.Typer(
    name="metadata",
    help="LOTSA dataset metadata management.",
    no_args_is_help=True,
)


def _validate_exclusive(
    dataset: str, phase: int, out: Output,
) -> None:
    """Enforce exactly one of --dataset / --phase."""
    has_dataset = bool(dataset)
    has_phase = phase > 0
    if has_dataset == has_phase:
        out.error("Provide exactly one of --dataset or --phase")
        raise typer.Exit(1)


@metadata_app.command("hf-fetch")
def hf_fetch(
    ctx: typer.Context,
    dataset: str = typer.Option("", help="Single dataset name"),
    phase: int = typer.Option(0, help="Phase number (1-6)"),
    timeout: int = typer.Option(30, help="Per-request timeout in seconds"),
) -> None:
    """Fetch HF structural metadata for LOTSA datasets."""
    out: Output = ctx.obj["output"]
    _validate_exclusive(dataset, phase, out)

    if dataset:
        try:
            result = lotsa.hf_fetch_metadata(dataset, timeout=timeout)
            out.raw_json(result)
        except (SourceError, ValueError) as exc:
            out.error(str(exc))
            raise typer.Exit(1)
    else:
        try:
            lotsa.validate_phase(phase)
            datasets = lotsa.load_phase_datasets(phase)
        except (ValueError, StoreError) as exc:
            out.error(str(exc))
            raise typer.Exit(1)

        results, failures = [], []
        for entry in datasets:
            name = entry["dataset_name"]
            try:
                meta = lotsa.hf_fetch_metadata(name, timeout=timeout)
                results.append(meta)
                out.info(f"{name}: OK")
            except SourceError as exc:
                failures.append(name)
                out.warn(f"{name}: {exc}")
            time.sleep(1)

        out.raw_json(results)
        if failures:
            out.warn(f"{len(failures)} failed: {', '.join(failures)}")


@metadata_app.command("enrich")
def enrich(
    ctx: typer.Context,
    phase: int = typer.Option(..., help="Phase number (1-6)"),
    force: bool = typer.Option(False, "--force", help="Rebuild from coarse"),
) -> None:
    """Generate enriched JSONL by merging coarse registry with HF metadata."""
    out: Output = ctx.obj["output"]

    try:
        lotsa.validate_phase(phase)
        coarse = lotsa.load_phase_datasets(phase)
    except (ValueError, StoreError) as exc:
        out.error(str(exc))
        raise typer.Exit(1)

    if force:
        base = {}
    else:
        existing = lotsa.load_enriched_datasets(phase)
        base = {e["dataset_name"]: e for e in existing}

    results = []
    succeeded, failed = 0, 0
    failed_datasets = []

    for entry in coarse:
        name = entry["dataset_name"]
        existing_row = base.get(name, entry)
        try:
            hf_meta = lotsa.hf_fetch_metadata(name)
            merged = lotsa.merge_hf_into_entry(existing_row, hf_meta)
            results.append(merged)
            succeeded += 1
            out.info(f"{name}: OK")
        except SourceError as exc:
            results.append(existing_row)
            failed += 1
            failed_datasets.append(name)
            out.warn(f"{name}: {exc}")
        time.sleep(1)

    try:
        output_path = lotsa.save_enriched(phase, results)
    except (StoreError, OSError) as exc:
        out.error(f"Failed to write enriched file: {exc}")
        raise typer.Exit(1)
    out.raw_json({
        "phase": phase,
        "total": len(coarse),
        "succeeded": succeeded,
        "failed": failed,
        "failed_datasets": failed_datasets,
        "output_path": str(output_path),
    })


@metadata_app.command("prepare")
def prepare(
    ctx: typer.Context,
    dataset: str = typer.Option(..., help="Dataset name"),
    arrow_file: list[Path] = typer.Option(
        [],
        "--arrow-file",
        exists=True,
        dir_okay=False,
        help="Arrow shard; repeat for multiple shards",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        dir_okay=False,
        help="Metadata briefing JSON path",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing briefing"),
) -> None:
    """Prepare deterministic facts for Agent-authored strict metadata."""
    out: Output = ctx.obj["output"]
    if not arrow_file:
        out.error("Provide at least one --arrow-file")
        raise typer.Exit(1)
    output_path = output or lotsa.metadata_briefing_path(dataset)
    if output_path.exists() and not force:
        out.error(f"Metadata briefing already exists: {output_path} (use --force)")
        raise typer.Exit(1)
    try:
        briefing = lotsa.build_metadata_briefing(dataset, arrow_file)
        lotsa.write_json(output_path, briefing)
    except (OSError, StoreError, ValueError) as exc:
        out.error(str(exc))
        raise typer.Exit(1)
    out.raw_json({
        "dataset": dataset,
        "briefing_path": str(output_path),
        "offline_metadata_path": briefing["agent_output"]["path"],
        "base_series_count": briefing["arrow_facts"]["base_series_count"],
        "runtime_series_count": briefing["arrow_facts"]["runtime_series_count"],
    })


@metadata_app.command("validate")
def validate(
    ctx: typer.Context,
    offline_metadata_file: Path = typer.Option(
        ...,
        "--offline-metadata-file",
        exists=True,
        dir_okay=False,
        help="Agent-authored strict OfflineMetadata JSON",
    ),
    arrow_file: list[Path] = typer.Option(
        [],
        "--arrow-file",
        exists=True,
        dir_okay=False,
        help="Arrow shard; repeat for multiple shards",
    ),
) -> None:
    """Validate strict metadata against the Arrow inputs."""
    out: Output = ctx.obj["output"]
    if not arrow_file:
        out.error("Provide at least one --arrow-file")
        raise typer.Exit(1)
    try:
        result = lotsa.validate_offline_metadata(offline_metadata_file, arrow_file)
    except (OSError, ValueError) as exc:
        out.error(str(exc))
        raise typer.Exit(1)
    out.raw_json(result)


@metadata_app.command("render-card")
def render_card(
    ctx: typer.Context,
    offline_metadata_file: Path = typer.Option(
        ...,
        "--offline-metadata-file",
        exists=True,
        dir_okay=False,
        help="Validated strict OfflineMetadata JSON",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        dir_okay=False,
        help="Optional Markdown output path",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing card"),
) -> None:
    """Render optional Markdown documentation from strict metadata."""
    out: Output = ctx.obj["output"]
    try:
        metadata = lotsa.load_offline_metadata(offline_metadata_file)
        output_path = output or lotsa.metadata_card_path(
            metadata.dataset_semantics.dataset_name
        )
        if output_path.exists() and not force:
            raise ValueError(f"Metadata card already exists: {output_path} (use --force)")
        content = lotsa.render_metadata_card(metadata, offline_metadata_file.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    except (OSError, ValueError) as exc:
        out.error(str(exc))
        raise typer.Exit(1)
    out.raw_json({
        "dataset_id": metadata.dataset_semantics.dataset_name,
        "card_path": str(output_path),
        "source": str(offline_metadata_file),
    })


@metadata_app.command("list")
def metadata_list(ctx: typer.Context) -> None:
    """List all available LOTSA dataset names."""
    out: Output = ctx.obj["output"]
    names = set(lotsa.list_arrow_datasets())
    for phase in sorted(lotsa.PHASE_MAP):
        try:
            coarse = lotsa.load_phase_datasets(phase)
        except StoreError:
            coarse = []
        for entry in [*coarse, *lotsa.load_enriched_datasets(phase)]:
            if entry.get("dataset_name"):
                names.add(entry["dataset_name"])
    ordered_names = sorted(names)
    if out.json_mode:
        out.raw_json(ordered_names)
    else:
        for n in ordered_names:
            print(n)


@metadata_app.command("status")
def metadata_status(
    ctx: typer.Context,
    phase: int = typer.Option(0, help="Filter to specific phase (1-6)"),
) -> None:
    """Show metadata coverage status for LOTSA datasets."""
    out: Output = ctx.obj["output"]

    phase_filter = phase if phase > 0 else None
    if phase_filter is not None:
        try:
            lotsa.validate_phase(phase_filter)
        except ValueError as exc:
            out.error(str(exc))
            raise typer.Exit(1)

    status = lotsa.get_metadata_status(phase=phase_filter)

    if out.json_mode:
        out.raw_json(status)
    else:
        columns = [
            "dataset_name",
            "phase",
            "hf_enriched",
            "briefing_exists",
            "offline_metadata_exists",
            "schema_valid",
            "card_exists",
        ]
        out.table("LOTSA Metadata Status", status, columns)


__all__ = ["metadata_app"]
