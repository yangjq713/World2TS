"""Thin CLI entry point for the explicit offline-discover v3 workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from world2ts_core.offline_discover.workflow import PHASES, run_phase
from world2ts_core.output import Output


def offline_discover(
    ctx: typer.Context,
    phase: str = typer.Option(
        ...,
        "--phase",
        help="Workflow phase: " + ", ".join(PHASES),
    ),
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        file_okay=False,
        help="Greenfield offline-discover v3 run directory.",
    ),
    dataset_id: str = typer.Option(
        "",
        "--dataset-id",
        help="Dataset ID; required only for prepare.",
    ),
    arrow_file: list[Path] = typer.Option(
        [],
        "--arrow-file",
        help="Arrow shard; repeat for prepare.",
    ),
    offline_metadata_file: Optional[Path] = typer.Option(
        None,
        "--offline-metadata-file",
        help="Strict OfflineMetadata JSON; required only for prepare.",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry failed search windows during agent-loop.",
    ),
) -> None:
    """Execute one explicit Agent-driven offline-discover v3.3 phase."""
    out: Output = ctx.obj["output"]
    try:
        result = run_phase(
            phase=phase,
            run_dir=run_dir,
            dataset_id=dataset_id,
            arrow_files=arrow_file,
            metadata_file=offline_metadata_file,
            retry_failed=retry_failed,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        out.error(str(exc))
        raise typer.Exit(1)
    out.raw_json(result)
