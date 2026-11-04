"""Fetch online series into the canonical offline input pair."""

from pathlib import Path

import typer

from world2ts_core.collection import load_collection_plan, materialize_collection
from world2ts_core.output import Output


def collect(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ...,
        "--plan",
        help="Strict Agent-authored collection plan JSON.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="New directory for data.arrow and offline_metadata.json.",
    ),
):
    """Collect remote series as Arrow plus strict OfflineMetadata.

    The Agent selects the exact source, series, time range, frequency, semantic
    metadata, and adapter options in the plan. The CLI executes that plan
    without source fallback and writes exactly two offline-compatible files.

    Examples:
      world2ts-core --json collect --plan collection_plan.json --output-dir collected/finance_daily
    """
    out: Output = ctx.obj["output"]

    try:
        collection_plan = load_collection_plan(plan)
        out.info(
            f"Collecting {len(collection_plan.symbols)} series from exact source "
            f"{collection_plan.source}..."
        )
        summary = materialize_collection(collection_plan, output_dir)
    except Exception as exc:
        out.error(str(exc))
        raise typer.Exit(1) from None

    out.success(
        f"Materialized {summary['item_count']} series into {Path(output_dir).expanduser()}"
    )
    out.raw_json(summary)
