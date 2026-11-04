"""world2ts-core sources — Inspect registered data source capabilities."""

import json
import sys

import typer

from world2ts_core.bootstrap import ensure_sources_registered
from world2ts_core.output import Output
from world2ts_core.registry import list_event_sources, list_series_sources

sources_app = typer.Typer(name="sources", help="Inspect data source capabilities.")


@sources_app.command("list")
def sources_list(
    ctx: typer.Context,
    domain: str = typer.Option("", help="Filter by domain (e.g. finance, crypto)"),
    event_type: str = typer.Option("", help="Filter by event type (e.g. earnings_release, outbreak)"),
):
    """List all registered data sources and their capabilities."""
    out: Output = ctx.obj["output"]
    ensure_sources_registered()

    event_caps = list_event_sources(
        domain=domain or None,
        event_type=event_type or None,
    )
    series_caps = list_series_sources(
        domain=domain or None,
    )

    if out.json_mode:
        data = {
            "event_sources": [c.model_dump(mode="json") for c in event_caps],
            "series_sources": [c.model_dump(mode="json") for c in series_caps],
        }
        sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        sys.stdout.flush()
    else:
        # TTY: Rich tables
        event_rows = []
        for c in event_caps:
            event_rows.append({
                "Name": c.name,
                "Kind": c.kind,
                "Domains": ", ".join(c.domains),
                "Event Types": ", ".join(c.event_types[:3]),
                "Geo": ", ".join(c.geo),
                "Resolution": c.time_resolution,
                "History": f"{c.history_days}d",
                "Reliability": c.reliability,
            })
        if event_rows:
            out.table(
                "Event Sources",
                event_rows,
                ["Name", "Kind", "Domains", "Event Types", "Geo", "Resolution", "History", "Reliability"],
            )

        series_rows = []
        for c in series_caps:
            series_rows.append({
                "Name": c.name,
                "Domains": ", ".join(c.domains),
                "Frequencies": ", ".join(c.frequencies),
                "Geo": ", ".join(c.geo),
                "History": f"{c.history_days}d",
                "Reliability": c.reliability,
            })
        if series_rows:
            out.table(
                "Series Sources",
                series_rows,
                ["Name", "Domains", "Frequencies", "Geo", "History", "Reliability"],
            )
