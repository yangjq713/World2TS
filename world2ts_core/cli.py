"""World2TS-core CLI entry point."""

import typer

from world2ts_core.commands import (
    collect,
    config_cmd,
)
from world2ts_core.commands import offline_discover as offline_discover_cmd
from world2ts_core.commands.metadata import metadata_app
from world2ts_core.commands.sources import sources_app
from world2ts_core.output import Output

_HELP = """\
World2TS-core builds auditable event context for time-series datasets.

CLI = hands (mechanical execution). Agent = brain (semantic decisions).

Unified pipeline:
  Agent collection plan -> collect remote data into Arrow + strict metadata
  -> offline-discover prepare -> Agent semantics/search/curation -> publish

Discovery phases (local Arrow + strict OfflineMetadata, time-only):
  prepare -> validate-semantics -> validate-strategy -> user strategy review
  intents -> agent-loop -> validate-canonical-events -> project
  audit -> user quality review -> publish

Use `world2ts-core <command> --help` for command-specific options.
"""

app = typer.Typer(name="world2ts-core", help=_HELP, no_args_is_help=True)


@app.callback()
def main(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output JSON to stdout (diagnostics go to stderr)",
    ),
):
    """Global options."""
    ctx.ensure_object(dict)
    ctx.obj["output"] = Output(force_json=json_output)


app.command("collect")(collect.collect)
app.command("offline-discover")(offline_discover_cmd.offline_discover)
app.add_typer(config_cmd.config_app)
app.add_typer(sources_app)
app.add_typer(metadata_app)
