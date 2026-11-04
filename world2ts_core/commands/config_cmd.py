"""world2ts-core config — Manage configuration."""

import typer
from world2ts_core.config import load_config, set_config as _set_config
from world2ts_core.output import Output

config_app = typer.Typer(name="config", help="Manage World2TS-core configuration.")


@config_app.command("show")
def config_show(ctx: typer.Context):
    """Show current configuration from all layers."""
    out: Output = ctx.obj["output"]
    cfg = load_config()
    if not cfg:
        out.info("No configuration found. Copy world2ts.example.toml to world2ts.toml")
        return
    out.raw_json(cfg)


@config_app.command("set")
def config_set(
    ctx: typer.Context,
    key: str = typer.Argument(help="Config key (e.g. http_proxy)"),
    value: str = typer.Argument(help="Config value"),
):
    """Set a config value in the project config file."""
    out: Output = ctx.obj["output"]
    _set_config(key, value)
    out.success(f"Set {key} in world2ts.toml")
