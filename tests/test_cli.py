"""CLI smoke tests."""

from typer.testing import CliRunner
from typer.main import get_command

from world2ts_core.cli import app

runner = CliRunner()

COMMANDS = [
    "collect",
    "offline-discover",
    "config show",
    "config set",
    "sources list",
    "metadata hf-fetch",
    "metadata enrich",
    "metadata prepare",
    "metadata validate",
    "metadata render-card",
    "metadata list",
    "metadata status",
]


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "collect" in result.output
    assert "offline-discover" in result.output
    assert "metadata" in result.output

    root = get_command(app)
    assert root.name == "world2ts-core"
    assert "World2TS-core" in result.output
    assert set(root.commands) == {
        "collect",
        "offline-discover",
        "config",
        "sources",
        "metadata",
    }
    assert set(root.commands["metadata"].commands) == {
        "hf-fetch",
        "enrich",
        "prepare",
        "validate",
        "render-card",
        "list",
        "status",
    }


def test_all_commands_have_help():
    for cmd in COMMANDS:
        result = runner.invoke(app, cmd.split() + ["--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"


def test_collect_help_exposes_only_unified_input_contract():
    result = runner.invoke(app, ["collect", "--help"])

    assert result.exit_code == 0
    assert "--plan" in result.output
    assert "--output-dir" in result.output
    for legacy_option in ("--source", "--symbols", "--start", "--end", "--domain"):
        assert legacy_option not in result.output


def test_offline_discover_help_has_no_legacy_metadata_flags():
    result = runner.invoke(app, ["offline-discover", "--help"])

    assert result.exit_code == 0
    for phase in (
        "prepare",
        "validate-strategy",
        "agent-loop",
        "validate-canonical-events",
        "project",
        "audit",
        "publish",
    ):
        assert phase in result.output
    assert "--metadata-file" not in result.output
    assert "--dataset-info-file" not in result.output
    assert "--item-map-file" not in result.output
    assert "anchor" not in result.output.casefold()
    assert "changepoint" not in result.output.casefold()
    assert "full fallback" not in result.output.casefold()


def test_offline_discover_rejects_mechanical_full_phase(tmp_path):
    result = runner.invoke(
        app,
        [
            "--json",
            "offline-discover",
            "--phase",
            "full",
            "--run-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "--phase must be one of:" in result.output
