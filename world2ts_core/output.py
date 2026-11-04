"""Dual-mode output system.

TTY mode: Rich tables, panels, progress bars.
JSON mode (--json or pipe): clean JSON to stdout, diagnostics to stderr.

Rule: in JSON mode, stdout contains ONLY valid JSON. Never decorative text.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pydantic
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

_PYDANTIC_V2 = int(pydantic.VERSION.split(".")[0]) >= 2

# Test-only override: set to True/False by the with_tty/no_tty fixtures to
# control TTY detection without fighting pytest's capsys stdout replacement.
# None means "use sys.stdout.isatty()" (normal production behaviour).
_IS_TTY_OVERRIDE: bool | None = None


def _model_to_json_str(model: BaseModel) -> str:
    """Serialize a Pydantic model to a JSON string, compatible with v1 and v2."""
    if _PYDANTIC_V2:
        return model.model_dump_json()  # type: ignore[attr-defined]
    return model.json()


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model to a dict, compatible with v1 and v2."""
    if _PYDANTIC_V2:
        return model.model_dump(mode="json")  # type: ignore[attr-defined]
    return json.loads(model.json())


def _write_stdout(text: str) -> None:
    payload = text.encode("utf-8")
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(payload)
        buffer.flush()
        return
    sys.stdout.write(payload.decode("utf-8"))
    sys.stdout.flush()


class Output:
    """Centralized output handler. One instance per CLI invocation."""

    def __init__(self, force_json: bool = False):
        is_tty = _IS_TTY_OVERRIDE if _IS_TTY_OVERRIDE is not None else sys.stdout.isatty()
        self.json_mode = force_json or not is_tty
        self._err = Console(stderr=True, highlight=False)

    @property
    def _out(self) -> Console:
        """Rich console targeting the current sys.stdout.

        Created lazily on each access so that pytest's capsys fixture can
        intercept output correctly. force_terminal=True ensures ANSI markup
        is emitted even when stdout is not a real TTY (e.g. during tests).
        """
        return Console(
            file=sys.stdout,
            highlight=False,
            force_terminal=not self.json_mode,
        )

    def record(self, model: BaseModel) -> None:
        """Print a single Pydantic model."""
        json_str = _model_to_json_str(model)
        if self.json_mode:
            _write_stdout(json.dumps(json.loads(json_str), ensure_ascii=False, indent=2) + "\n")
        else:
            self._out.print_json(json_str)

    def records(self, models: list[BaseModel], title: str = "") -> None:
        """Print a list of Pydantic models."""
        if self.json_mode:
            data = [_model_to_dict(m) for m in models]
            _write_stdout(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        else:
            if title:
                self._out.print(f"[bold]{title}[/bold] ({len(models)} items)")
            for m in models:
                self._out.print_json(_model_to_json_str(m))

    def table(self, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
        """Print tabular data."""
        if self.json_mode:
            _write_stdout(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        else:
            t = Table(title=title)
            for col in columns:
                t.add_column(col)
            for row in rows:
                t.add_row(*[str(row.get(c, "")) for c in columns])
            self._out.print(t)

    def raw_json(self, data: Any) -> None:
        """Print arbitrary JSON-serializable data."""
        if self.json_mode:
            _write_stdout(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        else:
            self._out.print_json(json.dumps(data, ensure_ascii=False))

    def panel(self, title: str, content: str) -> None:
        """Print a panel / summary block."""
        if self.json_mode:
            _write_stdout(json.dumps({"title": title, "content": content}, ensure_ascii=False) + "\n")
        else:
            self._out.print(Panel(content, title=title))

    def success(self, msg: str) -> None:
        self._err.print(f"[green]v[/green] {msg}")

    def error(self, msg: str) -> None:
        self._err.print(f"[red]x[/red] {msg}")

    def warn(self, msg: str) -> None:
        self._err.print(f"[yellow]![/yellow] {msg}")

    def info(self, msg: str) -> None:
        self._err.print(f"[dim]>[/dim] {msg}")

    def progress(self, total: int | None = None, description: str = "Processing") -> Progress | None:
        """Return a Rich Progress context manager (TTY only).

        In JSON mode returns None. Caller should check and iterate normally.
        """
        if self.json_mode:
            return None
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self._err,
        )
