"""Shared test fixtures and test-only compatibility helpers."""

import inspect
from pathlib import Path
from typing import IO

import pytest
from typer.testing import CliRunner as TyperCliRunner


class _TtyStdout:
    """Thin wrapper around a file-like object that claims to be a TTY.

    Used by the ``with_tty`` fixture to override ``sys.stdout.isatty()``
    regardless of whether pytest's capsys has replaced sys.stdout with a
    ``CaptureIO`` wrapper (which always returns False from ``isatty``).
    """

    def __init__(self, wrapped: IO[str]) -> None:
        self._wrapped = wrapped

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        return self._wrapped.write(s)

    def flush(self) -> None:
        self._wrapped.flush()

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    """Temporary store directory for tests."""
    store = tmp_path / "store"
    store.mkdir()
    return store


@pytest.fixture
def no_tty(monkeypatch):
    """Force non-TTY mode for output tests."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)


@pytest.fixture
def with_tty(monkeypatch):
    """Force TTY mode for output tests.

    Patches the sys module's stdout isatty at the point where Output reads it,
    using a side-channel flag rather than fighting capsys's stdout replacement.
    """
    import world2ts_core.output as _out_mod
    monkeypatch.setattr(_out_mod, "_IS_TTY_OVERRIDE", True)


@pytest.fixture(autouse=False)
def _clear_tty_override():
    """Ensure _IS_TTY_OVERRIDE is reset after each test."""
    import world2ts_core.output as _out_mod
    yield
    _out_mod._IS_TTY_OVERRIDE = None


def make_cli_runner(**kwargs):
    """Build a Typer CliRunner across Click/Typer versions.

    Typer 0.20 / Click 8.2 removed the ``mix_stderr`` constructor argument,
    while older test code in this repo still passes it explicitly and expects
    ``result.output`` to contain stdout only when ``mix_stderr=False``.
    """
    mix_stderr = kwargs.get("mix_stderr", True)
    supported = inspect.signature(TyperCliRunner).parameters
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    runner = TyperCliRunner(**filtered)

    if "mix_stderr" in supported or mix_stderr:
        return runner

    class _CompatResult:
        """Restore Click < 8.2 ``result.output`` semantics for tests."""

        def __init__(self, result) -> None:
            self._result = result

        @property
        def output(self) -> str:
            return self._result.stdout

        def __getattr__(self, name: str):
            return getattr(self._result, name)

    class _CompatCliRunner:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped

        def invoke(self, *args, **kwargs):
            return _CompatResult(self._wrapped.invoke(*args, **kwargs))

        def __getattr__(self, name: str):
            return getattr(self._wrapped, name)

    return _CompatCliRunner(runner)
