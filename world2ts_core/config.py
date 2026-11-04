"""Layered TOML configuration.

Precedence (highest first):
  1. CLI flags (handled by Typer, not this module)
  2. Environment variables: WORLD2TS_<KEY> (uppercase)
  3. Project config: ./world2ts.toml
  4. User config: ~/.config/world2ts/config.toml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


_DEFAULT_USER_PATH = Path.home() / ".config" / "world2ts" / "config.toml"
_DEFAULT_PROJECT_PATH = Path("world2ts.toml")
_ENV_PREFIX = "WORLD2TS_"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _read_env() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            config_key = key[len(_ENV_PREFIX):].lower()
            result[config_key] = value
    return result


def load_config(
    project_path: Path = _DEFAULT_PROJECT_PATH,
    user_path: Path = _DEFAULT_USER_PATH,
) -> dict[str, Any]:
    """Load and merge config from all layers."""
    cfg: dict[str, Any] = {}
    cfg.update(_read_toml(user_path))
    cfg.update(_read_toml(project_path))
    cfg.update(_read_env())
    return cfg


def get(
    key: str,
    default: Any = None,
    project_path: Path = _DEFAULT_PROJECT_PATH,
    user_path: Path = _DEFAULT_USER_PATH,
) -> Any:
    """Get a single config value."""
    return load_config(project_path, user_path).get(key, default)


def set_config(
    key: str,
    value: str,
    config_path: Path = _DEFAULT_PROJECT_PATH,
) -> None:
    """Write a key-value pair to the project config file."""
    existing: dict[str, Any] = {}
    if config_path.is_file():
        with open(config_path, "rb") as f:
            existing = tomllib.load(f)
    existing[key] = value
    lines = [f'{k} = {_toml_value(v)}' for k, v in sorted(existing.items())]
    config_path.write_text("\n".join(lines) + "\n")


def _toml_value(v: Any) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
