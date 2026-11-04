"""Tests for layered TOML config."""

from pathlib import Path

from world2ts_core import config
from world2ts_core.config import get, load_config, set_config


def test_default_config_paths_use_world2ts_names():
    assert config._DEFAULT_PROJECT_PATH == Path("world2ts.toml")
    assert config._DEFAULT_USER_PATH.parts[-3:] == (".config", "world2ts", "config.toml")


def test_empty_config_returns_defaults():
    cfg = load_config(project_path=Path("/nonexistent"), user_path=Path("/nonexistent"))
    assert cfg == {}


def test_project_config_loaded(tmp_path: Path):
    conf = tmp_path / "world2ts.toml"
    conf.write_text('http_proxy = "http://127.0.0.1:7890"\nhttp_timeout = 60\n')
    cfg = load_config(project_path=conf, user_path=Path("/nonexistent"))
    assert cfg["http_proxy"] == "http://127.0.0.1:7890"
    assert cfg["http_timeout"] == 60


def test_user_config_loaded(tmp_path: Path):
    conf = tmp_path / "config.toml"
    conf.write_text('store_root = "/custom/store"\n')
    cfg = load_config(project_path=Path("/nonexistent"), user_path=conf)
    assert cfg["store_root"] == "/custom/store"


def test_project_overrides_user(tmp_path: Path):
    user_conf = tmp_path / "user.toml"
    user_conf.write_text('http_timeout = 10\n')
    proj_conf = tmp_path / "project.toml"
    proj_conf.write_text('http_timeout = 60\n')
    cfg = load_config(project_path=proj_conf, user_path=user_conf)
    assert cfg["http_timeout"] == 60


def test_env_overrides_file(tmp_path: Path, monkeypatch):
    conf = tmp_path / "world2ts.toml"
    conf.write_text('http_proxy = "from-file"\n')
    monkeypatch.setenv("WORLD2TS_HTTP_PROXY", "from-env")
    cfg = load_config(project_path=conf, user_path=Path("/nonexistent"))
    assert cfg["http_proxy"] == "from-env"


def test_set_config_writes_toml(tmp_path: Path):
    conf = tmp_path / "world2ts.toml"
    set_config("http_proxy", "new-value", config_path=conf)
    assert "new-value" in conf.read_text()
    cfg = load_config(project_path=conf, user_path=Path("/nonexistent"))
    assert cfg["http_proxy"] == "new-value"


def test_get_with_default():
    val = get("nonexistent_key", default="fallback",
              project_path=Path("/nonexistent"), user_path=Path("/nonexistent"))
    assert val == "fallback"
