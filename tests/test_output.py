"""Tests for dual-mode output system."""

import json
from io import BytesIO
import sys

from pydantic import BaseModel

from world2ts_core.output import Output


class SampleModel(BaseModel):
    name: str
    value: int


class TestOutputJsonMode:
    """When --json or non-TTY, output must be valid JSON to stdout."""

    def test_record_outputs_json(self, capsys, no_tty):
        out = Output(force_json=True)
        model = SampleModel(name="test", value=42)
        out.record(model)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == {"name": "test", "value": 42}

    def test_records_outputs_json_array(self, capsys, no_tty):
        out = Output(force_json=True)
        models = [SampleModel(name="a", value=1), SampleModel(name="b", value=2)]
        out.records(models)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "a"

    def test_table_outputs_json_array(self, capsys, no_tty):
        out = Output(force_json=True)
        rows = [{"col1": "a", "col2": "1"}, {"col1": "b", "col2": "2"}]
        out.table("Test", rows, ["col1", "col2"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert len(parsed) == 2

    def test_success_goes_to_stderr(self, capsys, no_tty):
        out = Output(force_json=True)
        out.success("done")
        captured = capsys.readouterr()
        assert captured.out == ""  # stdout is clean
        assert "done" in captured.err

    def test_error_goes_to_stderr(self, capsys, no_tty):
        out = Output(force_json=True)
        out.error("failed")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "failed" in captured.err

    def test_auto_detects_non_tty(self, no_tty):
        out = Output(force_json=False)
        assert out.json_mode is True

    def test_records_write_utf8_bytes_to_stdout_buffer(self, monkeypatch, no_tty):
        class FakeStdout:
            def __init__(self):
                self.buffer = BytesIO()

            def isatty(self):
                return False

            def write(self, text):
                raise AssertionError("json_mode should write UTF-8 bytes via stdout.buffer")

            def flush(self):
                pass

        fake_stdout = FakeStdout()
        monkeypatch.setattr(sys, "stdout", fake_stdout)

        out = Output(force_json=True)
        models = [SampleModel(name="Here’s Boston weather", value=1)]
        out.records(models)

        decoded = fake_stdout.buffer.getvalue().decode("utf-8")
        parsed = json.loads(decoded)
        assert parsed[0]["name"] == "Here’s Boston weather"


class TestOutputRichMode:
    """When TTY, output uses Rich formatting."""

    def test_record_outputs_rich(self, capsys, with_tty):
        out = Output(force_json=False)
        model = SampleModel(name="test", value=42)
        out.record(model)
        captured = capsys.readouterr()
        assert out.json_mode is False
        assert "test" in captured.out

    def test_force_json_overrides_tty(self, capsys, with_tty):
        out = Output(force_json=True)
        model = SampleModel(name="test", value=42)
        out.record(model)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["name"] == "test"
