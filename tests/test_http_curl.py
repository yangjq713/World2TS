from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from world2ts_core.errors import SourceError
from world2ts_core.http_curl import _STATUS_MARKER, curl_get, curl_post


@patch("world2ts_core.http_curl.resolve_curl_binary", return_value="C:\\Windows\\System32\\curl.exe")
@patch("world2ts_core.http_curl.subprocess.run")
def test_curl_get_decodes_utf8_bytes_from_windows_curl(mock_run, mock_resolve):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout='{"title":"Here’s Boston weather"}'.encode("utf-8"),
        stderr=b"",
    )

    raw = curl_get("http://example.com", retries=1)

    assert "Here’s Boston weather" in raw


@patch("world2ts_core.http_curl.resolve_curl_binary", return_value="/usr/bin/curl")
@patch("world2ts_core.http_curl.subprocess.run")
def test_curl_get_strips_status_marker_and_returns_body(mock_run, _resolve):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=f"hello body{_STATUS_MARKER}200".encode("utf-8"),
        stderr=b"",
    )
    assert curl_get("http://example.com", retries=1) == "hello body"


@patch("world2ts_core.http_curl.time.sleep")
@patch("world2ts_core.http_curl.resolve_curl_binary", return_value="/usr/bin/curl")
@patch("world2ts_core.http_curl.subprocess.run")
def test_curl_get_raises_on_http_error_status(mock_run, _resolve, _sleep):
    # curl exits 0 for HTTP 432 (Tavily quota); the status marker must turn it
    # into a loud SourceError instead of returning the error body as success.
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=f'{{"detail":{{"error":"usage limit"}}}}{_STATUS_MARKER}432'.encode("utf-8"),
        stderr=b"",
    )
    with pytest.raises(SourceError, match="432"):
        curl_get("http://example.com", retries=1)


@patch("world2ts_core.http_curl.time.sleep")
@patch("world2ts_core.http_curl.resolve_curl_binary", return_value="/usr/bin/curl")
@patch("world2ts_core.http_curl.subprocess.run")
def test_curl_post_raises_on_http_error_status(mock_run, _resolve, _sleep):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=f'{{"detail":{{"error":"usage limit"}}}}{_STATUS_MARKER}432'.encode("utf-8"),
        stderr=b"",
    )
    with pytest.raises(SourceError, match="432"):
        curl_post("http://example.com", {"q": "x"}, retries=1)


@patch("world2ts_core.http_curl.time.sleep")
@patch("world2ts_core.http_curl.resolve_curl_binary", return_value="/usr/bin/curl")
@patch("world2ts_core.http_curl.subprocess.run")
def test_curl_get_retries_when_proxy_returns_empty_body(mock_run, _resolve, _sleep):
    """Proxy may return returncode=0 with empty body — should retry to recover."""
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout=b"", stderr=b""),
        MagicMock(returncode=0, stdout=b"hello world", stderr=b""),
    ]

    raw = curl_get("http://example.com", retries=2)

    assert raw == "hello world"
    assert mock_run.call_count == 2


@patch("world2ts_core.http_curl.time.sleep")
@patch("world2ts_core.http_curl.resolve_curl_binary", return_value="/usr/bin/curl")
@patch("world2ts_core.http_curl.subprocess.run")
def test_curl_get_raises_when_all_attempts_return_empty(mock_run, _resolve, _sleep):
    """Exhausted retries with empty body must raise, not silently return ''."""
    mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(SourceError, match="empty"):
        curl_get("http://example.com", retries=2)

    assert mock_run.call_count == 2
