"""Shared curl helpers for World2TS HTTP-bound sources."""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Iterable

from world2ts_core.config import get as config_get
from world2ts_core.errors import SourceError


def resolve_curl_binary() -> str:
    """Return a usable curl binary path for the current environment."""
    configured = str(config_get("curl_binary", default="")).strip()
    if configured:
        return configured

    for candidate in ("curl.exe", "curl", "/usr/bin/curl"):
        resolved = shutil.which(candidate) if not candidate.startswith("/") else candidate
        if resolved:
            return resolved

    raise SourceError(
        "No curl binary found. Set WORLD2TS_CURL_BINARY or install curl on this system."
    )


_STATUS_MARKER = "__WORLD2TS_HTTP_STATUS__"


def _split_status(stdout: str) -> "tuple[str, int]":
    """Separate the response body from the appended `-w` HTTP status marker.

    When no marker is present (e.g. older mocks or curl builds), the status is
    reported as 0 so callers fall back to returncode/body checks.
    """
    if _STATUS_MARKER in stdout:
        body, _, status_str = stdout.rpartition(_STATUS_MARKER)
        try:
            return body, int(status_str.strip())
        except ValueError:
            return body, 0
    return stdout, 0


def curl_get(
    url: str,
    *,
    timeout: int = 30,
    retries: int = 3,
    headers: Iterable[str] | None = None,
) -> str:
    """Fetch a URL with curl while respecting configured proxies.

    Captures the HTTP status code via `-w` and raises SourceError on any
    status >= 400. `curl -s` exits 0 even for HTTP 4xx/5xx (and HTTP-200
    application-error envelopes are handled by callers), so without this an
    error body — e.g. a Tavily quota error — would be returned as if it were
    a successful, empty result. Fail loudly instead.
    """
    cmd = [resolve_curl_binary(), "-s", "--max-time", str(timeout),
           "-w", f"{_STATUS_MARKER}%{{http_code}}"]
    for header in headers or []:
        cmd += ["-H", header]

    proxy = str(config_get("http_proxy", default="")).strip()
    if proxy:
        cmd += ["-x", proxy]
    cmd.append(url)

    last_result = None
    for attempt in range(retries):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                timeout=timeout + 5,
            )
        except FileNotFoundError as exc:
            raise SourceError(f"curl executable is not available: {cmd[0]}") from exc
        except subprocess.TimeoutExpired:
            if attempt == retries - 1:
                raise
            time.sleep(2)
            continue

        last_result = result
        body, http_status = _split_status(_decode_output(result.stdout))
        if result.returncode == 0 and http_status >= 400:
            raise SourceError(f"HTTP {http_status} from {url}: {body.strip()[:200]}")
        if result.returncode == 0 and body.strip():
            return body
        if attempt < retries - 1:
            time.sleep(2)

    if last_result is None:
        raise SourceError("curl request did not execute")
    if last_result.returncode != 0:
        raise SourceError(
            f"curl failed (code {last_result.returncode}): {_decode_output(last_result.stderr)[:200]}"
        )
    raise SourceError(
        f"curl returned an empty body after {retries} attempt(s) "
        f"(likely a proxy or upstream drop)"
    )


def curl_post(
    url: str,
    body: dict,
    *,
    timeout: int = 30,
    retries: int = 3,
    headers: Iterable[str] | None = None,
) -> str:
    """POST JSON to URL with curl while respecting configured proxies.

    Captures the HTTP status code via `-w` and raises SourceError on any
    status >= 400, so an error envelope (e.g. a Tavily HTTP 432 quota error)
    fails loudly instead of being parsed by the caller as an empty result.
    """
    import json as _json
    body_str = _json.dumps(body)
    cmd = [
        resolve_curl_binary(), "-s", "--max-time", str(timeout),
        "-w", f"{_STATUS_MARKER}%{{http_code}}",
        "-X", "POST",
        "-H", "Content-Type: application/json",
    ]
    for header in headers or []:
        cmd += ["-H", header]
    proxy = str(config_get("http_proxy", default="")).strip()
    if proxy:
        cmd += ["-x", proxy]
    cmd += ["-d", body_str, url]

    last_result = None
    for attempt in range(retries):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                timeout=timeout + 5,
            )
        except FileNotFoundError as exc:
            raise SourceError(f"curl executable is not available: {cmd[0]}") from exc
        except subprocess.TimeoutExpired:
            if attempt == retries - 1:
                raise
            time.sleep(2)
            continue

        last_result = result
        resp_body, http_status = _split_status(_decode_output(result.stdout))
        if result.returncode == 0 and http_status >= 400:
            raise SourceError(f"HTTP {http_status} from {url}: {resp_body.strip()[:200]}")
        if result.returncode == 0 and resp_body.strip():
            return resp_body
        if attempt < retries - 1:
            time.sleep(2)

    if last_result is None:
        raise SourceError("curl POST request did not execute")
    if last_result.returncode != 0:
        raise SourceError(
            f"curl POST failed (code {last_result.returncode}): "
            f"{_decode_output(last_result.stderr)[:200]}"
        )
    raise SourceError(
        f"curl POST returned an empty body after {retries} attempt(s) "
        f"(likely a proxy or upstream drop)"
    )


def _decode_output(payload: bytes | str | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")
