"""The HERE polling client, against stubbed sessions only. No test touches the network.

The key used here is a fake one written to a temporary properties file; the real
`phone/local.properties` is never read by any test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import poll_here_feed as phf  # noqa: E402

FAKE_KEY = "FAKE-key_0123456789abcdefXYZ"


class FakeResponse:
    def __init__(self, status: int, source_updated: str | None = "2026-10-15T12:00:36Z",
                 headers: dict | None = None):
        self.status_code = status
        body = {"sourceUpdated": source_updated, "results": []} if source_updated else {}
        self.content = json.dumps(body).encode()
        self.headers = {"Date": "Thu, 15 Oct 2026 12:01:30 GMT", "Age": "3",
                        "Cache-Control": "max-age=30", "ETag": '"abc"', **(headers or {})}


class FakeSession:
    """Returns the queued responses in order; records what it was asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        item = self.responses.pop(0) if self.responses else FakeResponse(200)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def wall(self):
        return self.now

    def mono(self):
        return self.now - 1_700_000_000.0

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


def _poll(tmp_path, responses, **kwargs):
    session = FakeSession(responses)
    clock = FakeClock()
    defaults = dict(areas=[phf.AREAS["nj"]], interval_s=5.0, duration_s=600.0,
                    max_requests=100, out_dir=tmp_path / "s1", key=FAKE_KEY,
                    wall=clock.wall, mono=clock.mono, sleep=clock.sleep,
                    clock_probe=lambda: None, script_sha=lambda: "deadbeef")
    defaults.update(kwargs)
    result = phf.poll(session, **defaults)
    return session, result


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestKey:
    def test_the_key_is_read_from_a_properties_file(self, tmp_path):
        path = tmp_path / "local.properties"
        path.write_text(f"sdk.dir=/x\nhere.apiKey={FAKE_KEY}\nother=1\n")
        assert phf.read_key(path) == FAKE_KEY

    def test_a_missing_key_is_a_refusal(self, tmp_path):
        path = tmp_path / "local.properties"
        path.write_text("sdk.dir=/x\n")
        with pytest.raises(SystemExit):
            phf.read_key(path)


class TestRequest:
    def test_the_key_is_only_in_params_and_never_in_the_recorded_url(self):
        url, params, recorded = phf.build_request(phf.AREAS["nj"], FAKE_KEY)
        assert params["apiKey"] == FAKE_KEY
        assert FAKE_KEY not in url
        assert FAKE_KEY not in recorded
        assert "apiKey" not in recorded
        assert "circle%3A40.3525%2C-74.6239%3Br%3D500" in recorded

    def test_redact_removes_the_key_from_an_exception_string(self):
        message = (f"HTTPSConnectionPool: Max retries exceeded with url: /v7/flow?in=x"
                   f"&apiKey={FAKE_KEY} (Caused by timeout)")
        cleaned = phf.redact(message, FAKE_KEY)
        assert FAKE_KEY not in cleaned and "REDACTED" in cleaned

    def test_redact_also_removes_the_url_encoded_key(self):
        key = "a+b/c="
        assert key not in phf.redact("apiKey=a%2Bb%2Fc%3D", key)
        assert "a%2Bb%2Fc%3D" not in phf.redact("apiKey=a%2Bb%2Fc%3D", key)


class TestSelfCheck:
    def test_a_file_containing_the_key_is_deleted_and_raises(self, tmp_path):
        clean = tmp_path / "ok.jsonl"
        clean.write_text("nothing here\n")
        dirty = tmp_path / "leak.jsonl"
        dirty.write_text(f"url?apiKey={FAKE_KEY}\n")
        with pytest.raises(RuntimeError, match="leak.jsonl"):
            phf.self_check([clean, dirty], FAKE_KEY)
        assert not dirty.exists()
        assert clean.exists()

    def test_clean_files_pass(self, tmp_path):
        clean = tmp_path / "ok.jsonl"
        clean.write_text("nothing here\n")
        phf.self_check([clean], FAKE_KEY)


class TestPoll:
    def test_max_requests_3_stops_after_3(self, tmp_path):
        session, result = _poll(tmp_path, [FakeResponse(200)] * 10, max_requests=3)
        assert len(session.calls) == 3
        assert result["stop_reason"] == "max_requests"
        requests = [r for r in _records(tmp_path / "s1" / "requests.jsonl")
                    if r.get("type") == "request"]
        assert [r["seq"] for r in requests] == [0, 1, 2]

    def test_a_429_stops_at_once_and_is_not_retried(self, tmp_path):
        session, result = _poll(tmp_path, [FakeResponse(200), FakeResponse(429),
                                           FakeResponse(200)])
        assert len(session.calls) == 2
        assert result["stop_reason"] == "http_429"

    def test_five_consecutive_failures_stop(self, tmp_path):
        failures = [FakeResponse(503, None)] * 3 + [RuntimeError(f"boom apiKey={FAKE_KEY}")] * 2
        session, result = _poll(tmp_path, [FakeResponse(200)] + failures)
        assert len(session.calls) == 6
        assert result["stop_reason"] == "consecutive_failures"
        text = (tmp_path / "s1" / "requests.jsonl").read_text()
        assert FAKE_KEY not in text and "REDACTED" in text

    def test_records_carry_the_fields_and_the_first_body_of_each_version(self, tmp_path):
        responses = [FakeResponse(200, "2026-10-15T12:00:36Z"),
                     FakeResponse(200, "2026-10-15T12:00:36Z"),
                     FakeResponse(200, "2026-10-15T12:01:36Z")]
        _poll(tmp_path, responses, max_requests=3)
        lines = _records(tmp_path / "s1" / "requests.jsonl")
        assert lines[0]["type"] == "header"
        assert lines[0]["script_sha"] == "deadbeef"
        requests = [r for r in lines if r.get("type") == "request"]
        first = requests[0]
        for field in ("seq", "area", "t_send_wall", "t_send_mono", "t_recv_wall",
                      "t_recv_mono", "status", "Date", "Age", "Cache-Control", "ETag",
                      "sourceUpdated", "body_sha256", "body_bytes", "rtt_s", "request_url",
                      "body_file"):
            assert field in first, field
        assert [bool(r["body_file"]) for r in requests] == [True, False, True]
        assert len(list((tmp_path / "s1" / "bodies").iterdir())) == 2
        clock = [r for r in lines if r.get("type") == "clock_check"]
        assert len(clock) == 1 and "offset_s" in clock[0]
        assert FAKE_KEY not in (tmp_path / "s1" / "requests.jsonl").read_text()

    def test_two_areas_are_interleaved_half_an_interval_apart(self, tmp_path):
        areas = [phf.AREAS["nj"], phf.AREAS["mainz"]]
        session, _ = _poll(tmp_path, [FakeResponse(200)] * 4, areas=areas, max_requests=4)
        lines = [r for r in _records(tmp_path / "s1" / "requests.jsonl")
                 if r.get("type") == "request"]
        assert [r["area"] for r in lines] == ["nj", "mainz", "nj", "mainz"]
        sends = [r["t_send_wall"] for r in lines]
        assert [round(b - a, 6) for a, b in zip(sends, sends[1:])] == [2.5, 2.5, 2.5]

    def test_the_duration_ends_the_session(self, tmp_path):
        session, result = _poll(tmp_path, [FakeResponse(200)] * 100, duration_s=20.0)
        assert result["stop_reason"] == "duration"
        assert len(session.calls) == 4  # at 0, 5, 10, 15 s

    def test_coverage_requests_store_their_bodies(self, tmp_path):
        session, result = _poll(tmp_path, [FakeResponse(200)] * 5, max_requests=3,
                                coverage=list(phf.COVERAGE.values()))
        coverage = [r for r in _records(tmp_path / "s1" / "requests.jsonl")
                    if r.get("type") == "request" and r["area"].startswith("coverage")]
        assert len(coverage) == 2 and all(r["body_file"] for r in coverage)
        assert "r%3D7000" in coverage[0]["request_url"]
        assert len(session.calls) == 3  # the cap counts the coverage requests


def test_main_refuses_without_the_budget_confirmation(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(phf, "read_key", lambda path: called.append(path) or FAKE_KEY)
    with pytest.raises(SystemExit):
        phf.main(["--session-id", "x", "--out", str(tmp_path)])
    assert called == []  # the key is not even read
