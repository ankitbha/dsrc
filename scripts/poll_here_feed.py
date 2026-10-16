#!/usr/bin/env python3
"""Poll the HERE v7 flow feed from a stationary laptop to measure its update process.

    .venv/bin/python scripts/poll_here_feed.py --session-id afternoon \\
        --duration-s 7200 --budget-confirmed

Plan section 2 (`plans/2026-10-15-feed-agreement-plan.md`). One request per 5 s per
area, the two areas interleaved 2.5 s apart, for a stated duration. Every request is a
JSONL record under `outputs/feed/polling/{session_id}/requests.jsonl`; the body of the
first response of each new `sourceUpdated` per area is kept under `bodies/`.
`analyze_feed_process.py` turns the records into `results/feed/feed_process.json`.

THE KEY IS SHARED WITH NASH PRODUCTION (`plans/plan_task21_here_client.md`, item 2) and
its monthly allowance is not recorded anywhere in the repository, so:

* the script refuses to run without `--budget-confirmed`, which states that the author
  has checked the request count against the account;
* `--max-requests` (default 2,900) is a hard cap that counts every request sent,
  coverage requests included;
* it stops at the first HTTP 429, at five consecutive non-200 responses, or at the cap,
  and never retries;
* a 10-minute dry session (`--duration-s 600`) should run first and its 429 count be
  read before a 2 h session.

Key handling: the key is read from `phone/local.properties` at start, held in one local
variable, and passed only as the `apiKey` entry of the request's params. The recorded
`request_url` is built without it. Every string written passes through `redact`, which
covers exception messages from `requests` (they can embed the full URL). Before exit
every file written is re-read, and one containing the key is deleted and the run fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOW_URL = "https://data.traffic.hereapi.com/v7/flow"
DEFAULT_KEY_FILE = REPO_ROOT / "phone" / "local.properties"
KEY_PROPERTY = "here.apiKey"
DEFAULT_MAX_REQUESTS = 2900
CONSECUTIVE_FAILURE_LIMIT = 5


@dataclass(frozen=True)
class Area:
    name: str
    lat: float
    lon: float
    radius_m: float


#: Plan section 2.1. `nj` is the centroid of the 93 query centres of
#: run_20260908_170849; `mainz` the midpoint of exit link 218 through the Vissim
#: georeference (derived, not verified: plan open item 8).
AREAS: dict[str, Area] = {
    "nj": Area("nj", 40.3525, -74.6239, 500.0),
    "mainz": Area("mainz", 49.9312, 8.3091, 500.0),
}
#: One 7 km request at the Mainz exit and one at entry 289, bodies kept: the coverage
#: check that the network's roads are returned.
COVERAGE: dict[str, Area] = {
    "coverage_mainz_exit": Area("coverage_mainz_exit", 49.9312, 8.3091, 7000.0),
    "coverage_mainz_289": Area("coverage_mainz_289", 49.9600, 8.1529, 7000.0),
}


def read_key(path: Path) -> str:
    """`here.apiKey` from a Java properties file; a missing key is a refusal."""
    for line in Path(path).read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == KEY_PROPERTY and value.strip():
            return value.strip()
    raise SystemExit(f"{path} has no {KEY_PROPERTY}")


def build_request(area: Area, key: str) -> tuple[str, dict[str, str], str]:
    """`(url, params, recorded_url)`. The key is in `params` only."""
    query = {"in": f"circle:{area.lat:.4f},{area.lon:.4f};r={area.radius_m:.0f}",
             "locationReferencing": "shape"}
    recorded = f"{FLOW_URL}?{urlencode(query)}"
    return FLOW_URL, {**query, "apiKey": key}, recorded


def redact(text: str, key: str) -> str:
    """Replace the key, raw or URL-encoded, with REDACTED."""
    if not key:
        return text
    for form in {key, quote(key, safe=""), quote(key, safe="").lower()}:
        text = text.replace(form, "REDACTED")
    return text


def self_check(paths: Sequence[Path], key: str) -> None:
    """Re-read every written file; delete any that contains the key, then raise."""
    leaked = []
    forms = {key.encode(), quote(key, safe="").encode()}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        data = path.read_bytes()
        if any(form in data for form in forms):
            path.unlink()
            leaked.append(path.name)
    if leaked:
        raise RuntimeError(f"the key was found in {leaked}; those files were deleted")


def _script_sha() -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _clock_probe() -> str | None:
    """`sntp -t 5 time.apple.com`, if the tool exists; its output, not interpreted."""
    try:
        result = subprocess.run(["sntp", "-t", "5", "time.apple.com"], capture_output=True,
                                text=True, timeout=15)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _http_date(value: str | None) -> float | None:
    if not value:
        return None
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        return None


def poll(session, *, areas: Sequence[Area], interval_s: float, duration_s: float,
         max_requests: int, out_dir: Path, key: str,
         coverage: Sequence[Area] = (),
         wall: Callable[[], float] = time.time,
         mono: Callable[[], float] = time.monotonic,
         sleep: Callable[[float], None] = time.sleep,
         clock_probe: Callable[[], str | None] = _clock_probe,
         script_sha: Callable[[], str | None] = _script_sha,
         session_id: str | None = None) -> dict[str, Any]:
    """Run one session. `session` is a `requests.Session` or a test double."""
    out_dir = Path(out_dir)
    bodies = out_dir / "bodies"
    bodies.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "requests.jsonl"
    written: list[Path] = [records_path]
    seen_versions: dict[str, set[str]] = {}
    state = {"sent": 0, "failures": 0, "clock_checked": False}
    stop_reason = None

    def write(handle, record: dict) -> None:
        handle.write(redact(json.dumps(record), key) + "\n")
        handle.flush()

    def send(handle, area: Area, keep_body: bool) -> int | None:
        url, params, recorded = build_request(area, key)
        seq = state["sent"]
        state["sent"] += 1
        t_send_wall, t_send_mono = wall(), mono()
        record: dict[str, Any] = {"type": "request", "seq": seq, "area": area.name,
                                  "t_send_wall": t_send_wall, "t_send_mono": t_send_mono,
                                  "request_url": recorded}
        status = None
        try:
            response = session.get(url, params=params, timeout=(5, 10))
            t_recv_wall, t_recv_mono = wall(), mono()
            status = int(response.status_code)
            content = response.content or b""
            source_updated = None
            try:
                source_updated = json.loads(content).get("sourceUpdated")
            except (ValueError, AttributeError):
                pass
            headers = response.headers
            record.update({
                "t_recv_wall": t_recv_wall, "t_recv_mono": t_recv_mono, "status": status,
                "Date": headers.get("Date"), "Age": headers.get("Age"),
                "Cache-Control": headers.get("Cache-Control"), "ETag": headers.get("ETag"),
                "Last-Modified": headers.get("Last-Modified"),
                "sourceUpdated": source_updated,
                "body_sha256": hashlib.sha256(content).hexdigest(),
                "body_bytes": len(content), "rtt_s": t_recv_mono - t_send_mono,
                "body_file": None,
            })
            new_version = (status == 200 and source_updated is not None
                           and source_updated not in seen_versions.setdefault(area.name, set()))
            if new_version:
                seen_versions[area.name].add(source_updated)
            if status == 200 and (new_version or keep_body):
                body_path = bodies / f"{area.name}_{seq:06d}.json"
                body_path.write_text(redact(content.decode("utf-8", "replace"), key))
                written.append(body_path)
                record["body_file"] = str(body_path.relative_to(out_dir))
            if status == 200 and not state["clock_checked"]:
                server = _http_date(headers.get("Date"))
                write(handle, {"type": "clock_check", "t_recv_wall": t_recv_wall,
                               "server_date": headers.get("Date"),
                               "offset_s": (t_recv_wall - server) if server else None,
                               "resolution_s": 1.0})
                state["clock_checked"] = True
        except Exception as error:  # noqa: BLE001 - recorded, redacted, counted
            t_recv_wall, t_recv_mono = wall(), mono()
            record.update({"t_recv_wall": t_recv_wall, "t_recv_mono": t_recv_mono,
                           "status": None, "error": redact(repr(error), key),
                           "rtt_s": t_recv_mono - t_send_mono, "body_file": None,
                           "sourceUpdated": None})
        write(handle, record)
        state["failures"] = 0 if status == 200 else state["failures"] + 1
        return status

    with records_path.open("w") as handle:
        write(handle, {"type": "header", "session_id": session_id or out_dir.name,
                       "started_wall": wall(), "platform": platform.platform(),
                       "python": sys.version.split()[0], "script_sha": script_sha(),
                       "sntp": clock_probe(), "areas": [a.__dict__ for a in areas],
                       "interval_s": interval_s, "duration_s": duration_s,
                       "max_requests": max_requests,
                       "coverage": [a.__dict__ for a in coverage]})
        for area in coverage:
            if state["sent"] >= max_requests:
                stop_reason = "max_requests"
                break
            status = send(handle, area, keep_body=True)
            if status == 429:
                stop_reason = "http_429"
                break
        start = wall()
        slot = 0
        offset = interval_s / max(len(areas), 1)
        while stop_reason is None:
            area = areas[slot % len(areas)]
            due = start + (slot // len(areas)) * interval_s + (slot % len(areas)) * offset
            if due - start >= duration_s:
                stop_reason = "duration"
                break
            if state["sent"] >= max_requests:
                stop_reason = "max_requests"
                break
            now = wall()
            if due > now:
                sleep(due - now)
            status = send(handle, area, keep_body=False)
            slot += 1
            if status == 429:
                stop_reason = "http_429"
            elif state["failures"] >= CONSECUTIVE_FAILURE_LIMIT:
                stop_reason = "consecutive_failures"
        write(handle, {"type": "footer", "stop_reason": stop_reason, "sent": state["sent"],
                       "ended_wall": wall()})
    self_check(written, key)
    return {"stop_reason": stop_reason, "sent": state["sent"], "records": str(records_path),
            "files": [str(p) for p in written]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--areas", nargs="+", choices=tuple(AREAS), default=list(AREAS))
    parser.add_argument("--coverage", action="store_true",
                        help="send the two 7 km Mainz coverage requests first")
    parser.add_argument("--key-file", default=str(DEFAULT_KEY_FILE))
    parser.add_argument("--out", default="outputs/feed/polling")
    parser.add_argument("--budget-confirmed", action="store_true",
                        help="the author has checked this session's request count against "
                             "the production account's allowance")
    args = parser.parse_args(argv)
    if not args.budget_confirmed:
        raise SystemExit("refusing to poll: the key is shared with Nash production and the "
                         "budget has not been confirmed (--budget-confirmed)")
    planned = (len(args.areas) * int(args.duration_s // args.interval_s)
               + (len(COVERAGE) if args.coverage else 0))
    print(f"session {args.session_id}: up to {min(planned, args.max_requests)} requests "
          f"(cap {args.max_requests})")
    key = read_key(Path(args.key_file))
    import requests

    out_dir = (REPO_ROOT / args.out / args.session_id) if not Path(args.out).is_absolute() \
        else Path(args.out) / args.session_id
    with requests.Session() as session:
        result = poll(session, areas=[AREAS[a] for a in args.areas],
                      interval_s=args.interval_s, duration_s=args.duration_s,
                      max_requests=args.max_requests, out_dir=out_dir, key=key,
                      coverage=list(COVERAGE.values()) if args.coverage else (),
                      session_id=args.session_id)
    print(f"stopped: {result['stop_reason']} after {result['sent']} requests; "
          f"key self-check passed on {len(result['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
