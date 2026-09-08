"""Keep the HERE response bodies, because they cannot be fetched again.

Why this exists. `_read_here` receives every response body, hands it to the feed,
which extracts `downstream_congestion` and `free_flow_mps`, and the body is then
gone. On 2026-09-08 a drive made 143 successful calls of about 55 KB each and
retained two floats per call. **A HERE response is unrepeatable**: nobody can ask
what the congestion on that road was at that minute, ever again. So a parse bug
found later, or a field the extractor did not take, costs the whole drive rather
than an afternoon of reprocessing.

Video frames had the same problem and were fixed the same way, so the shape here is
deliberately the same: payloads on disk, and a sidecar index naming what each one
is. A reader needs no code from this repository to use either.

The index is written AFTER the body, by the thread that wrote it, so a line exists
only for a body that is really on disk. The opposite order names files that may not
exist -- which is the failure the video index was built to avoid.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class HereLogger:
    """Writes `here/NNNNNN.json` plus a line per body in `here_index.jsonl`."""

    #: The sidecar naming what each stored body is.
    INDEX_NAME = "here_index.jsonl"
    #: The subdirectory the bodies go in.
    BODY_DIR = "here"

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.body_dir = self.run_dir / self.BODY_DIR
        self.body_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.run_dir / self.INDEX_NAME
        self._index = open(self.index_path, "a", buffering=1)
        self.written = 0
        self.bytes_written = 0
        #: Every body this logger was handed and failed to store, and why. Counted
        #: AND named: knowing three were lost is far less use than knowing which,
        #: because a reader can then exclude those calls instead of doubting all of
        #: them.
        self.failed = 0
        self.last_error: str | None = None

    def write(self, *, status: int, body: bytes, query_lat: float | None = None,
              query_lon: float | None = None, query_radius_m: float | None = None,
              received_t_mono: float | None = None, tick_id: int | None = None,
              request_url: str | None = None) -> bool:
        """Store one body. Returns False on failure, never raises.

        Never raises because the caller is the HERE reader thread: an exception here
        would take that thread down for the rest of the drive and stop the feed
        entirely, which is a far worse outcome than an unstored body. `_read_here`
        already learned this the hard way with an `OverflowError` out of `float()`.
        """
        seq = self.written
        name = f"{seq:06d}.json"
        try:
            path = self.body_dir / name
            # Written whole then renamed, so a reader never sees a partial body and
            # mistakes a truncated JSON for a malformed response.
            tmp = path.with_suffix(".json.part")
            with open(tmp, "wb") as fh:
                fh.write(body)
            os.replace(tmp, path)
            entry = {
                "seq": seq,
                "file": f"{self.BODY_DIR}/{name}",
                "status": int(status),
                "bytes": len(body),
                "query_lat": query_lat,
                "query_lon": query_lon,
                "query_radius_m": query_radius_m,
                "received_t_mono": received_t_mono,
                "tick_id": tick_id,
                "request_url": request_url,
            }
            self._index.write(json.dumps(entry) + "\n")
        except Exception as exc:  # noqa: BLE001 - see the docstring
            self.failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        self.written += 1
        self.bytes_written += len(body)
        return True

    def to_record(self) -> dict[str, Any]:
        """What a reader needs to trust, or distrust, the stored bodies.

        Surfaced because `VideoLogger.dropped_frames` was incremented and read by
        nothing, so a run could lose data and say so nowhere -- leaving "none lost"
        and "loss not reported" indistinguishable.
        """
        return {
            "written": self.written,
            "bytes_written": self.bytes_written,
            "failed": self.failed,
            "last_error": self.last_error,
            "index": self.INDEX_NAME,
            "body_dir": self.BODY_DIR,
            "complete": self.failed == 0,
        }

    def close(self) -> None:
        try:
            self._index.close()
        except Exception:  # noqa: BLE001
            pass
