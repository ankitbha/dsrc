#!/usr/bin/env python3
"""Watch a drive's health and say when it changes.

    python3 scripts/watch_drive.py                 # loop, 15s, 4h cap
    python3 scripts/watch_drive.py --once          # one verdict, for a shell
    python3 scripts/watch_drive.py --interval 30 --max-minutes 90

Two outputs, on purpose. `~/dsrc_logs/drive_health.jsonl` gets a record only when
the verdict or its reason CHANGES, so an hour of a healthy drive is a handful of
lines rather than 240 identical ones -- a log nobody can skim is a log nobody reads,
and this project has a measured case of 134 identical escalations burying the one
that mattered. `~/dsrc_logs/drive_health.latest.json` is overwritten every tick, so
a reader always has the current state without seeking to the end of a file.

Exit status on `--once` is the verdict, so it composes: 0 OK, 1 DEGRADED, 2 BROKEN,
3 UNKNOWN. UNKNOWN is not 0. A caller must not be able to treat "could not check" as
"fine" just by testing for success.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "deployment", "jetson"))

import yaml  # noqa: E402

from drive_health import BROKEN, DEGRADED, OK, UNKNOWN, assess, gather  # noqa: E402

EXIT = {OK: 0, DEGRADED: 1, BROKEN: 2, UNKNOWN: 3}


def ntp_synchronised() -> bool | None:
    """None, not False, when it cannot be determined -- see task 65."""
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized",
                              "--value"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    value = out.stdout.strip()
    return True if value == "yes" else False if value == "no" else None


def video_enabled(config_path: str) -> bool:
    try:
        return bool(yaml.safe_load(open(config_path))["logio"]["video"])
    except Exception:
        return False


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default=os.path.expanduser("~/dsrc_logs"))
    ap.add_argument("--config",
                    default=os.path.join(here, "..", "deployment", "jetson", "config.yaml"))
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--max-minutes", type=float, default=240.0,
                    help="hard cap, checked inside the loop so the watcher cannot "
                         "outlive the drive even if its terminal goes away")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    changes = os.path.join(args.log_dir, "drive_health.jsonl")
    latest = os.path.join(args.log_dir, "drive_health.latest.json")
    enabled = video_enabled(args.config)

    def tick() -> tuple[str, str]:
        health = assess(gather(args.log_dir, enabled, ntp_synchronised()))
        record = health.to_record()
        try:
            os.makedirs(args.log_dir, exist_ok=True)
            tmp = latest + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(record, fh)
            os.replace(tmp, latest)          # atomic: never a half-written state
        except OSError:
            pass
        return record["verdict"], record["line"]

    if args.once:
        verdict, line = tick()
        print(line)
        return EXIT[verdict]

    # Computed before the loop and compared inside it, so a killed parent cannot
    # leave this running past its welcome.
    deadline = time.monotonic() + args.max_minutes * 60.0
    previous: str | None = None
    while time.monotonic() < deadline:
        verdict, line = tick()
        if line != previous:                 # only a CHANGE is worth a line
            stamped = f"{time.strftime('%H:%M:%S')}  {line}"
            print(stamped, flush=True)
            try:
                with open(changes, "a") as fh:
                    fh.write(json.dumps({"at_wall": time.time(), "line": line}) + "\n")
            except OSError:
                pass
            previous = line
        time.sleep(args.interval)
    print(f"{time.strftime('%H:%M:%S')}  watcher reached its {args.max_minutes:.0f} "
          f"minute cap and stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
