#!/usr/bin/env python3
"""Record what an install step just put on the phone.

A4 (validation round 2): given a run directory alone, the phone's APK was
not attributable to anything -- `run_demo.py`'s `summary["build"]` reads
`apk_sha256` from the file THIS script writes,
`deployment/jetson/models/installed_apk.json`. Untracked, same as every
other model artifact (`models/.gitignore` is `*`).

A5 (validation round 3): the local APK's own hash is not evidence of what
is installed. `adb install -r other.apk` run without this script leaves
the old hash in place forever, and even at write time,
`record_installed_apk.py good.apk --serial X` after installing `bad.apk`
recorded `good.apk`'s hash and reported success -- nothing here ever read
the DEVICE. `sha256`, the field `_build_provenance` treats as
authoritative, is therefore the device-side `sha256sum` of `pm path`'s own
result, queried with `--serial`; the local file's hash is kept beside it
as `local_apk_sha256`, corroborating rather than authoritative.
`last_update_time` (`dumpsys package`'s own field) is recorded too, so
`run_demo.py` can re-check at run time whether the device still matches
what this sidecar was written against.

Run this on whichever host just ran `adb install -r <apk>` -- the same
host `run_demo.py` itself runs on, so the file lands in the tree that will
read it back:

    python3 scripts/record_installed_apk.py <apk_path> --serial <serial>

`--serial` is required to produce an authoritative `sha256`: without a
device to query, nothing here has read the phone at all, and the local
file's hash is not a substitute (that is the defect this fix exists for).
Omitting it still writes `local_apk_sha256` and leaves `sha256`/
`last_update_time` null, named rather than silently guessed at.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "deployment" / "jetson" / "models" / "installed_apk.json"
PACKAGE = "com.dsrc.phone"

# B23 (validation round 4, acted on rather than skipped): `lastUpdateTime`
# is read here at INSTALL time and again by `run_demo.py`'s
# `_live_apk_last_update_time` at RUN time, and the two values are then
# COMPARED for equality -- a regex hardened in one copy and not the other
# would make that comparison assert a reinstall that never happened, so
# the extraction itself lives in one shared module rather than two
# separate regexes that happen to agree today.
sys.path.insert(0, str(REPO_ROOT / "deployment" / "jetson"))
from dumpsys_util import parse_last_update_time  # noqa: E402


def _dumpsys_versions(serial: str) -> tuple[int | None, str | None]:
    """versionCode/versionName off `dumpsys package com.dsrc.phone`, or
    `(None, None)` when adb cannot answer -- this is corroborating
    provenance, not required for the sha256 record to exist at all."""
    try:
        result = subprocess.run(
            ["adb", "-s", serial, "shell", "dumpsys", "package", PACKAGE],
            capture_output=True, text=True, timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    code_match = re.search(r"versionCode=(\d+)", result.stdout)
    name_match = re.search(r"versionName=(\S+)", result.stdout)
    code = int(code_match.group(1)) if code_match else None
    name = name_match.group(1) if name_match else None
    return code, name


def _dumpsys_last_update_time(serial: str) -> str | None:
    """`lastUpdateTime` off `dumpsys package com.dsrc.phone`, verbatim
    (e.g. `"2026-09-05 00:58:05"`) -- the field `run_demo.py`'s
    `_build_provenance` re-reads at run time to tell whether the device
    still matches what this sidecar was written against (A5, validation
    round 3). `None` when adb cannot answer or the field is absent. The
    extraction itself is `dumpsys_util.parse_last_update_time`, shared
    with that re-check rather than duplicated (B23, validation round 4)."""
    try:
        result = subprocess.run(
            ["adb", "-s", serial, "shell", "dumpsys", "package", PACKAGE],
            capture_output=True, text=True, timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_last_update_time(result.stdout)


def _device_apk_sha256(serial: str) -> str | None:
    """The DEVICE's own `sha256sum` of the installed APK -- `pm path`'s
    result, hashed on the device, not the local file this run installed
    from (A5, validation round 3): a stale local file, or a mismatch
    between what was pushed and what actually took, both read as a
    correct install under the old, local-only hash.

    This app has one split (`splits=[base]`, verified on real hardware),
    so `pm path` returns exactly one `package:`-prefixed line; only the
    first such line is used, which is also what a split APK's base module
    would report first.
    """
    try:
        path_result = subprocess.run(
            ["adb", "-s", serial, "shell", "pm", "path", PACKAGE],
            capture_output=True, text=True, timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if path_result.returncode != 0:
        return None
    device_path = None
    for line in path_result.stdout.splitlines():
        line = line.strip()
        if line.startswith("package:"):
            device_path = line[len("package:"):]
            break
    if not device_path:
        return None
    try:
        hash_result = subprocess.run(
            ["adb", "-s", serial, "shell", "sha256sum", device_path],
            capture_output=True, text=True, timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if hash_result.returncode != 0:
        return None
    parts = hash_result.stdout.split()
    return parts[0] if parts else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("apk_path", type=Path)
    parser.add_argument(
        "--serial",
        help="query the DEVICE for its installed APK's own sha256sum, lastUpdateTime, "
             "versionCode/versionName -- without this, sha256 and last_update_time are "
             "null (A5, validation round 3: the local file's hash is not authoritative)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.apk_path.exists():
        print(f"{args.apk_path} does not exist", file=sys.stderr)
        return 1

    local_apk_sha256 = hashlib.sha256(args.apk_path.read_bytes()).hexdigest()
    device_sha256 = None
    last_update_time = None
    version_code = version_name = None
    if args.serial:
        version_code, version_name = _dumpsys_versions(args.serial)
        last_update_time = _dumpsys_last_update_time(args.serial)
        device_sha256 = _device_apk_sha256(args.serial)
        if device_sha256 is None:
            print(
                "warning: could not read the device's own installed APK hash "
                "(pm path/sha256sum failed) -- sha256 will be null", file=sys.stderr,
            )
    else:
        print(
            "warning: no --serial given -- sha256 will be null; local_apk_sha256 is "
            "not a substitute for reading the device", file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "sha256": device_sha256,
        "local_apk_sha256": local_apk_sha256,
        "last_update_time": last_update_time,
        "apk_name": args.apk_path.name,
        "version_code": version_code,
        "version_name": version_name,
        "serial": args.serial,
    }, indent=2))
    print(f"wrote {args.out}")
    print(
        f"sha256={device_sha256} local_apk_sha256={local_apk_sha256} "
        f"last_update_time={last_update_time} version_code={version_code} "
        f"version_name={version_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
