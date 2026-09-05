#!/usr/bin/env python3
"""Record what an install step just put on the phone.

A4 (validation round 2): given a run directory alone, the phone's APK was
not attributable to anything -- `run_demo.py`'s `summary["build"]` now
reads `apk_sha256` from the file THIS script writes,
`deployment/jetson/models/installed_apk.json`. Untracked, same as every
other model artifact (`models/.gitignore` is `*`).

Run this on whichever host just ran `adb install -r <apk>` -- the same
host `run_demo.py` itself runs on, so the file lands in the tree that will
read it back:

    python3 scripts/record_installed_apk.py <apk_path> --serial <serial>
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


def _dumpsys_versions(serial: str) -> tuple[int | None, str | None]:
    """versionCode/versionName off `dumpsys package com.dsrc.phone`, or
    `(None, None)` when adb cannot answer -- this is corroborating
    provenance, not required for the sha256 record to exist at all."""
    try:
        result = subprocess.run(
            ["adb", "-s", serial, "shell", "dumpsys", "package", "com.dsrc.phone"],
            capture_output=True, text=True, timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    code_match = re.search(r"versionCode=(\d+)", result.stdout)
    name_match = re.search(r"versionName=(\S+)", result.stdout)
    code = int(code_match.group(1)) if code_match else None
    name = name_match.group(1) if name_match else None
    return code, name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("apk_path", type=Path)
    parser.add_argument("--serial", help="also query dumpsys for versionCode/versionName")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.apk_path.exists():
        print(f"{args.apk_path} does not exist", file=sys.stderr)
        return 1

    sha256 = hashlib.sha256(args.apk_path.read_bytes()).hexdigest()
    version_code = version_name = None
    if args.serial:
        version_code, version_name = _dumpsys_versions(args.serial)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "sha256": sha256,
        "apk_name": args.apk_path.name,
        "version_code": version_code,
        "version_name": version_name,
        "serial": args.serial,
    }, indent=2))
    print(f"wrote {args.out}")
    print(f"sha256={sha256} version_code={version_code} version_name={version_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
