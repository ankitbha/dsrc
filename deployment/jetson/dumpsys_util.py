"""Shared parsing for `adb shell dumpsys package <pkg>` output.

B23 (validation round 4, acted on rather than skipped): two call sites
read `lastUpdateTime` off the same command's output and then compare
their two readings for EQUALITY -- `run_demo.py`'s `_build_provenance`
(`_live_apk_last_update_time`, read NOW) against
`scripts/record_installed_apk.py`'s own recorded value
(`_dumpsys_last_update_time`, read at install time). A regex hardened in
one copy and not the other would make that comparison assert a reinstall
that never happened -- a correctness risk specific to the two values
being COMPARED, not just displayed independently side by side, which is
what makes this duplication worth closing where the validator's other,
purely-cosmetic duplication findings were not.

This module is the one place the extraction lives. `run_demo.py` imports
it normally (both live in `deployment/jetson/`, the same package).
`scripts/record_installed_apk.py` reaches in via `sys.path`, the same way
this task's own test files reach into `scripts/` from `tests/` -- CLI in
/scripts, /nash for modules; a script reaching into a module is the
allowed direction, not the reverse.
"""
from __future__ import annotations

import re


def parse_last_update_time(dumpsys_output: str) -> str | None:
    """`lastUpdateTime=...` off one `dumpsys package <pkg>` capture,
    verbatim and stripped (e.g. `"2026-09-05 00:58:05"`), or `None` when
    the field is absent from the output."""
    match = re.search(r"lastUpdateTime=(.+)", dumpsys_output)
    return match.group(1).strip() if match else None
