"""dumpsys_util.parse_last_update_time -- B23 (validation round 4, acted
on rather than skipped): the one shared extraction both `run_demo.py`'s
`_live_apk_last_update_time` and `scripts/record_installed_apk.py`'s
`_dumpsys_last_update_time` call, since their two readings are compared
for equality and a regex difference between them would assert a
reinstall that never happened.
"""

from __future__ import annotations

from dumpsys_util import parse_last_update_time

#: Verbatim `adb -s ZY227VV4XC shell dumpsys package com.dsrc.phone`
#: (this session, 2026-09-05) -- the same real capture used elsewhere in
#: this task's fixtures.
REAL_DUMPSYS_OUTPUT = (
    "    versionCode=1 minSdk=29 targetSdk=35\n"
    "    versionName=0.1\n"
    "    splits=[base]\n"
    "    firstInstallTime=2026-09-05 00:58:05\n"
    "    lastUpdateTime=2026-09-05 00:58:05\n"
)


def test_parses_the_real_device_output():
    assert parse_last_update_time(REAL_DUMPSYS_OUTPUT) == "2026-09-05 00:58:05"


def test_is_none_on_unrecognised_output():
    assert parse_last_update_time("nothing relevant here") is None


def test_is_none_on_empty_output():
    assert parse_last_update_time("") is None


def test_strips_surrounding_whitespace():
    assert parse_last_update_time("lastUpdateTime=  2026-09-05 00:58:05  \n") == "2026-09-05 00:58:05"
