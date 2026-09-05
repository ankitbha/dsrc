"""scripts/record_installed_apk.py: the sidecar A4 (validation round 2)
reads `apk_sha256` from, and A5 (validation round 3) made read the DEVICE
rather than the local file it happened to be run with.

Imported the same way `test_observation_audit.py` imports
`scripts/audit_observation_fields.py`'s `json_safe`: `sys.path` gains
`scripts/` and the module is imported bare, since it is a script rather
than a package.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import record_installed_apk as ria  # noqa: E402


#: Verbatim off ZY227VV4XC (Step 0, this session; re-captured 2026-09-05 to
#: add `lastUpdateTime`, which A5's `_dumpsys_last_update_time` reads).
REAL_DUMPSYS_OUTPUT = (
    "    versionCode=1 minSdk=29 targetSdk=35\n"
    "    versionName=0.1\n"
    "    splits=[base]\n"
    "    firstInstallTime=2026-09-05 00:58:05\n"
    "    lastUpdateTime=2026-09-05 00:58:05\n"
)

#: Verbatim `adb -s ZY227VV4XC shell pm path com.dsrc.phone` (this session).
REAL_PM_PATH_OUTPUT = (
    "package:/data/app/~~QTsXfk3jV6LbKousGlS8QQ==/"
    "com.dsrc.phone-KakTd_o6eVFvp4Xee90S8A==/base.apk\n"
)

#: Verbatim `adb -s ZY227VV4XC shell sha256sum <that path>` (this session) --
#: matches the coordinator's own A5 repro exactly.
REAL_DEVICE_SHA256 = "e179e95282ced6a99a71b03547350079254ceda3caabbbd2778dc7c15f327602"
REAL_SHA256SUM_OUTPUT = (
    f"{REAL_DEVICE_SHA256}  /data/app/~~QTsXfk3jV6LbKousGlS8QQ==/"
    f"com.dsrc.phone-KakTd_o6eVFvp4Xee90S8A==/base.apk\n"
)


class _Result:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def _dispatching_run(pm_path_out=REAL_PM_PATH_OUTPUT, sha256sum_out=REAL_SHA256SUM_OUTPUT,
                      dumpsys_out=REAL_DUMPSYS_OUTPUT):
    """A fake `subprocess.run` that answers `pm path`, `sha256sum` and
    `dumpsys package` differently, the way a real `adb` does -- A5's
    `_device_apk_sha256` issues TWO different adb calls, so a fixture that
    returns the same fixed string for both (the pre-A5 tests' shape)
    cannot exercise it at all.
    """
    def fake_run(args, *, capture_output, text, timeout):
        if "sha256sum" in args:
            return _Result(sha256sum_out)
        if "path" in args:
            return _Result(pm_path_out)
        if "dumpsys" in args:
            return _Result(dumpsys_out)
        raise AssertionError(f"unexpected adb invocation: {args}")
    return fake_run


def test_dumpsys_versions_parses_the_real_device_output():
    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = _dispatching_run()
    try:
        code, name = ria._dumpsys_versions("ZY227VV4XC")
    finally:
        ria.subprocess.run = ria_subprocess_run
    assert code == 1
    assert name == "0.1"


def test_dumpsys_versions_is_none_none_when_adb_is_unreachable():
    def raising_run(args, *, capture_output, text, timeout):
        raise FileNotFoundError("no adb")

    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = raising_run
    try:
        assert ria._dumpsys_versions("ZY227VV4XC") == (None, None)
    finally:
        ria.subprocess.run = ria_subprocess_run


def test_dumpsys_versions_is_none_none_on_unrecognised_output():
    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = _dispatching_run(dumpsys_out="nothing relevant here")
    try:
        assert ria._dumpsys_versions("ZY227VV4XC") == (None, None)
    finally:
        ria.subprocess.run = ria_subprocess_run


# -- A5 (validation round 3): _dumpsys_last_update_time ---------------------


def test_dumpsys_last_update_time_parses_the_real_device_output():
    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = _dispatching_run()
    try:
        assert ria._dumpsys_last_update_time("ZY227VV4XC") == "2026-09-05 00:58:05"
    finally:
        ria.subprocess.run = ria_subprocess_run


def test_dumpsys_last_update_time_is_none_on_unrecognised_output():
    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = _dispatching_run(dumpsys_out="nothing relevant here")
    try:
        assert ria._dumpsys_last_update_time("ZY227VV4XC") is None
    finally:
        ria.subprocess.run = ria_subprocess_run


def test_dumpsys_last_update_time_is_none_when_adb_is_unreachable():
    def raising_run(args, *, capture_output, text, timeout):
        raise FileNotFoundError("no adb")

    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = raising_run
    try:
        assert ria._dumpsys_last_update_time("ZY227VV4XC") is None
    finally:
        ria.subprocess.run = ria_subprocess_run


# -- A5 (validation round 3): _device_apk_sha256 -----------------------------


def test_device_apk_sha256_parses_the_real_pm_path_and_sha256sum_output():
    """Confirmed against real hardware (this session): the coordinator's
    own A5 repro's hash, byte-identical."""
    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = _dispatching_run()
    try:
        assert ria._device_apk_sha256("ZY227VV4XC") == REAL_DEVICE_SHA256
    finally:
        ria.subprocess.run = ria_subprocess_run


def test_device_apk_sha256_is_none_when_pm_path_has_no_package_line():
    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = _dispatching_run(pm_path_out="\n")
    try:
        assert ria._device_apk_sha256("ZY227VV4XC") is None
    finally:
        ria.subprocess.run = ria_subprocess_run


def test_device_apk_sha256_is_none_when_pm_path_fails():
    def fake_run(args, *, capture_output, text, timeout):
        if "path" in args:
            return _Result("", returncode=1)
        raise AssertionError(f"unexpected adb invocation: {args}")

    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = fake_run
    try:
        assert ria._device_apk_sha256("ZY227VV4XC") is None
    finally:
        ria.subprocess.run = ria_subprocess_run


def test_device_apk_sha256_is_none_when_sha256sum_fails():
    def fake_run(args, *, capture_output, text, timeout):
        if "path" in args:
            return _Result(REAL_PM_PATH_OUTPUT)
        if "sha256sum" in args:
            return _Result("", returncode=1)
        raise AssertionError(f"unexpected adb invocation: {args}")

    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = fake_run
    try:
        assert ria._device_apk_sha256("ZY227VV4XC") is None
    finally:
        ria.subprocess.run = ria_subprocess_run


def test_device_apk_sha256_is_none_when_adb_is_unreachable():
    def raising_run(args, *, capture_output, text, timeout):
        raise FileNotFoundError("no adb")

    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = raising_run
    try:
        assert ria._device_apk_sha256("ZY227VV4XC") is None
    finally:
        ria.subprocess.run = ria_subprocess_run


# -- main() -------------------------------------------------------------


def test_main_leaves_sha256_null_without_a_serial(tmp_path, monkeypatch, capsys):
    """A5 (validation round 3): the pre-fix behaviour recorded the LOCAL
    file's hash as `sha256` even with no device involved at all -- the
    exact "even at write time" half of the finding. Now `sha256` is null
    (nothing read a device), and `local_apk_sha256` carries the local
    hash instead, named as corroborating rather than authoritative.
    """
    apk = tmp_path / "app-debug.apk"
    apk.write_bytes(b"not a real apk, just bytes to hash")
    expected_local_sha256 = hashlib.sha256(apk.read_bytes()).hexdigest()
    out = tmp_path / "installed_apk.json"

    monkeypatch.setattr(sys, "argv", ["record_installed_apk.py", str(apk), "--out", str(out)])
    rc = ria.main()
    assert rc == 0

    record = json.loads(out.read_text())
    assert record["sha256"] is None
    assert record["local_apk_sha256"] == expected_local_sha256
    assert record["last_update_time"] is None
    assert record["apk_name"] == "app-debug.apk"
    assert record["version_code"] is None
    assert record["version_name"] is None
    assert record["serial"] is None
    assert "no --serial given" in capsys.readouterr().err


def test_main_records_the_device_sha256_as_authoritative_when_a_serial_is_given(
    tmp_path, monkeypatch,
):
    """The core of A5: `good.apk --serial X` after installing `bad.apk`
    must not report `good.apk`'s hash as the truth -- `sha256` comes from
    the DEVICE (`REAL_DEVICE_SHA256`, the coordinator's own verified
    value), independent of and different from whatever local file this
    run happened to point at.
    """
    apk = tmp_path / "good.apk"
    apk.write_bytes(b"a different local apk than what is actually installed")
    local_sha256 = hashlib.sha256(apk.read_bytes()).hexdigest()
    out = tmp_path / "installed_apk.json"

    monkeypatch.setattr(ria.subprocess, "run", _dispatching_run())
    monkeypatch.setattr(
        sys, "argv",
        ["record_installed_apk.py", str(apk), "--serial", "ZY227VV4XC", "--out", str(out)],
    )
    rc = ria.main()
    assert rc == 0

    record = json.loads(out.read_text())
    assert record["sha256"] == REAL_DEVICE_SHA256
    assert record["local_apk_sha256"] == local_sha256
    assert record["sha256"] != record["local_apk_sha256"], (
        "the whole point: the device's hash and the local file's hash are "
        "different quantities and must not collapse to one field"
    )
    assert record["last_update_time"] == "2026-09-05 00:58:05"
    assert record["version_code"] == 1
    assert record["version_name"] == "0.1"
    assert record["serial"] == "ZY227VV4XC"


def test_main_warns_and_leaves_sha256_null_when_the_device_cannot_be_read(
    tmp_path, monkeypatch, capsys,
):
    """`--serial` given but adb cannot answer at all (bad serial, device
    unplugged): `sha256` must not silently fall back to the local file's
    hash -- that is the defect this fix exists for -- it stays null, and
    a reader is told why via stderr rather than left to assume the field
    is simply missing by design.
    """
    apk = tmp_path / "app-debug.apk"
    apk.write_bytes(b"some apk bytes")
    out = tmp_path / "installed_apk.json"

    def raising_run(args, *, capture_output, text, timeout):
        raise FileNotFoundError("no adb")

    monkeypatch.setattr(ria.subprocess, "run", raising_run)
    monkeypatch.setattr(
        sys, "argv",
        ["record_installed_apk.py", str(apk), "--serial", "NOSUCHSERIAL", "--out", str(out)],
    )
    rc = ria.main()
    assert rc == 0

    record = json.loads(out.read_text())
    assert record["sha256"] is None
    assert record["local_apk_sha256"] is not None
    assert "could not read the device" in capsys.readouterr().err


def test_main_refuses_a_missing_apk(tmp_path, monkeypatch, capsys):
    out = tmp_path / "installed_apk.json"
    monkeypatch.setattr(
        sys, "argv",
        ["record_installed_apk.py", str(tmp_path / "nope.apk"), "--out", str(out)],
    )
    rc = ria.main()
    assert rc == 1
    assert not out.exists()
    assert "does not exist" in capsys.readouterr().err
