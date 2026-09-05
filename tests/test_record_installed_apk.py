"""scripts/record_installed_apk.py: the sidecar A4 (validation round 2)
reads `apk_sha256` from.

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


#: Verbatim off ZY227VV4XC (Step 0, this session).
REAL_DUMPSYS_OUTPUT = "    versionCode=1 minSdk=29 targetSdk=35\n    versionName=0.1\n"


def test_dumpsys_versions_parses_the_real_device_output():
    def fake_run(args, *, capture_output, text, timeout):
        class Result:
            stdout = REAL_DUMPSYS_OUTPUT

        return Result()

    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = fake_run
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
    def fake_run(args, *, capture_output, text, timeout):
        class Result:
            stdout = "nothing relevant here"

        return Result()

    ria_subprocess_run = ria.subprocess.run
    ria.subprocess.run = fake_run
    try:
        assert ria._dumpsys_versions("ZY227VV4XC") == (None, None)
    finally:
        ria.subprocess.run = ria_subprocess_run


def test_main_writes_the_real_sha256_without_a_serial(tmp_path, monkeypatch):
    apk = tmp_path / "app-debug.apk"
    apk.write_bytes(b"not a real apk, just bytes to hash")
    expected_sha256 = hashlib.sha256(apk.read_bytes()).hexdigest()
    out = tmp_path / "installed_apk.json"

    monkeypatch.setattr(sys, "argv", ["record_installed_apk.py", str(apk), "--out", str(out)])
    rc = ria.main()
    assert rc == 0

    record = json.loads(out.read_text())
    assert record["sha256"] == expected_sha256
    assert record["apk_name"] == "app-debug.apk"
    assert record["version_code"] is None
    assert record["version_name"] is None
    assert record["serial"] is None


def test_main_writes_dumpsys_versions_when_a_serial_is_given(tmp_path, monkeypatch):
    apk = tmp_path / "app-debug.apk"
    apk.write_bytes(b"some apk bytes")
    out = tmp_path / "installed_apk.json"

    def fake_run(args, *, capture_output, text, timeout):
        class Result:
            stdout = REAL_DUMPSYS_OUTPUT

        return Result()

    monkeypatch.setattr(ria.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv",
        ["record_installed_apk.py", str(apk), "--serial", "ZY227VV4XC", "--out", str(out)],
    )
    rc = ria.main()
    assert rc == 0

    record = json.loads(out.read_text())
    assert record["version_code"] == 1
    assert record["version_name"] == "0.1"
    assert record["serial"] == "ZY227VV4XC"


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
