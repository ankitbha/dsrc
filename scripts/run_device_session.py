#!/usr/bin/env python3
"""Drive a real sensing session on a device and bring the evidence back.

Three things about this app shape the approach.

The service is `exported="false"`, so `am start-foreground-service` is refused and
the only way in is the UI -- which is also how the app is actually used.

The activity never goes idle: it redraws roughly every 250 ms, so `uiautomator dump`
returns "could not get idle state" against it and writes no file. `dumpsys activity
top` reads the view tree without waiting for idle, so that is what reads state here.
uiautomator *is* used for permission dialogs, because permissioncontroller does go
idle.

Button positions move. The status text above them grows and shrinks by a line as the
state changes, which shifts both buttons down and up; a coordinate captured a second
ago can miss. So buttons are pressed by focus (DPAD + ENTER) rather than by tapping a
remembered point, which is also resolution-independent and so works on any handset.

Everything is bounded: every wait has a deadline inside its loop. Nothing here sends a
`here` query, and a build without a HERE key disables that modality outright, so the
phone cannot call HERE.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

PKG = "com.dsrc.phone"

GRANTS = (
    "android.permission.CAMERA",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.POST_NOTIFICATIONS",
)

# Answered in order of preference. "While using the app" over "Only this time",
# because a one-time grant is revoked when the app is backgrounded.
ALLOW_IDS = (
    "permission_allow_foreground_only_button",
    "permission_allow_button",
    "permission_allow_one_time_button",
)


def sh(serial: str, *args: str, timeout: float = 120.0) -> str:
    done = subprocess.run(
        ["adb", "-s", serial, *args], capture_output=True, text=True, timeout=timeout
    )
    return done.stdout


def views(serial: str) -> str:
    """Our activity's view tree. Does not require an idle window."""
    tree = sh(serial, "shell", "dumpsys", "activity", "top")
    marker = f"ACTIVITY {PKG}"
    return tree[tree.index(marker) :] if marker in tree else ""


def buttons(serial: str) -> list[str]:
    return re.findall(r"android\.widget\.Button\{[^}]*\}", views(serial))


def focus_index(serial: str) -> int | None:
    """Which button currently has focus.

    A view line is `Button{hash FLAGS PRIVATEFLAGS bounds ...}`, and focus is the
    second character of the private-flags block: `.F....ID` focused, `........` not.
    """
    for i, view in enumerate(buttons(serial)):
        parts = view.split()
        if len(parts) > 2 and len(parts[2]) > 1 and parts[2][1] == "F":
            return i
    return None


def press(serial: str, index: int, deadline: float) -> bool:
    """Move focus onto button `index` and activate it."""
    sh(serial, "shell", "input", "keyevent", "KEYCODE_DPAD_DOWN")
    while time.time() < deadline:
        at = focus_index(serial)
        if at == index:
            sh(serial, "shell", "input", "keyevent", "KEYCODE_ENTER")
            return True
        if at is None:
            sh(serial, "shell", "input", "keyevent", "KEYCODE_DPAD_DOWN")
        elif at < index:
            sh(serial, "shell", "input", "keyevent", "KEYCODE_DPAD_DOWN")
        else:
            sh(serial, "shell", "input", "keyevent", "KEYCODE_DPAD_UP")
        time.sleep(0.4)
    return False


def clear_dialogs(serial: str, deadline: float) -> int:
    """Answer runtime permission dialogs. permissioncontroller does go idle."""
    answered = 0
    while time.time() < deadline:
        sh(serial, "shell", "rm", "-f", "/sdcard/_d.xml")
        sh(serial, "shell", "uiautomator", "dump", "/sdcard/_d.xml")
        xml = sh(serial, "shell", "cat", "/sdcard/_d.xml")
        if "permissioncontroller" not in xml:
            return answered
        for ident in ALLOW_IDS:
            m = re.search(rf'resource-id="[^"]*{ident}"', xml)
            if not m:
                continue
            window = xml[max(0, m.start() - 800) : m.start() + 800]
            b = re.findall(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', window)
            if b:
                x1, y1, x2, y2 = map(int, b[-1])
                sh(serial, "shell", "input", "tap", str((x1 + x2) // 2), str((y1 + y2) // 2))
                answered += 1
                time.sleep(1.5)
                break
        else:
            return answered
    return answered


def transitions(serial: str) -> list[str]:
    log = sh(serial, "logcat", "-d", "-s", "SensingService:*", timeout=180)
    return re.findall(r"([A-Z_]+ -> [A-Z_]+ on \w+)", log)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # A real handset locks itself while the rig is being set up, and a locked screen
    # keeps the activity off the foreground -- `am start` returns fine and `dumpsys
    # activity top` then shows the keyguard, so the buttons simply are not there. An
    # emulator never locks, so nothing upstream of a real device can catch this.
    sh(args.serial, "shell", "input", "keyevent", "KEYCODE_WAKEUP")
    sh(args.serial, "shell", "wm", "dismiss-keyguard")
    # `stayon usb` depends on the phone actually charging over the cable, and this
    # one dozed off mid-setup with it set. `true` holds the screen regardless.
    sh(args.serial, "shell", "svc", "power", "stayon", "true")
    time.sleep(2.0)
    awake = sh(args.serial, "shell", "dumpsys", "power")
    if "mWakefulness=Awake" not in awake:
        print("device did not wake", flush=True)
        return 4

    for perm in GRANTS:
        sh(args.serial, "shell", "pm", "grant", PKG, perm)
    sh(args.serial, "shell", "am", "force-stop", PKG)
    sh(args.serial, "logcat", "-c")
    sh(args.serial, "shell", "am", "start", "-n", f"{PKG}/.MainActivity")

    # Wait for the activity rather than assuming a fixed settle: a cold start on a
    # handset is slower than on an emulator by enough to matter.
    up = time.time() + 60
    while time.time() < up and not buttons(args.serial):
        time.sleep(1.0)

    if not press(args.serial, 0, time.time() + 45):
        print("could not focus START", flush=True)
        (args.out / "logcat.txt").write_text(sh(args.serial, "logcat", "-d", "-v", "time", timeout=180))
        (args.out / "views.txt").write_text(views(args.serial))
        sh(args.serial, "exec-out", "screencap", "-p")
        return 2
    answered = clear_dialogs(args.serial, time.time() + 90)
    print(f"answered {answered} permission dialog(s)", flush=True)

    # MainActivity resumes the Start the user asked for once permissions land, so a
    # second press is only needed when no dialog appeared and the first was swallowed.
    deadline = time.time() + 40
    while time.time() < deadline and not any("-> RUNNING" in t for t in transitions(args.serial)):
        time.sleep(2.0)
    if not any("-> RUNNING" in t for t in transitions(args.serial)):
        press(args.serial, 0, time.time() + 20)
        clear_dialogs(args.serial, time.time() + 30)
        deadline = time.time() + 40
        while time.time() < deadline and not any("-> RUNNING" in t for t in transitions(args.serial)):
            time.sleep(2.0)

    running = any("-> RUNNING" in t for t in transitions(args.serial))
    print(f"reached RUNNING: {running}", flush=True)
    if not running:
        (args.out / "logcat.txt").write_text(sh(args.serial, "logcat", "-d", "-v", "time", timeout=180))
        return 3

    time.sleep(args.seconds)
    press(args.serial, 1, time.time() + 30)
    time.sleep(6.0)

    (args.out / "logcat.txt").write_text(sh(args.serial, "logcat", "-d", "-v", "time", timeout=240))
    listing = sh(args.serial, "shell", "run-as", PKG, "ls", "-l", "files/sessions")
    (args.out / "sessions_ls.txt").write_text(listing)
    for name in re.findall(r"(\S+\.jsonl)", listing):
        body = sh(args.serial, "shell", "run-as", PKG, "cat", f"files/sessions/{name}", timeout=600)
        (args.out / name).write_text(body)
        print(f"pulled {name} ({len(body)} bytes)", flush=True)

    print("transitions: " + " | ".join(transitions(args.serial)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
