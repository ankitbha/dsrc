#!/usr/bin/env python3
"""Watch a drive and restart the phone app if it dies.

    python3 scripts/keep_drive_alive.py --mode live
    python3 scripts/keep_drive_alive.py --mode live --max-minutes 120

WHAT IT WILL AND WILL NOT DO, because the difference is the whole design.

It restarts the app **only when the app is provably gone**: adb answers, and
`pidof` comes back empty. If adb itself cannot be reached the state is UNKNOWN and
it does nothing at all -- acting on "I cannot tell" is how a watchdog starts a
second session on top of a live one, and this project already has a defect where
`usb2` reported `connected` through a total outage.

It does **not** restart on a stalled tick stream. A stall recovered on its own
during the 2026-09-08 drives, and restarting through a recovery would have destroyed
a session that was about to come back. Stalls are reported, not acted on.

It restarts at most `--max-restarts` times with a cooldown between them. A phone too
hot to hold a session will die again immediately, and a tight restart loop adds the
CPU load that is causing the problem. Hammering a thermally dead handset makes it
worse and hides why.

Only actions and changes are printed. A quiet drive produces a quiet log, because a
log nobody can skim is a log nobody reads.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_session_module():
    """Reuse the button navigation that already works, rather than a second copy.

    `run_device_session.py` is not importable by name (a hyphen-free filename that
    is still a script, not a package), so it is loaded by path. Copying `press` and
    `focus_index` here would mean two implementations of the one thing that has to
    agree with the activity's button order -- and that order changed the day the
    second Start button was added.
    """
    path = os.path.join(HERE, "run_device_session.py")
    spec = importlib.util.spec_from_file_location("run_device_session", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rds = _load_session_module()

UNKNOWN, ALIVE, DEAD = "UNKNOWN", "ALIVE", "DEAD"


def app_state(serial: str, package: str) -> tuple[str, str]:
    """ALIVE, DEAD, or UNKNOWN -- three outcomes, never two."""
    try:
        devices = rds.sh(serial, "devices")
    except Exception as exc:  # noqa: BLE001
        return UNKNOWN, f"adb devices failed: {type(exc).__name__}"
    if f"{serial}\tdevice" not in devices:
        return UNKNOWN, "the handset is not attached, so the app cannot be assessed"
    try:
        pid = rds.sh(serial, "shell", "pidof", package).strip()
    except Exception as exc:  # noqa: BLE001
        return UNKNOWN, f"pidof failed: {type(exc).__name__}"
    if pid:
        return ALIVE, f"pid {pid}"
    return DEAD, "adb answers and pidof is empty"


def restart_app(serial: str, package: str, mode: str, log) -> bool:
    """Wake the screen, open the activity, press the Start for `mode`."""
    log(f"restarting the app in {mode} mode")
    # The screen has to be on and unlocked for the activity to take focus, and the
    # button press goes through the view tree, which is the only way in: the service
    # is exported="false" so `am start-foreground-service` is refused.
    rds.sh(serial, "shell", "input", "keyevent", "KEYCODE_WAKEUP")
    rds.sh(serial, "shell", "input", "keyevent", "KEYCODE_MENU")
    rds.sh(serial, "shell", "am", "start", "-n", f"{package}/.MainActivity")
    time.sleep(3.0)
    button = rds.MODE_BUTTON[mode]
    if not rds.press(serial, button, time.time() + 45):
        log("could not reach the Start button; leaving it alone")
        return False
    rds.clear_dialogs(serial, time.time() + 30)
    deadline = time.time() + 40
    while time.time() < deadline:
        if any("-> RUNNING" in t for t in rds.transitions(serial)):
            log("app reached RUNNING")
            return True
        time.sleep(2.0)
    log("pressed Start but the app did not reach RUNNING")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default="a1411577")
    ap.add_argument("--package", default="com.dsrc.phone")
    ap.add_argument("--mode", choices=sorted(rds.MODE_BUTTON), default="live")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--max-minutes", type=float, default=180.0)
    ap.add_argument("--max-restarts", type=int, default=5)
    ap.add_argument("--cooldown-s", type=float, default=120.0,
                    help="minimum gap between restarts; a handset too hot to hold a "
                         "session dies again at once, and restarting in a tight loop "
                         "adds the load that is causing it")
    args = ap.parse_args()

    def log(message: str) -> None:
        print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)

    # Computed before the loop and checked inside it, so a killed parent cannot
    # leave this running past its welcome.
    deadline = time.monotonic() + args.max_minutes * 60.0
    restarts = 0
    last_restart = 0.0
    previous: str | None = None

    log(f"watching {args.package} on {args.serial}, mode {args.mode}, "
        f"up to {args.max_restarts} restarts, {args.max_minutes:.0f} minute cap")

    while time.monotonic() < deadline:
        state, detail = app_state(args.serial, args.package)
        line = f"{state}: {detail}"
        if line != previous:
            log(line)
            previous = line

        if state == DEAD:
            if restarts >= args.max_restarts:
                log(f"app is dead but {restarts} restarts already used; not trying "
                    f"again -- something is wrong that restarting does not fix")
            elif time.monotonic() - last_restart < args.cooldown_s:
                waited = time.monotonic() - last_restart
                log(f"app is dead, waiting out the cooldown "
                    f"({waited:.0f}s of {args.cooldown_s:.0f}s)")
            else:
                restarts += 1
                last_restart = time.monotonic()
                restart_app(args.serial, args.package, args.mode, log)
                previous = None      # force the next state to be reported

        time.sleep(args.interval)

    log(f"reached the {args.max_minutes:.0f} minute cap; {restarts} restart(s) made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
