#!/usr/bin/env python3
"""Run a command holding an exclusive lock on the one attached Android device.

An instrumented Gradle run reinstalls com.dsrc.phone, and installing a package
force-stops any running process of it. Two runs against one emulator therefore
kill each other: the loser reports "Test run failed to complete. Instrumentation
run failed due to Process crashed", or silently finishes a partial subset -- and
the test that happens to be executing when the install lands is reported as the
failure, whether or not anything is wrong with it. That is a diagnosis pointing
at innocent code, which is worse than a plain crash.

    Killing 22150:com.dsrc.phone (adj 0): stop com.dsrc.phone due to installPackageLI
    Process 22150 exited due to signal 9 (Killed)
    Crash of app com.dsrc.phone running instrumentation

Wrap every device-touching command in this and they queue instead.

    python3 scripts/with_device.py -- ./phone/gradlew -p phone :app:connectedDebugAndroidTest

The wait is bounded: a caller that blocks forever behind a hung run looks
identical to a hang of its own, and there is no timeout(1) on macOS to lean on.
"""

import fcntl
import os
import pathlib
import subprocess
import sys
import time

LOCK = pathlib.Path(os.environ.get("DSRC_DEVICE_LOCK", "/tmp/dsrc-android-device.lock"))
WAIT_SECONDS = float(os.environ.get("DSRC_DEVICE_LOCK_WAIT", "2700"))


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("usage: with_device.py -- <command> [args...]", file=sys.stderr)
        return 2

    handle = LOCK.open("a+")
    deadline = time.monotonic() + WAIT_SECONDS
    announced = False
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                # Refusing beats running anyway. Running anyway is the failure
                # mode this wrapper exists to remove.
                print(
                    f"device busy for {WAIT_SECONDS:.0f}s; not running (holder: "
                    f"{_holder(handle)})",
                    file=sys.stderr,
                )
                return 75  # EX_TEMPFAIL
            if not announced:
                print(f"waiting for the device; held by {_holder(handle)}", file=sys.stderr)
                announced = True
            time.sleep(2)

    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} argv={' '.join(argv)}\n")
    handle.flush()
    try:
        return subprocess.call(argv)
    finally:
        # Released by the close the interpreter does anyway, but a SIGKILL of
        # this process also releases it -- the lock lives on the descriptor, so
        # it cannot be left held by a run that is no longer there.
        handle.seek(0)
        handle.truncate()
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _holder(handle) -> str:
    try:
        handle.seek(0)
        return handle.read().strip() or "unknown"
    except OSError:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
