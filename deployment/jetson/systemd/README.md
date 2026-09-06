# The in-car drive service

What this exists for: in the car there is no shell. Until this unit existed, the only
way to start `run_demo.py` was an `ssh` session, which needs the Jetson to have
internet, which needs a laptop and a hotspot in the car. The phone half was already a
button — `MainActivity` has "Start sensing" — so the Jetson was the whole gap.

With the unit installed and enabled the sequence is:

1. 12 V on. The Jetson boots and the service starts, blocking in `adb wait-for-device`.
2. Plug the phone in, if it is not already.
3. Tap **Start sensing**.

That is the whole procedure. Nothing else has to happen, and nothing needs a network.

## Why the order does not matter

`SessionHolder` on the phone retries a refused dial indefinitely — its own test is
named *a refused dial is retried until one succeeds*. So tapping Start before the
Jetson is listening is fine; the phone keeps dialling and connects when the runtime
comes up. The eight `link.dial_failed` events at the start of every bench run to date
are this behaviour working.

## Install

A **user** unit, so none of this needs root. Installing to `/etc` requires a password
an unattended deploy does not have; lingering does not.

```sh
loginctl enable-linger "$USER"          # so it starts at boot with nobody logged in
mkdir -p ~/.config/systemd/user
cp deployment/jetson/systemd/dsrc-drive.env     ~/.config/dsrc-drive.env
cp deployment/jetson/systemd/dsrc-drive.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable dsrc-drive      # start at boot
systemctl --user start dsrc-drive       # or right now
```

`~/.config/dsrc-drive.env` is a copy, not a symlink into the checkout: a deploy rsync
must not be able to change how the next drive runs.

## Choosing shadow or live

**From the app.** The activity has two Start buttons, `Start (shadow)` and
`Start (live)`, and the phone carries the choice to the Jetson in its handshake. One
service configuration therefore serves both kinds of drive, and switching between
them is stopping the session and starting another — no shell, no restart, no edit.

It is a request, not a setting: the Jetson decides, applies it before the first tick,
and records both the mode and who asked in `summary.json` under
`sensing.mode.mode_origin`, which reads `phone_request` or `command_line`. The rule
that the Jetson owns every sensing decision, and that every decision reaches its log,
is unchanged.

`DSRC_RUN_ARGS` in `~/.config/dsrc-drive.env` still sets the **default**, used when
the handset expresses no preference — a build older than the two buttons, for
instance. Adding `--live-rates` there changes only that default.

**Mid-session switching is deliberately not possible.** A drive that changes mode
part-way is scored wrongly today: `eval_run` treats `ever_live` as a property of a
whole drive, so the shadow segment of a promoted drive loses its caveat and its
unapplied commands read as a rate shortfall. Choosing at the start keeps
`flip_count` at 0 and each drive one kind throughout.

## Checking it without a screen

```sh
systemctl --user status dsrc-drive
tail -f ~/dsrc_logs/dsrc-drive.log
```

The runtime's output goes to that file and **not** to the journal:
`journalctl --user -u dsrc-drive` reports "No journal files were found" on this box,
so the usual check shows nothing. The unit also sets `PYTHONUNBUFFERED=1`, without
which Python block-buffers when its output is not a terminal and the file stays
empty for minutes.

`activating (start-pre)` means it is waiting for a handset — that is the normal state
between powering the Jetson and plugging the phone in, not a fault.

The phone's own display is the other check, and the one available in the driver's
seat: it shows the advisory, which only updates when frames are reaching the Jetson
and decisions are coming back.

## The two settings that matter most

`--rebind-timeout-s 0` waits indefinitely for a phone that went away. The default of
120 s **ends the run**, and in a car nothing is there to start it again — an incoming
call, an app restart or a thermal shutdown lasting longer than two minutes would
otherwise cost the drive silently.

`--duration-s 0` means no wall-clock limit, so a drive ends when you stop it rather
than when a timer someone guessed at expires.

## What this does not solve

The service restarts the runtime, not the phone. If the app dies — which
`pm revoke` demonstrated it will, in task 46's injection — the phone must be
restarted by tapping Start again. There is no way for the Jetson to start the app;
the USB link is adb with the Jetson as host, and there is no channel the other way.

Logging is roughly 150 MB per hour into `~/dsrc_logs`. At 647 GB free that is not a
constraint for any single drive, but nothing prunes it.
