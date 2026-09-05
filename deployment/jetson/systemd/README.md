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

Edit `DSRC_RUN_ARGS` in `~/.config/dsrc-drive.env` and restart the service. Shadow is
the default; adding `--live-rates` makes the phone apply the commanded rates, which
is task 51 and not task 50. Nothing else in the line should need changing.

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
