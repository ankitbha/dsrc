# Paused — task 20 (IMU), mid round 3

HEAD `2f8d44a`, tree clean, every suite green:
252 transport + app JVM + 54 instrumented + 952 Python, 0 failed.

## The blocking unknown, first

**`scripts/remutate.py` reported `*** SURVIVED ***` for the entry
`stats: an outbound framing refusal counts before it describes`, and that entry is caught
5 runs out of 5 when I apply the same mutation and run the test directly.**

So one of two things is true and I do not know which:

1. The harness's verdict is wrong for this entry — it runs the whole `:transport:test` and
   keys on the process exit code, so something in the wider run is masking the failure
   (another test's output, a task that short-circuits, an exit code that is not what I
   think it is).
2. The pin is genuinely intermittent under the fuller run, and five direct runs were luck.

This matters more than the finding it sits on. The harness exists *because* pins have
lapsed silently three times on this project, and its own verdict has now disagreed with a
direct measurement. Until that is resolved, **every `CAUGHT` it reports is worth less than
it looks** — including the ones I have been quoting as evidence in commit messages.

The way to settle it: make the harness report *which test failed* rather than only the
process exit code, the way `remutate_device.py` already does (it parses the JUnit XML). If
the entry then names the right test, the harness was miscounting; if it names nothing, the
pin is intermittent and the probe needs a deterministic trigger instead of a sampling loop.

## What is done

Task 19 is struck and pushed. Task 20 is implemented and has been through three validation
rounds; rounds 1 and 2 are fully closed, round 3 is closed except the items below.

Round 3 returned twenty findings. Closed: F1–F6, F7–F10, F13, F14, F15–F19, F20.

## What is open on task 20

Recorded in the plan under "Open, and deliberately not closed here":

- **The timebase verdict is taken from one event and is irreversible.** One late first
  delivery over two seconds stops IMU capture for the whole drive, with no retry. Nothing
  has measured first-event delivery latency on hardware, so the threshold is a guess about
  a distribution nobody has looked at.
- **Nothing pins that `stop()` unregisters the two sensor listeners.** `quitSafely()` ends
  the thread either way, so the thread census reports success on a leak. The `dumpsys
  sensorservice` test written for this was removed after three failed repairs — the note in
  `ImuCaptureTest` says why. The dump's live-connection section is the likelier instrument
  than its event ring.
- **The teardown log line has no test.** It is the sole reader of five diagnostic fields
  and deleting every interpolation leaves both suites green.

Still to do: round 4, then strike 20 in `plans/task_list.md`, push, and start task 21.

## The thing worth carrying forward

Twelve of the tests I wrote for task 20 could not fail, and every one was found by
mutation rather than by reading them. The recurring shapes, in the order they bit:

- inputs that make two implementations agree by accident (ages that only rise, so "keep the
  max" and "keep the last" are the same function);
- a boundary stepped over rather than landed on;
- a test that derives its inputs from the constant it is meant to pin, so widening that
  constant moves the test with it;
- a premise never active — the test named for the diverged branch drove both clocks equal,
  so the moot branch decided;
- an identity pinned in one direction, blind to the sum exceeding the whole.

And three rounds each found the same structural hole one layer further out: the decisions
were unreachable, then the caller was, then the instrument built to reach the caller
counted registrations without identifying them. The assertion that finally closed it was
the simplest available one — that a sample comes *out*.

One "fix" was reverted in round 3 because it was not one: taking the magnitude of the clock
gap looked like protection against a transposed clock pair and supplied none, since for a
negative gap the attribution branch reduces algebraically to the branch above it. That is
asserted now instead of argued.

## To resume

`python3 /Users/ankit_nash/Desktop/ankit_summer_2026/.pipeline/watch.py` prints the position;
`.pipeline/state.json` carries the same detail in `next_action`. Start with the harness
question at the top of this file — not with round 4.
