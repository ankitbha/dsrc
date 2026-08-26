# Task 30 — Shadow / live mode

## The short version

Half of this exists. `ConfigApplier` on the phone already honours the flag, and
correctly: a shadow command "changes nothing at all — not the rates, not the query,
and not `current`, which is what the phone is actually running", and it counts what
it shadowed. What is missing is the Jetson deciding which mode it is in, marking the
command, recording the decision, and letting the mode be flipped mid-drive.

**The property that makes shadow mode worth anything: the mode must never reach the
decision.** Task 43 checks that "logged shadow decisions match what live gating
produces on the same input". If `SensingController.decide` could see the mode, that
check would be comparing a function against itself and would pass no matter what.
So the mode is applied strictly after the decision, on its way out — it selects the
`shadow` field of the `RateCommand` and nothing else.

## Scope boundary

In: the mode holder, runtime flipping, the flag on the outgoing command, and the
decision log that task 35 scores from.

Out: sending (task 31); the scoring itself (35); the colocated correctness check
(43). Also out: any change to `SensingController.decide` — see above, its being
mode-blind is the point.

## What shadow mode actually gives you, and what it does not

This is the part worth getting on paper before anyone reads a shadow log as a
prediction.

In shadow the phone keeps running whatever it was last set to — the reference
rates — because nothing is applied. So a shadow drive samples at full rate
throughout, which is exactly what task 35 wants: "the shadow-mode decision log
emitted alongside the full-rate reference, so every candidate policy can be scored
against identical traffic from one drive."

But the controller's inputs are then **full-rate inputs**. In live, a decision to
drop the camera to 1 Hz changes what the next tick sees: fewer frames, a different
`local_density_bin`, possibly a different policy margin. So:

- Shadow predicts the **decision function** exactly: same inputs in, same decision
  out, which is what task 43 can check tick by tick.
- Shadow does **not** predict the **trajectory**. A live drive feeds its own reduced
  observations back into the next decision, and a shadow drive never does. The two
  diverge after the first change, and no amount of shadow logging closes that.

Recommended: say so in the record rather than leaving it to be discovered when a
shadow-scored policy behaves differently live. `shadow` on the wire means "recorded,
not applied"; it does not mean "what live would have done for the rest of the drive".

## Decisions taken

**Mode-blind decision, mode-aware dispatch.** `decide()` keeps its signature. A
separate step takes a `Decision` and a mode and produces the `RateCommand`. Nothing
else may read the mode.

**Every flip is recorded with when.** A drive that changed mode and cannot say at
which tick is unattributable afterwards, and this project has already paid for that
once. The record carries the flips, not just the final state.

**Flipping is safe from another thread.** The flag is read on the tick loop and
written by whatever flips it — a debug endpoint, a signal, a key. One atomic read
per tick, so a flip cannot split a single decision between two modes.

**Live is not the default.** A process that comes up gating for real because nobody
said otherwise is the wrong failure. Shadow is the default and live is chosen.

## The work

1. A `ModeHolder`: current mode, thread-safe flip, a log of flips with timestamps.
2. `command_for(decision, mode, now)` building the `RateCommand` with `shadow` set,
   and nothing else conditioned on the mode.
3. A decision record carrying the mode, the flips, and the caveat above, so a shadow
   log is readable a month later.

## Tests

- The same `Inputs` produce identical rates, trigger and query in both modes; only
  `shadow` differs. This is the property task 43 rests on.
- A flip mid-drive changes the flag on the next command and not the decision.
- Flipping from another thread never yields a command whose rates and flag came from
  different modes.
- The default is shadow.
- The record names every flip and when, not just where it ended.
- A shadow command is a legal `rate_cmd` — round-tripped through the real codec, as
  task 29's queries are.

## Needs sign-off

- Shadow as the default, so a process must be told to gate for real.
- That the record states plainly that shadow predicts the decision function and not
  the trajectory, since that limit bounds what task 35's scoring can claim.
