# Tasks 42, 43 and 47 — measured results of the USB campaign

## The short version

Three USB drives of 180 s each ran on 2026-09-05 at commit
`660e8d628cefc0dbc2a1302b121b1e49d72df37c`, with `source_tree_check: matched` on all
three, so the numbers below are attributable to that tree.

**The 200 ms figure on `e2e_ms` p95 is met in every run and pooled.** Pooled over all
2,684 ticks the p95 is **116.19 ms**, which is 83.81 ms below the 200 ms target and
99.44 ms below the tailnet baseline's 215.63 ms. The three runs individually give
90.72, 139.52 and 111.52 ms. The largest of the three, 139.52 ms, is still 60.48 ms
below the target.

**The three runs disagree by 48.80 ms at p95 and the disagreement has one named
source.** `e2e_ms` p95 differs between run 1 and run 2 by 48.80 ms; `enqueue_to_wire`
p95, a phone-side queueing segment measured on the phone's clock alone, differs
between the same two runs by 46.40 ms. Every other stage's p95 differs between runs
by less than 2 ms. The spread is phone-side queueing, not the link.

**The USB wire segment is not separable from its own conversion bound.** `transport`
p50 is 1.97, 2.16 and 2.03 ms across the three runs, against conversion bounds of
1.95, 1.86 and 2.06 ms for the same three runs. In run 3 the measured value is
0.03 ms *below* its own bound. What the campaign establishes about the wire hop is
that it is of the order of the bound, not a specific duration.

**`link_ms` is 18 times its bound, so that percentile is resolved.** Pooled
`round_trip` `link_ms` p50 is 36.83 ms against a bound p50 of 1.97 ms. `link_ms` spans
the whole phone-side path (capture to encode start, encode, encode to enqueue,
enqueue to wire) plus the wire hop, and over USB it is dominated by the phone-side
part: of the pooled 36.83 ms p50, the wire hop contributes about 2 ms.

**Tasks 43 and 47 pass on every tick of all three runs**, with zero mismatches in
every check.

## What ran

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| run directory | `run_20260905_113757` | `run_20260905_114546` | `run_20260905_115156` |
| ticks | 899 | 900 | 885 |
| duration | 179.6 s | 180.4 s | 180.2 s |
| delivered camera rate | 5.006 Hz | 4.989 Hz | 4.911 Hz |
| median tick rate | 4.95 Hz | 4.95 Hz | 4.95 Hz |
| session peer | `127.0.0.1:42843` | `127.0.0.1:56301` | `127.0.0.1:51873` |
| `over_tailnet` (start, end) | false, false | false, false | false, false |
| HERE calls placed | 0 | 0 | 0 |

Command: `run_demo.py --phone --usb --usb-serial ZY227VV4XC --headless --duration-s 180
--phone-wait-s 300 --print-every 10`, with the phone driven by
`scripts/run_device_session.py --serial ZY227VV4XC --seconds 195`. Both devices on a
desk, cabled on `usb:1-2.2`. The phone reported battery level 100% before run 1 and
after every run. Its reported temperature rose from 25.4 °C before run 1 to 26.5 °C
after run 1, 27.5 °C after run 2 and 27.8 °C after run 3.

The plan's D11 asks for 180 s "matching task 32's run". Task 32's tailnet baseline
(`run_20260902_183446`) in fact ran 300.1 s and produced 1,229 ticks. The median tick
rate is 4.95 Hz on the baseline and on all three USB runs, so the comparison is
like-for-like on rate and not on duration.

## Build provenance

Identical on all three runs:

```
commit                       660e8d628cefc0dbc2a1302b121b1e49d72df37c
dirty                        false
source_tree_check            matched
apk_last_update_time_check   matched
apk_sha256                   e179e95282ced6a99a71b03547350079254ceda3caabbbd2778dc7c15f327602
detector_engine_sha256       302c641fcc0715f5a078e02ad9fb80ef0636863b69d2a046343202dc5da8c404
policy bundle                random_init(seed=0), trained: false, contract 918ec57cf2f2e1db
```

The policy bundle is untrained, so the advisory values in these runs are placeholders.
Latency is unaffected: `infer` p50 is 0.55-0.56 ms in every run.

## Task 42 — `e2e_ms` against the 200 ms target

Both populations, per run and pooled. "Converted only" is the subset of ticks whose
`link_ms` resolved, which is what `timebase.converted` marks.

| | n | p50 (ms) | p95 (ms) | mean (ms) | max (ms) |
|---|---|---|---|---|---|
| run 1, all ticks | 899 | 65.47 | **90.72** | 66.64 | 145.09 |
| run 1, converted only | 898 | 65.51 | **90.73** | 66.66 | 145.09 |
| run 2, all ticks | 900 | 71.41 | **139.52** | 80.40 | 229.73 |
| run 2, converted only | 897 | 71.42 | **139.45** | 80.35 | 229.73 |
| run 3, all ticks | 885 | 67.59 | **111.52** | 72.56 | 199.90 |
| run 3, converted only | 881 | 67.59 | **111.40** | 72.56 | 199.90 |
| pooled, all ticks | 2,684 | 67.90 | **116.19** | 73.21 | 229.73 |
| pooled, converted only | 2,676 | 67.91 | **114.99** | 73.19 | 229.73 |
| tailnet baseline, all ticks | 1,229 | 96.15 | 215.63 | 115.85 | 649.85 |
| tailnet baseline, converted only | 1,204 | 96.80 | 218.70 | 117.57 | 649.85 |

**The target is met.** Every run's p95 is below 200 ms on both populations, by margins
of 109.28, 60.48 and 88.48 ms. Pooled, the p95 is 83.81 ms below the target on all
ticks and 85.01 ms below it on converted-only ticks.

**Against the baseline.** Pooled USB `e2e_ms` p95 is 99.44 ms below the tailnet
baseline's 215.63 ms on all ticks, and 103.71 ms below its 218.70 ms on converted-only
ticks. The 15.63 ms gap the tailnet run left against the target is closed with 83.81 ms
to spare.

The two populations differ by at most 0.12 ms within any single run, and by 1.20 ms
pooled, because only 1 to 4 ticks per run failed to convert. The distinction that
mattered on the baseline (25 unconverted ticks out of 1,229) is close to immaterial
over USB.

## Task 42 — `link_ms`, partitioned by `timebase.source`, with bounds

Never pooled across sources. `bound_ms` is the per-tick conversion bound the same
timebase estimate carries, quoted beside the value it bounds.

| population | source | n | link p50 | link p95 | bound p50 | bound p95 |
|---|---|---|---|---|---|---|
| run 1 | `round_trip` | 897 | 34.51 | 59.74 | 1.95 | 2.70 |
| run 1 | `one_way` | 1 | 37.58 | 37.58 | 7.84 | 7.84 |
| run 2 | `round_trip` | 896 | 40.55 | 107.99 | 1.86 | 2.63 |
| run 2 | `one_way` | 1 | 31.93 | 31.93 | 13.76 | 13.76 |
| run 3 | `round_trip` | 880 | 36.75 | 79.98 | 2.06 | 2.71 |
| run 3 | `one_way` | 1 | 36.08 | 36.08 | 63.70 | 63.70 |
| pooled | `round_trip` | 2,673 | 36.83 | 85.21 | 1.97 | 2.68 |
| pooled | `one_way` | 3 | 36.08 | 37.43 | 13.76 | 58.70 |
| tailnet baseline | `round_trip` | 1,198 | 65.68 | 185.78 | 8.69 | 9.63 |
| tailnet baseline | `one_way` | 6 | 62.01 | 101.89 | 52.64 | 52.66 |

All values in milliseconds. The baseline row is recomputed here from the baseline run's
own `metadata.jsonl` by the same partition, because the figure the plan quotes
(185.38 ms, n=1,204) was produced by a code path that pooled the two sources — the
plan's open item 4. Partitioned, the baseline's `round_trip` p95 is 185.78 ms over
n=1,198.

**The delta the plan states it is testing.** Pooled USB `round_trip` p95 is 85.21 ms
against the baseline's 185.78 ms, a reduction of 100.57 ms. Each run individually is
below the baseline: 59.74, 107.99 and 79.98 ms.

**What the `one_way` rows can and cannot support.** One tick per run took the one-way
estimator. In run 3 that tick's `link_ms` is 36.08 ms against a bound of 63.70 ms —
the bound exceeds the value, so that sample resolves nothing. With n=1 per run these
rows are reported for completeness, not as a measurement.

Ticks with no `link_ms` at all: 1 in run 1, 3 in run 2, 4 in run 3, in every case
because the offset window held 2 to 4 samples rather than enough to convert.

## Task 42 — the converted stages beside their bounds

`transport` and `return` are the only two stages that cross the two clocks. Values are
p50/p95 in milliseconds.

| stage | run 1 value | run 1 bound | run 2 value | run 2 bound | run 3 value | run 3 bound |
|---|---|---|---|---|---|---|
| `transport` | 1.97 / 3.67 | 1.95 / 2.70 | 2.16 / 3.75 | 1.86 / 2.63 | 2.03 / 3.58 | 2.06 / 2.72 |
| `return` | 3.14 / 9.14 | 1.90 / 2.63 | 2.90 / 9.73 | 1.81 / 2.57 | 3.06 / 8.31 | 2.00 / 2.66 |

On the tailnet baseline the same two stages read `transport` 26.33 / 127.29 against a
bound of 8.70 / 9.65, and `return` 9.43 / 14.90 against a bound of 8.64 / 9.64.

**`transport` over USB is at the resolution limit of the instrument.** Its p50 exceeds
its own bound p50 by 0.02 ms in run 1 and 0.30 ms in run 2, and falls 0.03 ms below it
in run 3. The correct statement is that the USB wire hop is of the order of 2 ms and
the measurement cannot place it more precisely than its bound allows. On the tailnet
baseline the same stage's p50 exceeded its bound by 17.64 ms, which is why the
baseline's `transport` figure is a measurement and the USB one is not.

`return` p50 exceeds its bound p50 by 1.24, 1.09 and 1.06 ms in the three runs. That
segment is resolved, but by roughly one bound-width.

## Task 42 — the full stage table

p50 / p95 in milliseconds. `capture` is an instant and carries no duration.

| stage | clock basis | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| `capture_to_encode_start` | measured, phone | 4.18 / 6.87 | 4.14 / 7.58 | 4.16 / 6.82 |
| `encode` | measured, phone | 7.82 / 11.70 | 7.80 / 13.40 | 7.79 / 11.43 |
| `encode_done_to_enqueue` | measured, phone | 10.54 / 19.69 | 10.61 / 19.57 | 10.10 / 19.61 |
| `enqueue_to_wire` | measured, phone | 7.76 / **30.94** | 14.57 / **77.34** | 10.79 / **52.58** |
| `transport` | converted | 1.97 / 3.67 | 2.16 / 3.75 | 2.03 / 3.58 |
| `jpeg_decode` | measured, jetson | 10.90 / 11.29 | 10.88 / 11.35 | 10.89 / 11.33 |
| `detect` | measured, jetson | 17.81 / 18.92 | 17.77 / 18.99 | 17.78 / 18.72 |
| `track` | measured, jetson | 0.05 / 0.06 | 0.05 / 0.05 | 0.05 / 0.05 |
| `fuse` | measured, jetson | 0.05 / 0.06 | 0.05 / 0.06 | 0.05 / 0.06 |
| `infer` | measured, jetson | 0.56 / 0.65 | 0.55 / 0.63 | 0.56 / 0.64 |
| `decode` | measured, jetson | 0.05 / 0.05 | 0.05 / 0.05 | 0.05 / 0.05 |
| `return` | converted | 3.14 / 9.14 | 2.90 / 9.73 | 3.06 / 8.31 |
| `render` | measured, phone | 105.34 / 197.68 | 103.29 / 209.49 | 104.06 / 202.24 |

**Where the run-to-run disagreement lives.** `e2e_ms` p95 rises from run 1 to run 2 by
48.80 ms. `enqueue_to_wire` p95 rises between the same two runs by 46.40 ms, which is
95.1% of it. From run 1 to run 3, `e2e_ms` p95 rises by 20.80 ms and
`enqueue_to_wire` p95 rises by 21.64 ms. Of the stages inside `e2e_ms`, no other one's
p95 moves by more than 1.97 ms between any pair of runs, and that largest mover is
`encode`. `enqueue_to_wire` is the interval between a frame
entering the phone's send queue and its bytes reaching the wire, measured on the
phone's clock at both ends, so it is neither a link effect nor a conversion artefact.

`render` is phone receipt to the first `AdvisoryHolder.current()` that returned the
advisory, and is not part of `e2e_ms`. It is measured on 715, 708 and 700 of the
ticks; the remainder have no `advisory_shown` line for that capture stamp.

## Task 42 — `jetson_ms` and the gate that exists in code

| | n | p50 | p95 | source |
|---|---|---|---|---|
| run 1 | 899 | 31.28 | 32.31 | measured |
| run 2 | 900 | 31.24 | 32.49 | measured |
| run 3 | 885 | 31.20 | 32.28 | measured |
| tailnet baseline | 1,229 | 31.31 | 32.98 | measured |

`GATE_JETSON_P95_MS` is `jetson_ms.p95 < 200.0`. It passes in all three runs, by 167.69,
167.51 and 167.72 ms. Jetson compute did not change between the tailnet path and USB:
the three p95 values are within 0.70 ms of the baseline's.

## Task 42 — the `usb` record

Identical in all three runs:

```
serial                       ZY227VV4XC
transport_id                 1
reverse_spec                 tcp:47811 tcp:47811
adb_version                  Android Debug Bridge version 1.0.41
reverses_reestablished       0
reverse_reestablish_failures 0
reverses_swept               0
address                      127.0.0.1:47811
```

`adb reverse --list` was empty before each run, showed `UsbFfs tcp:47811 tcp:47811`
after the acceptor bound, and was empty again after each run. `adb devices -l` reported
`ZY227VV4XC device usb:1-2.2` before and after every run. No mapping leaked between
runs.

## The cadence anomaly (risk R8) did not appear

The anomaly the plan guards against offers a uniform fraction of every commanded rate.
The delivered camera rate was 5.006, 4.989 and 4.911 Hz against a commanded
`camera_hz` whose time-weighted mean was 4.988, 4.988 and 4.768 Hz. No run delivered a
fraction of its commanded rates. The 1-second failure-scan cadence held to a p50
interval of 1.0001 s in all three runs.

Achieved-versus-commanded per channel could not be scored, for the reason the plan
anticipates: all three drives ran in shadow mode, and `eval_run._comparable` refuses the
comparison when the mode is shadow on every decision. The delivered camera rate above
is computed from tick count over duration, not from that table.

## Task 43 — results

`check_shadow_commands.py <run_dir> --serial ZY227VV4XC`, run on the Jetson immediately
after each drive so the `ConfigApplier` teardown line was still in `logcat`. Output in
each run directory as `shadow_command_check.json`.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| drive mode | shadow | shadow | shadow |
| ticks checked | 899 | 900 | 885 |
| command replay mismatches | 0 | 0 | 0 |
| logged `sensing.shadow` flag | ok | ok | ok |
| `commands_sent` (Jetson counter) | 37 | 38 | 38 |
| `applied` (phone counter) | 0 | 0 | 0 |
| `shadowed` (phone counter) | 37 | 38 | 38 |
| `overall_ok` | true | true | true |

Step 43.6's requirement — `applied == 0` and `shadowed == commands_sent` — holds in all
three runs. The wall-clock offset between the two devices at teardown was 0.916, 0.921
and 0.933 s.

## Task 47 — results

`scripts/observation_parity.py --run-dir <run_dir>`, run on the laptop against the
pulled run directories, with the resulting `observation_run_check.json` copied back to
the Jetson run directory.

The ledger classified all 39 slots across 7 scenes with no class mismatches and no
provenance mismatches. Against the live runs:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| ticks checked | 899 | 900 | 885 |
| 47.8 re-encode mismatches | 0 | 0 | 0 |
| 47.9 constant mismatches | none | none | none |
| `ok` | true | true | true |

Step 47.8 holds on every tick of all three runs: the logged `encoded` vector equals a
re-encode of that tick's logged `obs`. Step 47.9 holds on every tick: each
`substituted` and `structurally_absent` slot carries the constant the ledger names.

## Log integrity

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| ticks read / ticks reported | 899 / 899 | 900 / 900 | 885 / 885 |
| tick ids absent from log | 0 | 0 | 0 |
| unparseable lines | 0 | 0 | 0 |
| `log_complete` | true | true | true |
| dropped records | 0 | 0 | 0 |
| writer failure | none | none | none |
| advisories the phone logged / matched | 899 / 899 | 900 / 900 | 885 / 885 |
| Jetson ticks with no returned advisory | 0 | 0 | 0 |

## `overall_pass` is false in all three runs, for reasons the USB path did not cause

| gate | run 1 | run 2 | run 3 | tailnet baseline |
|---|---|---|---|---|
| `latency_jetson_p95` (< 200 ms) | pass, 32.3 | pass, 32.5 | pass, 32.3 | pass, 33.0 |
| `throughput_median` (>= 25 Hz) | fail, 5.0 | fail, 5.0 | fail, 5.0 | fail, 5.0 |
| `gps_fresh` (>= 0.95) | not evaluable, 0.0 | not evaluable, 0.0 | not evaluable, 0.0 | fail, 0.933 |
| `gps_speed_rmse` (< 1.0 m/s) | not evaluable | not evaluable | not evaluable | not evaluable |
| `perception_coverage` (>= 0.5) | fail, 0.0 | fail, 0.0 | fail, 0.0 | fail, 0.0 |
| `tick_coverage_absent_from_log` | pass, 0.0 | pass, 0.0 | pass, 0.0 | absent from that report |
| `tick_coverage_never_produced` | pass, 0.0 | pass, 0.0 | pass, 0.0 | absent from that report |

`throughput_median` fails because the drive runs at the commanded camera rate of about
5 Hz while the gate asks for 25 Hz; it fails identically on the tailnet baseline.
`perception_coverage` is 0.0 because the phone's camera faced a desk and no vehicle was
in frame. `gps_fresh` is 0.0 with `pass: null` because the handset held no fix indoors,
where the baseline, taken with some sky view, reached 0.933. None of these three is a
statement about the transport path.

## Caveats

The `transport` stage over USB cannot be quoted as a duration: its p50 and its
conversion bound are within 0.30 ms of each other in every run, and in run 3 the value
is below the bound.

The tailnet baseline's `link_ms` figure of 185.38 ms in the plan pools `round_trip` and
`one_way`. The 185.78 ms used here is the `round_trip` partition recomputed from that
run's `metadata.jsonl`; the two differ by 0.40 ms.

The policy bundle is untrained in all three runs, so nothing here says anything about
advisory quality.

The failure sources that fired differ between runs, in the same direction as
`enqueue_to_wire`:

| source | run 1 | run 2 | run 3 |
|---|---|---|---|
| `gps.not_fresh` | 181 | 181 | 181 |
| `clock.proxied` | 2 | 6 | 8 |
| `phone.dropped` | 0 | 3 | 17 |
| `wire.seq_gaps` | 0 | 5 | 5 |
| `camera.blind_ticks` | 0 | 0 | 2 |

`gps.not_fresh` fired on all 181 scan passes of every run because the handset held no
fix indoors. `phone.dropped` counts frames the phone discarded rather than sent, and it
is zero in the run with the lowest `enqueue_to_wire` p95 and non-zero in the other two.
`camera_dropped_frames` in `summary.json` is 0 for all three runs.
