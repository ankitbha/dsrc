"""The loopback harness's gate, tested.

The harness carries a pass/fail exit code and a reconciliation, and it had no
tests at all. That is the structural reason its gate could be wrong twice: the
first version passed on a conversion off by 67 hours, and its replacement passed
on one off by 900 ms. Both were found by someone mutating the library and reading
the report, not by anything that runs.

`_report` is driven directly here rather than through a live session, so a gate
clause can be exercised without waiting for a run and without depending on the
machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_loopback_pipeline import (  # noqa: E402
    CONVERGENCE_BUDGET_S,
    LINK_BOUND_MULTIPLE,
    LINK_FLOOR_MS,
    MIN_CONVERTED_FRACTION_AFTER_CONVERGENCE,
    _account,
    _report,
)
from transport.channels import Channel  # noqa: E402


class _Row:
    """A per-channel stats row shaped like the transport's."""

    def __init__(self, n: int, *, dropped_outbound: int = 0, dropped_inbound: int = 0,
                 delivered: int | None = None) -> None:
        self.queued = n
        self.sent = n - dropped_outbound
        self.dropped_outbound = dropped_outbound
        self.abandoned_outbound = 0
        self.received = n - dropped_outbound
        self.dropped_inbound = dropped_inbound
        self.delivered = (
            self.received - dropped_inbound if delivered is None else delivered
        )
        self.seq_gaps = 0
        self.missing_seqs = 0


class _Stats:
    def __init__(self, n: int, **kwargs) -> None:
        self._row = _Row(n, **kwargs)
        self.channels = _Channels(self._row)


class _Channels:
    def __init__(self, row) -> None:
        self._row = row

    def __getitem__(self, _channel):
        return self._row


class _Recorder:
    def to_record(self) -> dict:
        return {}


class _Pipeline:
    class stats:
        @staticmethod
        def snapshot() -> dict:
            return {}


def a_tick(index: int, link_ms: float, *, bound_ms: float | None = 0.4,
           converted: bool = True, fresh: bool = True) -> dict:
    return {
        "t_s": index * 0.1,
        "frame_id": index,
        "jetson_ms": 1.0,
        "link_ms": link_ms if converted else None,
        "e2e_ms": 1.0 + link_ms,
        "timebase": {
            "converted": converted,
            "proxy": not converted,
            "bound_ms": bound_ms if converted else None,
            "estimate_id": 1,
            "link_ms": link_ms if converted else None,
        },
        "gps_fresh": fresh,
        "gps_age_s": 0.001,
        "ego_speed_source": "measured_converted" if converted else "measured_arrival_proxy",
        "advisory": "x",
    }


CHANNELS = (Channel.CAMERA, Channel.GPS)


def account_for(count: int, *, pending=0, pending_in=0, decode_errors=0,
                **stats_kwargs) -> dict:
    stats = _Stats(count, **stats_kwargs)
    return _account(
        stats, stats,
        pending=dict.fromkeys(CHANNELS, pending),
        pending_in=dict.fromkeys(CHANNELS, pending_in),
        decode_errors=dict.fromkeys(CHANNELS, decode_errors),
    )


def report_for(ticks: list[dict], *, n_messages: int | None = None,
               first_converted_at: float | None = 0.0, **stats_kwargs) -> dict:
    count = len(ticks) if n_messages is None else n_messages
    return _report(
        ticks, 10.0, 243_264_000_000_000,
        {"frames": count, "fixes": count, "pongs": count, "invalid": 0},
        {"received": len(ticks), "unmatched": 0, "frame_ids": {t["frame_id"] for t in ticks}},
        _Recorder(), _Recorder(), _Recorder(), _Recorder(), _Pipeline(),
        first_converted_at,
        phone_stats=_Stats(count, **stats_kwargs),
        jetson_stats=_Stats(count, **stats_kwargs),
        account=account_for(count, **stats_kwargs),
    )


# -- the gate ----------------------------------------------------------------


def test_a_healthy_run_passes():
    report = report_for([a_tick(i, 0.3) for i in range(60)])
    assert report["usable"] is True, report["gate_detail"]
    assert report["gate_detail"]["converted_fraction_after_convergence"] == 1.0


@pytest.mark.parametrize("wrong_by_ms", [900.0, -900.0, 50.0, -50.0])
def test_a_conversion_wrong_by_more_than_its_bound_fails(wrong_by_ms):
    """The hole that survived the first fix. A flat 1000 ms ceiling was 2,600x
    the segment this link produces and 2,300x the bound the same run reports, so
    every realistic conversion bug lived inside it -- a stale estimate, one term
    with the wrong sign, a systematically asymmetric link.

    Charged against the run's own reported uncertainty instead.
    """
    report = report_for([a_tick(i, wrong_by_ms, bound_ms=0.4) for i in range(60)])
    assert report["usable"] is False, report["gate_detail"]


def test_a_negative_link_segment_fails_on_its_sign_not_its_size():
    """`abs(link_p95)` accepted a -899 ms flight time without objection, leaving
    the only real check in another module's constant. A negative segment is not
    merely large, it is impossible."""
    report = report_for([a_tick(i, -80.0, bound_ms=0.4) for i in range(60)])
    assert report["usable"] is False
    assert report["gate_detail"]["link_min_ms"] == pytest.approx(-80.0)


def test_a_minority_of_wrong_conversions_is_not_hidden_behind_a_percentile():
    """At n=91 a p95 leaves four values above it, so four grossly wrong
    conversions passed both clauses. The gate uses max, not p95."""
    ticks = [a_tick(i, 0.3) for i in range(87)] + [a_tick(i, 500.0) for i in range(87, 91)]
    report = report_for(ticks)
    assert report["usable"] is False
    assert report["gate_detail"]["link_max_ms"] == pytest.approx(500.0)


def test_a_run_that_fell_back_after_converging_fails():
    """`converted_and_fresh == len(converted)` was satisfied by 1 == 1, so a
    timebase that converged once and then went unusable for the rest of the drive
    passed. Plan section 9's stated risk is exactly the proxy being in use for
    longer than expected."""
    ticks = [a_tick(0, 0.3)] + [
        a_tick(i, 0.0, converted=False) for i in range(1, 91)
    ]
    report = report_for(ticks, first_converted_at=0.0)
    assert report["usable"] is False
    assert (
        report["gate_detail"]["converted_fraction_after_convergence"]
        < MIN_CONVERTED_FRACTION_AFTER_CONVERGENCE
    )


def test_a_healthy_short_run_is_not_failed_for_its_convergence_prefix():
    """The timebase needs ~1.1 s, so a whole-run fraction rejected a perfectly
    healthy two-second run -- and plan section 8.4, "the first ten seconds,
    recorded deliberately", is short by design. A gate that fails a healthy rig
    is a false claim about the run.

    Half the ticks proxied because they came BEFORE the first conversion; every
    tick after it converted.
    """
    ticks = [a_tick(i, 0.0, converted=False) for i in range(10)] + [
        a_tick(i, 0.3) for i in range(10, 21)
    ]
    for tick in ticks[:10]:
        tick["t_s"] = 0.05 * tick["frame_id"]
    for tick in ticks[10:]:
        tick["t_s"] = 1.2 + 0.05 * tick["frame_id"]
    report = report_for(ticks, first_converted_at=1.2)
    assert report["usable"] is True, report["gate_detail"]
    assert report["gate_detail"]["ticks_before_convergence"] == 10
    assert report["gate_detail"]["converted_fraction_after_convergence"] == 1.0


def test_a_converted_tick_that_is_not_fresh_fails():
    ticks = [a_tick(i, 0.3, fresh=(i > 0)) for i in range(60)]
    report = report_for(ticks)
    assert report["usable"] is False
    assert report["gate_detail"]["converted_and_fresh"] == 59


def test_a_run_with_no_ticks_fails():
    report = report_for([])
    assert report["usable"] is False


def test_the_ceiling_never_drops_below_its_floor():
    """A very tight bound must not make the gate hair-trigger on ordinary
    jitter."""
    report = report_for([a_tick(i, 1.0, bound_ms=0.001) for i in range(60)])
    assert report["gate_detail"]["link_ceiling_ms"] == LINK_FLOOR_MS
    assert report["usable"] is True

    # And a wide bound raises it proportionally rather than by a constant.
    wide = report_for([a_tick(i, 3.0, bound_ms=2.0) for i in range(60)])
    assert wide["gate_detail"]["link_ceiling_ms"] == pytest.approx(
        LINK_BOUND_MULTIPLE * 2.0
    )
    assert wide["usable"] is True


# -- the account -------------------------------------------------------------


def test_the_account_closes_and_still_names_what_was_lost():
    """`dropped_outbound` is a TERM of the identity, so the account closes by
    construction exactly when a depth-1 LATEST_WINS channel drops. Closing is
    not the same as losing nothing, and a reader given only a true flag would
    conclude it was."""
    report = report_for([a_tick(i, 0.3) for i in range(40)],
                        n_messages=100, dropped_outbound=60)
    account = report["transport"]
    assert account["reconciled"] is True, account["reconciliation"]
    assert account["lost_total"] == 120, "the loss is not stated"
    assert account["lost_by_channel"]["camera"] == 60


def test_a_record_that_arrived_and_was_never_delivered_is_a_gap():
    """The clause that catches a malformed message: received but not handed to a
    consumer. It was pulled into the record and never checked."""
    report = report_for([a_tick(i, 0.3) for i in range(40)],
                        n_messages=60, delivered=41)
    assert report["transport"]["reconciled"] is False
    assert any("delivered" in gap for gap in report["transport"]["reconciliation"])


def test_frames_still_in_flight_do_not_break_the_identity():
    """`queued` increments at enqueue and `sent` at write, and the stats are
    sampled while the writer may still be draining -- so without an in-flight
    term the flag went false for no defect.

    The term comes from the session's live queue depth. Derived as
    `queued - accounted` it made this identity a tautology: `accounted +
    (queued - accounted)` is `queued` for every input, so the clause could not
    fail while its comment claimed it verified something.
    """
    stats = _Stats(60)
    stats._row.sent = 59  # one still sitting in the outbound queue
    stats._row.received = 59
    stats._row.delivered = 59
    account = _account(
        stats, stats,
        pending=dict.fromkeys(CHANNELS, 1),
        pending_in=dict.fromkeys(CHANNELS, 0),
        decode_errors=dict.fromkeys(CHANNELS, 0),
    )
    assert account["reconciled"] is True, account["reconciliation"]
    assert account["upstream"]["camera"]["in_flight"] == 1


def test_the_outbound_identity_can_actually_fail():
    """The tautology test: with a real queue depth, a frame that vanished
    without being sent, dropped, abandoned or queued is a gap."""
    stats = _Stats(60)
    stats._row.sent = 55  # five unaccounted for, and nothing in the queue
    stats._row.received = 55
    stats._row.delivered = 55
    account = _account(
        stats, stats,
        pending=dict.fromkeys(CHANNELS, 0),
        pending_in=dict.fromkeys(CHANNELS, 0),
        decode_errors=dict.fromkeys(CHANNELS, 0),
    )
    assert account["reconciled"] is False
    assert any("queued 60" in gap for gap in account["reconciliation"])


def test_a_malformed_message_is_caught_by_the_router_not_by_delivered():
    """The session counts `delivered` when the frame leaves it, and the router
    rejects an undecodable one afterwards -- so a run where every message failed
    to decode reconciled clean on the inbound identity alone. The comment used to
    claim otherwise."""
    clean = account_for(60, decode_errors=0)
    assert clean["reconciled"] is True

    broken = account_for(60, decode_errors=60)
    assert broken["reconciled"] is False
    assert any("failed to decode" in gap for gap in broken["reconciliation"])


def test_records_still_in_the_inbound_queue_do_not_break_the_identity():
    """The receive-side twin: without it the inbound check is flaky rather than
    wrong, because `_report` runs after the readers are joined."""
    stats = _Stats(60)
    stats._row.delivered = 57
    account = _account(
        stats, stats,
        pending=dict.fromkeys(CHANNELS, 0),
        pending_in=dict.fromkeys(CHANNELS, 3),
        decode_errors=dict.fromkeys(CHANNELS, 0),
    )
    assert account["reconciled"] is True, account["reconciliation"]


def prefixed_run(prefix_ticks: int, converged_at: float, *, spacing: float = 0.1) -> dict:
    """A run that proxies for `prefix_ticks` and then converts for the rest."""
    ticks = []
    for index in range(prefix_ticks):
        tick = a_tick(index, 0.0, converted=False)
        tick["t_s"] = round(index * spacing, 3)
        ticks.append(tick)
    for index in range(prefix_ticks, prefix_ticks + 40):
        tick = a_tick(index, 0.3)
        tick["t_s"] = round(converged_at + (index - prefix_ticks) * spacing, 3)
        ticks.append(tick)
    return report_for(ticks, first_converted_at=converged_at)


def test_a_long_proxy_prefix_fails_even_if_everything_after_it_converted():
    """The hole my own short-run fix opened.

    Measuring the fraction only after the first conversion stopped a healthy 2 s
    run failing -- and made the prefix free at ANY length. Measured on real code
    with a 0.25 Hz sync cadence: convergence at 16.1 s, 67% of the run's
    ego-speed decisions on the arrival proxy, and the gate said usable. The
    whole-run fraction it replaced would have caught that, so the fix has to be
    two clauses rather than one replacing the other.
    """
    late = prefixed_run(prefix_ticks=161, converged_at=16.1)
    assert late["usable"] is False, late["gate_detail"]
    assert late["gate_detail"]["first_converted_tick_at_s"] > CONVERGENCE_BUDGET_S
    # And the post-convergence fraction is perfect, which is the point: that
    # clause cannot see this failure.
    assert late["gate_detail"]["converted_fraction_after_convergence"] == 1.0


def test_a_short_proxy_prefix_still_passes():
    """The case the prefix bound must not break: convergence takes ~1.1 s and
    every tick before it is expected, not a fault."""
    prompt = prefixed_run(prefix_ticks=11, converged_at=1.1)
    assert prompt["usable"] is True, prompt["gate_detail"]
    assert prompt["gate_detail"]["ticks_before_convergence"] == 11


def test_one_converted_tick_at_the_very_end_fails():
    """The route MIN_CONVERTED_FRACTION was introduced to close, which the
    post-convergence form re-opened: 299 proxied ticks then one conversion gives
    a post-convergence fraction of 1.0."""
    ticks = []
    for index in range(299):
        tick = a_tick(index, 0.0, converted=False)
        tick["t_s"] = round(index * 0.1, 3)
        ticks.append(tick)
    last = a_tick(299, 0.3)
    last["t_s"] = 29.9
    ticks.append(last)
    report = report_for(ticks, first_converted_at=29.9)
    assert report["usable"] is False


def test_the_convergence_boundary_is_compared_on_equal_rounding():
    """`t_s` is stored rounded to 3 dp and was compared against an unrounded
    threshold, so the first converted tick fell out of the post-convergence
    population -- harmless to the fraction, but it made
    `ticks_before_convergence` wrong by one in every run and turned a one-tick
    run's verdict into a rounding coin flip."""
    ticks = [a_tick(i, 0.0, converted=False) for i in range(3)]
    for index, tick in enumerate(ticks):
        tick["t_s"] = round(index * 0.1, 3)
    converged = a_tick(3, 0.3)
    converged["t_s"] = 0.3  # what the trace stores
    ticks.append(converged)
    # An unrounded threshold a hair above the stored value, as the real run gives.
    report = report_for(ticks, first_converted_at=0.30000000000000004)
    assert report["gate_detail"]["ticks_before_convergence"] == 3, (
        "the converted tick was counted as part of the prefix"
    )
    assert report["gate_detail"]["converted_fraction_after_convergence"] == 1.0


def test_intermittent_fallback_after_convergence_fails_at_the_stated_threshold():
    """The threshold's magnitude was unpinned: anything from ~0.012 to 0.9
    passed, because the only fell-back case had a fraction of 1/91. This pins a
    fraction just under it."""
    total = 100
    fell_back = 12  # 88% converted, under the 0.9 threshold
    ticks = []
    for index in range(total):
        converted = index >= fell_back
        tick = a_tick(index, 0.3 if converted else 0.0, converted=converted)
        tick["t_s"] = round(index * 0.1, 3)
        ticks.append(tick)
    # Converged on the first tick, then fell back for twelve.
    ticks[0] = a_tick(0, 0.3)
    ticks[0]["t_s"] = 0.0
    report = report_for(ticks, first_converted_at=0.0)
    fraction = report["gate_detail"]["converted_fraction_after_convergence"]
    assert 0.85 < fraction < MIN_CONVERTED_FRACTION_AFTER_CONVERGENCE, fraction
    assert report["usable"] is False


def test_the_fraction_counts_only_the_post_convergence_population():
    """Both earlier fraction tests gave the same answer whichever numerator was
    used, so counting the whole run survived. This separates them: 11 proxied
    then 40 converted is 40/40 after convergence and 40/51 over the run."""
    report = prefixed_run(prefix_ticks=11, converged_at=1.1)
    assert report["gate_detail"]["converted_fraction_after_convergence"] == 1.0, (
        "the numerator or denominator is counting the convergence prefix"
    )
