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
    LINK_BOUND_MULTIPLE,
    LINK_FLOOR_MS,
    MIN_CONVERTED_FRACTION,
    _report,
)


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


def report_for(ticks: list[dict], *, n_messages: int | None = None, **stats_kwargs) -> dict:
    count = len(ticks) if n_messages is None else n_messages
    return _report(
        ticks, 10.0, 243_264_000_000_000,
        {"frames": count, "fixes": count, "pongs": count, "invalid": 0},
        {"received": len(ticks), "unmatched": 0, "frame_ids": {t["frame_id"] for t in ticks}},
        _Recorder(), _Recorder(), _Recorder(), _Recorder(), _Pipeline(), 1.0,
        phone_stats=_Stats(count, **stats_kwargs), jetson_stats=_Stats(count, **stats_kwargs),
    )


# -- the gate ----------------------------------------------------------------


def test_a_healthy_run_passes():
    report = report_for([a_tick(i, 0.3) for i in range(60)])
    assert report["usable"] is True, report["gate_detail"]
    assert report["gate_detail"]["converted_fraction"] == 1.0


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


def test_a_run_that_mostly_proxied_fails():
    """A timebase that converged once and then went unusable for the rest of the
    drive is not a run that exercised the conversion -- and plan section 9's
    stated risk is exactly the proxy being in use for longer than expected.
    `converted_and_fresh == len(converted)` was satisfied by 1 == 1."""
    ticks = [a_tick(0, 0.3)] + [
        a_tick(i, 0.0, converted=False) for i in range(1, 91)
    ]
    report = report_for(ticks)
    assert report["usable"] is False
    assert report["gate_detail"]["converted_fraction"] < MIN_CONVERTED_FRACTION


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
    term the flag went false for no defect."""
    stats = _Stats(60)
    stats._row.sent = 59  # one still queued when stats() was sampled
    stats._row.received = 59
    stats._row.delivered = 59
    report = _report(
        [a_tick(i, 0.3) for i in range(40)], 10.0, 0,
        {"frames": 60}, {"received": 40, "unmatched": 0, "frame_ids": set(range(40))},
        _Recorder(), _Recorder(), _Recorder(), _Recorder(), _Pipeline(), 1.0,
        phone_stats=stats, jetson_stats=stats,
    )
    assert report["transport"]["reconciled"] is True, report["transport"]["reconciliation"]
    assert report["transport"]["upstream"]["camera"]["in_flight"] == 1
