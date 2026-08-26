"""Which fields the feed owns, and the mappings it must not make.

The camera sees a few hundred metres now; the feed sees kilometres, minutes ago.
Fusion here is ownership, not averaging -- averaging two sources that observe
different parts of the state produces a number neither measured.
"""

from __future__ import annotations

import math

import pytest

from perception import feed_fusion
from perception.feed_fusion import Decline, FeedOwnership, own
from sensors.here_feed import FlowLink, FlowReading, Outcome


def link(speed: float | None = 6.0, free_flow: float | None = 30.0,
         jam: float | None = 4.0) -> FlowLink:
    return FlowLink(
        points=((51.49, -0.20),), speed_mps=speed, free_flow_mps=free_flow,
        jam_factor=jam, confidence=0.9, traversability="open", length_m=800.0,
    )


def reading(**over) -> FlowReading:
    fields = dict(
        outcome=Outcome.OK, link=link(), response_age_s=2.0,
        response_age_bound_s=0.01, response_age_is_proxy=False,
        link_distance_m=400.0, link_cross_track_m=10.0,
    )
    fields.update(over)
    return FlowReading(**fields)


class TestTheMapping:

    def test_congestion_is_the_speed_ratio_and_not_the_jam_factor(self):
        # The whole point. `jamFactor / 10` makes the ranges agree while the
        # quantities stay different: the simulator's field is `jam_fraction`, the
        # proportion of a segment that is jammed, and a jamFactor of 5 means
        # "notably slow", not "half the segment is stopped". Chosen so the two
        # mappings give clearly different answers, and a later "simplification" to
        # the tempting one fails here.
        owned = own(reading(link=link(speed=6.0, free_flow=30.0, jam=1.0)))

        assert owned.downstream_congestion == pytest.approx(0.8)   # 1 - 6/30
        assert owned.downstream_congestion != pytest.approx(0.1)   # jam_factor / 10

    def test_a_free_flowing_link_reports_no_congestion(self):
        owned = own(reading(link=link(speed=30.0, free_flow=30.0)))
        assert owned.downstream_congestion == pytest.approx(0.0)

    def test_a_stopped_link_reports_full_congestion_and_is_clamped(self):
        # speed above free flow is not negative congestion.
        assert own(reading(link=link(speed=0.0, free_flow=30.0))).downstream_congestion == 1.0
        assert own(reading(link=link(speed=40.0, free_flow=30.0))).downstream_congestion == 0.0

    def test_the_free_flow_speed_is_passed_through_as_itself(self):
        # This one IS the same quantity as the simulator's, m/s to m/s, so it is a
        # substitution rather than an approximation.
        assert own(reading(link=link(free_flow=27.8))).free_flow_mps == pytest.approx(27.8)

    def test_a_jam_factor_alone_is_not_a_fallback(self):
        # Reaching for jamFactor because it is present is exactly the substitution
        # this module refuses. Without both speeds, the feed owns nothing.
        owned = own(reading(link=link(speed=None, free_flow=30.0, jam=9.5)))
        assert owned.owns_congestion is False
        assert owned.declined == Decline.NO_SPEEDS

    def test_a_free_flow_too_low_to_divide_by_is_declined(self):
        owned = own(reading(link=link(speed=0.5, free_flow=1.0)))
        assert owned.declined == Decline.NO_FREE_FLOW


class TestWhenTheFeedOwnsNothing:

    def test_no_reading_at_all(self):
        owned = own(None)
        assert owned.owns_congestion is False
        assert owned.declined == Decline.NO_READING

    def test_every_non_ok_outcome_declines_rather_than_contributing_zero(self):
        # Zero in this field reads as "clear road ahead". A run with no traffic
        # data must hand the field on, not answer it.
        for outcome in (Outcome.NO_RESPONSE_YET, Outcome.HTTP_ERROR, Outcome.UNPARSEABLE,
                        Outcome.NO_LINK_MATCHED, Outcome.NO_LINK_AHEAD, Outcome.STALE,
                        Outcome.UNUSABLE_FIX):
            owned = own(reading(outcome=outcome, link=None))
            assert owned.owns_congestion is False, outcome
            assert owned.declined == Decline.NOT_OK

    def test_a_match_off_to_the_side_is_not_our_road(self):
        # Task 27's cone widens with range and admits 2.6 km of lateral offset at
        # the horizon. Without this gate a motorway kilometres aside supplies the
        # congestion the driver is told about.
        owned = own(reading(link_cross_track_m=feed_fusion.MAX_CROSS_TRACK_M + 1.0))
        assert owned.declined == Decline.OFF_CORRIDOR

        near = own(reading(link_cross_track_m=feed_fusion.MAX_CROSS_TRACK_M - 1.0))
        assert near.owns_congestion is True

    def test_a_reading_with_no_cross_track_is_declined_rather_than_assumed_on_road(self):
        assert own(reading(link_cross_track_m=None)).declined == Decline.OFF_CORRIDOR


class TestAging:

    def test_an_old_reading_hands_the_field_on_rather_than_decaying_it(self):
        # The rejected alternative is interpolating toward neutral as the value
        # ages, which manufactures a middle number that neither source measured and
        # that reads as moderate congestion.
        owned = own(reading(response_age_s=feed_fusion.MAX_FEED_AGE_S + 1.0))
        assert owned.declined == Decline.STALE
        assert owned.downstream_congestion is None

    def test_the_bound_is_charged_against_the_limit(self):
        # `ObservationBuilder` adds `uncertainty_s` before comparing; a reading
        # just inside the window whose bound pushes it out is not fresh.
        inside = own(reading(response_age_s=feed_fusion.MAX_FEED_AGE_S - 1.0,
                             response_age_bound_s=0.1))
        assert inside.owns_congestion is True

        outside = own(reading(response_age_s=feed_fusion.MAX_FEED_AGE_S - 1.0,
                              response_age_bound_s=2.0))
        assert outside.declined == Decline.STALE

    def test_a_stamp_from_this_clocks_future_is_not_fresh(self):
        # `PhoneGpsReader.is_stale` states the rule: the freshness predicates in
        # this codebase must not disagree about that. Task 27 found the feed's own
        # predicate breaking it; this is the fourth and must not.
        owned = own(reading(response_age_s=-(feed_fusion.MAX_FEED_AGE_S + 1.0)))
        assert owned.declined == Decline.STALE

    def test_a_proxied_stamp_still_owns_but_says_so(self):
        # Proxying is normal in the opening seconds. The field is usable; a run
        # that cannot tell it was aged on a proxy cannot say how much of its feed
        # was aged on a guess.
        owned = own(reading(response_age_is_proxy=True))
        assert owned.owns_congestion is True
        assert owned.is_proxy is True
        assert feed_fusion.to_record(owned)["age_is_proxy"] is True


class TestTheRecord:

    def test_a_drive_where_the_feed_never_owned_a_field_says_why(self):
        record = feed_fusion.to_record(own(None))
        assert record["owns_congestion"] is False
        assert record["declined"] == Decline.NO_READING
        assert record["downstream_congestion"] is None

    def test_a_contributing_drive_records_the_value_and_its_age(self):
        record = feed_fusion.to_record(own(reading(response_age_s=3.25)))
        assert record["owns_congestion"] is True
        assert record["age_s"] == 3.25
        assert 0.0 <= record["downstream_congestion"] <= 1.0


class TestTheBuilderChain:
    """Ownership resolved inside `ObservationBuilder`, with the vector unchanged.

    Task 47 wants field-for-field parity with the simulator, so nothing here adds,
    removes or renames a field -- the feed either supplies an existing one or hands
    it on.
    """

    def builder(self):
        from perception.observation_builder import BuilderConfig, ObservationBuilder
        return ObservationBuilder(BuilderConfig())

    def test_a_run_with_no_feed_is_exactly_what_it_was(self):
        # The regression that matters most: ingestion must change nothing for a run
        # without a phone. Same fields, same values, same provenance.
        from sensors.gps_reader import GpsFix

        gps = GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=20.0,
                     heading_deg=90.0, t_mono=100.0)
        without = self.builder().build([], gps, t_mono=100.0)
        with_none = self.builder().build([], gps, t_mono=100.0, feed=None)

        assert without.obs == with_none.obs
        assert without.field_sources == with_none.field_sources
        assert with_none.obs["downstream_congestion_estimate"] == 0.0

    def test_a_feed_reading_supplies_the_congestion_field_and_names_itself(self):
        from sensors.gps_reader import GpsFix

        gps = GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=20.0,
                     heading_deg=90.0, t_mono=100.0)
        result = self.builder().build([], gps, t_mono=100.0,
                                      feed=reading(link=link(speed=6.0, free_flow=30.0)))

        assert result.obs["downstream_congestion_estimate"] == pytest.approx(0.8)
        # Not "measured": it is derived from a service's estimate of something this
        # vehicle cannot see, and a reader must be able to tell those apart.
        assert result.field_sources["downstream_congestion_estimate"] == feed_fusion.SOURCE_FEED

    def test_a_declined_reading_leaves_the_field_where_it_was(self):
        from sensors.gps_reader import GpsFix

        gps = GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=20.0,
                     heading_deg=90.0, t_mono=100.0)
        result = self.builder().build([], gps, t_mono=100.0,
                                      feed=reading(outcome=Outcome.STALE, link=None))

        assert result.obs["downstream_congestion_estimate"] == 0.0
        assert result.field_sources["downstream_congestion_estimate"] == "fallback_neutral"

    def test_the_feed_does_not_touch_the_bottleneck_flag(self):
        # It looks like exactly what `link_distance_m` is, and it is not: the
        # simulator uses 0.0 for "in a bottleneck segment" and inf otherwise, so
        # the policy has only ever seen {0, inf} there. A real distance in that
        # field is the same error as jamFactor/10, one field along.
        from sensors.gps_reader import GpsFix

        gps = GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=20.0,
                     heading_deg=90.0, t_mono=100.0)
        result = self.builder().build([], gps, t_mono=100.0, feed=reading())

        assert math.isinf(result.obs["distance_to_downstream_bottleneck"])
        assert result.field_sources["distance_to_downstream_bottleneck"] == "sim_parity"
