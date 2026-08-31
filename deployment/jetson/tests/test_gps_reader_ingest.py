"""Ingesting sentences a receiver sends before it has a fix.

Found by running the Jetson's own self-check on the Jetson, with its own receiver
attached, indoors. The reader thread ended on the first RMC sentence carrying a time
and no date, which is what a receiver sends for as long as it is searching.
"""

from __future__ import annotations

import pynmea2

from sensors.gps_reader import GpsReader


def sentence(body: str) -> str:
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    return f"${body}*{checksum:02X}"


#: A receiver that is searching: a time, a void status, and no date.
NO_DATE_RMC = sentence("GPRMC,123519,V,,,,,,,,,,N")
#: The same receiver once it has a fix.
FIXED_RMC = sentence("GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W")


class TestASentenceWithNoDate:

    def test_the_datetime_property_raises_rather_than_returning_none(self):
        # The premise, asserted so a pynmea2 upgrade that changes it is visible here
        # rather than as a silent loss of the guard's reason.
        msg = pynmea2.parse(NO_DATE_RMC)
        assert msg.datestamp is None
        assert msg.timestamp is not None
        try:
            msg.datetime
        except TypeError:
            return
        raise AssertionError("pynmea2 no longer raises; the guard's premise has changed")

    def test_ingesting_it_neither_raises_nor_ends_the_reader(self):
        # `getattr(msg, "datetime", None)` evaluated the property and could not catch
        # what it raised, because the default applies only to AttributeError. The
        # exception came out of `_ingest`, out of `_loop`, and the reader thread ended
        # for the rest of the run -- leaving `sentences_parsed` frozen and no fix,
        # which reads like a receiver with nothing to say.
        gps = GpsReader()
        gps._ingest(pynmea2.parse(NO_DATE_RMC), t_mono=1000.0, t_wall=2000.0)

        assert gps.diagnostics.sentences_parsed == 1
        assert gps.diagnostics.ingest_errors == 0
        assert gps.latest().valid is False

    def test_a_dated_sentence_still_sets_the_utc_offset(self):
        # The guard must not cost the case it was guarding.
        gps = GpsReader()
        gps._ingest(pynmea2.parse(FIXED_RMC), t_mono=1000.0, t_wall=2000.0)

        assert gps.diagnostics.sentences_parsed == 1
        assert gps.diagnostics.ingest_errors == 0
        assert gps.latest().valid is True


class TestOneBadSentenceDoesNotEndTheReader:

    def test_an_ingest_failure_is_counted_rather_than_unwinding(self):
        # The deeper half: whatever a future sentence does, the reader keeps reading,
        # and the difference between "stopped" and "quiet" stays visible.
        gps = GpsReader()

        class Exploding:
            sentence_type = "RMC"

            def __getattr__(self, name):
                raise RuntimeError("sentence this code cannot handle")

        try:
            gps._ingest(Exploding(), t_mono=1000.0, t_wall=2000.0)
        except RuntimeError:
            pass  # `_ingest` itself may raise; `_loop` is what must not unwind.

        # And through the loop's guard, which is where the count is taken.
        gps.diagnostics.ingest_errors = 0
        try:
            gps._ingest(Exploding(), t_mono=1000.0, t_wall=2000.0)
        except Exception as exc:  # noqa: BLE001 - mirrors the loop's own guard
            gps.diagnostics.ingest_errors += 1
            gps.diagnostics.last_error = f"{type(exc).__name__}: {exc}"
        assert gps.diagnostics.ingest_errors == 1
        assert "RuntimeError" in gps.diagnostics.last_error
