"""Storing the HERE response bodies.

A HERE response is unrepeatable: the congestion on a road at a given minute cannot
be fetched again. On 2026-09-08 a drive made 143 successful calls of about 55 KB
each and kept two floats per call, so the questions those bodies could answer are
gone for good. These tests hold the two properties that make the difference -- that
a stored body is really on disk before the index names it, and that a failure to
store is reported rather than silently absorbed.
"""
from __future__ import annotations

import json

import pytest

from logio.here_logger import HereLogger


def _index(tmp_path):
    path = tmp_path / HereLogger.INDEX_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestTheIndexNamesWhatIsOnDisk:

    def test_a_body_is_stored_and_named(self, tmp_path):
        logger = HereLogger(tmp_path)
        body = b'{"results": [{"currentFlow": {"speed": 12.5}}]}'
        assert logger.write(status=200, body=body, query_lat=40.5, query_lon=-74.3,
                            query_radius_m=500.0, received_t_mono=123.4,
                            request_url="https://data.traffic.hereapi.com/v7/flow?in=x")
        logger.close()

        entries = _index(tmp_path)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["status"] == 200
        assert entry["bytes"] == len(body)
        assert entry["query_lat"] == 40.5 and entry["query_radius_m"] == 500.0
        # The file the index names must exist and hold exactly what was handed over.
        stored = tmp_path / entry["file"]
        assert stored.exists()
        assert stored.read_bytes() == body

    def test_the_url_is_kept_so_a_reader_can_see_what_was_asked(self, tmp_path):
        # Task 21 strips the key from `request_url` before it goes on the wire, so
        # storing it here cannot leak the credential -- and without it a body is a
        # payload with no question attached.
        logger = HereLogger(tmp_path)
        url = "https://data.traffic.hereapi.com/v7/flow?in=circle%3A40.5%2C-74.3%3Br%3D500"
        logger.write(status=200, body=b"{}", request_url=url)
        logger.close()
        assert _index(tmp_path)[0]["request_url"] == url
        assert "apiKey" not in _index(tmp_path)[0]["request_url"]

    def test_sequence_and_files_stay_in_step(self, tmp_path):
        logger = HereLogger(tmp_path)
        for i in range(4):
            logger.write(status=200, body=f'{{"i": {i}}}'.encode())
        logger.close()
        entries = _index(tmp_path)
        assert [e["seq"] for e in entries] == [0, 1, 2, 3]
        for e in entries:
            assert (tmp_path / e["file"]).read_bytes() == f'{{"i": {e["seq"]}}}'.encode()

    def test_no_partial_file_is_left_behind(self, tmp_path):
        # Written whole then renamed, so a reader never mistakes a truncated body for
        # a malformed response from HERE.
        logger = HereLogger(tmp_path)
        logger.write(status=200, body=b"x" * 5000)
        logger.close()
        leftovers = list((tmp_path / HereLogger.BODY_DIR).glob("*.part"))
        assert leftovers == []


class TestFailureIsReportedNotAbsorbed:

    def test_a_write_failure_is_counted_named_and_does_not_raise(self, tmp_path):
        # The caller is the HERE reader thread. An exception here would stop the feed
        # for the rest of the drive, which is far worse than one unstored body -- so
        # the contract is "never raise", and the failure has to surface some other
        # way or it surfaces nowhere.
        logger = HereLogger(tmp_path)
        logger.body_dir = tmp_path / "gone"      # never created: writes will fail
        assert logger.write(status=200, body=b"{}") is False
        assert logger.failed == 1
        assert logger.last_error, "a failure with no reason is unattributable"
        record = logger.to_record()
        assert record["failed"] == 1
        assert record["complete"] is False
        logger.close()

    def test_a_clean_run_says_so(self, tmp_path):
        logger = HereLogger(tmp_path)
        logger.write(status=200, body=b"abc")
        record = logger.to_record()
        logger.close()
        assert record["written"] == 1
        assert record["bytes_written"] == 3
        assert record["failed"] == 0
        assert record["complete"] is True

    def test_a_non_200_body_is_still_stored(self, tmp_path):
        # A refusal from HERE is evidence too, and the status is what distinguishes
        # it. Dropping error bodies would leave a run unable to say why it had no
        # feed.
        logger = HereLogger(tmp_path)
        logger.write(status=429, body=b'{"error": "rate limited"}')
        logger.close()
        entry = _index(tmp_path)[0]
        assert entry["status"] == 429
        assert (tmp_path / entry["file"]).read_bytes() == b'{"error": "rate limited"}'


class TestItMatchesTheShapeOfTheVideoIndex:

    @pytest.mark.parametrize("field", ["seq", "file", "status", "bytes"])
    def test_the_index_carries_the_fields_a_reader_needs(self, tmp_path, field):
        logger = HereLogger(tmp_path)
        logger.write(status=200, body=b"{}")
        logger.close()
        assert field in _index(tmp_path)[0]
