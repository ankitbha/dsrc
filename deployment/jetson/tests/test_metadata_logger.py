"""The tick log's writer, which no test referenced.

`MetadataLogger` had zero tests in this directory, and it is the only collection in
the pipeline that grows per tick with no cap: measured at 5,251 bytes for a
three-vehicle record, which is 567 MB/hour at 30 fps. Its own docstring says a
stalled card is survivable -- "records are dropped past 50k pending and counted in
drop stats" -- and a FULL card was not: the first failed write killed the writer
thread, the queue filled, and `close()` blocked forever on a queue nothing drained.
"""

from __future__ import annotations

import json
import threading

from logio.metadata_logger import MetadataLogger


def _logger(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return MetadataLogger(run_dir)


class BrokenFile:
    """A card with no space left, which is what killed the writer thread."""

    def write(self, _):
        raise OSError(28, "No space left on device")

    def flush(self):
        pass

    def close(self):
        pass


class TestCloseAlwaysReturns:

    def test_close_returns_when_the_writer_has_died(self, tmp_path):
        # In `run_demo` everything after `logger.close()` is unreachable if it
        # blocks: the telemetry close, the window close, the summary line, and
        # `phone.stop()` -- which that file says must run or a successful drive
        # leaves the accept socket bound and the responder thread running.
        logger = _logger(tmp_path)

        logger._file = BrokenFile()
        logger.write({"type": "tick", "tick_id": 0})

        done = threading.Event()
        threading.Thread(target=lambda: (logger.close(), done.set()),
                         daemon=True).start()
        assert done.wait(10.0), "close() blocked on a queue nothing was draining"
        assert logger.writer_failure is not None
        assert "No space left" in logger.writer_failure

    def test_a_full_queue_does_not_block_close(self, tmp_path):
        # The queue at its 50k cap with a dead writer is the state the hang needed.
        logger = _logger(tmp_path)

        logger._file = BrokenFile()
        for i in range(200):
            logger.write({"type": "tick", "tick_id": i})

        done = threading.Event()
        threading.Thread(target=lambda: (logger.close(), done.set()),
                         daemon=True).start()
        assert done.wait(10.0), "close() blocked"

    def test_records_still_queued_when_the_writer_stopped_are_counted(self, tmp_path):
        # Counting only the offers refused at the door under-reported by exactly the
        # queue depth: fifty thousand records, some 250 MB of ticks, reported as zero.
        logger = _logger(tmp_path)

        logger._file = BrokenFile()
        for i in range(50):
            logger.write({"type": "tick", "tick_id": i})
        logger.close()
        assert logger.dropped_records > 0, (
            "records lost with the writer dead were reported as zero dropped"
        )


class TestTheHealthyPathStillWorks:

    def test_records_reach_the_file_and_close_flushes_them(self, tmp_path):
        logger = _logger(tmp_path)
        for i in range(5):
            logger.write({"type": "tick", "tick_id": i})
        logger.close()

        lines = (logger.path).read_text().splitlines()
        assert len(lines) == 5
        assert [json.loads(line)["tick_id"] for line in lines] == [0, 1, 2, 3, 4]
        assert logger.writer_failure is None
        assert logger.dropped_records == 0
