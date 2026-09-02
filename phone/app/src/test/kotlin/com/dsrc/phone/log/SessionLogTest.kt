package com.dsrc.phone.log

import com.dsrc.transport.Json
import com.dsrc.transport.JsonValue
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.file.Files

class SessionLogTest {

    private lateinit var dir: File

    @Before
    fun makeDir() {
        dir = Files.createTempDirectory("session-log").toFile()
    }

    @After
    fun removeDir() {
        dir.deleteRecursively()
    }

    private fun log(maxBytes: Long = SessionLog.MAX_BYTES, depth: Int = SessionLog.QUEUE_DEPTH) =
        SessionLog(File(dir, "drive.jsonl"), maxBytes = maxBytes, queueDepth = depth)

    private fun lines() = File(dir, "drive.jsonl").let {
        if (it.exists()) it.readLines() else emptyList()
    }

    @Test
    fun `every offered line reaches the file, one per line`() {
        val log = log()
        log.start()
        repeat(20) { log.offer("""{"ch":"gps","seq":$it}""") }
        log.stop()

        assertEquals(20, lines().size)
        assertEquals("""{"ch":"gps","seq":0}""", lines().first())
        assertEquals("""{"ch":"gps","seq":19}""", lines().last())
        assertTrue("${log.stats}", log.stats.complete)
    }

    @Test
    fun `stopping flushes what the writer still had`() {
        // Joined on stop, unlike the other workers here. The point of this thread is the
        // file it leaves behind: returning before it flushed would mean the last seconds of
        // a drive are missing from the artifact the drive was for.
        val log = log()
        log.start()
        repeat(50) { log.offer("""{"seq":$it}""") }
        log.stop()

        assertEquals(50, lines().size)
    }

    @Test
    fun `a full queue drops and counts, it does not block`() {
        // A log that stalled the camera to keep itself complete would corrupt the very
        // measurement it exists to record. Never started, so nothing drains: every offer
        // past the queue depth must be refused rather than waiting for room.
        val log = log(depth = 4)
        log.start()
        // Fill past the depth faster than the writer can drain, then check the accounting
        // rather than a timing: what matters is that no offer blocked.
        repeat(500) { log.offer("""{"seq":$it}""") }
        log.stop()

        val stats = log.stats
        assertEquals("every line is either written or dropped", 500, stats.written + stats.droppedQueueFull)
        assertTrue("nothing was dropped, so the queue never filled: $stats", stats.droppedQueueFull > 0)
        assertFalse("a lossy log must not claim to be complete", stats.complete)
    }

    @Test
    fun `the file stops at its cap rather than growing without bound`() {
        val log = log(maxBytes = 100)
        log.start()
        repeat(50) { log.offer("0123456789") }   // 11 bytes each with the newline
        log.stop()

        assertTrue("the file grew past its cap: ${log.stats.bytes}", log.stats.bytes <= 100)
        assertTrue("nothing was dropped at the cap: ${log.stats}", log.stats.droppedAtCap > 0)
        assertFalse(log.stats.complete)
    }

    @Test
    fun `the start of the drive is what survives the cap, not the end`() {
        // Stopping at the cap rather than rotating is a choice. The start is where the
        // setup, the first fixes and the timebase exchange are -- the part an analysis most
        // often needs -- and a file that stops early and says so never claims to be
        // complete, where one that silently rotates does.
        val log = log(maxBytes = 33)
        log.start()
        repeat(10) { log.offer("line-$it") }     // 8 bytes each with the newline
        log.stop()

        val written = lines()
        assertEquals(listOf("line-0", "line-1", "line-2", "line-3"), written)
    }

    @Test
    fun `offering before start writes nothing`() {
        // The writer is what creates the file; taking lines with nothing to drain them
        // would grow the queue against a thread that never arrives.
        val log = log()
        repeat(10) { log.offer("""{"seq":$it}""") }
        log.start()
        log.stop()

        assertEquals(0, lines().size)
    }

    @Test
    fun `starting twice does not start a second writer`() {
        val log = log()
        log.start()
        log.start()
        repeat(10) { log.offer("""{"seq":$it}""") }
        log.stop()

        // Each line is still written once -- two writers draining one queue do not
        // duplicate, because the poll is atomic -- so this does not pin the guard, and
        // saying so is better than implying it does. What the guard actually prevents is
        // `stop()` joining only the second thread and leaving the first writing after the
        // session ended, which no deterministic test here reaches.
        assertEquals(10, lines().size)
        assertEquals(10, lines().toSet().size)
    }

    @Test
    fun `the stats name the file, so an analysis can find it`() {
        val log = log()
        log.start()
        log.offer("""{"seq":1}""")
        log.stop()

        assertTrue(log.stats.path.endsWith("drive.jsonl"))
    }

    @Test
    fun `a line offered after stop is counted, not dropped in silence`() {
        // Teardown releases resources in an order, and the link outlives the log by several
        // steps -- so the transport can write after the log has stopped. Returning without
        // counting meant the file could be short and still call itself complete, which is
        // the one thing this class has to be right about.
        val log = log()
        log.start()
        log.offer("""{"seq":0}""")
        log.stop()
        repeat(25) { log.offer("""{"seq":$it}""") }

        val stats = log.stats
        assertEquals(1, lines().size)
        assertEquals(25, stats.droppedNotRunning)
        assertFalse("a short file must not call itself complete: $stats", stats.complete)
    }

    @Test
    fun `offering before start is counted too`() {
        val log = log()
        repeat(3) { log.offer("""{"seq":$it}""") }
        log.start()
        log.stop()

        assertEquals(3, log.stats.droppedNotRunning)
        assertFalse(log.stats.complete)
    }

    // -- the two inbound line shapes -------------------------------------------

    private fun anAdvisoryHeader(captureMonoNs: Long = 555) = JsonValue.Obj(
        mapOf(
            "ch" to JsonValue.Text("advisory"),
            "seq" to JsonValue.Num(3),
            "t_mono_ns" to JsonValue.Num(100),
            "t_wall_ns" to JsonValue.Num(200),
            "n" to JsonValue.Num(0),
            "t_capture_mono_ns" to JsonValue.Num(captureMonoNs),
        )
    )

    @Test
    fun `an inbound line carries the receipt stamps and the header verbatim`() {
        val log = log()
        log.start()
        log.offerInbound(recvMonoNs = 1_000, recvWallNs = 2_000, header = anAdvisoryHeader())
        log.stop()

        val decoded = Json.decode(lines().single()) as JsonValue.Obj
        assertEquals("in", (decoded.entries.getValue("dir") as JsonValue.Text).value)
        assertEquals(1_000L, (decoded.entries.getValue("recv_mono_ns") as JsonValue.Num).value)
        assertEquals(2_000L, (decoded.entries.getValue("recv_wall_ns") as JsonValue.Num).value)

        // Verbatim, the same "cannot disagree with what was decoded" contract offer()
        // already gives the outbound side -- every field the header carried is still there.
        val header = decoded.entries.getValue("header") as JsonValue.Obj
        assertEquals("advisory", (header.entries.getValue("ch") as JsonValue.Text).value)
        assertEquals(555L, (header.entries.getValue("t_capture_mono_ns") as JsonValue.Num).value)
    }

    @Test
    fun `an inbound line is not mistaken for an outbound one`() {
        // Outbound lines are bare headers with no "dir" key at all -- this is the whole
        // discriminator an offline reader keys on.
        val log = log()
        log.start()
        log.offer("""{"ch":"camera","seq":1}""")
        log.offerInbound(1, 2, anAdvisoryHeader())
        log.stop()

        val decodedOutbound = Json.decode(lines()[0]) as JsonValue.Obj
        val decodedInbound = Json.decode(lines()[1]) as JsonValue.Obj
        assertFalse("dir" in decodedOutbound.entries)
        assertEquals("in", (decodedInbound.entries.getValue("dir") as JsonValue.Text).value)
    }

    @Test
    fun `an advisory_shown line names the advisory and when it was first shown`() {
        val log = log()
        log.start()
        log.offerAdvisoryShown(captureMonoNs = 555, shownMonoNs = 1_250)
        log.stop()

        val decoded = Json.decode(lines().single()) as JsonValue.Obj
        assertEquals("shown", (decoded.entries.getValue("dir") as JsonValue.Text).value)
        assertEquals(555L, (decoded.entries.getValue("t_capture_mono_ns") as JsonValue.Num).value)
        assertEquals(1_250L, (decoded.entries.getValue("shown_mono_ns") as JsonValue.Num).value)
    }

    @Test
    fun `an inbound line offered before start is counted the same way an outbound one is`() {
        val log = log()
        log.offerInbound(1, 2, anAdvisoryHeader())
        log.offerAdvisoryShown(555, 600)
        log.start()
        log.stop()

        assertEquals(2, log.stats.droppedNotRunning)
        assertEquals(0, lines().size)
    }

    // -- the fourth line shape: offerFailure -------------------------------------

    @Test
    fun `a failure line is written and is a distinct shape`() {
        val log = log()
        log.start()
        log.offer("""{"ch":"gps","seq":1}""")
        log.offerFailure("link.dial_failed", atMonoNs = 100, atWallNs = 200, detail = "refused")
        log.offerAdvisoryShown(captureMonoNs = 1, shownMonoNs = 2)
        log.offerInbound(1, 2, anAdvisoryHeader())
        log.stop()

        val decoded = lines().map { Json.decode(it) as JsonValue.Obj }
        val failLine = decoded.single { (it.entries["dir"] as? JsonValue.Text)?.value == "fail" }
        assertEquals("link.dial_failed", (failLine.entries.getValue("kind") as JsonValue.Text).value)
        assertEquals(100L, (failLine.entries.getValue("at_mono_ns") as JsonValue.Num).value)
        assertEquals(200L, (failLine.entries.getValue("at_wall_ns") as JsonValue.Num).value)
        assertEquals(1L, (failLine.entries.getValue("n") as JsonValue.Num).value)
        assertEquals("refused", (failLine.entries.getValue("detail") as JsonValue.Text).value)
        assertEquals(0L, (failLine.entries.getValue("suppressed") as JsonValue.Num).value)

        // The three existing shapes are unchanged byte for byte.
        val outbound = decoded.first { "dir" !in it.entries }
        assertEquals("""{"ch":"gps","seq":1}""", Json.encode(outbound))
        val shown = decoded.single { (it.entries["dir"] as? JsonValue.Text)?.value == "shown" }
        assertEquals(1L, (shown.entries.getValue("t_capture_mono_ns") as JsonValue.Num).value)
        val inbound = decoded.single { (it.entries["dir"] as? JsonValue.Text)?.value == "in" }
        assertEquals(1L, (inbound.entries.getValue("recv_mono_ns") as JsonValue.Num).value)
    }

    @Test
    fun `the rate cap holds and counts what it suppressed`() {
        // D12's stated loss: within one second, only the first of a kind is written;
        // the rest are held back and their count travels forward onto the next
        // ACCEPTED line of that kind, whenever that is.
        val log = log()
        log.start()
        repeat(100) { log.offerFailure("link.dial_failed", atMonoNs = 1_000_000_000L, atWallNs = 0) }
        // A different second: this one is accepted, and carries what the first
        // second held back.
        log.offerFailure("link.dial_failed", atMonoNs = 2_000_000_000L, atWallNs = 0)
        log.stop()

        val decoded = lines().map { Json.decode(it) as JsonValue.Obj }
        assertEquals("only one line per kind per second reaches the file", 2, decoded.size)
        val suppressed = decoded.map { (it.entries.getValue("suppressed") as JsonValue.Num).value }
        assertEquals(listOf(0L, 99L), suppressed)
    }

    @Test
    fun `a saturated failure rate does not displace frame headers`() {
        // The direction test for behaviour change 3. A queue this shallow drops the
        // frame HEADER when full, which is the file's whole reason to exist -- so
        // the property worth pinning is that a burst of failures cannot compete
        // for that queue's depth no matter how large the burst is. All 5,000 share
        // one second, so the rate cap holds every one but the first back before
        // any of them reaches `queue.offer()` at all: `droppedQueueFull` staying
        // at zero is the proof that nothing here ever pressured the queue a real
        // header offer shares.
        val log = log(depth = 4)
        log.start()
        repeat(5_000) { log.offerFailure("link.dial_failed", atMonoNs = 0, atWallNs = 0) }
        log.stop()

        val stats = log.stats
        assertEquals("only the first of the burst should ever have reached the queue", 1L, stats.written)
        assertEquals(4_999L, stats.failuresSuppressed)
        assertEquals(
            "a burst that never touched queue.offer() still reported queue pressure",
            0L, stats.droppedQueueFull,
        )
    }

    @Test
    fun `the per-kind lifetime cap is separate from the per-second one`() {
        val log = log()
        log.start()
        // One per second, well past the lifetime cap.
        for (second in 0 until SessionLog.MAX_LINES_PER_KIND + 10) {
            log.offerFailure("link.dial_failed", atMonoNs = second * 1_000_000_000L, atWallNs = 0)
        }
        log.stop()

        val written = lines().map { Json.decode(it) as JsonValue.Obj }
        assertEquals(SessionLog.MAX_LINES_PER_KIND, written.size)
        // Ten occurrences arrived one per second, past the lifetime cap, each on
        // its own second -- so each is suppressed individually rather than
        // sharing a per-second bucket with anything else.
        assertEquals(10L, log.stats.failuresSuppressed)
    }

}
