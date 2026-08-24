package com.dsrc.phone.log

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

}
