package com.dsrc.phone.sensors

import com.dsrc.phone.config.SensingConfig
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraPipelineTest {

    /** Runs encode work inline, so a test observes the finished state without waiting. */
    private val inline = Executor { it.run() }

    private val ms = 1_000_000L

    private fun pipeline(hz: Double = 5.0, executor: Executor = inline) =
        CameraPipeline(SensingConfig(cameraHz = hz), executor)

    private fun offer(
        p: CameraPipeline,
        tsNs: Long,
        packs: AtomicInteger? = null,
        compresses: AtomicInteger? = null,
        compress: ((ByteArray, Int, Int, Int) -> ByteArray)? = null,
    ) = p.offer(
        timestampNs = tsNs,
        width = 4,
        height = 4,
        pack = { packs?.incrementAndGet(); ByteArray(24) },
        compress = compress ?: { bytes, _, _, _ -> compresses?.incrementAndGet(); bytes },
    )

    // -- rate ----------------------------------------------------------------

    @Test
    fun `a commanded rate is honoured over a simulated minute`() {
        val p = pipeline(hz = 5.0)
        val sourcePeriod = 1_000_000_000L / 30
        for (i in 0 until 30 * 60) offer(p, i * sourcePeriod)
        val stats = p.stats
        assertEquals(30 * 60L, stats.seen)
        assertTrue("expected ~300 accepted at 5 Hz, got ${stats.accepted}", stats.accepted in 299..301)
        assertEquals("the rest were gated", stats.seen - stats.accepted, stats.gated)
    }

    @Test
    fun `a rejected frame costs no pixel copy`() {
        // The reason the gate runs before packing: at 30 Hz into a 5 Hz target, five in
        // six frames must cost nothing at all.
        val packs = AtomicInteger()
        val p = pipeline(hz = 5.0)
        val sourcePeriod = 1_000_000_000L / 30
        for (i in 0 until 60) offer(p, i * sourcePeriod, packs = packs)
        assertEquals("pack must run only for accepted frames", p.stats.accepted.toInt(), packs.get())
        assertTrue(packs.get() < 15)
    }

    @Test
    fun `a rejected frame costs no compression`() {
        val compresses = AtomicInteger()
        val p = pipeline(hz = 5.0)
        val sourcePeriod = 1_000_000_000L / 30
        for (i in 0 until 60) offer(p, i * sourcePeriod, compresses = compresses)
        assertEquals(p.stats.accepted.toInt(), compresses.get())
    }

    @Test
    fun `a rate change is honoured mid-stream`() {
        val p = pipeline(hz = 1.0)
        offer(p, 0)
        assertEquals(1, p.stats.accepted)
        p.setRate(10.0)
        assertEquals(10.0, p.rateHz, 0.0)
        assertTrue(offer(p, 100 * ms))
        assertEquals(2, p.stats.accepted)
    }

    // -- frame ids -----------------------------------------------------------

    @Test
    fun `frame ids are consecutive over the accepted frames`() {
        val p = pipeline(hz = 1000.0)
        val ids = mutableListOf<Long>()
        for (i in 1..50) {
            offer(p, i * ms)
            p.drain()?.let { ids.add(it.frameId) }
        }
        assertEquals("every accepted frame should have been drained", 50, ids.size)
        assertEquals((1L..50L).toList(), ids)
    }

    @Test
    fun `frame ids count accepted frames, not frames the sensor produced`() {
        // So a gap means "lost after acceptance", which is actionable, rather than "the
        // camera runs faster than the commanded rate", which is normal.
        val p = pipeline(hz = 5.0)
        val sourcePeriod = 1_000_000_000L / 30
        var first: Long? = null
        var last: Long? = null
        for (i in 0 until 60) {
            if (offer(p, i * sourcePeriod)) {
                val frame = p.drain()!!
                if (first == null) first = frame.frameId
                last = frame.frameId
            }
        }
        assertEquals(1L, first)
        assertEquals("ids must not skip the gated frames", p.stats.accepted, last)
    }

    // -- timestamps ----------------------------------------------------------

    @Test
    fun `capture stamps are non-decreasing and are the ones offered`() {
        val p = pipeline(hz = 1000.0)
        var previous = Long.MIN_VALUE
        for (i in 1..30) {
            val ts = i * 2 * ms
            offer(p, ts)
            val frame = p.drain()!!
            assertEquals("the stamp must be carried through unchanged", ts, frame.captureMonoNs)
            assertTrue(frame.captureMonoNs >= previous)
            previous = frame.captureMonoNs
        }
    }

    // -- failure -------------------------------------------------------------

    @Test
    fun `an encode failure costs one frame and the stream continues`() {
        val p = pipeline(hz = 1000.0)
        var calls = 0
        val flaky: (ByteArray, Int, Int, Int) -> ByteArray = { b, _, _, _ ->
            calls++
            if (calls == 2) throw IllegalStateException("bad frame") else b
        }
        for (i in 1..4) offer(p, i * ms, compress = flaky)
        val stats = p.stats
        assertEquals(4, stats.accepted)
        assertEquals(1, stats.encodeFailures)
        assertEquals(3, stats.encoded)
        assertEquals("nothing left in flight with an inline executor", 0, stats.inFlight)
    }

    @Test
    fun `an encode failure does not stop later frames arriving`() {
        val p = pipeline(hz = 1000.0)
        var first = true
        offer(p, ms, compress = { b, _, _, _ -> if (first) { first = false; throw RuntimeException("x") } else b })
        assertNull("the failed frame produced nothing", p.drain())
        offer(p, 2 * ms)
        assertTrue("a later frame still arrives", p.drain() != null)
    }

    // -- stopping ------------------------------------------------------------

    @Test
    fun `a stopped pipeline accepts nothing`() {
        val p = pipeline(hz = 1000.0)
        offer(p, ms)
        p.stop()
        assertFalse(offer(p, 100 * ms))
        assertEquals(1, p.stats.accepted)
    }

    @Test
    fun `stopping discards the held frame`() {
        val p = pipeline(hz = 1000.0)
        offer(p, ms)
        p.stop()
        assertNull(p.drain())
    }

    @Test
    fun `a frame queued before a stop is not encoded after it`() {
        // A frame encoded after teardown would land in a buffer nobody drains, and its
        // pixels were copied from a camera buffer already handed back.
        val queued = mutableListOf<Runnable>()
        val deferred = Executor { queued.add(it) }
        val p = pipeline(hz = 1000.0, executor = deferred)
        offer(p, ms)
        assertEquals(1, queued.size)
        p.stop()
        queued.forEach { it.run() }
        assertEquals("the queued encode must drop out", 0, p.stats.encoded)
        assertNull(p.drain())
        assertTrue("the accounting must still balance: ${p.stats.buffer}", p.stats.buffer.balances)
    }

    // -- accounting ----------------------------------------------------------

    @Test
    fun `a pack failure is counted, not propagated`() {
        // YuvPacker refuses a geometry it cannot handle and is called from the pack
        // lambda, so a bad camera geometry lands exactly here. Letting it out left
        // `accepted` incremented with no matching outcome, so total failure was
        // indistinguishable from an encoder backlog.
        val p = pipeline(hz = 1000.0)
        repeat(10) { i ->
            p.offer(
                timestampNs = (i + 1) * ms,
                width = 4,
                height = 4,
                pack = { throw IllegalArgumentException("bad geometry") },
                compress = { b, _, _, _ -> b },
            )
        }
        val stats = p.stats
        assertEquals(10, stats.accepted)
        assertEquals(10, stats.packFailures)
        assertEquals(0, stats.encoded)
        assertEquals("nothing may be left in flight", 0, stats.inFlight)
    }

    @Test
    fun `a pack failure does not stop later frames`() {
        val p = pipeline(hz = 1000.0)
        var first = true
        p.offer(ms, 4, 4, { if (first) { first = false; throw RuntimeException("x") } else ByteArray(24) }, { b, _, _, _ -> b })
        offer(p, 2 * ms)
        assertTrue("a later frame still arrives", p.drain() != null)
    }

    @Test
    fun `a frame refused because the pipeline stopped is not reported as rate limiting`() {
        // Both were `gated`, which is documented as the normal cost of a commanded rate
        // below the sensor's -- so a teardown read as rate limiting.
        val p = pipeline(hz = 1000.0)
        offer(p, ms)
        p.stop()
        repeat(10) { offer(p, (it + 2) * ms) }
        val stats = p.stats
        assertEquals(10, stats.refusedStopped)
        assertEquals("none of those were rate-limited", 0, stats.gated)
        assertTrue("every delivered frame must be accounted for: $stats", stats.balances)
    }

    @Test
    fun `the balance identity can actually fail`() {
        // The anti-vacuity control. While `gated` was derived from the other counters the
        // identity reduced to `seen == seen`, so it held for every input -- including
        // seen=5 with accepted=100 -- and every test asserting it passed for nothing.
        val consistent = CameraPipeline.Stats(
            seen = 10, accepted = 4, encoded = 4, encodeFailures = 0, packFailures = 0,
            refusedStopped = 2, gated = 4, abandoned = 0,
            buffer = FrameBuffer.Stats(4, 3, 1, 0, false),
        )
        assertTrue(consistent.balances)

        val inconsistent = consistent.copy(seen = 11)
        assertFalse("an unaccounted frame must show up", inconsistent.balances)
        assertFalse("as must a miscounted one", consistent.copy(gated = 3).balances)
        assertFalse(consistent.copy(accepted = 100).balances)
    }

    @Test
    fun `a frame abandoned after a stop is counted, not left in flight`() {
        // Same shape as the pack early-return: without a counter the frame leaves
        // `accepted` incremented and no outcome recorded, so `inFlight` never returns to
        // zero and a teardown reads as an encoder backlog. onSensingDown logs the stats
        // before stopping, so that is exactly when it would be read.
        val queued = mutableListOf<Runnable>()
        val p = pipeline(hz = 1000.0, executor = Executor { queued.add(it) })
        repeat(3) { offer(p, (it + 1) * ms) }
        p.stop()
        queued.forEach { it.run() }

        val stats = p.stats
        assertEquals(3, stats.accepted)
        assertEquals(3, stats.abandoned)
        assertEquals(0, stats.encoded)
        assertEquals("nothing may be left in flight after a teardown", 0, stats.inFlight)
    }

    @Test
    fun `every frame the camera delivered is accounted for under exactly one heading`() {
        val p = pipeline(hz = 5.0)
        val sourcePeriod = 1_000_000_000L / 30
        for (i in 0 until 120) offer(p, i * sourcePeriod)
        p.stop()
        for (i in 120 until 150) offer(p, i * sourcePeriod)
        val stats = p.stats
        assertEquals(150, stats.seen)
        assertTrue("$stats", stats.balances)
        assertEquals(stats.seen, stats.accepted + stats.gated + stats.refusedStopped)
    }

    @Test
    fun `the counters balance over a lossy run`() {
        val p = pipeline(hz = 1000.0)
        for (i in 1..200) {
            offer(p, i * ms)
            if (i % 4 == 0) p.drain()
        }
        val stats = p.stats
        assertEquals(200, stats.seen)
        assertEquals(200, stats.accepted)
        assertEquals(0, stats.inFlight)
        assertTrue("buffer accounting must balance: ${stats.buffer}", stats.buffer.balances)
    }

    @Test
    fun `latest-wins keeps the newest frame when the drain is slow`() {
        val p = pipeline(hz = 1000.0)
        for (i in 1..5) offer(p, i * ms)
        val frame = p.drain()!!
        assertEquals("the newest accepted frame survives", 5L, frame.frameId)
        assertEquals("four were displaced", 4, p.stats.buffer.dropped)
    }
}
