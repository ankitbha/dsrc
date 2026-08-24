package com.dsrc.phone.sensors

import com.dsrc.phone.config.SensingConfig
import com.dsrc.transport.ImuSample
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ImuPipelineTest {

    private val ms = 1_000_000L

    private fun pipeline(
        hz: Double = 50.0,
        accept: Boolean = true,
    ): Pair<ImuPipeline, MutableList<ImuSample>> {
        val out = mutableListOf<ImuSample>()
        val p = ImuPipeline(SensingConfig(imuHz = hz)) { out.add(it); accept }
        return p to out
    }

    private fun reading(
        captureMonoNs: Long,
        gyroAgeNs: Long = 0,
        ax: Double = 0.1,
        gx: Double = 0.01,
        accuracy: Long? = 3,
    ) = ImuReading(
        captureMonoNs = captureMonoNs,
        ax = ax, ay = 0.2, az = 9.8,
        gx = gx, gy = 0.02, gz = 0.03,
        accuracy = accuracy,
        gyroAgeNs = gyroAgeNs,
    )

    // -- rate ----------------------------------------------------------------

    @Test
    fun `a commanded rate is honoured over a simulated minute`() {
        val (p, _) = pipeline(hz = 50.0)
        // Offered at 200 Hz for 60 s: four times the commanded rate, so three quarters
        // must be gated.
        var t = 0L
        repeat(60 * 200) {
            p.offer(reading(t))
            t += 5 * ms
        }
        val stats = p.stats
        // 50 Hz for 60 s is 3,000. The gate anchors on the first accepted sample, so one
        // period of slack either way.
        assertTrue("accepted ${stats.accepted}, wanted about 3000", stats.accepted in 2_990..3_010)
        assertTrue("$stats", stats.balances)
    }

    @Test
    fun `a re-commanded rate re-anchors rather than drifting`() {
        val (p, _) = pipeline(hz = 10.0)
        var t = 0L
        repeat(100) { p.offer(reading(t)); t += 10 * ms }
        val atFirstRate = p.stats.accepted

        p.setRate(50.0)
        repeat(100) { p.offer(reading(t)); t += 10 * ms }

        // 10 Hz over the first second takes about 10 of 100; 50 Hz over the second takes
        // about 50. The point is that the second stretch is roughly five times the first,
        // not that either number is exact.
        val atSecondRate = p.stats.accepted - atFirstRate
        assertTrue(
            "first=$atFirstRate second=$atSecondRate: the new rate did not take effect",
            atSecondRate > atFirstRate * 3,
        )
    }

    // -- pairing -------------------------------------------------------------

    @Test
    fun `an accelerometer event with no gyro reading yet is unpaired, not zero-filled`() {
        // Zeros are a reading. A sample carrying gx=gy=gz=0 says "not rotating", which on
        // a vehicle mid-corner is a confident lie, and ImuSample has no null for those
        // axes -- so the only honest answer is to drop it. Counting it is what stops a
        // startup gap from looking like a rate that came up slow.
        val (p, out) = pipeline()
        p.offerUnpaired()
        p.offerUnpaired()

        val stats = p.stats
        assertEquals("nothing may be sent", 0, out.size)
        assertEquals(2, stats.unpaired)
        assertEquals("an unpaired event is not an accepted one", 0, stats.accepted)
        assertEquals("nor a gated one", 0, stats.gated)
        assertTrue("$stats", stats.balances)
    }

    @Test
    fun `the gyro age rides along and is summarised over the samples that went out`() {
        // The pairing is an approximation, so it carries its own error term. Measured on
        // the samples that were actually sent: a gated sample's staleness costs nobody
        // anything, and averaging it in would understate the error on what the Jetson sees.
        val (p, _) = pipeline(hz = 1_000.0)
        // Descending, deliberately. With ages that only ever rise, "keep the maximum" and
        // "keep the last" give the same answer, and a mutation swapping one for the other
        // survived on exactly that.
        p.offer(reading(0, gyroAgeNs = 4 * ms))
        p.offer(reading(10 * ms, gyroAgeNs = 2 * ms))

        val stats = p.stats
        assertEquals(2, stats.accepted)
        assertEquals("the mean of 4 ms and 2 ms", 3 * ms, stats.gyroAgeMeanNs)
        assertEquals("the max must survive a smaller sample after it", 4 * ms, stats.gyroAgeMaxNs)
    }

    @Test
    fun `a gyro half older than one commanded period is counted as stale`() {
        // At 50 Hz a period is 20 ms. A gyro reading older than that was taken before the
        // previous accelerometer event, so the two halves of the sample describe different
        // moments by more than the sample rate itself -- which is the point at which the
        // pairing stops being an approximation and starts being wrong.
        val (p, _) = pipeline(hz = 50.0)
        p.offer(reading(0, gyroAgeNs = 5 * ms))
        // Exactly one period: *not* stale, because "older than one period" is strict. The
        // boundary is asserted rather than stepped over -- with only 5 ms and 25 ms, `>`
        // and `>=` give the same answer and a mutation between them survived.
        p.offer(reading(100 * ms, gyroAgeNs = 20 * ms))
        p.offer(reading(200 * ms, gyroAgeNs = 25 * ms))

        val stats = p.stats
        assertEquals(3, stats.accepted)
        assertEquals(
            "only the 25 ms one is stale at a 20 ms period; 20 ms itself is not",
            1,
            stats.staleGyroSamples,
        )
    }

    // -- stamps and ordering -------------------------------------------------

    @Test
    fun `the sample carries the reading's capture stamp and axes unchanged`() {
        val (p, out) = pipeline()
        p.offer(reading(captureMonoNs = 12_345, ax = 1.5, gx = 0.75, accuracy = 2))

        assertEquals(1, out.size)
        val sample = out.first()
        assertEquals(12_345L, sample.captureMonoNs)
        assertEquals(1.5, sample.ax, 1e-9)
        assertEquals(0.75, sample.gx, 1e-9)
        assertEquals(2L, sample.accuracy)
    }

    @Test
    fun `a null accuracy survives, because not every device reports one`() {
        val (p, out) = pipeline()
        p.offer(reading(1, accuracy = null))
        assertEquals(null, out.first().accuracy)
    }

    @Test
    fun `a capture stamp that goes backwards is counted, not corrected`() {
        // Same reasoning as GpsPipeline's nonMonotonicFixes: the receiver's arithmetic is
        // built on these being monotonic, and a silent correction is a measurement nobody
        // can trust. Counted on `seen`, before the gate, so an out-of-order event that the
        // gate would have dropped anyway is still visible.
        val (p, _) = pipeline(hz = 1_000.0)
        p.offer(reading(100 * ms))
        p.offer(reading(50 * ms))

        assertEquals(1, p.stats.nonMonotonicSamples)
    }

    // -- accounting ----------------------------------------------------------

    @Test
    fun `a sample the transport accepts is delivered, not refused`() {
        // Both halves, because the analogous split in GpsPipeline had only the refusing
        // half tested -- and swapping delivered++ for refusedBySink++ passed 241 tests.
        val (p, out) = pipeline(hz = 1_000.0, accept = true)
        repeat(3) { p.offer(reading(it * 10L * ms)) }

        val stats = p.stats
        assertEquals(3, out.size)
        assertEquals(3, stats.delivered)
        assertEquals(0, stats.refusedBySink)
        assertTrue("$stats", stats.acceptedBalances)
    }

    @Test
    fun `a sample the transport refuses is counted apart from a gated one`() {
        val (p, _) = pipeline(hz = 1_000.0, accept = false)
        repeat(3) { p.offer(reading(it * 10L * ms)) }

        val stats = p.stats
        assertEquals(3, stats.refusedBySink)
        assertEquals(0, stats.delivered)
        assertEquals("a refusal by the sink is not a gate rejection", 0, stats.gated)
        assertTrue("$stats", stats.acceptedBalances)
    }

    @Test
    fun `a sample offered after stop is refused and counted where it happened`() {
        val (p, out) = pipeline()
        p.stop()
        assertFalse(p.offer(reading(1)))
        p.offerUnpaired()

        val stats = p.stats
        assertEquals(0, out.size)
        assertEquals("both paths count against the stop", 2, stats.refusedStopped)
        assertEquals(0, stats.accepted)
        assertEquals("an event after the stop is not unpaired, it is refused", 0, stats.unpaired)
        assertTrue("$stats", stats.balances)
    }

    @Test
    fun `every heading is asserted, not only the sum`() {
        // A sum is blind to a swap between its own terms -- that is how a delivery failure
        // came to be filed as a success elsewhere in this codebase and the identity held
        // perfectly. So each heading gets its own number alongside the balance.
        val (p, _) = pipeline(hz = 100.0)
        p.offer(reading(0))                    // accepted
        p.offer(reading(1 * ms))               // gated: inside the 10 ms period
        p.offerUnpaired()                      // unpaired
        p.stop()
        p.offer(reading(500 * ms))             // refusedStopped

        val stats = p.stats
        assertEquals(4, stats.seen)
        assertEquals(1, stats.accepted)
        assertEquals(1, stats.gated)
        assertEquals(1, stats.unpaired)
        assertEquals(1, stats.refusedStopped)
        assertTrue("$stats", stats.balances)
    }

    @Test
    fun `the balance identities can actually fail`() {
        // The identities are built from the terms they check, so a test that only ever
        // feeds them consistent numbers proves nothing about them.
        assertTrue(
            ImuPipeline.Stats(
                seen = 4, accepted = 1, gated = 1, refusedStopped = 1, unpaired = 1,
                delivered = 1, refusedBySink = 0, nonMonotonicSamples = 0,
                staleGyroSamples = 0, gyroAgeMeanNs = 0, gyroAgeMaxNs = 0,
            ).balances
        )
        assertFalse(
            ImuPipeline.Stats(
                seen = 5, accepted = 1, gated = 1, refusedStopped = 1, unpaired = 1,
                delivered = 1, refusedBySink = 0, nonMonotonicSamples = 0,
                staleGyroSamples = 0, gyroAgeMeanNs = 0, gyroAgeMaxNs = 0,
            ).balances
        )
        assertFalse(
            ImuPipeline.Stats(
                seen = 1, accepted = 1, gated = 0, refusedStopped = 0, unpaired = 0,
                delivered = 0, refusedBySink = 0, nonMonotonicSamples = 0,
                staleGyroSamples = 0, gyroAgeMeanNs = 0, gyroAgeMaxNs = 0,
            ).acceptedBalances
        )
    }

    @Test
    fun `the gyro age mean is zero when nothing was accepted, not a division by zero`() {
        val (p, _) = pipeline()
        p.offerUnpaired()
        assertEquals(0, p.stats.gyroAgeMeanNs)
    }

    @Test
    fun `isStopped tracks stop, and is not a constant`() {
        val (p, _) = pipeline()
        assertFalse("a running pipeline is not stopped", p.isStopped)
        p.stop()
        assertTrue("a stopped pipeline says so", p.isStopped)
    }
}
