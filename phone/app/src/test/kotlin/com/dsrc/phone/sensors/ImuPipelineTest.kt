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
        // Bounded on both sides. `> first * 3` admits a gate running at ten times the
        // commanded rate exactly as readily as one running at the commanded rate, so it
        // did not distinguish "the new rate took effect" from "the rate is wrong".
        val atSecondRate = p.stats.accepted - atFirstRate
        assertTrue(
            "first=$atFirstRate second=$atSecondRate: wanted about 5x, which is 50 Hz over 10",
            atSecondRate in (atFirstRate * 4)..(atFirstRate * 6),
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
        // Distinct counts per heading, and that is the whole point. The first version drove
        // all four to 1, which made the four assertions mutually interchangeable: swapping
        // `gated` and `unpaired` in the production code -- a true swap, both directions --
        // passed this test alone with the mutation in place. A test named for swap-blindness
        // that is itself swap-blind is worse than none, because it is cited as coverage.
        val (p, _) = pipeline(hz = 100.0)
        p.offer(reading(0))                    // accepted: 1
        repeat(2) { p.offer(reading(1L * ms)) } // gated: 2, inside the 10 ms period
        repeat(3) { p.offerUnpaired() }        // unpaired: 3
        p.stop()
        repeat(4) { p.offer(reading(500 * ms)) } // refusedStopped: 4

        val stats = p.stats
        assertEquals(10, stats.seen)
        assertEquals(1, stats.accepted)
        assertEquals(2, stats.gated)
        assertEquals(3, stats.unpaired)
        assertEquals(4, stats.refusedStopped)
        assertTrue("$stats", stats.balances)
    }

    @Test
    fun `the pairing statistics cover the samples that went out, not the ones gated away`() {
        // Every test carrying a nonzero age ran at a rate where nothing was gated, so
        // `accepted == seen` and "measured on the accepted" was indistinguishable from
        // "measured on everything". Moving the statistics block above the gate, or dividing
        // by `seen`, both survived the suite.
        //
        // At 10 Hz with offers 1 ms apart, only the first of each burst is accepted.
        val (p, _) = pipeline(hz = 10.0)
        p.offer(reading(0, gyroAgeNs = 2 * ms))            // accepted
        // 150 ms, comfortably past the 100 ms period. At exactly one period these are not
        // stale under `>` wherever the counter sits, so the stale assertion below was inert
        // and moving the whole block above the gate survived.
        p.offer(reading(1 * ms, gyroAgeNs = 150 * ms))     // gated, and very stale
        p.offer(reading(2 * ms, gyroAgeNs = 150 * ms))     // gated, and very stale

        val stats = p.stats
        assertEquals(1, stats.accepted)
        assertEquals(2, stats.gated)
        assertEquals("only the accepted sample's age counts", 2 * ms, stats.gyroAgeMeanNs)
        assertEquals(2 * ms, stats.gyroAgeMaxNs)
        assertEquals("a gated sample's staleness costs nobody anything", 0, stats.staleGyroSamples)
    }

    @Test
    fun `a reversal wholly inside a gated run is still seen`() {
        // The comment claimed an out-of-order event the gate would have dropped "is still
        // visible". The count sits before the gate, but the baseline only advanced on an
        // accepted sample, so a reversal between two gated events was invisible: at 10 Hz,
        // offers at 0, 50 ms, 40 ms gave nonMonotonicSamples = 0.
        val (p, _) = pipeline(hz = 10.0)
        p.offer(reading(0))          // accepted, anchors the gate
        p.offer(reading(50 * ms))    // gated
        p.offer(reading(40 * ms))    // gated, and a reversal against the one before it

        assertEquals("a reversal between two gated events is still a reversal", 1, p.stats.nonMonotonicSamples)
    }

    @Test
    fun `two events at the same instant are not a reversal`() {
        // The boundary was stepped over: no test ever offered a duplicate stamp, so `<` and
        // `<=` gave the same answer. Equal stamps are two events in the same nanosecond,
        // which is a resolution limit rather than an ordering failure.
        val (p, _) = pipeline(hz = 1_000.0)
        p.offer(reading(100 * ms))
        p.offer(reading(100 * ms))

        assertEquals(0, p.stats.nonMonotonicSamples)
    }

    @Test
    fun `the balance identities can actually fail`() {
        // The identities are built from the terms they check, so a test that only ever
        // feeds them consistent numbers proves nothing about them.
        assertTrue(
            ImuPipeline.Stats(
                seen = 4, accepted = 1, gated = 1, refusedStopped = 1, unpaired = 1,
                delivered = 1, refusedBySink = 0, nonMonotonicSamples = 0,
                staleGyroSamples = 0, gyroAgeMeanNs = 0, gyroAgeMaxNs = 0, rateHz = 50.0,
            ).balances
        )
        assertFalse(
            ImuPipeline.Stats(
                seen = 5, accepted = 1, gated = 1, refusedStopped = 1, unpaired = 1,
                delivered = 1, refusedBySink = 0, nonMonotonicSamples = 0,
                staleGyroSamples = 0, gyroAgeMeanNs = 0, gyroAgeMaxNs = 0, rateHz = 50.0,
            ).balances
        )
        assertFalse(
            ImuPipeline.Stats(
                seen = 1, accepted = 1, gated = 0, refusedStopped = 0, unpaired = 0,
                delivered = 0, refusedBySink = 0, nonMonotonicSamples = 0,
                staleGyroSamples = 0, gyroAgeMeanNs = 0, gyroAgeMaxNs = 0, rateHz = 50.0,
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

    @Test
    fun `the stats carry the rate they were measured at, and it follows a re-command`() {
        // Added so staleGyroSamples is interpretable, then asserted by nothing -- which is
        // the shape this whole round was about, so it would have been a poor place to stop.
        // A constant survived every other test.
        val (p, _) = pipeline(hz = 50.0)
        assertEquals(50.0, p.stats.rateHz, 1e-9)
        p.setRate(10.0)
        assertEquals(
            "the rate on the record must be the one in force, not the one at construction",
            10.0,
            p.stats.rateHz,
            1e-9,
        )
    }


    @Test
    fun `staleness is judged against the live period, not a fixed one`() {
        // Pinned at 50 Hz only, where a literal 20 ms agrees with the real period -- so
        // hard-coding that literal survived. The same age is stale at one rate and not at
        // another, which is the whole reason `rateHz` is on the record.
        val slow = pipeline(hz = 10.0).first          // 100 ms period
        slow.offer(reading(0, gyroAgeNs = 50 * ms))
        assertEquals("50 ms is well inside a 100 ms period", 0, slow.stats.staleGyroSamples)

        val fast = pipeline(hz = 100.0).first         // 10 ms period
        fast.offer(reading(0, gyroAgeNs = 50 * ms))
        assertEquals("the same age is stale at 100 Hz", 1, fast.stats.staleGyroSamples)
    }

    @Test
    fun `the balance identities fail in both directions`() {
        // Pinned in one direction only: every case supplied a left side that was too large,
        // so weakening `==` to `<=` survived and a sum EXCEEDING seen -- one event counted
        // under two headings, which is what a double-count looks like -- went unnoticed.
        assertFalse(
            "the parts must not exceed the whole",
            ImuPipeline.Stats(
                seen = 2, accepted = 2, gated = 1, refusedStopped = 0, unpaired = 0,
                delivered = 2, refusedBySink = 0, nonMonotonicSamples = 0,
                staleGyroSamples = 0, gyroAgeMeanNs = 0, gyroAgeMaxNs = 0, rateHz = 50.0,
            ).balances
        )
        assertFalse(
            ImuPipeline.Stats(
                seen = 1, accepted = 1, gated = 0, refusedStopped = 0, unpaired = 0,
                delivered = 1, refusedBySink = 1, nonMonotonicSamples = 0,
                staleGyroSamples = 0, gyroAgeMeanNs = 0, gyroAgeMaxNs = 0, rateHz = 50.0,
            ).acceptedBalances
        )
    }

    @Test
    fun `the reversal baseline is the previous event, not a high-water mark`() {
        // Unpinned in one direction: a high-water mark survived, and it counts a different
        // thing -- every event below the maximum ever seen, rather than every event below
        // the one before it. The documented policy is about delivery order.
        val (p, _) = pipeline(hz = 1_000.0)
        p.offer(reading(0))
        p.offer(reading(50 * ms))
        p.offer(reading(40 * ms))     // a reversal against 50 ms
        p.offer(reading(45 * ms))     // an advance against 40 ms, but below the 50 ms peak

        assertEquals(
            "a high-water mark would count two; the previous event counts one",
            1,
            p.stats.nonMonotonicSamples,
        )
    }

}
