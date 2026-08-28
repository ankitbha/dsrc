package com.dsrc.phone.sensors

import com.dsrc.phone.config.SensingConfig
import com.dsrc.transport.GpsRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GpsPipelineTest {

    private val ms = 1_000_000L

    private fun reading(
        fixNs: Long,
        receiptNs: Long = fixNs + 40 * 1_000_000L,
        valid: Boolean = true,
    ) = GpsReading(
        record = if (valid) {
            GpsPipeline.record(
                fixMonoNs = fixNs, latitude = 51.5074, longitude = -0.1278,
                speedMps = 13.4, headingDeg = 91.2, satellites = 9,
                hdop = 0.9, altitudeM = 35.0, utcEpochNs = 1_755_648_000_000_000_000,
            )
        } else {
            GpsRecord.noFix(fixNs)
        },
        fixMonoNs = fixNs,
        receiptMonoNs = receiptNs,
    )

    private fun pipeline(hz: Double = 1.0, accept: Boolean = true): Pair<GpsPipeline, MutableList<GpsReading>> {
        val sent = mutableListOf<GpsReading>()
        val pipe = GpsPipeline(SensingConfig(gpsHz = hz)) { sent.add(it); accept }
        return pipe to sent
    }

    // -- both clocks ---------------------------------------------------------

    @Test
    fun `fix time and receipt time are both carried and differ`() {
        // The task asks for both, and neither substitutes for the other: the difference is
        // the location stack's own latency.
        val r = reading(fixNs = 1_000 * ms, receiptNs = 1_040 * ms)
        assertEquals(1_000 * ms, r.fixMonoNs)
        assertEquals(1_040 * ms, r.receiptMonoNs)
        assertEquals(40 * ms, r.deliveryLatencyNs)
    }

    @Test
    fun `only the fix time reaches the wire`() {
        // The frozen contract has no receipt field; t_capture_mono_ns is the fix.
        val r = reading(fixNs = 7_000, receiptNs = 9_000)
        assertEquals(7_000, r.record.captureMonoNs)
    }

    @Test
    fun `receipt is never before the fix`() {
        val r = reading(fixNs = 1_000 * ms)
        assertTrue("receipt $r", r.receiptMonoNs >= r.fixMonoNs)
        assertTrue(r.deliveryLatencyNs >= 0)
    }

    // -- the no-fix path -----------------------------------------------------

    @Test
    fun `a missing position maps to the all-null record`() {
        val record = GpsPipeline.record(
            fixMonoNs = 5, latitude = null, longitude = null, speedMps = 3.0,
            headingDeg = 90.0, satellites = 4, hdop = 2.0, altitudeM = 10.0,
            utcEpochNs = 1_000,
        )
        assertFalse("without a position there is no fix", record.valid)
        assertEquals(0, record.fixQuality)
        assertEquals(0, record.satellites)
        assertNull("everything else must be dropped, not partially kept", record.speedMps)
        assertNull(record.headingDeg)
        assertNull(record.hdop)
        assertNull(record.altitudeM)
        assertNull(record.utcEpochNs)
        assertEquals("the stamp is still ours to report", 5, record.captureMonoNs)
    }

    @Test
    fun `half a position is still no position`() {
        for (pair in listOf(51.5 to null, null to -0.1)) {
            val record = GpsPipeline.record(
                fixMonoNs = 1, latitude = pair.first, longitude = pair.second,
                speedMps = null, headingDeg = null, satellites = 0, hdop = null,
                altitudeM = null, utcEpochNs = null,
            )
            assertFalse("lat=${pair.first} lon=${pair.second}", record.valid)
        }
    }

    @Test
    fun `a valid fix never carries fix_quality zero`() {
        // Zero is the wire's "no fix", so a valid record must not use it.
        val record = GpsPipeline.record(
            fixMonoNs = 1, latitude = 51.5, longitude = -0.1, speedMps = null,
            headingDeg = null, satellites = 0, hdop = null, altitudeM = null, utcEpochNs = null,
        )
        assertTrue(record.valid)
        assertTrue("fix_quality was ${record.fixQuality}", record.fixQuality > 0)
    }

    @Test
    fun `a fix with no speed or altitude is still a fix`() {
        // Stationary devices report no speed or heading, and a 2D fix has no altitude.
        val record = GpsPipeline.record(
            fixMonoNs = 1, latitude = 51.5, longitude = -0.1, speedMps = null,
            headingDeg = null, satellites = 5, hdop = 1.2, altitudeM = null, utcEpochNs = 99,
        )
        assertTrue(record.valid)
        assertNull(record.speedMps)
        assertNull(record.altitudeM)
        assertEquals(5, record.satellites)
    }

    // -- rate ----------------------------------------------------------------

    @Test
    fun `the commanded rate is honoured and the same gate is used as the camera`() {
        // One rate implementation across modalities: a second would drift from the first,
        // and this is the piece that took four attempts to get right.
        val (pipe, sent) = pipeline(hz = 1.0)
        // A platform delivering at 5 Hz into a 1 Hz target.
        for (i in 0 until 50) pipe.offer(reading(fixNs = i * 200 * ms))
        assertEquals(50, pipe.stats.seen)
        assertTrue("expected about 10, got ${sent.size}", sent.size in 9..11)
        assertEquals(sent.size.toLong(), pipe.stats.accepted)
    }

    @Test
    fun `a rate change takes effect`() {
        val (pipe, _) = pipeline(hz = 0.2)
        pipe.offer(reading(fixNs = 0))
        pipe.setRate(10.0)
        assertEquals(10.0, pipe.rateHz, 0.0)
        assertTrue("the new period is 100 ms", pipe.offer(reading(fixNs = 100 * ms)))
    }

    // -- accounting ----------------------------------------------------------

    @Test
    fun `every reading is under exactly one heading`() {
        val (pipe, _) = pipeline(hz = 1.0)
        for (i in 0 until 40) pipe.offer(reading(fixNs = i * 250 * ms))
        pipe.stop()
        for (i in 40 until 50) pipe.offer(reading(fixNs = i * 250 * ms))
        val stats = pipe.stats
        assertEquals(50, stats.seen)
        assertEquals(10, stats.refusedStopped)
        assertTrue("$stats", stats.balances)
    }

    @Test
    fun `a reading the transport refuses is counted apart from a gated one`() {
        // Unrelated causes: one is the commanded rate, the other a full queue or a dead
        // session.
        val (pipe, _) = pipeline(hz = 1000.0, accept = false)
        for (i in 1..5) pipe.offer(reading(fixNs = i * 10 * ms))
        val stats = pipe.stats
        assertEquals(5, stats.accepted)
        assertEquals(5, stats.refusedBySink)
        assertEquals(0, stats.delivered)
        assertEquals(0, stats.gated)
        assertTrue("$stats", stats.acceptedBalances)
    }

    @Test
    fun `the balance identities can actually fail`() {
        // Anti-vacuity: with every term counted where it happens rather than derived,
        // an inconsistent set must be visible as such.
        val consistent = GpsPipeline.Stats(
            seen = 10, accepted = 4, gated = 5, refusedStopped = 1,
            delivered = 3, refusedBySink = 1, invalidFixes = 0, nonMonotonicFixes = 0,
        )
        assertTrue(consistent.balances)
        assertTrue(consistent.acceptedBalances)
        assertFalse(consistent.copy(seen = 11).balances)
        assertFalse(consistent.copy(gated = 4).balances)
        assertFalse(consistent.copy(delivered = 2).acceptedBalances)
    }

    @Test
    fun `an invalid fix is forwarded and counted, not discarded`() {
        // The Jetson needs to know the phone has no fix; silence and no-fix are different
        // facts, and only one of them is actionable.
        val (pipe, sent) = pipeline(hz = 1000.0)
        pipe.offer(reading(fixNs = 10 * ms, valid = false))
        assertEquals(1, sent.size)
        assertFalse(sent.first().record.valid)
        assertEquals(1, pipe.stats.invalidFixes)
    }

    @Test
    fun `an out-of-order fix stamp is counted rather than hidden`() {
        // The receiver's freshness arithmetic assumes these stamps are monotonic, so a
        // platform handing us an out-of-order update is worth surfacing.
        val (pipe, _) = pipeline(hz = 1000.0)
        pipe.offer(reading(fixNs = 100 * ms))
        pipe.offer(reading(fixNs = 50 * ms))
        assertEquals(1, pipe.stats.nonMonotonicFixes)
    }

    @Test
    fun `a reversal between two gated fixes is counted too`() {
        // Every other test here runs at 1000 Hz, where the gate never gates -- so
        // whether the baseline advanced before or after it was unobservable, and it
        // advanced after. A reversal entirely inside a gated stretch counted nothing.
        //
        // The rate gate can only ever LOWER a rate: it drops fixes the provider has
        // already delivered. So on any drive where the commanded gps_hz is below what
        // the platform produces -- the normal case -- this is the only shape a
        // reversal can take. What is being detected is a property of the delivery, and
        // the gate has no business in it.
        val (pipe, _) = pipeline(hz = 1.0)
        pipe.offer(reading(fixNs = 0))            // accepted, opens the gate's period
        pipe.offer(reading(fixNs = 500 * ms))     // gated
        pipe.offer(reading(fixNs = 400 * ms))     // gated, and out of order
        assertEquals(2, pipe.stats.gated)
        assertEquals(1, pipe.stats.nonMonotonicFixes)
    }

    @Test
    fun `a stopped pipeline forwards nothing`() {
        val (pipe, sent) = pipeline(hz = 1000.0)
        pipe.stop()
        assertFalse(pipe.offer(reading(fixNs = 10 * ms)))
        assertTrue(sent.isEmpty())
    }

    // -- the wire form -------------------------------------------------------

    @Test
    fun `a produced record survives the wire round trip`() {
        val record = GpsPipeline.record(
            fixMonoNs = 1_000_000_001, latitude = 51.5074, longitude = -0.1278,
            speedMps = 13.4, headingDeg = 91.2, satellites = 9, hdop = 0.9,
            altitudeM = 35.0, utcEpochNs = 1_755_648_000_000_000_000,
        )
        assertEquals(record, GpsRecord.fromWire(record.toExtensions(), ByteArray(0)))
    }

    @Test
    fun `a no-fix record survives the wire round trip`() {
        val record = GpsRecord.noFix(1_000_000_002)
        assertEquals(record, GpsRecord.fromWire(record.toExtensions(), ByteArray(0)))
    }

    @Test
    fun `isStopped tracks stop, and is not a constant`() {
        // The whole stats-ordering claim in SensingService rests on this accessor, and no
        // test named it: reducing it to `get() = true` left 241 JVM and all 53 instrumented
        // tests green while fully restoring the defect it guards -- abandoned,
        // refusedStopped and the buffer's discarded structurally zero at the only place
        // production reads them. Both ends asserted, because `= true` and `= false` are
        // each satisfied by half of this.
        val (p, _) = pipeline()
        assertFalse("a running pipeline is not stopped", p.isStopped)
        p.stop()
        assertTrue("a stopped pipeline says so", p.isStopped)
    }


    @Test
    fun `a reading the transport accepts is counted as delivered, not as refused`() {
        // The other half, and it was missing. The only test touching this split runs with
        // accept = false and asserts refusedBySink == 5, delivered == 0 -- both of which a
        // mutant that increments refusedBySink on the success path satisfies exactly. So
        // every delivered fix could be reported as refused by the transport, permanently,
        // with 241 tests green. `acceptedBalances` cannot see it either: it is a sum over
        // the two terms, and a transfer between them leaves the sum alone.
        //
        // The same swaps in CameraFrameSender and CameraPipeline are both caught; GPS was
        // the one modality without the pin.
        val (p, delivered) = pipeline(accept = true)
        repeat(3) { p.offer(reading(it * 1_000_000_000L)) }

        val stats = p.stats
        assertEquals("the sink took them", 3, delivered.size)
        assertEquals("and they are counted as delivered", 3, stats.delivered)
        assertEquals("with nothing refused", 0, stats.refusedBySink)
        assertTrue("$stats", stats.acceptedBalances)
    }

}
