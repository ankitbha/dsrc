package com.dsrc.phone.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The decisions [ImuSource] used to make inside a callback nothing could invoke.
 *
 * Round 1 deleted the gyroscope registration, the unregistration, the unpaired branch and
 * the timebase gate all at once and watched 267 JVM plus 53 instrumented tests stay green
 * while the `imu` channel transmitted nothing for a whole session. These are the tests
 * that make that impossible.
 */
class ImuPairingTest {

    private val ms = 1_000_000L
    private val second = 1_000_000_000L

    /** Clocks agreeing: no accumulated suspend, delivery a millisecond after capture. */
    private fun accel(
        pairing: ImuPairing,
        captureNs: Long,
        appNowNs: Long = captureNs + ms,
        monoNowNs: Long = captureNs + ms,
        accuracy: Long? = 3,
    ) = pairing.onAccelerometer(captureNs, 0.1, 0.2, 9.8, accuracy, appNowNs, monoNowNs)

    // -- pairing -------------------------------------------------------------

    @Test
    fun `a sample carries the latest gyro reading, not the first`() {
        // The mutation that kept the *first* reading survived the whole suite, because
        // nothing ever supplied a second one.
        val pairing = ImuPairing()
        pairing.onGyro(0, 0.01, 0.02, 0.03)
        pairing.onGyro(10 * ms, 0.44, 0.55, 0.66)

        val outcome = accel(pairing, 20 * ms)
        val reading = (outcome as ImuOutcome.Paired).reading
        assertEquals(0.44, reading.gx, 1e-9)
        assertEquals(0.66, reading.gz, 1e-9)
        assertEquals("the age is measured from the latest gyro, not the first", 10 * ms, reading.gyroAgeNs)
    }

    @Test
    fun `an accelerometer event before any gyro event is unpaired`() {
        assertEquals(ImuOutcome.Unpaired, accel(ImuPairing(), 5 * ms))
    }

    @Test
    fun `the sample's capture stamp is the event's, not the delivery instant`() {
        // Driven apart deliberately: with the two equal this assertion cannot fail, which
        // is the trap that made a camera receipt test unfailable in an earlier round.
        val pairing = ImuPairing()
        pairing.onGyro(0, 0.0, 0.0, 0.0)
        val reading = (accel(pairing, captureNs = 7 * ms, appNowNs = 900 * ms, monoNowNs = 900 * ms)
            as ImuOutcome.Paired).reading
        assertEquals(7 * ms, reading.captureMonoNs)
    }

    @Test
    fun `a gyro stamped after the accelerometer gives a magnitude, and is counted`() {
        // Both sensors register on one handler at the same commanded period, so a gyro
        // sample captured after an accelerometer sample can be delivered before it. Left
        // signed, the age cancelled real error in the mean: +18 ms and -18 ms averaged to
        // zero and reported a perfect pairing.
        val pairing = ImuPairing()
        pairing.onGyro(30 * ms, 0.0, 0.0, 0.0)
        val reading = (accel(pairing, captureNs = 12 * ms) as ImuOutcome.Paired).reading

        assertEquals("the statistic is about magnitude", 18 * ms, reading.gyroAgeNs)
        assertEquals("and the direction gets its own counter", 1, pairing.outOfOrderPairings)
    }

    @Test
    fun `an in-order pairing is not counted as out of order`() {
        val pairing = ImuPairing()
        pairing.onGyro(0, 0.0, 0.0, 0.0)
        accel(pairing, 10 * ms)
        assertEquals(0, pairing.outOfOrderPairings)
    }

    // -- the timebase --------------------------------------------------------

    @Test
    fun `a wrong timebase refuses, counts, and keeps refusing`() {
        // Sticky, so the decision is taken once. And counted, because it was not: a
        // wrong-clock session and a device with no IMU at all produced identical
        // statistics, distinguishable only by a field in a log line.
        val pairing = ImuPairing()
        pairing.onGyro(0, 0.0, 0.0, 0.0)

        // Delivery stamp far in the past relative to the event: not one clock.
        assertEquals(ImuOutcome.WrongTimebase, accel(pairing, captureNs = 10 * second, appNowNs = 0, monoNowNs = 0))
        assertEquals(ImuTimebase.MISMATCHED, pairing.timebase)

        // A later event that would have looked fine is still refused.
        assertEquals(ImuOutcome.WrongTimebase, accel(pairing, captureNs = 20 * second))
        assertEquals(2, pairing.refusedWrongTimebase)
    }

    @Test
    fun `a matching timebase decides once and stops re-deciding`() {
        val pairing = ImuPairing()
        pairing.onGyro(0, 0.0, 0.0, 0.0)
        assertTrue(accel(pairing, 10 * ms) is ImuOutcome.Paired)
        assertEquals(ImuTimebase.MATCHED, pairing.timebase)

        // A later event whose delta would fail on its own is accepted, because the
        // question was settled. Without the sticky branch this would be refused.
        assertTrue(accel(pairing, captureNs = 0, appNowNs = 10 * second, monoNowNs = 10 * second) is ImuOutcome.Paired)
        assertEquals(0, pairing.refusedWrongTimebase)
    }

    @Test
    fun `the monotonic-clock bug is caught once the clocks have diverged`() {
        // The bug this exists for. SensorEvent.timestamp on System.nanoTime rather than
        // elapsedRealtimeNanos does not differ by an epoch -- it differs by the device's
        // accumulated suspend time, which starts at zero and grows. A phone that had
        // suspended for 90 s stamps 90 s in the past relative to the app clock.
        val suspended = 90 * second
        val capture = 100 * second
        val appNow = capture + suspended + ms   // app clock: boottime, so ahead by the suspend
        val monoNow = capture + ms              // the clock the sensor is actually using

        assertEquals(
            "an event on the monotonic clock must not be read as elapsedRealtime",
            ImuTimebase.MISMATCHED,
            ImuPairing.verdictFor(
                deliveryDeltaNs = appNow - capture,
                clockGapNs = appNow - monoNow,
            ),
        )
    }

    @Test
    fun `a modest suspend is exactly the case the old bound admitted`() {
        // 1.5 s of accumulated suspend sat comfortably inside a two-second bound and was
        // accepted, and every sample after it carried a silent constant 1.5 s error --
        // about 21 m of road at 50 km/h.
        val gap = 1_500 * ms
        assertEquals(
            ImuTimebase.MISMATCHED,
            ImuPairing.verdictFor(deliveryDeltaNs = gap + ms, clockGapNs = gap, maxDeliveryNs = 2 * second),
        )
    }

    @Test
    fun `when the two clocks have barely diverged the question is moot, and it says yes`() {
        // The ambiguous case is the harmless one: whichever clock the sensor uses, the
        // stamp is right to within the gap. Refusing here would stop the IMU on a
        // freshly-booted device for an error of a few milliseconds.
        assertEquals(
            ImuTimebase.MATCHED,
            ImuPairing.verdictFor(deliveryDeltaNs = 5 * ms, clockGapNs = 3 * ms),
        )
    }

    @Test
    fun `a stamp ahead of a clock read after it is refused whatever the gap`() {
        for (gap in listOf(0L, 10 * second)) {
            assertEquals(
                ImuTimebase.MISMATCHED,
                ImuPairing.verdictFor(deliveryDeltaNs = -ms, clockGapNs = gap),
            )
        }
    }

    @Test
    fun `both sides of the delivery bound are asserted`() {
        // Asserting only the accepting side passes for a function that accepts everything,
        // and only the refusing side for one that refuses everything.
        assertEquals(
            ImuTimebase.MATCHED,
            ImuPairing.verdictFor(ImuPairing.MAX_PLAUSIBLE_DELIVERY_NS, clockGapNs = 0),
        )
        assertEquals(
            ImuTimebase.MISMATCHED,
            ImuPairing.verdictFor(ImuPairing.MAX_PLAUSIBLE_DELIVERY_NS + 1, clockGapNs = 0),
        )
    }

    @Test
    fun `the clock gap is recorded on the accepting branch too`() {
        // An accepting verdict still has to say how wrong it could have been. The previous
        // version computed the offset and then discarded its magnitude, so 3 ms and 1.9 s
        // were indistinguishable to everything downstream.
        val pairing = ImuPairing()
        pairing.onGyro(0, 0.0, 0.0, 0.0)
        accel(pairing, captureNs = 0, appNowNs = 5 * ms, monoNowNs = 2 * ms)

        assertEquals(ImuTimebase.MATCHED, pairing.timebase)
        assertEquals(5 * ms, pairing.timebaseOffsetNs)
        assertEquals("the gap a wrong attribution would have cost", 3 * ms, pairing.clockGapNs)
    }
}
