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
        assertEquals("and the direction gets its own counter", 1, pairing.outOfOrderPairings.get())
    }

    @Test
    fun `an in-order pairing is not counted as out of order`() {
        val pairing = ImuPairing()
        pairing.onGyro(0, 0.0, 0.0, 0.0)
        accel(pairing, 10 * ms)
        assertEquals(0, pairing.outOfOrderPairings.get())
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
        assertEquals(2, pairing.refusedWrongTimebase.get())

        // Recorded on the REFUSING branch, which is the branch that has a reader: the log
        // line in stopBecauseOfTimebase is the only diagnostic a session that lost its IMU
        // produces, and it prints these two. Assigning them only on the matched path left
        // both mutations alive and that line printing zeroes -- a fix whose whole purpose
        // was to explain a refusal, blank on refusal.
        assertEquals("the delta that decided it", -10 * second, pairing.timebaseOffsetNs)
        assertEquals("and the gap it was weighed against", 0, pairing.clockGapNs)
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
        assertEquals(0, pairing.refusedWrongTimebase.get())
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

    @Test
    fun `a stamp near neither clock is refused, not attributed to the closer one`() {
        // The "or near neither" half of the attribution, and the conjunct no test reached:
        // dropping `fromApp <= maxDeliveryNs` from the diverged branch survived 279 tests.
        // A delivery delta of 3 s against a 10 s clock gap is nearer the app clock than the
        // monotonic one, and is still not delivery latency -- it is a stamp from neither.
        assertEquals(
            ImuTimebase.MISMATCHED,
            ImuPairing.verdictFor(deliveryDeltaNs = 3 * second, clockGapNs = 10 * second),
        )
        // And the same shape inside the bound *is* attributed, so the assertion above is
        // about the bound rather than about the comparison next to it.
        assertEquals(
            ImuTimebase.MATCHED,
            ImuPairing.verdictFor(deliveryDeltaNs = ms, clockGapNs = 10 * second),
        )
    }

    @Test
    fun `the clock-gap threshold decides, and both sides of it are asserted`() {
        // 50 ms was justified in a comment and asserted by nothing, so the constant could
        // move without a test noticing -- and it is the number that decides whether the
        // attribution is attempted at all.
        //
        // At exactly the threshold the question is still moot, so a delta that would lose
        // the attribution is accepted. One nanosecond above it, the attribution runs and
        // that same delta is refused.
        // Literal 50 ms, not the constant. Deriving the inputs from the value under test
        // makes the test move with it: widening the threshold tenfold left this passing,
        // because `halfway` widened too. A test that reads its own subject cannot pin it.
        assertEquals(
            "the constant is 50 ms, and this test is about that number",
            50_000_000L,
            ImuPairing.MAX_TOLERABLE_CLOCK_GAP_NS,
        )
        val halfway = 26 * ms
        assertEquals(
            "at the threshold the clocks are close enough that it does not matter",
            ImuTimebase.MATCHED,
            ImuPairing.verdictFor(deliveryDeltaNs = halfway, clockGapNs = 50 * ms),
        )
        assertEquals(
            "one nanosecond further apart, the attribution runs and this stamp loses it",
            ImuTimebase.MISMATCHED,
            ImuPairing.verdictFor(deliveryDeltaNs = halfway, clockGapNs = 50 * ms + 1),
        )
    }

    @Test
    fun `a tie between the two clocks is refused rather than guessed`() {
        // delta exactly half the gap is equidistant. Refusing is the right call -- there is
        // no evidence either way -- and `<` rather than `<=` is what holds it there, which
        // nothing asserted.
        // Inside the delivery bound, or the bound decides first and the comparison is never
        // reached -- which is what a 10 s gap did: 5 s of "latency" fails the 2 s bound
        // whatever the tie says. A 2 s gap puts the midpoint at 1 s, comfortably inside it.
        val gap = 2 * second
        assertEquals(
            ImuTimebase.MISMATCHED,
            ImuPairing.verdictFor(deliveryDeltaNs = gap / 2, clockGapNs = gap),
        )
    }

    @Test
    fun `the delivery bound is honoured on the diverged branch too, and is injectable`() {
        // The `maxDeliveryNs` constructor seam existed and no test used it: every test
        // built ImuPairing(), so the mutation replacing the parameter with the constant
        // survived. Driven here through an instance, which is the only thing that reaches
        // that parameter at all.
        // On the DIVERGED branch, which the first version never reached: it drove
        // appNowNs == monoNowNs, so the gap was zero, the moot branch decided, and the
        // injected bound never met the conjunct it exists for. A one-second gap puts this
        // past the 50 ms threshold and into the attribution.
        val gap = second
        val tight = ImuPairing(maxDeliveryNs = 10 * ms)
        tight.onGyro(0, 0.0, 0.0, 0.0)
        assertEquals(
            "20 ms of latency is outside a 10 ms bound even when it is the nearer clock",
            ImuOutcome.WrongTimebase,
            tight.onAccelerometer(0, 0.1, 0.2, 9.8, 3, appNowNs = 20 * ms, monoNowNs = 20 * ms - gap),
        )

        val loose = ImuPairing()
        loose.onGyro(0, 0.0, 0.0, 0.0)
        assertTrue(
            "the same event is accepted under the default bound",
            loose.onAccelerometer(0, 0.1, 0.2, 9.8, 3, appNowNs = 20 * ms, monoNowNs = 20 * ms - gap)
                is ImuOutcome.Paired,
        )
    }


    @Test
    fun `the accelerometer's accuracy reaches the reading`() {
        // Pinned only where the test built the ImuReading itself, so the class that
        // actually carries the value off the event was uncovered: replacing it with null
        // survived.
        val pairing = ImuPairing()
        pairing.onGyro(0, 0.0, 0.0, 0.0)
        assertEquals(
            2L,
            (pairing.onAccelerometer(ms, 0.1, 0.2, 9.8, accuracy = 2, appNowNs = 2 * ms, monoNowNs = 2 * ms)
                as ImuOutcome.Paired).reading.accuracy,
        )
    }

    @Test
    fun `the timebase is decided even when there is no gyro to pair with`() {
        // The order of the two guards was unpinned, and swapping them defers the decision
        // until the first pairable event -- so a session whose gyroscope never registers
        // never runs the gate at all, and a wrong clock goes unreported on exactly the
        // device that has something else wrong with it too.
        val pairing = ImuPairing()
        assertEquals(
            ImuOutcome.WrongTimebase,
            pairing.onAccelerometer(10 * second, 0.1, 0.2, 9.8, 3, appNowNs = 0, monoNowNs = 0),
        )
        assertEquals(ImuTimebase.MISMATCHED, pairing.timebase)
    }

    @Test
    fun `a negative clock gap gives the same verdict either way it is read`() {
        // Round 3 asked whether a large negative gap "fails open" through the moot branch.
        // It does reach that branch -- and it makes no difference, which is worth an
        // assertion rather than an argument, because a magnitude guard was written here on
        // the strength of the concern and had to come back out.
        //
        // With the gap negative, `fromMono = delta + |gap|` always exceeds `fromApp =
        // delta`, so the attribution branch reduces to the same `delta <= maxDelivery` test
        // the moot branch applies. Both readings agree for every negative gap.
        for (delta in listOf(ms, 3 * second, 90 * second)) {
            assertEquals(
                "delta=$delta: the two readings of a negative gap must agree",
                ImuPairing.verdictFor(deliveryDeltaNs = delta, clockGapNs = 0),
                ImuPairing.verdictFor(deliveryDeltaNs = delta, clockGapNs = -90 * second),
            )
        }
    }


    @Test
    fun `all six axes reach the reading in their own fields`() {
        // The layer above `sample()`, and it had the same hole: gy = gz and ay = az both
        // survived the suite. Pinning only the pipeline would leave this one open, which is
        // why the axis family needs an assertion at each layer rather than one at the end.
        val pairing = ImuPairing()
        pairing.onGyro(0, 4.0, 5.0, 6.0)
        val reading = (pairing.onAccelerometer(
            captureNs = ms, x = 1.0, y = 2.0, z = 3.0,
            accuracy = 3, appNowNs = 2 * ms, monoNowNs = 2 * ms,
        ) as ImuOutcome.Paired).reading

        assertEquals(1.0, reading.ax, 1e-9)
        assertEquals(2.0, reading.ay, 1e-9)
        assertEquals(3.0, reading.az, 1e-9)
        assertEquals(4.0, reading.gx, 1e-9)
        assertEquals(5.0, reading.gy, 1e-9)
        assertEquals(6.0, reading.gz, 1e-9)
    }

}
