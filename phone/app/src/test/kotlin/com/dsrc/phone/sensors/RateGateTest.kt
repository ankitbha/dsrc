package com.dsrc.phone.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RateGateTest {

    private val ms = 1_000_000L

    @Test
    fun `the first frame is always accepted`() {
        assertTrue(RateGate(10.0).accept(0L))
        assertTrue(RateGate(10.0).accept(123_456_789L))
    }

    @Test
    fun `a frame inside the period is rejected`() {
        val gate = RateGate(10.0)
        assertTrue(gate.accept(0))
        assertFalse(gate.accept(50 * ms))
        assertFalse(gate.accept(99 * ms))
    }

    @Test
    fun `a frame at the period boundary is accepted`() {
        val gate = RateGate(10.0)
        assertTrue(gate.accept(0))
        assertTrue(gate.accept(100 * ms))
    }

    @Test
    fun `a source that is an exact multiple hits the target rate exactly`() {
        // 30 Hz source, 10 Hz target: every third frame, no drift over a minute.
        val gate = RateGate(10.0)
        var accepted = 0
        val sourcePeriod = 1_000_000_000L / 30
        for (i in 0 until 1800) {
            if (gate.accept(i * sourcePeriod)) accepted++
        }
        assertEquals("60 s at 10 Hz", 600, accepted)
    }

    @Test
    fun `a source that is not a multiple stays within one frame of the target`() {
        // The undershoot case: scheduling each slot from *now* rather than from the
        // previous slot gives 7.5 Hz here instead of 10.
        val gate = RateGate(10.0)
        var accepted = 0
        val sourcePeriod = 1_000_000_000L / 24
        val seconds = 10
        for (i in 0 until 24 * seconds) {
            if (gate.accept(i * sourcePeriod)) accepted++
        }
        assertTrue("expected ~${10 * seconds}, got $accepted", accepted in (10 * seconds - 1)..(10 * seconds + 1))
    }

    @Test
    fun `a stall does not produce a catch-up burst`() {
        // The deficit case, and the one that matters on a throttling phone: a gate that
        // advances slots unconditionally owes 50 frames after a 5 s stall at 10 Hz and
        // pays them back as fast as the camera can deliver.
        val gate = RateGate(10.0)
        assertTrue(gate.accept(0))

        val afterStall = 5_000 * ms
        assertTrue("the first frame after a stall is due", gate.accept(afterStall))

        assertFalse("no burst", gate.accept(afterStall + 1 * ms))
        assertFalse("no burst", gate.accept(afterStall + 50 * ms))
        assertFalse("no burst", gate.accept(afterStall + 99 * ms))
        assertTrue("back to a normal cadence", gate.accept(afterStall + 100 * ms))
    }

    @Test
    fun `a stall costs exactly one period, not the whole backlog`() {
        val gate = RateGate(20.0)
        assertTrue(gate.accept(0))
        val stalled = 3_000 * ms
        assertTrue(gate.accept(stalled))
        var accepted = 0
        // Feed one hour of frames at 100 Hz; a burst would show as far more than the
        // 20 Hz the gate was asked for.
        for (i in 1..1000) {
            if (gate.accept(stalled + i * 10 * ms)) accepted++
        }
        assertTrue("expected ~200 over 10 s at 20 Hz, got $accepted", accepted in 199..201)
    }

    @Test
    fun `a rate change takes effect on the next decision`() {
        val gate = RateGate(1.0)
        assertTrue(gate.accept(0))
        assertFalse(gate.accept(100 * ms))
        gate.setRate(10.0)
        assertTrue("the new period is 100 ms", gate.accept(101 * ms))
        assertEquals(10.0, gate.hz, 0.0)
    }

    @Test
    fun `a rate change does not re-arm the gate`() {
        // Otherwise repeated commands would drive the camera faster than any rate
        // actually asked for.
        val gate = RateGate(1.0)
        assertTrue(gate.accept(0))
        repeat(50) { gate.setRate(1.0) }
        assertFalse("still inside the period", gate.accept(500 * ms))
    }

    @Test
    fun `a large rate increase takes effect without waiting out the old period`() {
        // 0.2 Hz to 10 Hz: leaving the old slot alone would stall for the remaining
        // five seconds, which is indistinguishable from ignoring the command.
        val gate = RateGate(0.2)
        assertTrue(gate.accept(0))
        gate.setRate(10.0)
        assertFalse("still not earlier than the new period", gate.accept(99 * ms))
        assertTrue("one new period after the last frame", gate.accept(100 * ms))
    }

    @Test
    fun `a rate decrease is honoured immediately`() {
        val gate = RateGate(10.0)
        assertTrue(gate.accept(0))
        gate.setRate(1.0)
        assertFalse(gate.accept(100 * ms))
        assertFalse(gate.accept(999 * ms))
        assertTrue(gate.accept(1_000 * ms))
    }

    @Test
    fun `a rate change never emits sooner than the new rate allows`() {
        // The property that makes re-anchoring safe, swept over pairs of rates.
        for (from in listOf(0.5, 1.0, 5.0, 30.0)) {
            for (to in listOf(0.5, 1.0, 5.0, 30.0)) {
                val gate = RateGate(from)
                assertTrue(gate.accept(0))
                gate.setRate(to)
                val newPeriod = RateGate.periodNsFor(to)
                assertFalse(
                    "$from -> $to emitted early",
                    gate.accept(newPeriod - 1),
                )
                assertTrue("$from -> $to never emitted", gate.accept(newPeriod))
            }
        }
    }

    @Test
    fun `the period matches the rate`() {
        assertEquals(1_000_000_000L, RateGate.periodNsFor(1.0))
        assertEquals(100_000_000L, RateGate.periodNsFor(10.0))
        assertEquals(20_000_000L, RateGate.periodNsFor(50.0))
        assertEquals(1_000_000L, RateGate.periodNsFor(1000.0))
    }

    @Test
    fun `the period at the maximum rate is a millisecond, not a rounding artefact`() {
        // Asserting `>= 1` here proves nothing: the value is 1,000,000.
        assertEquals(1_000_000L, RateGate.periodNsFor(RateGate.MAX_HZ))
        assertEquals(1_000_001L, RateGate.periodNsFor(999.999))
    }

    @Test
    fun `a rate low enough to overflow a Long period accepts once and then never`() {
        // 1e-11 Hz is inside the wire's (0, 1000] range and its period exceeds
        // Long.MAX_VALUE nanoseconds. Adding it to a timestamp wrapped negative, which
        // stood the gate permanently open -- a command meaning "almost never" produced
        // full-rate capture, the exact inversion of a rate limit.
        val gate = RateGate(1e-11)
        assertEquals(Long.MAX_VALUE, gate.periodNanos)
        assertTrue("the first frame is still due", gate.accept(1_000_000L))
        var accepted = 0
        for (i in 2..1_000) if (gate.accept(i * 1_000_000L)) accepted++
        assertEquals("nothing further may be accepted", 0, accepted)
    }

    @Test
    fun `the slot arithmetic saturates instead of wrapping`() {
        assertEquals(Long.MAX_VALUE, RateGate.addSaturating(Long.MAX_VALUE, 1))
        assertEquals(Long.MAX_VALUE, RateGate.addSaturating(Long.MAX_VALUE - 5, 100))
        assertEquals(Long.MAX_VALUE, RateGate.addSaturating(1, Long.MAX_VALUE))
        assertEquals(10L, RateGate.addSaturating(4, 6))
        assertEquals(0L, RateGate.addSaturating(0, 0))
    }

    @Test
    fun `a timestamp near the top of the range does not unlock the gate`() {
        val gate = RateGate(1.0)
        assertTrue(gate.accept(Long.MAX_VALUE - 2_000_000_000L))
        assertFalse("the next slot must not have wrapped", gate.accept(Long.MAX_VALUE - 1_900_000_000L))
    }

    @Test
    fun `a rate change is visible to another thread`() {
        // The other half of the synchronization: without it the fields are plain and
        // non-volatile, so a change made on one thread can stay invisible to the
        // analyzer indefinitely.
        val gate = RateGate(1.0)
        gate.accept(0)
        val setter = Thread { gate.setRate(1000.0) }
        setter.start()
        setter.join()
        assertEquals(1_000_000L, gate.periodNanos)
        assertTrue("the new rate must apply on this thread too", gate.accept(1_000_000L))
    }

    @Test
    fun `re-sending the same rate does not disturb the schedule`() {
        // Re-anchoring unconditionally turns "schedule from the previous slot" into
        // "schedule from now", which is the 25% undershoot the class exists to avoid.
        // A peer that simply repeats the current rate would silently lose a quarter of
        // the frame rate, and task 22 makes that peer real.
        val sourcePeriod = 1_000_000_000L / 30
        val plain = RateGate(10.0)
        var withoutCommands = 0
        for (i in 0 until 300) if (plain.accept(i * sourcePeriod)) withoutCommands++

        val recommanded = RateGate(10.0)
        var withCommands = 0
        for (i in 0 until 300) {
            if (recommanded.accept(i * sourcePeriod)) withCommands++
            recommanded.setRate(10.0)
        }
        assertEquals("re-commanding the same rate must cost nothing", withoutCommands, withCommands)
        assertEquals(100, withCommands)
    }

    @Test
    fun `a repeated rate command still reports the rate`() {
        val gate = RateGate(10.0)
        gate.accept(0)
        gate.setRate(10.0)
        assertEquals(10.0, gate.hz, 0.0)
    }

    @Test
    fun `the reset boundary admits no extra frame`() {
        // `advanced <= nowNs` rather than `<`. At the boundary the strict version arms a
        // slot at nowNs itself, which the very next call satisfies.
        val gate = RateGate(10.0)
        assertTrue(gate.accept(0))
        // Exactly one period late: `advanced` equals `nowNs`, so the reset branch applies.
        assertTrue(gate.accept(200 * ms))
        assertFalse("the slot must be a full period out, not at now", gate.accept(200 * ms))
        assertFalse(gate.accept(299 * ms))
        assertTrue(gate.accept(300 * ms))
    }

    @Test
    fun `a negative first timestamp is treated as a real stamp`() {
        // The sentinel this replaced was Long.MIN_VALUE, which is a legal value in the
        // domain; the rewrite's justification went untested.
        val gate = RateGate(10.0)
        assertTrue("the first frame is due whatever the clock reads", gate.accept(Long.MIN_VALUE))
        assertFalse(gate.accept(Long.MIN_VALUE + 50 * ms))
        assertTrue(gate.accept(Long.MIN_VALUE + 100 * ms))
    }

    @Test
    fun `a rate change after a negative first stamp re-anchors to it`() {
        val gate = RateGate(1.0)
        assertTrue(gate.accept(Long.MIN_VALUE))
        gate.setRate(10.0)
        assertFalse(gate.accept(Long.MIN_VALUE + 99 * ms))
        assertTrue("re-anchored to the real stamp, not to a sentinel", gate.accept(Long.MIN_VALUE + 100 * ms))
    }

    @Test
    fun `concurrent callers cannot both be admitted in one slot`() {
        // One analyzer thread today, but setRate is public and reachable from any
        // thread, and unsynchronised non-volatile fields make a change invisible.
        //
        // 2000 rounds, not 20: at 20 the race is too rare to observe, and a version of
        // this test with 20 rounds failed to notice `synchronized` being removed across
        // 240 consecutive rounds -- an assurance with no detection power.
        repeat(2_000) {
            val gate = RateGate(1.0)
            val admitted = java.util.concurrent.atomic.AtomicInteger()
            val start = java.util.concurrent.CountDownLatch(1)
            val threads = (1..4).map {
                Thread {
                    start.await()
                    if (gate.accept(1_000_000L)) admitted.incrementAndGet()
                }
            }
            threads.forEach { it.start() }
            start.countDown()
            threads.forEach { it.join() }
            assertEquals("exactly one caller may win the slot", 1, admitted.get())
        }
    }

    @Test
    fun `a rate outside the wire's range is refused`() {
        for (bad in listOf(0.0, -1.0, 1000.1, Double.NaN, Double.POSITIVE_INFINITY)) {
            val threw = runCatching { RateGate.periodNsFor(bad) }.isFailure
            assertTrue("rate $bad should be refused", threw)
        }
    }

    @Test
    fun `the wire's boundary rates are accepted`() {
        RateGate.periodNsFor(1000.0)
        RateGate.periodNsFor(0.001)
    }

    @Test
    fun `a non-monotonic timestamp does not unlock the gate`() {
        // A clock that goes backwards must not be read as a period having elapsed.
        val gate = RateGate(10.0)
        assertTrue(gate.accept(1_000 * ms))
        assertFalse(gate.accept(500 * ms))
        assertFalse(gate.accept(0))
    }
}
