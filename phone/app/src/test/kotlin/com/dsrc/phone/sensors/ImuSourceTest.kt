package com.dsrc.phone.sensors

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The timebase policy, which is the only part of [ImuSource] a test can settle.
 *
 * Whether a given handset actually has the vendor bug is not something the emulator can
 * tell us -- its virtual sensors report the same clock, so the mismatched branch is inert
 * there. What is testable, and what matters, is what we do when the numbers say the two
 * clocks differ: refuse, rather than guess a correction.
 */
class ImuSourceTest {

    private val ms = 1_000_000L
    private val second = 1_000_000_000L

    @Test
    fun `a small positive delivery delta means the clocks agree`() {
        // The app clock is read at delivery and the event was captured before it, so a
        // healthy difference is exactly delivery latency: small and positive.
        for (delta in listOf(0L, 1L, ms, 50 * ms, second)) {
            assertEquals(
                "a delivery delta of ${delta}ns is ordinary latency",
                ImuTimebase.MATCHED,
                ImuSource.verdictFor(delta),
            )
        }
    }

    @Test
    fun `a negative delta is a mismatch, not a rounding artefact`() {
        // The sensor stamp being *ahead* of a clock read afterwards is as damning as a
        // huge gap: it cannot happen on one monotonic clock, so it is two.
        for (delta in listOf(-1L, -ms, -second)) {
            assertEquals(
                "a sensor stamp ${-delta}ns in the future is not the same clock",
                ImuTimebase.MISMATCHED,
                ImuSource.verdictFor(delta),
            )
        }
    }

    @Test
    fun `a delta far larger than any delivery latency is a different epoch`() {
        // What this detects is a different *epoch*, which on a device up for any length of
        // time is seconds to hours. A phone up for a day gives about 86,400 s.
        for (delta in listOf(3 * second, 60 * second, 86_400 * second)) {
            assertEquals(ImuTimebase.MISMATCHED, ImuSource.verdictFor(delta))
        }
    }

    @Test
    fun `the bound itself is deliberately generous, and both sides of it are checked`() {
        // Two seconds: far above any real delivery latency, far below the epoch difference
        // being looked for. A tight bound would turn one slow first delivery into a false
        // alarm and stop the IMU on a healthy device -- and the check runs once, so a false
        // alarm is permanent for the session.
        //
        // Both sides, because asserting only the accepting side would pass for a function
        // that accepts everything, and only the refusing side for one that refuses
        // everything.
        assertEquals(
            ImuTimebase.MATCHED,
            ImuSource.verdictFor(ImuSource.MAX_PLAUSIBLE_DELIVERY_NS),
        )
        assertEquals(
            ImuTimebase.MISMATCHED,
            ImuSource.verdictFor(ImuSource.MAX_PLAUSIBLE_DELIVERY_NS + 1),
        )
    }

    @Test
    fun `the two verdicts are distinct, so a constant cannot satisfy the suite`() {
        // Guards against the shape where every assertion above happens to want the same
        // answer: `verdictFor` returning one enum value for everything is a passing
        // implementation of half a test suite.
        assertEquals(
            setOf(ImuTimebase.MATCHED, ImuTimebase.MISMATCHED),
            setOf(ImuSource.verdictFor(ms), ImuSource.verdictFor(-ms)),
        )
    }
}
