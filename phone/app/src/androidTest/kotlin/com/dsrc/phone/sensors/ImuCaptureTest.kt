package com.dsrc.phone.sensors

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.dsrc.phone.SensingService
import com.dsrc.phone.SensingState
import com.dsrc.phone.SensingStatus
import com.dsrc.phone.config.SensingConfig
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * That the IMU actually produces samples on a device.
 *
 * Three validation rounds found the same hole from three angles. Round 1: the decisions
 * lived in a `SensorEventListener` nothing could invoke. Round 2: moving them to
 * [ImuPairing] made the decisions reachable and left the *caller* unreachable. Round 3: an
 * instrumented test that read `dumpsys sensorservice` counted registrations without
 * identifying them, so swapping the gyroscope for the magnetometer passed — the count
 * stayed at two, `hasGyro` never became true, every event took the unpaired branch, and
 * the `imu` channel carried nothing for the whole session.
 *
 * What was missing at every step was an assertion that a sample came *out*. That is what
 * this is. The wrong sensor cannot satisfy it, and neither can a listener that never
 * dispatches, a pairing that never pairs, or a timebase gate that refuses everything.
 */
@RunWith(AndroidJUnit4::class)
class ImuCaptureTest {

    /**
     * Every other class that starts sensing has one of these, and this one did not.
     *
     * It passed regardless, because an earlier class in the same run had already granted
     * them — and run alone it failed with "timed out waiting for RUNNING", which reads as a
     * sensing defect rather than a missing grant.
     */
    @get:Rule
    val permissions: GrantPermissionRule = GrantPermissionRule.grant(
        android.Manifest.permission.CAMERA,
        android.Manifest.permission.ACCESS_FINE_LOCATION,
    )

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    /**
     * Start from stopped.
     *
     * Another class may leave sensing running, and `Start` offered from `RUNNING` is
     * ignored by the machine — so without this the service never comes up here and the
     * failure reads as "no samples" rather than "the test began in the wrong state".
     */
    @Before
    fun startFromIdle() {
        SensingService.stop(context)
        awaitState(SensingState.IDLE)
    }

    @After
    fun stopSensing() {
        SensingService.stop(context)
        awaitState(SensingState.IDLE)
    }

    @Test
    fun sensingProducesPairedImuSamples() {
        SensingService.start(context)
        awaitState(SensingState.RUNNING)

        assertTrue(
            "no IMU sample was accepted in 10 s: ${SensingService.liveImu?.stats}",
            pollUntil(10_000) { (SensingService.liveImu?.stats?.accepted ?: 0) > 0 },
        )

        val stats = requireNotNull(SensingService.liveImu).stats

        // Three of the four assertions here used to be entailed by the poll above.
        // `accepted >= 1` already forces `seen > 0` and `unpaired < seen`, since `seen` is
        // incremented on every offer and `unpaired` only on the unpaired path -- so they
        // could not fail once the poll had passed. And `refusedStopped` was asserted with
        // the message "the timebase was refused on a device whose clocks agree", which it
        // does not measure at all: a timebase refusal never reaches the pipeline. It is
        // counted in `ImuPairing.refusedWrongTimebase` and short-circuits into a teardown.
        //
        // What is left are the two facts the poll does not give: the timebase question was
        // actually settled and settled in favour of capturing, and the accounting balances.
        val source = requireNotNull(SensingService.liveImuSource)
        assertEquals(
            "the timebase was refused on a device whose clocks agree: " +
                "offset=${source.timebaseOffsetNs} gap=${source.clockGapNs}",
            ImuTimebase.MATCHED,
            source.timebase,
        )
        assertEquals(
            "events were discarded for a wrong timebase on a device whose clocks agree",
            0L,
            source.refusedWrongTimebase,
        )
        assertTrue("$stats", stats.balances)
    }

    @Test
    fun aRateCommandArrivingAfterAStopDoesNotReEngageTheSensors() {
        // The window a combined round found. A rate_cmd is applied on the transport's
        // delivery thread, which runs until the link stops at step 14 of teardown and is
        // not joined even then -- while the sources stop at step 3. So a command arriving
        // during teardown re-registered both sensors after their unregister, and since the
        // service nulls its reference in the same breath, nothing was left that could ever
        // switch them off: awake at the commanded rate for the life of the process,
        // delivering into a looper that had been quit, with no counter moving.
        //
        // The applier is now cleared first, which closes the wide window. This pins the
        // narrow one: even called directly, a stopped source must refuse.
        val source = ImuSource(
            context = context,
            config = SensingConfig(imuHz = 50.0),
            appClock = android.os.SystemClock::elapsedRealtimeNanos,
            monoClock = System::nanoTime,
        )
        source.start(onReading = {}, onUnpaired = {})
        assertEquals("the source did not come up at its commanded rate", 50.0, source.requestedHz, 1e-9)

        source.stop()
        source.setRate(200.0)

        assertEquals(
            "a rate command after the stop re-requested the sensors",
            50.0,
            source.requestedHz,
            1e-9,
        )
    }

    // Not here, and recorded rather than papered over: nothing pins that `stop()`
    // unregisters the two listeners. Deleting `unregisterListener` leaves both suites
    // green, because `quitSafely()` ends the thread either way and the thread census only
    // sees threads.
    //
    // A test built on `dumpsys sensorservice`'s Previous Registrations was written for it
    // and removed. The dump prints newest-first, so a positional slice read another
    // class's entries from nine seconds earlier; the ring is 200 lines shared by every
    // process on the device, so a count delta can fall mid-test and accuse innocent code,
    // which it did once; and the lines are not unique, so a set difference collapses two
    // registrations differing only by handle. Each repair exposed the next flaw in the
    // instrument rather than in the code. The live-connection section of that dump is the
    // more likely instrument, and settling that is a piece of work rather than a repair.

    private fun awaitState(state: SensingState) {
        assertTrue(
            "timed out waiting for $state, still ${SensingStatus.shared.state}",
            pollUntil(15_000) { SensingStatus.shared.state == state },
        )
    }

    private fun pollUntil(timeoutMs: Long, condition: () -> Boolean): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return true
            Thread.sleep(100)
        }
        return condition()
    }
}
