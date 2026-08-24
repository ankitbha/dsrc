package com.dsrc.phone.sensors

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.dsrc.phone.SensingService
import com.dsrc.phone.SensingState
import com.dsrc.phone.SensingStatus
import org.junit.After
import org.junit.Before
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * What the platform records about our sensor registrations.
 *
 * Rounds 1 and 2 both found the same hole from opposite sides: every decision in
 * `ImuSource` moved somewhere testable, and the *glue* stayed unreachable. Fourteen
 * mutations to the listener survived the whole JVM suite, and four of them applied at
 * once — never registering the gyroscope, never unregistering, dropping the unpaired
 * branch, dropping the timebase gate — left the instrumented suite at 53 passing while
 * the `imu` channel transmitted nothing for a whole session. The thread census cannot see
 * any of it, because `HandlerThread.start()` happens before and independently of the two
 * `registerListener` calls, and `quitSafely()` ends the thread whether or not the
 * listeners were ever removed.
 *
 * `dumpsys sensorservice` is the instrument, and this repo already uses the technique:
 * `GpsCaptureTest` reads `dumpsys location` to pin exactly this failure for GPS. The
 * sensor service keeps a per-connection record tagged with the registering class, plus an
 * event log of every registration and removal by sensor handle:
 *
 * ```
 * + 0x00000001 pid=4310 package=com.dsrc.phone.sensors.ImuSource samplingPeriod=20000us batchingPeriod=0us
 * - 0x00000001 pid=4310 package=com.dsrc.phone.sensors.ImuSource
 * ```
 *
 * So the platform will tell us which sensors we asked for, at what rate, with what
 * batching, and whether we gave them back — four claims that had no test at all.
 */
@RunWith(AndroidJUnit4::class)
class ImuCaptureTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    /**
     * Start from stopped.
     *
     * Another test class may leave sensing running, and `Start` offered from `RUNNING` is
     * ignored by the machine -- so without this the service never comes up here, no
     * registration is recorded, and the failure reads as "the platform logged nothing"
     * rather than "the test began in the wrong state".
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
    fun bothSensorsAreRegisteredAtTheCommandedRateAndGivenBackOnStop() {
        // Counted as a delta, not indexed. Other tests in this suite start sensing in the
        // same process, so the pid alone does not isolate this run -- and the event log is
        // a bounded history, so indexing into it by a position taken earlier misaligns the
        // moment it rotates. A count taken immediately either side of one start is immune
        // to both, as long as the ring does not rotate inside this test; it holds tens of
        // entries and this adds four.
        val addedBefore = additions().size
        val removedBefore = removals().size

        SensingService.start(context)
        awaitState(SensingState.RUNNING)
        assertTrue(
            "no registration appeared in the sensor service's log",
            pollUntil(10_000) { additions().size > addedBefore },
        )

        // Two sensors, not one. Deleting the gyroscope registration is the mutation that
        // leaves every accelerometer event on the unpaired branch, so the channel carries
        // nothing all session with both suites green.
        val added = additions().drop(addedBefore)
        assertEquals(
            "wanted an accelerometer and a gyroscope registration, got: $added",
            2,
            added.size,
        )

        // The rate we asked for, and no batching. Both are decisions the plan records and
        // neither was asserted anywhere: a batch hands over a burst whose timestamps are
        // right but whose arrival is late, and the rate gate would then pass one of the
        // burst and drop the rest, turning 50 Hz into one sample per batch.
        for (entry in added) {
            assertTrue(
                "the commanded 50 Hz did not reach the platform: $entry",
                entry.contains("samplingPeriod=20000us"),
            )
            assertTrue(
                "batching is on, which the rate gate cannot survive: $entry",
                entry.contains("batchingPeriod=0us"),
            )
        }

        SensingService.stop(context)
        awaitState(SensingState.IDLE)

        // Given back. `quitSafely()` ends the thread either way, so without this the
        // sensors stay live at the commanded rate for the life of the process, delivering
        // into a looper nobody is draining -- and the thread census reports success.
        assertTrue(
            "a sensor registration outlived the service: ${additions().drop(addedBefore)} " +
                "vs ${removals().drop(removedBefore)}",
            pollUntil(10_000) { removals().size - removedBefore >= added.size },
        )
    }

    /**
     * Event-log lines naming our source class **in this process**.
     *
     * Filtered by pid rather than watermarked by position. The sensor service's event log
     * is a bounded history that drops its oldest lines, so indexing into it by a count
     * taken earlier misaligns the moment it rotates -- the same defect as watermarking a
     * camera log by position, which this repo has already been bitten by once. A pid is
     * stable for the run and unique to it.
     */
    private fun sensorEvents(): List<String> {
        val pid = "pid=${android.os.Process.myPid()}"
        return shell("dumpsys sensorservice")
            .lines()
            .filter { it.contains(ImuSource::class.java.name) && it.contains(pid) }
    }

    private fun additions() = sensorEvents().filter { it.contains(" + ") }

    private fun removals() = sensorEvents().filter { it.contains(" - ") }

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

    private fun shell(command: String): String {
        val descriptor = InstrumentationRegistry.getInstrumentation()
            .uiAutomation.executeShellCommand(command)
        return android.os.ParcelFileDescriptor.AutoCloseInputStream(descriptor).use {
            it.readBytes().toString(Charsets.UTF_8)
        }
    }
}
