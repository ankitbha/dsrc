package com.dsrc.phone.sensors

import android.os.PowerManager
import com.dsrc.transport.PhoneTelemetry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TelemetryReporterTest {

    private val second = 1_000_000_000L

    private fun sample(
        delivered: Map<String, Long> = mapOf("camera_hz" to 0L, "gps_hz" to 0L, "imu_hz" to 0L, "here_hz" to 0L),
        dropped: Map<String, Long> = mapOf("camera" to 0L, "gps" to 0L, "imu" to 0L, "here" to 0L),
        status: String = "nominal",
        headroom: Double? = 0.42,
        hereCalls: Long = 0,
        hereErrors: Long = 0,
    ) = TelemetryReporter.Sample(status, headroom, delivered, dropped, hereCalls, hereErrors)

    private class Recorder {
        val sent = mutableListOf<PhoneTelemetry>()
        var accept = true
        fun sink(t: PhoneTelemetry): Boolean {
            sent.add(t)
            return accept
        }
    }

    private fun reporter(
        samples: List<TelemetryReporter.Sample>,
        stepNs: Long = second,
    ): Pair<TelemetryReporter, Recorder> {
        val recorder = Recorder()
        var index = 0
        var now = 0L
        val r = TelemetryReporter(
            monoClock = { now += stepNs; now },
            sample = { samples[minOf(index++, samples.size - 1)] },
            sink = recorder::sink,
        )
        return r to recorder
    }

    @Test
    fun `the first report is skipped, because a rate needs two readings`() {
        // A report built from one reading would divide by the time since some arbitrary
        // epoch, which is not a rate anyone asked about.
        val (r, recorder) = reporter(listOf(sample()))
        assertFalse(r.report())

        assertEquals(0, recorder.sent.size)
        assertEquals(1, r.stats.skipped)
    }

    @Test
    fun `achieved is the delivery rate between two readings`() {
        val (r, recorder) = reporter(
            listOf(
                sample(delivered = mapOf("camera_hz" to 0L, "gps_hz" to 0L, "imu_hz" to 0L, "here_hz" to 0L)),
                sample(delivered = mapOf("camera_hz" to 5L, "gps_hz" to 1L, "imu_hz" to 50L, "here_hz" to 1L)),
            )
        )
        r.report()
        assertTrue(r.report())

        val achieved = recorder.sent.single().achieved
        // One second between readings, so the deltas are the rates. Four distinct values,
        // so a modality reported under another's key is visible.
        assertEquals(5.0, achieved.getValue("camera_hz"), 1e-6)
        assertEquals(1.0, achieved.getValue("gps_hz"), 1e-6)
        assertEquals(50.0, achieved.getValue("imu_hz"), 1e-6)
        assertEquals(1.0, achieved.getValue("here_hz"), 1e-6)
    }

    @Test
    fun `a shortfall against the commanded rate is what the report is for`() {
        // The phone reports and the Jetson decides. A handset that managed 22 of a
        // commanded 50 says 22 -- it does not quietly lower its own command, because the
        // far side would then be comparing a model against inputs it never asked for and
        // cannot see it did not get.
        val (r, recorder) = reporter(
            listOf(
                sample(delivered = mapOf("camera_hz" to 0L, "gps_hz" to 0L, "imu_hz" to 0L, "here_hz" to 0L)),
                sample(delivered = mapOf("camera_hz" to 0L, "gps_hz" to 0L, "imu_hz" to 22L, "here_hz" to 0L)),
            )
        )
        r.report(); r.report()

        assertEquals(22.0, recorder.sent.single().achieved.getValue("imu_hz"), 1e-6)
    }

    @Test
    fun `a headroom the platform will not give is null, not NaN`() {
        // getThermalHeadroom returns NaN when it has no estimate. Canonical JSON refuses to
        // encode a NaN on both sides, so a NaN here would not produce a wrong number -- it
        // would fail the whole telemetry frame and take the thermal status down with it.
        // The phone would go quiet about being hot at the moment it was hottest.
        assertNull(ThermalReader.headroomOrNull(Float.NaN))
        assertNull(ThermalReader.headroomOrNull(Float.POSITIVE_INFINITY))
        assertNull(ThermalReader.headroomOrNull(-1.0f))
        assertNull(ThermalReader.headroomOrNull(1e9f))
        assertEquals(0.42, ThermalReader.headroomOrNull(0.42f)!!, 1e-6)
        // Above one is throttling, which is exactly the reading worth having.
        assertEquals(1.5, ThermalReader.headroomOrNull(1.5f)!!, 1e-6)
    }

    @Test
    fun `a null headroom still carries the thermal status`() {
        // The status is the part the Jetson can act on. Dropping the report because the
        // headroom was unavailable would lose it.
        val (r, recorder) = reporter(listOf(sample(headroom = null), sample(headroom = null)))
        r.report(); r.report()

        assertNull(recorder.sent.single().thermalHeadroom)
        assertEquals("nominal", recorder.sent.single().thermalStatus)
    }

    @Test
    fun `a counter that went backwards reports zero, not a negative rate`() {
        // A modality restarted underneath the reporter resets its counters. A negative rate
        // is a number nobody can act on, and the wire would carry it happily.
        val (r, recorder) = reporter(
            listOf(
                sample(delivered = mapOf("camera_hz" to 100L, "gps_hz" to 0L, "imu_hz" to 0L, "here_hz" to 0L)),
                sample(delivered = mapOf("camera_hz" to 3L, "gps_hz" to 0L, "imu_hz" to 0L, "here_hz" to 0L)),
            )
        )
        r.report(); r.report()

        assertEquals(0.0, recorder.sent.single().achieved.getValue("camera_hz"), 1e-9)
    }

    @Test
    fun `two readings in the same instant are skipped rather than divided by zero`() {
        // A rate of infinity is refused by the wire, and refusing the frame takes the
        // thermal status down with it.
        val (r, recorder) = reporter(listOf(sample(), sample()), stepNs = 0)
        r.report()
        assertFalse(r.report())

        assertEquals(0, recorder.sent.size)
        assertEquals(2, r.stats.skipped)
    }

    @Test
    fun `drops are reported per modality and gated frames are not among them`() {
        // A frame the gate rejected is the commanded rate working, not something the phone
        // failed to deliver. Counting it would make every healthy drive look lossy.
        val (r, recorder) = reporter(
            listOf(
                sample(),
                sample(dropped = mapOf("camera" to 3L, "gps" to 0L, "imu" to 1L, "here" to 2L)),
            )
        )
        r.report(); r.report()

        val dropped = recorder.sent.single().dropped
        assertEquals(3, dropped.getValue("camera"))
        assertEquals(0, dropped.getValue("gps"))
        assertEquals(1, dropped.getValue("imu"))
        assertEquals(2, dropped.getValue("here"))
    }

    @Test
    fun `here calls and errors ride along`() {
        val (r, recorder) = reporter(
            listOf(sample(), sample(hereCalls = 30, hereErrors = 1))
        )
        r.report(); r.report()

        assertEquals(30, recorder.sent.single().hereCalls)
        assertEquals(1, recorder.sent.single().hereErrors)
    }

    @Test
    fun `the headroom call is not made below the api level that has it`() {
        // getThermalHeadroom is API 30 and minSdk is 29 -- a guess about the handset, not a
        // device we have. Called unguarded it raises NoSuchMethodError inside a lambda whose
        // caller wraps it in runCatching, so the whole telemetry stream went silent for the
        // entire drive: no status, no achieved, no drops, and nothing logged. The status is
        // API 29 and would have been fine on its own, which is what makes losing it to the
        // headroom call the wrong trade.
        var called = false
        val onAndroid10 = ThermalReader.headroomIfSupported(sdkInt = 29) {
            called = true
            throw NoSuchMethodError("No virtual method getThermalHeadroom(I)F")
        }

        assertNull(onAndroid10)
        assertFalse("the call was made on a platform that does not have it", called)

        // And it is made where it exists, or the guard would silence a device that works.
        assertEquals(
            0.42,
            ThermalReader.headroomIfSupported(sdkInt = 30) { 0.42f }!!,
            1e-6,
        )

        // `headroomFrom` repeats this predicate rather than calling through, because lint
        // cannot trace a version check through a lambda. The two must agree at the
        // boundary, or the tested guard and the shipped one are different guards.
        assertEquals(
            "the boundary the test pins is the boundary the platform call is behind",
            android.os.Build.VERSION_CODES.R,
            30,
        )
        assertNull(ThermalReader.headroomIfSupported(sdkInt = 29) { 0.42f })
        assertNotNull(ThermalReader.headroomIfSupported(sdkInt = 30) { 0.42f })
    }

    @Test
    fun `the thermal status names are the wire's, not android's integers`() {
        assertEquals("nominal", ThermalReader.statusName(android.os.PowerManager.THERMAL_STATUS_NONE))
        assertEquals("severe", ThermalReader.statusName(android.os.PowerManager.THERMAL_STATUS_SEVERE))
        assertEquals("shutdown", ThermalReader.statusName(android.os.PowerManager.THERMAL_STATUS_SHUTDOWN))
        // A value the platform grows later becomes `unknown` rather than a stringified
        // integer, so a receiver keying on these need not guess whether "7" is a status.
        assertEquals("unknown", ThermalReader.statusName(99))
    }
}
