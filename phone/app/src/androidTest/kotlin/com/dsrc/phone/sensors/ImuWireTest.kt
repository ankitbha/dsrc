package com.dsrc.phone.sensors

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.dsrc.phone.SensingService
import com.dsrc.phone.SensingState
import com.dsrc.phone.SensingStatus
import com.dsrc.phone.config.LinkConfig
import com.dsrc.transport.Channels
import com.dsrc.transport.Frame
import com.dsrc.transport.ImuSample
import com.dsrc.transport.Protocol
import com.dsrc.phone.ui.AdvisoryHolder
import com.dsrc.transport.AdvisoryMessage
import com.dsrc.transport.RateCommand
import com.dsrc.transport.Session
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.IOException
import java.net.InetAddress
import java.net.ServerSocket
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * What the phone actually puts on the wire, read by a peer that is really listening.
 *
 * Four validation rounds found the same defect one layer further out each time: the
 * decisions were unreachable, then their caller was, then the instrument that reached the
 * caller counted registrations without identifying them, and then the test that asserted
 * "a sample comes out" asserted it came out of the *pipeline*. Replacing the sink with a
 * body that never calls the transport left every suite green, because `accepted` is
 * incremented before the sink runs and nothing downstream of it was ever observed.
 *
 * So this stands where the Jetson stands. It accepts a real session on the port the
 * service dials, decodes the frames that arrive, and asserts on their contents. A sample
 * that is counted but never sent does not reach here; nor does one sent on the wrong
 * channel; nor does one whose two sensor streams have been transposed.
 *
 * The physics is the assertion. A stationary phone's accelerometer reads one g and its
 * gyroscope reads nothing, so the two vectors are not interchangeable — which is what
 * makes a transposition visible without knowing the device's orientation.
 */
@RunWith(AndroidJUnit4::class)
class ImuWireTest {

    @get:Rule
    val permissions: GrantPermissionRule = GrantPermissionRule.grant(
        android.Manifest.permission.CAMERA,
        android.Manifest.permission.ACCESS_FINE_LOCATION,
    )

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private val frames = ConcurrentLinkedQueue<Frame>()
    private val sessions = ConcurrentLinkedQueue<Session>()
    private var server: ServerSocket? = null

    @Before
    fun listenAsTheJetson() {
        SensingService.stop(context)
        awaitState(SensingState.IDLE)

        // The port the service dials. Bound before sensing starts, so the very first
        // connection attempt succeeds and no reconnect backoff is involved.
        // 127.0.0.1 by name, not `getLoopbackAddress()`. That returns ::1 here, so the
        // peer bound on IPv6 loopback while the service dials IPv4 -- and the symptom was
        // ECONNREFUSED, indistinguishable from a phone that never tried. The probe below
        // used the same helper and so agreed with the bug.
        val socket = ServerSocket(LinkConfig.DEFAULT_PORT, 8, InetAddress.getByName(LinkConfig().host))
        server = socket
        Thread({ acceptLoop(socket) }, "imu-wire-peer").also { it.isDaemon = true; it.start() }

        // Prove the listener listens before blaming the phone for silence. A peer that
        // failed to bind, or bound somewhere the app cannot reach, produces exactly the
        // same symptom as a phone that sends nothing -- and that is a diagnosis pointing at
        // innocent code, which this project has already been caught by twice.
        java.net.Socket().use { probe ->
            probe.connect(
                java.net.InetSocketAddress(InetAddress.getByName(LinkConfig().host), LinkConfig.DEFAULT_PORT),
                2_000,
            )
        }
    }

    @After
    fun stopEverything() {
        SensingService.stop(context)
        awaitState(SensingState.IDLE)
        sessions.forEach { runCatching { it.close() } }
        runCatching { server?.close() }
    }

    private fun acceptLoop(socket: ServerSocket) {
        while (!socket.isClosed) {
            val client = try {
                socket.accept()
            } catch (e: IOException) {
                return
            }
            client.tcpNoDelay = true
            // Each connection on its own thread. `session.start()` blocks for the handshake,
            // and the reachability probe in @Before connects and closes without ever saying
            // hello -- so on one thread the loop sat in that dead handshake while the
            // service's real connection waited unaccepted, and the failure read as "the
            // link never came up".
            Thread({
                runCatching {
                    val session = Session(
                        input = client.getInputStream(),
                        output = client.getOutputStream(),
                        deviceId = "imu-wire-test",
                        role = Session.ROLE_JETSON,
                        monoClock = { System.nanoTime() },
                        wallClock = { System.currentTimeMillis() * 1_000_000L },
                        onFrame = { frames.add(it) },
                    )
                    sessions.add(session)
                    session.start()
                }
            }, "imu-wire-session").also { it.isDaemon = true; it.start() }
        }
    }

    @Test
    fun imuSamplesReachTheWireWithTheTwoSensorsInTheirOwnFields() {
        SensingService.start(context)
        awaitState(SensingState.RUNNING)

        assertTrue(
            "no frame arrived on the imu channel in 20 s; the samples are counted but not sent",
            pollUntil(20_000) { imuFrames().isNotEmpty() },
        )

        // Decoded, not merely counted. A frame on the wrong channel does not appear above,
        // and a frame that will not decode fails here.
        val samples = imuFrames().map { ImuSample.fromWire(it.header.entries, it.payload) }
        assertTrue("decoded ${samples.size} samples", samples.isNotEmpty())

        // The physics. Stationary, the accelerometer reads one g and the gyroscope reads
        // nothing, so the two vectors cannot be swapped without this failing -- and unlike
        // an axis-by-axis assertion it does not depend on how the device is lying.
        // Compared on the median so one noisy sample cannot decide it.
        val accel = samples.map { sqrt(it.ax * it.ax + it.ay * it.ay + it.az * it.az) }.sorted()
        val gyro = samples.map { sqrt(it.gx * it.gx + it.gy * it.gy + it.gz * it.gz) }.sorted()
        val accelMagnitude = accel[accel.size / 2]
        val gyroMagnitude = gyro[gyro.size / 2]

        assertTrue(
            "the accelerometer fields do not carry one g (|a| = $accelMagnitude); " +
                "if |g| = $gyroMagnitude is near 9.8 the two streams are transposed",
            abs(accelMagnitude - GRAVITY) < 2.0,
        )
        assertTrue(
            "the gyro fields carry $gyroMagnitude rad/s on a stationary device, which is " +
                "an accelerometer reading in the wrong fields",
            gyroMagnitude < 1.0,
        )

        // Which axis carries gravity, not just how much of it there is. A magnitude is
        // invariant under a swap between two accelerometer axes, so transposing
        // `values[1]` and `values[2]` in the listener survived every assertion above --
        // the last of the three layers where the axis family was free.
        //
        // The expected axis is this AVD's fixed orientation. A device lying differently
        // fails here loudly, with the readings in the message, rather than passing on a
        // magnitude that never noticed.
        val ax = samples.map { it.ax }.sorted()[samples.size / 2]
        val ay = samples.map { it.ay }.sorted()[samples.size / 2]
        val az = samples.map { it.az }.sorted()[samples.size / 2]
        assertTrue(
            "gravity is not on the axis this AVD reports it on: ax=$ax ay=$ay az=$az",
            abs(ay - GRAVITY) < 2.0 && abs(ax) < 5.0 && abs(az) < 5.0,
        )

        // And the stamps are the ones this task is about: on the app's clock, not the
        // delivery instant, and inside the session rather than at some epoch of their own.
        val now = android.os.SystemClock.elapsedRealtimeNanos()
        for (sample in samples) {
            assertTrue(
                "a capture stamp of ${sample.captureMonoNs} is not on elapsedRealtime (now $now)",
                sample.captureMonoNs in (now - 60_000_000_000L)..now,
            )
        }
    }

    @Test
    fun aRateCommandFromTheJetsonChangesARunningRate() {
        // The whole of task 22 in one assertion: a command arrives on the live link and the
        // modality's rate changes, with capture never restarting. Driven from the peer
        // rather than by calling the applier, because the routing -- decode, dispatch,
        // reach the right pipeline -- is the part that had no handler at all.
        SensingService.start(context)
        awaitState(SensingState.RUNNING)
        // A *running* session, not the first one. The reachability probe in @Before also
        // gets a session, and it is dead the moment the probe closes -- so `first()` picked
        // that one and the failure read as "the link never came up" while the phone was
        // happily sending hundreds of frames over the session next to it.
        assertTrue(
            "the link never came up, so no command could arrive",
            pollUntil(15_000) { sessions.any { it.isRunning } },
        )
        assertTrue(
            "sensing never produced a sample, so there is no rate to change",
            pollUntil(10_000) { (SensingService.liveImu?.stats?.accepted ?: 0) > 0 },
        )

        val before = requireNotNull(SensingService.liveImu).stats.rateHz
        val wanted = if (before == 20.0) 10.0 else 20.0
        val command = RateCommand(
            captureMonoNs = android.os.SystemClock.elapsedRealtimeNanos(),
            rates = mapOf("camera_hz" to 5.0, "gps_hz" to 1.0, "imu_hz" to wanted, "here_hz" to 0.2),
            trigger = "instrumented-test",
            shadow = false,
        )
        assertTrue(
            "the peer could not send the command",
            sessions.first { it.isRunning }.send(Channels.RATE_CMD, command.toExtensions()),
        )

        assertTrue(
            "the commanded rate never reached the imu pipeline: " +
                "${SensingService.liveImu?.stats?.rateHz} (was $before, wanted $wanted)",
            pollUntil(10_000) { SensingService.liveImu?.stats?.rateHz == wanted },
        )

        // And capture kept running across it. A rate change that restarted the modality
        // would show as the sample count going backwards or the thread being replaced.
        val after = requireNotNull(SensingService.liveImu).stats
        assertTrue("sensing stopped when the rate changed: $after", after.accepted > 0)
    }

    @Test
    fun aCommandRaisingTheRateIsHonouredOnTheWire() {
        // The direction the gate cannot serve. A RateGate only ever *drops* samples the
        // platform already produced, so raising imu_hz above what was requested at start
        // changed nothing on the wire while the pipeline reported the new rate as in force
        // -- commanded 200 Hz, measured 50, reported 200, with nothing on either side of
        // the link recording the difference. The source re-requests its period now.
        //
        // Measured by counting frames the peer actually received, because the pipeline's
        // own rateHz is exactly the number that was lying.
        SensingService.start(context)
        awaitState(SensingState.RUNNING)
        assertTrue(pollUntil(15_000) { sessions.any { it.isRunning } })
        assertTrue(pollUntil(10_000) { imuFrames().isNotEmpty() })

        // Above the rate the source was started at, which is the only direction that
        // discriminates. Commanding down and then back up does not: the source began at
        // 50 Hz, so a version that never re-requests is still sitting at 50 when the raise
        // arrives and produces the same answer. Both mutations survived that sequence.
        val baselineFrom = imuFrames().size
        Thread.sleep(3_000)
        val baseline = (imuFrames().size - baselineFrom) / 3.0

        command(imuHz = 200.0)
        Thread.sleep(1_000)
        val raisedFrom = imuFrames().size
        Thread.sleep(3_000)
        val raised = (imuFrames().size - raisedFrom) / 3.0

        assertTrue(
            "commanded 200 Hz from a 50 Hz baseline and measured $raised/s against " +
                "$baseline/s: a raise the gate cannot serve must reach the source",
            raised > baseline * 1.4,
        )
    }

    @Test
    fun anAdvisoryFromTheJetsonIsShownAndThenGoesStale() {
        // The two halves of task 23 that matter: an advisory arriving on the live link
        // reaches the display, and one that stops arriving leaves it. The second is the
        // point -- the transport keeps only the newest, but once nothing arrives there is
        // nothing to displace what is on screen, and a recommendation about road the driver
        // has covered looks exactly like a current one.
        SensingService.start(context)
        awaitState(SensingState.RUNNING)
        assertTrue(pollUntil(15_000) { sessions.any { it.isRunning } })

        val advisory = AdvisoryMessage(
            captureMonoNs = android.os.SystemClock.elapsedRealtimeNanos(),
            recSpeedMps = 13.4, recSpeedDisplay = 30.0, currentSpeedDisplay = 28.0,
            units = "mph", headwayTargetS = 2.0,
            laneText = "keep", mergeText = "", trafficText = "moderate",
            confidence = 0.87, confidenceLabel = "high",
            action = mapOf(
                "lane_preference" to "keep", "merge_mode" to "normal",
                "desired_speed_bin" to "nominal", "desired_headway_bin" to "normal",
            ),
        )
        assertTrue(
            "the peer could not send the advisory",
            sessions.first { it.isRunning }.send(Channels.ADVISORY, advisory.toExtensions()),
        )

        assertTrue(
            "the advisory never reached the display",
            pollUntil(10_000) {
                SensingService.advisories.current(android.os.SystemClock.elapsedRealtimeNanos()) != null
            },
        )
        val shown = SensingService.advisories.current(android.os.SystemClock.elapsedRealtimeNanos())!!
        assertEquals("the Jetson's number, not one the phone made", 30.0, shown.recSpeedDisplay, 1e-9)
        assertEquals("mph", shown.units)

        // Stopping the session takes it off immediately, without waiting for the expiry.
        // A driver who stopped is not being advised, and three seconds of a recommendation
        // that no longer applies is three seconds too many.
        SensingService.stop(context)
        awaitState(SensingState.IDLE)
        assertNull(
            "stopping the session left the advisory on the display",
            SensingService.advisories.current(android.os.SystemClock.elapsedRealtimeNanos()),
        )

        // And when it is not stopped, it leaves on its own because nothing more arrives.
        SensingService.start(context)
        awaitState(SensingState.RUNNING)
        assertTrue(pollUntil(15_000) { sessions.any { it.isRunning } })
        sessions.first { it.isRunning }.send(Channels.ADVISORY, advisory.toExtensions())
        assertTrue(
            pollUntil(10_000) {
                SensingService.advisories.current(android.os.SystemClock.elapsedRealtimeNanos()) != null
            },
        )
        Thread.sleep(AdvisoryHolder.MAX_AGE_NS / 1_000_000 + 500)
        assertNull(
            "a stale advisory is still on the display",
            SensingService.advisories.current(android.os.SystemClock.elapsedRealtimeNanos()),
        )
    }

    private fun command(imuHz: Double) {
        val sent = sessions.first { it.isRunning }.send(
            Channels.RATE_CMD,
            RateCommand(
                captureMonoNs = android.os.SystemClock.elapsedRealtimeNanos(),
                rates = mapOf(
                    "camera_hz" to 5.0, "gps_hz" to 1.0, "imu_hz" to imuHz, "here_hz" to 0.2,
                ),
                trigger = "instrumented-test",
                shadow = false,
            ).toExtensions(),
        )
        assertTrue("the peer could not send the command", sent)
    }

    private fun imuFrames() = frames.filter { it.channel == Channels.IMU }

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

    private companion object {
        /** One g, near enough for a check that only has to separate 9.8 from 0. */
        const val GRAVITY = 9.81
    }
}
