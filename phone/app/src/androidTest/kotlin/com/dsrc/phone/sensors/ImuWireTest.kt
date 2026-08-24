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
import com.dsrc.transport.Session
import org.junit.After
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
