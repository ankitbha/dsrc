package com.dsrc.phone.sensors

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
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
import com.dsrc.transport.PhoneTelemetry
import com.dsrc.transport.RateCommand
import com.dsrc.transport.Session
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assume.assumeTrue
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
        // Against the platform's OWN reading, not against a fixed orientation. The
        // assertion used to require gravity on Y, which is where the emulator holds it;
        // a handset lying on a desk reports it on Z, so the test failed on the target
        // hardware while the code under test was correct. Measured on moto g power:
        // ax=0.005 ay=-0.034 az=9.740.
        //
        // Reading the sensor directly and comparing axis for axis keeps what the
        // assertion was for -- a transposition of values[1] and values[2] in the
        // listener survives a magnitude check -- and drops what it was not for, which
        // is the orientation the device happens to be in.
        val ax = samples.map { it.ax }.sorted()[samples.size / 2]
        val ay = samples.map { it.ay }.sorted()[samples.size / 2]
        val az = samples.map { it.az }.sorted()[samples.size / 2]

        val direct = readAccelerometerDirectly()
        assertNotNull("the platform produced no accelerometer reading to compare against",
                      direct)
        val (dx, dy, dz) = direct!!
        // Generous per-axis tolerance: the two readings are taken at different instants
        // and a handheld device is never perfectly still.
        val tolerance = 3.0
        // A transposition of two axes is invisible wherever those two axes read alike:
        // swapping them moves each by their difference, and a per-axis comparison
        // cannot see a move smaller than its own tolerance. That is a property of the
        // attitude, not of the code, and no single assertion escapes it.
        //
        // So the pairs are named. Y/Z is the one this test exists for -- the listener's
        // `values[1]`/`values[2]` -- and it is required. X/Y and X/Z are reported
        // rather than required, because on a phone lying flat, which is the attitude
        // this test was recalibrated for, X and Y both read near zero: measured
        // ax=0.005 ay=-0.034, a difference of 0.039 against a tolerance of 3.0. An
        // assertion that claimed to cover that pair there would be one that cannot
        // fail.
        val pairs = mapOf(
            "values[1]/values[2] (Y/Z)" to abs(dy - dz),
            "values[0]/values[1] (X/Y)" to abs(dx - dy),
            "values[0]/values[2] (X/Z)" to abs(dx - dz),
        )
        val undetectable = pairs.filterValues { it < tolerance }.keys
        assumeTrue(
            "at this attitude Y and Z read alike (y=$dy z=$dz), so the transposition " +
                "this test exists for is not detectable; stand the device flatter or " +
                "more upright",
            pairs.getValue("values[1]/values[2] (Y/Z)") >= tolerance,
        )
        // Said out loud rather than left for a reader to work out from the attitude:
        // which swaps this run could not have caught.
        if (undetectable.isNotEmpty()) {
            android.util.Log.i(
                "ImuWireTest",
                "attitude ax=$dx ay=$dy az=$dz leaves these transpositions " +
                    "undetectable: ${undetectable.joinToString()}",
            )
        }
        assertTrue(
            "the wire's axes do not match the platform's: wire ax=$ax ay=$ay az=$az, " +
                "sensor x=$dx y=$dy z=$dz",
            abs(ax - dx) < tolerance && abs(ay - dy) < tolerance && abs(az - dz) < tolerance,
        )
        // And gravity is on exactly one axis, wherever the device is lying, so a reading
        // that lost a component does not pass by agreeing with an equally lost one.
        assertTrue(
            "no axis carries gravity: ax=$ax ay=$ay az=$az",
            listOf(ax, ay, az).any { abs(abs(it) - GRAVITY) < 3.0 },
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
    fun aCommandRaisingTheRateReachesTheSource() {
        // The direction the gate cannot serve. A RateGate only ever *drops* samples the
        // platform already produced, so raising imu_hz above what was requested at start
        // changed nothing while the pipeline reported the new rate as in force --
        // commanded 200 Hz, measured 50, reported 200, with nothing on either side of the
        // link recording the difference. The source re-requests its period now.
        //
        // Measured on `seen`, the count incremented in the sensor callback, so the
        // quantity is what the platform delivered to this process. Two quantities were
        // rejected for this, for opposite reasons.
        //
        // `stats.rateHz` is `gate.hz`, the value `setRate` stored. It is the number that
        // was lying in the original defect, so asserting on it would restore the exact
        // blindness this test exists to escape. The same goes for `ImuSource.requestedHz`.
        //
        // Frames the peer received -- what this test used to measure -- is bounded by the
        // smaller of the source rate and the socket's drain rate. On handset ZY227VV4XC
        // the socket drains about 50 frames/s and the 50 Hz baseline already sits there,
        // so a raise cannot show: measured 53.0/s and 53.3/s against a 56.8/s floor while
        // the source was running at the raised rate, with the session recording
        // `enqueued=756, sent=525, dropped=0` -- 231 samples were queued, not lost. That
        // makes the wire rate fail under two different conditions, a command that never
        // reached the source and a socket that cannot carry the result, with no way to
        // say which. The peer-received rate is still measured and logged below, because
        // the transport's ceiling is worth seeing; it is not what the assertion turns on.
        SensingService.start(context)
        awaitState(SensingState.RUNNING)
        assertTrue(pollUntil(15_000) { sessions.any { it.isRunning } })
        assertTrue(pollUntil(10_000) { imuFrames().isNotEmpty() })

        // Above the rate the source was started at, which is the only direction that
        // discriminates. Commanding down and then back up does not: the source began at
        // 50 Hz, so a version that never re-requests is still sitting at 50 when the raise
        // arrives and produces the same answer. Both mutations survived that sequence.
        // Two baseline windows, so the noise floor is measured on this device rather
        // than assumed. The delivered rate is bounded by the slower of the two sensors
        // and by the pairing, not by the commanded rate alone, so how big a raise looks
        // is a property of the handset.
        fun sourceSeen() = requireNotNull(SensingService.liveImu).stats.seen
        fun refusedBySink() = requireNotNull(SensingService.liveImu).stats.refusedBySink

        val firstFrom = sourceSeen()
        val firstWireFrom = imuFrames().size
        Thread.sleep(3_000)
        val first = (sourceSeen() - firstFrom) / 3.0
        val firstWire = (imuFrames().size - firstWireFrom) / 3.0
        val secondFrom = sourceSeen()
        val secondWireFrom = imuFrames().size
        Thread.sleep(3_000)
        val second = (sourceSeen() - secondFrom) / 3.0
        val secondWire = (imuFrames().size - secondWireFrom) / 3.0
        val baseline = maxOf(first, second)
        val noise = abs(first - second)
        val refusedBefore = refusedBySink()

        // Commanded at what this device can actually deliver. 200 Hz is above the
        // accelerometer's maximum on some handsets -- on moto g power the raise produced
        // 51.3 samples/s against a 50 Hz baseline, so the test failed on hardware that
        // was behaving correctly. The sensor's `minDelay` is the platform's own
        // statement of its fastest rate, so ask it rather than assume.
        val maxHz = fastestAccelerometerHz()
        assumeTrue(
            "this device's accelerometer tops out at ${"%.0f".format(maxHz)} Hz, which " +
                "is not enough above the ${"%.0f".format(baseline)}/s baseline for a " +
                "raise to be distinguishable",
            maxHz > baseline * 1.8,
        )
        val commanded = minOf(200.0, maxHz)

        command(imuHz = commanded)
        Thread.sleep(1_000)
        val raisedFrom = sourceSeen()
        val raisedWireFrom = imuFrames().size
        Thread.sleep(3_000)
        val raised = (sourceSeen() - raisedFrom) / 3.0
        val raisedWire = (imuFrames().size - raisedWireFrom) / 3.0

        // Above the noise this device actually shows, not above a fixed multiple. The
        // emulator's raise is large and the handset's is not: measured 51.7/s to 60.7/s,
        // a 17 per cent increase, which the previous 1.4x threshold rejected while the
        // code under test was doing its job. The floor is the larger of three times the
        // baseline-to-baseline variation and ten per cent, so a raise has to clear the
        // measurement rather than merely exceed it.
        val floor = baseline + maxOf(3.0 * noise, 0.10 * baseline)
        // Recorded rather than asserted: how much of the raise the link carried. The two
        // rates diverging is the transport's ceiling, not a fault in the rate command,
        // and a reader of this output should be able to see it without running anything.
        android.util.Log.i(
            "ImuWireTest",
            "source $first/s and $second/s -> $raised/s; " +
                "wire $firstWire/s and $secondWire/s -> $raisedWire/s",
        )
        assertTrue(
            "commanded ${"%.0f".format(commanded)} Hz from source baselines $first/s " +
                "and $second/s and measured $raised/s, which does not clear $floor/s: " +
                "a raise the gate cannot serve must reach the source (the wire carried " +
                "$raisedWire/s, which this assertion does not turn on)",
            raised > floor,
        )
        // Bought back from the wire measurement, which did cover it: the sink refusing
        // what the source produced. This does NOT cover the channel evicting a sample --
        // `Session.enqueue` drops the oldest and returns true, so an eviction is not a
        // refusal -- and the phone-side channel counters are not reachable from an
        // instrumented test, so that case is uncovered here and named rather than implied.
        assertEquals(
            "the sink refused ${refusedBySink() - refusedBefore} samples across the " +
                "raise; a raise that reaches the source and is then refused is not a " +
                "raise that took effect",
            refusedBefore,
            refusedBySink(),
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

    @Test
    fun telemetryReachesTheJetsonWithARealThermalReading() {
        // Task 24's deliverable, asserted where it lands. The phone reports and the Jetson
        // decides -- so what matters is that a frame arrives, that the thermal status is a
        // word from the wire's vocabulary rather than a stringified Android integer, and
        // that `achieved` is a rate the far side can compare against what it commanded.
        SensingService.start(context)
        awaitState(SensingState.RUNNING)
        assertTrue(pollUntil(15_000) { sessions.any { it.isRunning } })

        // Two reporting periods: the first establishes the baseline and sends nothing.
        assertTrue(
            "no telemetry arrived: ${SensingService.liveTelemetry?.stats}",
            pollUntil(15_000) { telemetryFrames().isNotEmpty() },
        )

        val report = PhoneTelemetry.fromWire(
            telemetryFrames().last().header.entries,
            telemetryFrames().last().payload,
        )
        assertTrue(
            "thermal_status is '${report.thermalStatus}', not a word from the wire's set",
            report.thermalStatus in setOf(
                "nominal", "light", "moderate", "severe", "critical", "emergency", "shutdown",
            ),
        )
        assertTrue(
            "achieved has no imu rate: ${report.achieved}",
            report.achieved.containsKey("imu_hz"),
        )
        // The IMU is running at 50 Hz by default and its frames are arriving, so the
        // reported rate must be a real one rather than the zero a broken baseline gives.
        assertTrue(
            "achieved imu_hz is ${report.achieved["imu_hz"]} while imu frames are arriving",
            report.achieved.getValue("imu_hz") > 1.0,
        )
    }

    @Test
    fun theSessionLogRecordsTheHeadersThatWentOnTheWire() {
        // Ground truth, asserted against the wire rather than against itself. The log's
        // whole design is that it records the encoded header, so a line in the file and a
        // frame the peer received are the same object -- and the way to show that is to
        // compare them, not to check the file is non-empty.
        SensingService.start(context)
        awaitState(SensingState.RUNNING)
        assertTrue(pollUntil(15_000) { sessions.any { it.isRunning } })
        assertTrue(pollUntil(10_000) { imuFrames().size > 20 })

        val received = imuFrames().take(20).map { frame ->
            com.dsrc.transport.Json.encode(
                com.dsrc.transport.Framing.withPayloadLength(frame.header, frame.payload.size),
            )
        }

        SensingService.stop(context)
        awaitState(SensingState.IDLE)

        // Every session file, not the newest by mtime. Earlier tests in this class each
        // leave one, filenames are second-resolution so several can share a name, and
        // picking "the newest" made this pass or fail on which file mtime happened to
        // favour -- it failed once with all twenty frames "absent" and passed unchanged on
        // the next run. The claim is that these frames were logged; searching every file
        // asserts exactly that and nothing about file selection.
        val logs = java.io.File(context.filesDir, "sessions").listFiles().orEmpty()
        assertTrue("no session log was written", logs.isNotEmpty())
        val lines = logs.flatMap { it.readLines() }.toSet()

        assertTrue("the log is empty", lines.isNotEmpty())
        val missing = received.filterNot { it in lines }
        assertTrue(
            "${missing.size} of 20 frames the peer received are absent from ${lines.size} " +
                "logged lines across ${logs.size} files.\n  wanted: ${missing.firstOrNull()}",
            missing.isEmpty(),
        )
    }

    private fun telemetryFrames() = frames.filter { it.channel == Channels.TELEMETRY }

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
    /** The accelerometer's fastest rate, as the platform states it. */
    private fun fastestAccelerometerHz(): Double {
        val manager = InstrumentationRegistry.getInstrumentation().targetContext
            .getSystemService(SensorManager::class.java)
        val sensor = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) ?: return 0.0
        // `minDelay` is the shortest interval the sensor supports, in microseconds. Zero
        // means on-change only, which cannot serve a rate at all.
        return if (sensor.minDelay <= 0) 0.0 else 1_000_000.0 / sensor.minDelay
    }

    /** One accelerometer reading straight from the platform, or null on timeout. */
    private fun readAccelerometerDirectly(timeoutMs: Long = 4_000): Triple<Double, Double, Double>? {
        val manager = InstrumentationRegistry.getInstrumentation().targetContext
            .getSystemService(SensorManager::class.java)
        val sensor = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) ?: return null
        val latch = java.util.concurrent.CountDownLatch(1)
        var reading: Triple<Double, Double, Double>? = null
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                if (reading == null) {
                    reading = Triple(event.values[0].toDouble(),
                                     event.values[1].toDouble(),
                                     event.values[2].toDouble())
                    latch.countDown()
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        manager.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        try {
            latch.await(timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS)
        } finally {
            manager.unregisterListener(listener)
        }
        return reading
    }

}
