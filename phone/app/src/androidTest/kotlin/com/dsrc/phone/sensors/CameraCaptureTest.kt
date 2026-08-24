package com.dsrc.phone.sensors

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import com.dsrc.phone.PermissionModel
import com.dsrc.phone.config.SensingConfig
import java.util.concurrent.Executors
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The real camera, on the emulator's virtual device.
 *
 * The pipeline's policy is unit-tested. What needs hardware is whether CameraX
 * actually delivers, whether the `ImageProxy` pool discipline holds over a sustained
 * run, and whether the platform JPEG encoder accepts what [YuvPacker] produces. The
 * pool one matters most: a leaked proxy stops the stream permanently once `maxImages`
 * are outstanding, and raises nothing, so the only symptom is frames ceasing.
 */
@RunWith(AndroidJUnit4::class)
class CameraCaptureTest {

    @get:Rule
    val permissions: GrantPermissionRule = GrantPermissionRule.grant(PermissionModel.CAMERA)

    private val context: Context get() = ApplicationProvider.getApplicationContext()
    private var harness: CameraHarness? = null

    @After
    fun tearDown() {
        harness?.stop()
        harness = null
    }

    /** Owns a lifecycle so CameraX has something to bind to outside a service. */
    private inner class CameraHarness(config: SensingConfig) {
        private val owner = TestLifecycleOwner()
        private val encoder = Executors.newSingleThreadExecutor()
        private val source = CameraXSource(context, owner, config)
        val pipeline = CameraPipeline(config, encoder)

        fun start() {
            owner.start()
            source.start(pipeline)
        }

        fun stop() {
            source.stop()
            pipeline.stop()
            encoder.shutdown()
            owner.stop()
        }

        /**
         * Stop the source only, leaving the lifecycle alive.
         *
         * This is production's shape: `SensingService` is the LifecycleOwner and
         * `onSensingDown` runs on a Stop intent with the service still alive, so nothing
         * destroys the lifecycle. Calling `owner.stop()` as well -- which the full teardown
         * does -- masks the source's own release entirely, because CameraX unbinds when a
         * lifecycle is destroyed whatever the source did. That is why deleting
         * `unbindAll()` from `stop()` survived.
         */
        fun stopSourceOnly() {
            source.stop()
        }

        val closeFailures: Long get() = source.closeFailures
        val bindFailures: Long get() = source.bindFailures
    }

    private fun start(config: SensingConfig): CameraHarness =
        CameraHarness(config).also { harness = it; it.start() }

    /** Drains until [atLeast] frames have arrived, or the deadline passes. */
    private fun awaitFrames(pipeline: CameraPipeline, atLeast: Int, timeoutMs: Long): Int {
        val deadline = System.currentTimeMillis() + timeoutMs
        var drained = 0
        while (System.currentTimeMillis() < deadline && drained < atLeast) {
            if (pipeline.drain() != null) drained++ else Thread.sleep(20)
        }
        return drained
    }

    private fun awaitOneFrame(pipeline: CameraPipeline, timeoutMs: Long): CapturedFrame? {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            pipeline.drain()?.let { return it }
            Thread.sleep(20)
        }
        return null
    }

    @Test
    fun theCameraDeliversFrames() {
        val h = start(SensingConfig(cameraHz = 10.0))
        val drained = awaitFrames(h.pipeline, 5, 20_000)
        assertTrue("expected frames, drained $drained (${h.pipeline.stats})", drained >= 5)
    }

    @Test
    fun framesCarryDecodableJpegAtTheReportedSize() {
        val h = start(SensingConfig(cameraHz = 10.0))
        val frame = awaitOneFrame(h.pipeline, 20_000)
        assertNotNull("no frame arrived: ${h.pipeline.stats}", frame)
        val f = frame!!

        assertEquals("jpeg", f.format)
        assertTrue("a JPEG should not be tiny", f.jpeg.size > 1_000)
        // Whole-file markers only. Neither these nor the dimension check below can see a
        // stride mistake or a chroma swap: both preserve the size and both decode
        // cleanly. JpegEncoderTest is what catches those, by driving a known colour
        // through the packer and the platform encoder.
        assertEquals("SOI", 0xFF.toByte(), f.jpeg[0])
        assertEquals("SOI", 0xD8.toByte(), f.jpeg[1])
        assertEquals("EOI", 0xFF.toByte(), f.jpeg[f.jpeg.size - 2])
        assertEquals("EOI", 0xD9.toByte(), f.jpeg[f.jpeg.size - 1])

        val bitmap = BitmapFactory.decodeByteArray(f.jpeg, 0, f.jpeg.size)
        assertNotNull("the JPEG must decode", bitmap)
        assertEquals("decoded width must match the reported width", f.width, bitmap!!.width)
        assertEquals("decoded height must match the reported height", f.height, bitmap.height)
    }

    @Test
    fun theCommandedRateIsHonoured() {
        val h = start(SensingConfig(cameraHz = 5.0))
        awaitFrames(h.pipeline, 1, 20_000)
        val before = h.pipeline.stats.accepted
        Thread.sleep(6_000)
        val accepted = h.pipeline.stats.accepted - before
        // Wide bounds on purpose: the virtual camera is not a metronome, and the claim
        // is that the gate limits, not that it is exact.
        assertTrue("expected roughly 30 in 6 s at 5 Hz, got $accepted", accepted in 15..45)
        assertTrue("the gate must be rejecting something (${h.pipeline.stats})", h.pipeline.stats.gated > 0)
    }

    @Test
    fun aSustainedRunDoesNotStall() {
        val h = start(SensingConfig(cameraHz = 10.0))
        awaitFrames(h.pipeline, 1, 20_000)
        val early = h.pipeline.stats.accepted
        Thread.sleep(20_000)
        val late = h.pipeline.stats.accepted
        assertTrue(
            "the stream stalled: accepted went $early -> $late over 20 s (${h.pipeline.stats})",
            late - early > 50,
        )
        assertEquals("every image must close cleanly", 0, h.closeFailures)
    }

    @Test
    fun theCameraCanBeRestarted() {
        val first = start(SensingConfig(cameraHz = 10.0))
        assertTrue(awaitFrames(first.pipeline, 3, 20_000) >= 3)
        first.stop()
        harness = null

        val second = start(SensingConfig(cameraHz = 10.0))
        assertTrue(
            "the camera must be released on stop, or a second bind gets nothing",
            awaitFrames(second.pipeline, 3, 20_000) >= 3,
        )
    }

    @Test
    fun theRequestedAspectRatioReachesTheCamera() {
        // Task 18's headline result had no test. Deleting the aspect-ratio strategy from
        // CameraXSource restores exactly the documented bug -- asking for 1280x720 on a
        // device that offers it returns 640x480 -- and all five tests here passed, because
        // framesCarryDecodableJpegAtTheReportedSize compares the decoded bitmap against the
        // frame's *own* width and height, which is self-consistent at any size.
        //
        // The ratio is what the fix controls, and asserting it is hardware-independent:
        // ResolutionSelector defaults to 4:3, so without the strategy a 16:9 request is
        // filtered out of the candidate set before any resolution rule runs. 640x480 is
        // 1.333 against a requested 1.778.
        val config = SensingConfig(cameraHz = 5.0, cameraWidth = 1280, cameraHeight = 720)
        val harness = start(config)
        val frame = awaitOneFrame(harness.pipeline, 15_000)
        assertNotNull("no frame arrived", frame)
        frame!!

        val requested = config.cameraWidth.toDouble() / config.cameraHeight
        val delivered = frame.width.toDouble() / frame.height
        assertEquals(
            "the requested ${config.cameraWidth}x${config.cameraHeight} ratio did not reach " +
                "the camera: got ${frame.width}x${frame.height}",
            requested,
            delivered,
            0.02,
        )

        // And on this emulator the exact size is available, so the request should be met
        // outright. Stated separately from the ratio because it is the one part of this
        // that is a property of the device rather than of the code.
        assertEquals("emulator offers 1280x720 exactly", 1280, frame.width)
        assertEquals("emulator offers 1280x720 exactly", 720, frame.height)
    }

    @Test
    fun theConfiguredJpegQualityReachesTheEncoder() {
        // jpegQuality reached no assertion anywhere: the pipeline's compress(..., quality)
        // could be hardcoded to 1 and CapturedFrame.quality set to null, and the whole
        // suite -- JVM and instrumented -- stayed green. CameraMeasurementTest logs
        // config.jpegQuality rather than the frame's.
        //
        // Two qualities, because asserting one value proves only that *a* number arrived.
        // A low quality must produce a materially smaller JPEG at the same resolution.
        val high = start(SensingConfig(cameraHz = 5.0, jpegQuality = 95))
        val highFrame = awaitOneFrame(high.pipeline, 15_000)
        assertNotNull("no frame at quality 95", highFrame)
        assertEquals("the frame must report the quality it was encoded at", 95, highFrame!!.quality)
        val highBytes = highFrame.jpeg.size
        high.stop()

        val low = start(SensingConfig(cameraHz = 5.0, jpegQuality = 30))
        val lowFrame = awaitOneFrame(low.pipeline, 15_000)
        assertNotNull("no frame at quality 30", lowFrame)
        assertEquals(30, lowFrame!!.quality)

        // A margin, not a bare `<`: the ordering inverted in 2 of 5 runs when the encoder
        // quality was hardcoded, so a bare inequality survived the mutation 3 times in 5.
        //
        // The threshold is 0.72, and both the number and the reasoning behind the previous
        // 0.8 were wrong. I justified 0.8 with "+/-5% scene noise", which is the
        // *intra-session* spread (q95 within one session ranges 27398-31820, about 16%) --
        // a figure this test never touches, because it compares the *first* frame of each
        // session. First-frame spread at fixed quality is 0.74% at q95 and 0.57% at q30, and
        // the true ratio is 0.65, so the real headroom was about twenty sigma rather than
        // eight. The looser threshold was not dangerous, it was undiscriminating:
        // `quality.coerceAtMost(85)` measures 0.77 and survived 0.8. At 0.72 it dies.
        //
        // The constant is specific to the 95-versus-30 pair. At 95 versus 85 the true ratio
        // is 0.85 and this assertion would fail, so it is not portable to other qualities.
        assertTrue(
            "quality 30 produced ${lowFrame.jpeg.size} bytes against $highBytes at 95, " +
                "a ratio of ${"%.3f".format(lowFrame.jpeg.size.toDouble() / highBytes)} -- " +
                "not the ~0.65 a real quality change gives, so the setting is not reaching " +
                "the encoder",
            lowFrame.jpeg.size < highBytes * 0.72,
        )
    }


    @Test
    fun stoppingReleasesTheCameraDevice() {
        // Deleting unbindAll() from CameraXSource.stop() survived, because bind() calls
        // unbindAll() itself before every bind -- so the next start released the camera and
        // the test's stated mechanism ("a second bind gets nothing") was delivered by a
        // different line entirely. What the line in stop() actually buys is that the camera
        // is released *when sensing stops* rather than whenever something next binds: on a
        // phone in a car that is the difference between the camera powering down and staying
        // on for the rest of the drive.
        //
        // The platform records it, so the platform is what gets asked.
        val harness = start(SensingConfig(cameraHz = 5.0))
        assertNotNull("no frame arrived", awaitOneFrame(harness.pipeline, 15_000))
        assertEquals("the camera should be connected while running", "CONNECT", lastCameraEvent())

        // Only the source, so the lifecycle stays alive -- production's shape, and the only
        // arrangement in which the source's own release is observable at all.
        harness.stopSourceOnly()

        val deadline = System.currentTimeMillis() + 10_000
        while (System.currentTimeMillis() < deadline && lastCameraEvent() != "DISCONNECT") {
            Thread.sleep(100)
        }
        assertEquals(
            "the camera was still held after stop(): ${cameraClientLog().take(3)}",
            "DISCONNECT",
            lastCameraEvent(),
        )
    }

    @Test
    fun stoppingImmediatelyAfterStartingDoesNotKillTheProcess() {
        // CameraXSource binds asynchronously, so a stop that lands before the future
        // completes used to bind against a destroyed lifecycle: an IllegalArgumentException
        // on the main thread, which is unrecoverable process death rather than a failed
        // test. The `stopped` flag guards it and deleting the flag survived the suite --
        // nothing raced the bind.
        //
        // This test passing at all is the assertion: if the process dies, the run fails.
        repeat(8) {
            val harness = start(SensingConfig(cameraHz = 5.0))
            // No wait: the bind future is deliberately still in flight.
            harness.stop()
        }
        // And the camera still works afterwards, so the races left nothing wedged.
        val harness = start(SensingConfig(cameraHz = 5.0))
        assertNotNull("the camera was wedged by the start/stop races", awaitOneFrame(harness.pipeline, 15_000))
    }

    @Test
    fun stoppingTheSourceDeclinesAPendingBind() {
        // Third attempt, and the first two failed for different reasons -- both of which I
        // wrote into the plan as "this cannot be observed". That record was wrong.
        //
        //   1. `bindFailures == 0` cannot discriminate, because the mutated bind *succeeds*:
        //      setAnalyzer does not reject a shut-down executor, so nothing throws. The
        //      docstring claiming "without it the bind is attempted and counted as a
        //      failure" was false.
        //   2. "no new CONNECT" was a *length* delta on a saturated ring buffer.
        //      `dumpsys media.camera`'s event log is fixed-capacity and already full at 90
        //      entries, so `log.size - before` is always 0 and `take(0).count { ... }` is
        //      identically zero. The content rotates correctly; only the count was pinned.
        //
        // Watermarking by the newest *entry* rather than by length discriminates: with the
        // flag, logcat shows "provider resolved after teardown; not binding" and no new
        // CONNECT; without it, the bind completes and the camera connects after the stop.
        val watermark = cameraClientLog().firstOrNull()
        val harness = start(SensingConfig(cameraHz = 5.0))
        // No wait: the provider future is deliberately still in flight, and the lifecycle
        // stays alive so only the flag can decline it.
        harness.stopSourceOnly()
        Thread.sleep(3_000)

        val fresh = entriesNewerThan(watermark)
        val connects = fresh.count { it.contains("CONNECT") && !it.contains("DISCONNECT") }
        assertEquals(
            "a bind completed after the source was stopped: ${fresh.take(3)}",
            0,
            connects,
        )
    }

    /**
     * Camera-service entries newer than a watermark entry.
     *
     * By content, not by count. The dump's event log is a saturated fixed-capacity ring, so
     * its length never changes and a size delta is always zero -- which made the previous
     * version of the assertion above inert.
     */
    private fun entriesNewerThan(watermark: String?): List<String> {
        val log = cameraClientLog()
        if (watermark == null) return log
        val index = log.indexOf(watermark)
        return if (index < 0) log else log.take(index)
    }

    /** The camera service's own client log for this package, most recent first. */
    private fun cameraClientLog(): List<String> {
        val dump = shell("dumpsys media.camera")
        return dump.lines()
            .filter { it.contains(context.packageName) && (it.contains("CONNECT") || it.contains("DISCONNECT")) }
            .map { it.trim() }
    }

    private fun lastCameraEvent(): String? =
        cameraClientLog().firstOrNull()?.let { if (it.contains("DISCONNECT")) "DISCONNECT" else "CONNECT" }

    private fun shell(command: String): String {
        val descriptor = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation()
            .uiAutomation.executeShellCommand(command)
        return android.os.ParcelFileDescriptor.AutoCloseInputStream(descriptor).use {
            it.readBytes().toString(Charsets.UTF_8)
        }
    }
}
