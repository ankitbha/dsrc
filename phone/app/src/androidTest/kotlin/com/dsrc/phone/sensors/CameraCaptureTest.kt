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

        val closeFailures: Long get() = source.closeFailures
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

        assertTrue(
            "quality 30 produced ${lowFrame.jpeg.size} bytes against ${highBytes} at 95, " +
                "so the setting is not reaching the encoder",
            lowFrame.jpeg.size < highBytes,
        )
    }

}
