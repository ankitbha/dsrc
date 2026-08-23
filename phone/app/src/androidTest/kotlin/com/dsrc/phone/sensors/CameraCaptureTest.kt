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
        // Whole-file markers. A stride mistake in the packer yields a JPEG that decodes
        // to a sheared or green-bottomed image rather than an error, so the dimension
        // check below is the part that would actually catch it.
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
}
