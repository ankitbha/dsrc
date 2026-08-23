package com.dsrc.phone.sensors

import android.content.Context
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import com.dsrc.phone.PermissionModel
import com.dsrc.phone.config.SensingConfig
import java.util.concurrent.Executors
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The measurements task 18's plan asks for, taken on the emulator's virtual camera.
 *
 * Separate from [CameraCaptureTest] because these are measurements rather than
 * assertions: they print numbers under the tag `CAMERA_MEASUREMENT` for the report and
 * assert only enough to prove the run happened. A measurement that silently produced
 * nothing must not read as a pass.
 *
 * The virtual camera renders a synthetic scene, so JPEG sizes here say nothing about a
 * road. The rate and drop numbers are about this code and do transfer.
 */
@RunWith(AndroidJUnit4::class)
class CameraMeasurementTest {

    @get:Rule
    val permissions: GrantPermissionRule = GrantPermissionRule.grant(PermissionModel.CAMERA)

    private val context: Context get() = ApplicationProvider.getApplicationContext()
    private var harness: Harness? = null

    @After
    fun tearDown() {
        harness?.stop()
        harness = null
    }

    private inner class Harness(config: SensingConfig) {
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
    }

    private fun start(config: SensingConfig) = Harness(config).also { harness = it; it.start() }

    private fun awaitFirstFrame(pipeline: CameraPipeline, timeoutMs: Long): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (pipeline.stats.accepted > 0) return true
            Thread.sleep(20)
        }
        return false
    }

    private fun log(line: String) = Log.i(TAG, line)

    @Test
    fun measureAchievedRateAcrossCommandedRates() {
        for (hz in listOf(1.0, 5.0, 15.0)) {
            val h = start(SensingConfig(cameraHz = hz))
            assertTrue("no frames at $hz Hz", awaitFirstFrame(h.pipeline, 20_000))
            val before = h.pipeline.stats
            val startMs = System.currentTimeMillis()
            Thread.sleep(10_000)
            val elapsedS = (System.currentTimeMillis() - startMs) / 1000.0
            val after = h.pipeline.stats

            val accepted = after.accepted - before.accepted
            val seen = after.seen - before.seen
            val achieved = accepted / elapsedS
            val sourceRate = seen / elapsedS
            log(
                "rate commanded=%.1f achieved=%.2f source=%.2f accepted=%d seen=%d gated=%d over %.1fs"
                    .format(hz, achieved, sourceRate, accepted, seen, seen - accepted, elapsedS)
            )
            h.stop()
            harness = null
        }
    }

    @Test
    fun measureJpegSizeAndEncodeCost() {
        val config = SensingConfig(cameraHz = 15.0, jpegQuality = 85)
        val h = start(config)
        assertTrue(awaitFirstFrame(h.pipeline, 20_000))

        val sizes = mutableListOf<Int>()
        var geometry: Triple<Int, Int, String>? = null
        val deadline = System.currentTimeMillis() + 15_000
        while (System.currentTimeMillis() < deadline && sizes.size < 60) {
            val frame = h.pipeline.drain()
            if (frame == null) {
                Thread.sleep(10)
            } else {
                sizes.add(frame.jpeg.size)
                // Captured inside the loop. Draining afterwards returns null, which
                // reported the geometry as "null" and told us nothing.
                if (geometry == null) geometry = Triple(frame.width, frame.height, frame.format)
            }
        }
        assertTrue("no frames measured", sizes.size >= 10)

        val sorted = sizes.sorted()
        fun pct(p: Double) = sorted[((sorted.size - 1) * p).toInt()]
        log(
            "jpeg n=%d quality=%d size_bytes min=%d p50=%d p95=%d max=%d mean=%.0f"
                .format(
                    sizes.size, config.jpegQuality, sorted.first(), pct(0.5), pct(0.95),
                    sorted.last(), sizes.average(),
                )
        )
        log(
            "frame geometry width=%d height=%d format=%s (configured %dx%d)".format(
                geometry!!.first, geometry.second, geometry.third,
                config.cameraWidth, config.cameraHeight,
            )
        )
    }

    @Test
    fun measureDropsUnderASlowDrain() {
        // latest_wins accounting: with the drain slower than the commanded rate, every
        // frame beyond the one held must be counted as dropped, not silently lost.
        val h = start(SensingConfig(cameraHz = 15.0))
        assertTrue(awaitFirstFrame(h.pipeline, 20_000))

        var drained = 0
        val deadline = System.currentTimeMillis() + 10_000
        while (System.currentTimeMillis() < deadline) {
            Thread.sleep(500)
            if (h.pipeline.drain() != null) drained++
        }
        val stats = h.pipeline.stats
        log(
            "slow drain: accepted=%d encoded=%d drained=%d dropped=%d holding=%s balances=%s"
                .format(
                    stats.accepted, stats.encoded, drained, stats.buffer.dropped,
                    stats.buffer.holding, stats.buffer.balances,
                )
        )
        assertTrue("the accounting must balance under loss: ${stats.buffer}", stats.buffer.balances)
        assertTrue("a slow drain should have dropped frames", stats.buffer.dropped > 0)
    }

    @Test
    fun measureASustainedRun() {
        val h = start(SensingConfig(cameraHz = 10.0))
        assertTrue(awaitFirstFrame(h.pipeline, 20_000))
        val before = h.pipeline.stats
        val startMs = System.currentTimeMillis()

        var drained = 0
        while (System.currentTimeMillis() - startMs < 30_000) {
            if (h.pipeline.drain() != null) drained++ else Thread.sleep(5)
        }
        val elapsedS = (System.currentTimeMillis() - startMs) / 1000.0
        val after = h.pipeline.stats
        log(
            "sustained %.0fs: seen=%d accepted=%d encoded=%d failures=%d drained=%d dropped=%d achieved=%.2f Hz"
                .format(
                    elapsedS, after.seen - before.seen, after.accepted - before.accepted,
                    after.encoded - before.encoded, after.encodeFailures, drained,
                    after.buffer.dropped - before.buffer.dropped,
                    (after.accepted - before.accepted) / elapsedS,
                )
        )
        assertTrue(
            "the stream stalled during the sustained run",
            after.accepted - before.accepted > 100,
        )
        assertTrue("encoding must not be failing", after.encodeFailures == 0L)
    }

    private companion object {
        const val TAG = "CAMERA_MEASUREMENT"
    }
}
