package com.dsrc.phone.sensors

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Size
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import com.dsrc.phone.config.SensingConfig
import java.util.concurrent.ExecutorService
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * The real camera, via CameraX `ImageAnalysis`.
 *
 * `ImageAnalysis` rather than `ImageCapture` because this is a continuous stream with
 * backpressure we want to control, not a series of shutter presses.
 * `STRATEGY_KEEP_ONLY_LATEST` is chosen to match the `camera` channel's `latest_wins`
 * policy exactly, so the camera and the wire agree about what loss means.
 *
 * The analyzer's one hard obligation is closing every [ImageProxy] exactly once.
 * CameraX hands out a bounded pool; leak one and the stream stops permanently after
 * `maxImages` with no exception raised, which is why it is closed in a `finally` and
 * why the pixels are copied out before the callback returns rather than being
 * compressed while the buffer is still held.
 */
class CameraXSource(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val config: SensingConfig,
) : CameraSource {

    private var provider: ProcessCameraProvider? = null
    private var analysisExecutor: ExecutorService? = null

    @Volatile
    private var stopped = false

    @Volatile
    var closeFailures: Long = 0
        private set

    @Volatile
    var bindFailures: Long = 0
        private set

    override fun start(pipeline: CameraPipeline) {
        stopped = false
        val analysis = Executors.newSingleThreadExecutor()
        analysisExecutor = analysis

        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            // This listener runs later, on the main thread, and the owner may be gone by
            // then -- a sensing session that stops before the provider resolves is
            // ordinary, not exceptional. bindToLifecycle on a destroyed lifecycle throws
            // IllegalArgumentException on the main thread, which is an unrecoverable
            // process death that no try/catch at the call site can reach, because the
            // call site returned long ago.
            if (stopped || lifecycleOwner.lifecycle.currentState == Lifecycle.State.DESTROYED) {
                Log.i(TAG, "camera provider resolved after teardown; not binding")
                return@addListener
            }
            try {
                bind(future.get(), analysis, pipeline)
            } catch (t: Throwable) {
                // Anything else the bind can raise -- no camera on the device, the
                // provider failing to initialise -- is counted rather than fatal.
                bindFailures++
                Log.e(TAG, "binding the camera failed", t)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bind(
        cameraProvider: ProcessCameraProvider,
        analysis: ExecutorService,
        pipeline: CameraPipeline,
    ) {
        provider = cameraProvider

        val analyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            // Without this the configured resolution is dead: the camera picks its own
            // default and the width/height in SensingConfig describe nothing.
            //
            // The fallback prefers *lower*. A device need not offer the requested size,
            // and preferring higher overshoots without limit -- asking the emulator for
            // 1280x720 got 1856x1392, nearly four times the pixels, which is four times
            // the encode cost and the payload on a link that has to carry it. Going
            // under is the safe direction for a phone; going over is not.
            //
            // Either way the configured size is a *request*. What arrives is whatever
            // the device chose, which is why CapturedFrame reports the ImageProxy's own
            // dimensions rather than the config's.
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(config.cameraWidth, config.cameraHeight),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER,
                        )
                    )
                    .build()
            )
            .build()

        analyzer.setAnalyzer(analysis) { image ->
            try {
                handle(image, pipeline)
            } catch (t: Throwable) {
                // A throw out of an analyzer is swallowed by CameraX, so it would
                // otherwise be a stream that quietly produces nothing.
                Log.e(TAG, "analyzer failed", t)
            } finally {
                try {
                    image.close()
                } catch (t: Throwable) {
                    closeFailures++
                    Log.e(TAG, "closing the image failed", t)
                }
            }
        }

        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            analyzer,
        )
        // Logged because the request and the result can differ, and only the result
        // matters for the encode cost and the payload size.
        Log.i(
            TAG,
            "camera bound: ${config.cameraHz} Hz target, requested " +
                "${config.cameraWidth}x${config.cameraHeight}",
        )
    }

    private fun handle(image: ImageProxy, pipeline: CameraPipeline) {
        val width = image.width
        val height = image.height
        // elapsedRealtimeNanos, not the sensor timestamp: it has to share a clock with
        // the transport's enqueue stamp for `t_mono_ns - t_capture_mono_ns` to mean
        // anything. See plan_task18_camera_capture.md O1.
        val timestampNs = android.os.SystemClock.elapsedRealtimeNanos()

        pipeline.offer(
            timestampNs = timestampNs,
            width = width,
            height = height,
            pack = {
                val y = image.planes[0]
                val u = image.planes[1]
                val v = image.planes[2]
                YuvPacker.toNv21(
                    y = y.buffer.toByteArray(),
                    u = u.buffer.toByteArray(),
                    v = v.buffer.toByteArray(),
                    width = width,
                    height = height,
                    yRowStride = y.rowStride,
                    uvRowStride = u.rowStride,
                    uvPixelStride = u.pixelStride,
                )
            },
            compress = JpegEncoder::compress,
        )
    }

    override fun stop() {
        // Set first, so a provider callback still in flight sees it and declines to
        // bind rather than racing the teardown.
        stopped = true

        // unbindAll must run on the main thread. Service callbacks already are, but a
        // caller on any other thread would otherwise get IllegalStateException("Not in
        // application's main thread") -- so the marshalling belongs here rather than
        // being an unwritten requirement on every caller.
        onMainThreadBlocking {
            provider?.unbindAll()
            provider = null
        }

        analysisExecutor?.shutdown()
        analysisExecutor = null
    }

    private fun onMainThreadBlocking(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            block()
            return
        }
        val latch = CountDownLatch(1)
        Handler(Looper.getMainLooper()).post {
            try {
                block()
            } finally {
                // Counted down in a finally: a throw inside the block would otherwise
                // hold the caller for the full timeout and hide the real error.
                latch.countDown()
            }
        }
        if (!latch.await(UNBIND_TIMEOUT_S, TimeUnit.SECONDS)) {
            Log.w(TAG, "timed out waiting for the camera to unbind")
        }
    }

    private companion object {
        const val TAG = "CameraXSource"
        const val UNBIND_TIMEOUT_S = 5L

        fun java.nio.ByteBuffer.toByteArray(): ByteArray {
            rewind()
            return ByteArray(remaining()).also { get(it) }
        }
    }
}
