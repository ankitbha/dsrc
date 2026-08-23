package com.dsrc.phone.sensors

import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicLong

/**
 * The camera path, with no Android in it.
 *
 * Sits between whatever produces frames and whatever drains them, and owns the three
 * things that decide whether the stream is honest: the rate gate, the frame ids, and
 * the counters.
 *
 * The split between packing and compressing is deliberate and is the reason this
 * takes two lambdas. Copying bytes out of a camera buffer is a memcpy and has to
 * happen before the buffer is handed back, so it runs on the caller's thread;
 * compressing is tens of milliseconds at 720p and would stall the camera's analyzer
 * if it ran there, so it goes to [encodeExecutor]. Gating before either means a
 * rejected frame costs neither.
 */
class CameraPipeline(
    config: com.dsrc.phone.config.SensingConfig,
    private val encodeExecutor: Executor,
    private val buffer: FrameBuffer = FrameBuffer(),
) {
    private val gate = RateGate(config.cameraHz)
    private val nextFrameId = AtomicLong(0)
    private val quality = config.jpegQuality

    @Volatile
    private var running = true

    private val seen = AtomicLong(0)
    private val accepted = AtomicLong(0)
    private val encoded = AtomicLong(0)
    private val encodeFailures = AtomicLong(0)
    private val packFailures = AtomicLong(0)
    private val refusedStopped = AtomicLong(0)
    private val gatedCount = AtomicLong(0)
    private val abandoned = AtomicLong(0)

    /**
     * Offer a frame.
     *
     * @param timestampNs the frame's capture stamp, on the device's monotonic clock
     * @param pack copies the pixels out; called on this thread, only if the frame is kept
     * @param compress turns packed bytes into JPEG; called on [encodeExecutor]
     * @return whether the frame was kept
     */
    fun offer(
        timestampNs: Long,
        width: Int,
        height: Int,
        pack: () -> ByteArray,
        compress: (ByteArray, Int, Int, Int) -> ByteArray,
    ): Boolean {
        seen.incrementAndGet()
        if (!running) {
            // Counted apart from the gate's rejections: a frame refused because sensing
            // stopped is not the ordinary cost of a commanded rate below the sensor's,
            // and reporting both as `gated` made a teardown look like rate limiting.
            refusedStopped.incrementAndGet()
            return false
        }
        if (!gate.accept(timestampNs)) {
            // Counted rather than derived. Deriving it as seen - accepted - refused made
            // the balance identity read `seen == seen`, so it held for every input --
            // including seen=5 with accepted=100 -- and the test asserting it passed
            // vacuously.
            gatedCount.incrementAndGet()
            return false
        }

        // The id counts frames the gate kept, not frames the sensor produced, so a gap
        // means "lost after acceptance" -- something worth acting on -- rather than
        // "the camera runs faster than the commanded rate", which is normal.
        val frameId = nextFrameId.incrementAndGet()
        accepted.incrementAndGet()

        val packed = try {
            pack()
        } catch (t: Throwable) {
            // YuvPacker refuses a geometry it cannot handle, and that call happens here.
            // Letting it propagate left `accepted` incremented with no matching encode
            // or failure, so total failure was indistinguishable from an encoder
            // backlog -- a broken stream reported as a busy one.
            packFailures.incrementAndGet()
            return false
        }
        encodeExecutor.execute {
            // Re-checked inside the task: stop() can land between submission and
            // execution, and a frame encoded after teardown would arrive in a buffer
            // nobody is draining.
            if (!running) {
                // Same shape as the pack early-return: without a counter this frame
                // leaves `accepted` incremented and no outcome recorded, so `inFlight`
                // never returns to zero and a teardown reads as an encoder backlog.
                abandoned.incrementAndGet()
                return@execute
            }
            try {
                val jpeg = compress(packed, width, height, quality)
                buffer.offer(
                    CapturedFrame(
                        frameId = frameId,
                        width = width,
                        height = height,
                        format = "jpeg",
                        quality = quality,
                        captureMonoNs = timestampNs,
                        jpeg = jpeg,
                    )
                )
                encoded.incrementAndGet()
            } catch (t: Throwable) {
                // One bad frame costs one frame. Letting it out would kill the encoder
                // thread and stop the stream silently, which is the failure mode the
                // protocol's drop-and-count rule exists to avoid.
                encodeFailures.incrementAndGet()
            }
        }
        return true
    }

    fun setRate(hz: Double) = gate.setRate(hz)

    val rateHz: Double get() = gate.hz

    fun drain(): CapturedFrame? = buffer.drain()

    /**
     * Stop accepting and discard anything held.
     *
     * Frames already queued on the executor check [running] and drop out, so a stop
     * does not have to wait for the encoder to finish a backlog.
     */
    fun stop() {
        running = false
        // Closing rather than clearing: a frame whose compression began before this
        // point finishes after it, and the buffer has to refuse the late arrival rather
        // than hold it for a session that has ended.
        buffer.close()
    }

    val stats: Stats
        get() = Stats(
            seen = seen.get(),
            accepted = accepted.get(),
            encoded = encoded.get(),
            encodeFailures = encodeFailures.get(),
            packFailures = packFailures.get(),
            refusedStopped = refusedStopped.get(),
            gated = gatedCount.get(),
            abandoned = abandoned.get(),
            buffer = buffer.stats,
        )

    data class Stats(
        val seen: Long,
        val accepted: Long,
        val encoded: Long,
        val encodeFailures: Long,
        val packFailures: Long,
        val refusedStopped: Long,
        /** Frames the gate rejected: the normal cost of a commanded rate below the sensor's. */
        val gated: Long,
        /** Frames dropped because sensing stopped between submission and encoding. */
        val abandoned: Long,
        val buffer: FrameBuffer.Stats,
    ) {
        /** Accepted frames whose outcome is not yet known. */
        val inFlight: Long
            get() = accepted - encoded - encodeFailures - packFailures - abandoned

        /**
         * Every frame the camera delivered is accounted for under exactly one heading.
         *
         * Each term is counted where it happens, not derived from the others -- a
         * derived term makes this an identity that cannot fail.
         */
        val balances: Boolean get() = seen == accepted + gated + refusedStopped
    }
}
