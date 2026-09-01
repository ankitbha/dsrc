package com.dsrc.phone.sensors

import com.dsrc.transport.CameraFrameMessage
import com.dsrc.transport.Channels
import com.dsrc.transport.JsonValue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Moves encoded frames from the camera pipeline onto the link.
 *
 * Its own thread, not the encoder's: a send that blocks on a socket write would hold up
 * the next JPEG, and the encoder is single-threaded precisely so `frame_id` stays
 * monotonic. Coupling them would make a slow link look like a slow encoder.
 *
 * It polls rather than being woken, because the alternative is a condition variable
 * inside [FrameBuffer] -- task 18 code that is queued for audit, and not worth disturbing
 * for a wakeup this cheap. At 5 Hz the buffer holds a frame for 200 ms and the poll
 * interval is 20 ms, so a frame waits 10 ms on average.
 *
 * There are two places a camera frame can be dropped and that is a real cost, not an
 * oversight: [FrameBuffer] is depth-1 latest-wins, and so is the `camera` channel queue.
 * The buffer counts its own drops and the channel counts its own, so nothing vanishes
 * unaccounted -- but a receiver reading only the sequence gaps sees the second kind and
 * not the first.
 */
class CameraFrameSender(
    /**
     * Where a frame comes from: [CameraPipeline.drain], normally.
     *
     * A function rather than the pipeline itself, because the loop below is the part with
     * behaviour worth pinning -- drain-until-empty, the sleep only when idle, the stop --
     * and a concrete pipeline cannot be made to yield a scripted sequence.
     */
    private val drain: () -> CapturedFrame?,
    private val send: (String, Map<String, JsonValue>, ByteArray, Boolean) -> Boolean,
    private val pollMs: Long = DEFAULT_POLL_MS,
    private val sleeper: (Long) -> Unit = { Thread.sleep(it) },
) {

    private val stopped = AtomicBoolean(false)
    private var thread: Thread? = null

    private val drained = AtomicLong(0)
    private val sent = AtomicLong(0)
    private val refused = AtomicLong(0)

    fun start() {
        // `check`, not `require`: starting twice is a state problem rather than a bad
        // argument, and IllegalStateException is what a caller would catch for it.
        check(thread == null) { "already started" }
        thread = Thread({ loop() }, "dsrc-camera-send").also {
            it.isDaemon = true
            it.start()
        }
    }

    fun stop() {
        stopped.set(true)
        thread?.interrupt()
    }

    /**
     * Drain and send until stopped.
     *
     * Three guards end this loop and every one of them is individually deletable with the
     * suite green: the `while` condition, the `if (stopped.get()) break` before the sleep,
     * and the `InterruptedException` catch. That is deliberate rather than accidental, but
     * only two of the three were ever written down, so the third read as an oversight.
     *
     * They are not equivalent. `stop()` sets the flag *and* interrupts, so on a healthy
     * stop the interrupt lands inside `sleeper` and the catch is what actually breaks --
     * which is why deleting the mid-loop check changes nothing measurable, and why a test
     * cannot distinguish them. The flag covers the case where the interrupt is consumed by
     * something inside `drain` or `send` before this loop sees it; the mid-loop check keeps
     * a drain that took longer than the poll interval from sleeping once more on the way
     * out. Each is cheap, and the failure mode they guard against is a send loop that keeps
     * running after teardown with nothing reading its output -- silent, and the reason this
     * class has a stop path at all.
     */
    private fun loop() {
        while (!stopped.get()) {
            // Drain until empty, then sleep. Taking one frame per sleep would cap the
            // send rate at 1/pollMs regardless of the commanded rate, which at 50 Hz --
            // legal on the wire -- would more than halve it.
            //
            // An earlier version skipped the sleep whenever a frame had moved. It made no
            // observable difference, because the inner loop has already emptied the buffer
            // by the time the question is asked, so no test could tell the two apart and
            // the branch was one nothing could pin.
            while (!stopped.get()) {
                val frame = drain() ?: break
                dispatch(frame)
            }
            if (stopped.get()) break
            try {
                sleeper(pollMs)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                break
            }
        }
    }

    /** Visible for a test that wants one frame moved without running the thread. */
    fun dispatch(frame: CapturedFrame): Boolean {
        drained.incrementAndGet()
        val message = CameraFrameMessage(
            captureMonoNs = frame.captureMonoNs,
            frameId = frame.frameId,
            width = frame.width.toLong(),
            height = frame.height.toLong(),
            format = frame.format,
            quality = frame.quality?.toLong(),
            encodeStartMonoNs = frame.encodeStartMonoNs,
            encodeDoneMonoNs = frame.encodeDoneMonoNs,
        )
        // Asked for on every camera frame, not conditionally: this is the busiest
        // channel and a departure stamp is what a receiver needs to measure the
        // network hop, and the added header cost is one int64 -- negligible against
        // a JPEG payload.
        val accepted = send(Channels.CAMERA, message.toExtensions(), frame.jpeg, true)
        if (accepted) sent.incrementAndGet() else refused.incrementAndGet()
        return accepted
    }

    val stats: Stats
        get() = Stats(drained = drained.get(), sent = sent.get(), refused = refused.get())

    data class Stats(val drained: Long, val sent: Long, val refused: Long) {
        /**
         * Every frame taken out of the buffer had exactly one outcome.
         *
         * Counted at three points rather than two with the third derived, so that a
         * frame lost between the drain and the send makes this false instead of being
         * absorbed into whichever term was computed from the others.
         */
        val balances: Boolean get() = drained == sent + refused
    }

    companion object {
        /** Well under the shortest sensible frame period, and not a busy loop. */
        const val DEFAULT_POLL_MS = 20L
    }
}
