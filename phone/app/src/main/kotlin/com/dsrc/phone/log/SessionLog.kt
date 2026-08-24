package com.dsrc.phone.log

import android.util.Log
import java.io.File
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * A local record of what the phone sent, for ground truth and post-hoc analysis.
 *
 * **It logs the frame headers, verbatim.** Not a summary, not a re-derivation from the
 * pipelines' counters — the canonical JSON that went on the wire, one object per line. That
 * is the whole design: a log built from anything else becomes a second source of truth, and
 * the first time it disagrees with what the Jetson received, nobody can say which is right.
 * Written from the header the transport encoded, it cannot disagree, because it *is* the
 * thing that was sent.
 *
 * Payload bytes are not written. A drive's JPEGs are gigabytes and the phone would fill its
 * own storage inside a few minutes; the header already carries `n`, the payload length, so
 * an analysis can tell a 25 kB frame from an empty one without keeping either.
 *
 * **One file spans every session of one service run, reconnects included.** A dropped link
 * in a moving car is what `SessionHolder` exists for, and each reconnect restarts every
 * channel's sequence at zero — so a file can hold `imu#0` twice with `t_mono_ns` still
 * monotonic across the seam. The hello is logged for exactly this reason: it is the frame
 * that names a session, so it is the marker an analyst joins on before pairing `(ch, seq)`
 * with a Jetson-side recording. Without it the duplicate keys are silent, and joining on
 * them pairs the wrong frames.
 *
 * Nothing here blocks a sensing thread. Offers go to a bounded queue and are dropped if it
 * is full — a log that stalled the camera to keep itself complete would corrupt the very
 * measurement it exists to record.
 */
class SessionLog(
    private val file: File,
    private val maxBytes: Long = MAX_BYTES,
    private val queueDepth: Int = QUEUE_DEPTH,
) {
    private val queue = ArrayBlockingQueue<String>(queueDepth)
    private val lock = Any()

    private var written = 0L
    private var bytes = 0L
    private var droppedQueueFull = 0L
    private var droppedNotRunning = 0L
    private var droppedAtCap = 0L
    private var failures = 0L

    @Volatile
    private var running = false

    @Volatile
    private var thread: Thread? = null

    fun start() {
        synchronized(lock) {
            if (running) return
            running = true
        }
        val worker = Thread({ drain() }, THREAD_NAME)
        worker.isDaemon = true
        thread = worker
        worker.start()
    }

    /**
     * Record one frame header.
     *
     * Never blocks. A full queue means the writer is behind — a slow filesystem, or a burst
     * — and the honest response is to drop this line and count it, because the alternative
     * is back-pressure onto whichever sensing thread happened to call.
     */
    fun offer(headerJson: String) {
        if (!running) {
            // Counted, not dropped in silence. A frame the transport wrote after the log
            // was stopped -- teardown releases them in an order, and the link outlives the
            // log by several steps -- incremented nothing at all, so the file could be
            // missing frames and still call itself complete. That is the one thing this
            // class has to be right about.
            synchronized(lock) { droppedNotRunning++ }
            return
        }
        if (!queue.offer(headerJson)) {
            synchronized(lock) { droppedQueueFull++ }
        }
    }

    private fun drain() {
        file.parentFile?.mkdirs()
        while (running || queue.isNotEmpty()) {
            val line = try {
                queue.poll(POLL_MS, TimeUnit.MILLISECONDS)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                return
            } ?: continue
            write(line)
        }
    }

    private fun write(line: String) {
        val size = line.length + 1L
        synchronized(lock) {
            if (bytes + size > maxBytes) {
                // Stop, and say so. The alternative is rotating the start of the drive
                // away, and the start is where the setup, the first fixes and the timebase
                // exchange are -- the part an analysis most often needs. A file that stops
                // early and records that it stopped never claims to be complete; one that
                // silently rotates does.
                droppedAtCap++
                return
            }
            bytes += size
        }
        try {
            file.appendText(line + "\n")
            synchronized(lock) { written++ }
        } catch (e: Exception) {
            synchronized(lock) {
                failures++
                bytes -= size
            }
            Log.e(TAG, "session log write failed; continuing", e)
        }
    }

    /** Stop taking lines and let the writer finish what it has. */
    fun stop() {
        running = false
        val worker = thread ?: return
        thread = null
        // Joined, unlike the other workers here, because the point of this thread is the
        // file it leaves behind: returning before it has flushed would mean the last
        // seconds of a drive are missing from the artifact the drive was for.
        worker.join(JOIN_MS)
    }

    val stats: Stats
        get() = synchronized(lock) {
            Stats(
                written = written,
                bytes = bytes,
                droppedQueueFull = droppedQueueFull,
                droppedNotRunning = droppedNotRunning,
                droppedAtCap = droppedAtCap,
                failures = failures,
                path = file.absolutePath,
            )
        }

    data class Stats(
        val written: Long,
        val bytes: Long,
        /** Lines dropped because the writer was behind. */
        val droppedQueueFull: Long,
        /** Lines offered after the log was stopped, or before it was started. */
        val droppedNotRunning: Long,
        /** Lines dropped because the file hit its size cap. */
        val droppedAtCap: Long,
        val failures: Long,
        val path: String,
    ) {
        /** Whether the file is a complete record of the session. */
        val complete: Boolean
            get() = droppedQueueFull == 0L && droppedAtCap == 0L &&
                droppedNotRunning == 0L && failures == 0L
    }

    companion object {
        const val THREAD_NAME = "dsrc-log"

        /**
         * The most a session may write.
         *
         * 256 MiB. At the default rates the headers are a few hundred bytes each and about
         * 60 frames a second, so roughly 20 kB/s — a drive would have to run for something
         * like three hours to reach it. Generous enough not to truncate a real session,
         * bounded enough that a runaway cannot fill a handset.
         */
        const val MAX_BYTES = 256L * 1024 * 1024

        /**
         * How many lines may wait for the writer.
         *
         * Two seconds of frames at the default rates. Deep enough to ride out a filesystem
         * pause, shallow enough that a genuinely stuck writer is visible as drops rather
         * than as unbounded memory.
         */
        const val QUEUE_DEPTH = 128

        private const val POLL_MS = 200L
        private const val JOIN_MS = 2_000L
        private const val TAG = "SessionLog"
    }
}
