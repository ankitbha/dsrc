package com.dsrc.phone.log

import android.os.SystemClock
import android.util.Log
import com.dsrc.transport.Json
import com.dsrc.transport.JsonValue
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
 * **Nothing prunes the directory.** The cap below is per file; the handset's storage is
 * per device, and every drive leaves another file. At roughly 20 kB/s of headers that is
 * about 72 MB an hour of driving, accumulating until someone clears it by hand. Bounding it
 * needs a retention decision — how long a drive's ground truth is worth keeping — which is
 * a research question rather than an engineering one, so it is named here rather than
 * guessed at. When the storage does fill, `appendText` throws and `failures` climbs, so the
 * log says so; by then the app's own space is gone.
 *
 * Nothing here blocks a sensing thread. Offers go to a bounded queue and are dropped if it
 * is full — a log that stalled the camera to keep itself complete would corrupt the very
 * measurement it exists to record.
 */
class SessionLog(
    private val file: File,
    private val maxBytes: Long = MAX_BYTES,
    private val queueDepth: Int = QUEUE_DEPTH,
    /** For the one failure this class reports about itself -- see [write]'s
     *  own catch. The same clock every other line here is stamped on. */
    private val monoClock: () -> Long = SystemClock::elapsedRealtimeNanos,
    private val wallClock: () -> Long = { System.currentTimeMillis() * 1_000_000L },
) {
    private val queue = ArrayBlockingQueue<String>(queueDepth)
    private val lock = Any()

    private var written = 0L
    private var bytes = 0L
    private var droppedQueueFull = 0L
    private var droppedNotRunning = 0L
    private var droppedAtCap = 0L
    private var failures = 0L

    // -- the fourth line shape's own rate cap (D12) --------------------------
    // One line per kind per second, plus a per-kind lifetime cap: the queue
    // above is 128 deep and shared with every frame header this device sends,
    // and a full queue drops the header -- the file's reason to exist. Guarded
    // by the same `lock` as the counters above; the state is small and the
    // critical section is short, so a second lock would only be a second thing
    // to get the ordering of right.
    private val lastAcceptedFailureSecond = mutableMapOf<String, Long>()
    private val suppressedSinceAccepted = mutableMapOf<String, Long>()
    private val writtenPerKind = mutableMapOf<String, Int>()
    private var failuresSuppressed = 0L

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
    fun offer(headerJson: String) = enqueueLine(headerJson)

    /**
     * Record one inbound frame, with the reader's own receipt stamps.
     *
     * A distinct shape from the outbound lines [offer] writes -- `{"dir":"in",...}` rather
     * than a bare header -- because an outbound line already *is* the canonical header this
     * device sent, verbatim, and an inbound line needs to say more than the header alone
     * can: when this device received it, on its own clock, which the header does not carry
     * at all. `header` is passed through as the frame's own extensions object, unmodified,
     * for the same "verbatim, cannot disagree" reason [offer] exists.
     */
    fun offerInbound(recvMonoNs: Long, recvWallNs: Long, header: JsonValue.Obj) {
        val wrapped = JsonValue.Obj(
            mapOf(
                "dir" to JsonValue.Text("in"),
                "recv_mono_ns" to JsonValue.Num(recvMonoNs),
                "recv_wall_ns" to JsonValue.Num(recvWallNs),
                "header" to header,
            )
        )
        enqueueLine(Json.encode(wrapped))
    }

    /**
     * Record the first instant `AdvisoryHolder.current()` returned a given advisory.
     *
     * Keyed on the advisory's own `t_capture_mono_ns` rather than a generated id, because
     * that is the same key an offline join already uses to pair an advisory with the
     * Jetson tick that produced it -- `run_phone_drive.py` and `eval_run.py`'s phone-log
     * join both key on it, and a third identifier here would just be one more thing to
     * keep in agreement with the other two.
     */
    fun offerAdvisoryShown(captureMonoNs: Long, shownMonoNs: Long) {
        val wrapped = JsonValue.Obj(
            mapOf(
                "dir" to JsonValue.Text("shown"),
                "t_capture_mono_ns" to JsonValue.Num(captureMonoNs),
                "shown_mono_ns" to JsonValue.Num(shownMonoNs),
            )
        )
        enqueueLine(Json.encode(wrapped))
    }

    /**
     * Record one failure occurrence: a condition detected off the link, at the
     * site that already counts it -- see [FailureKinds] for the closed set of
     * `kind`.
     *
     * Rate-capped to at most one line per `kind` per second and
     * [MAX_LINES_PER_KIND] per session, because a fault repeating faster than
     * that would otherwise compete with frame headers for the same 128-deep
     * queue, and a full queue drops the header -- this file's whole reason to
     * exist. An occurrence suppressed by either limit is not lost silently:
     * the count travels forward and rides on the next accepted line of the
     * same kind, in [suppressed].
     *
     * `atMonoNs` is [android.os.SystemClock.elapsedRealtimeNanos], the same
     * clock every other line in this file is stamped on, and it is not
     * converted -- an estimate-dependent bound on a number nobody differences
     * would only add uncertainty nothing here needs.
     */
    fun offerFailure(kind: String, atMonoNs: Long, atWallNs: Long, n: Long = 1, detail: String? = null) {
        val suppressedBefore: Long
        synchronized(lock) {
            val writtenSoFar = writtenPerKind.getOrDefault(kind, 0)
            val second = atMonoNs / 1_000_000_000L
            if (writtenSoFar >= MAX_LINES_PER_KIND || lastAcceptedFailureSecond[kind] == second) {
                suppressedSinceAccepted[kind] = (suppressedSinceAccepted[kind] ?: 0L) + 1L
                failuresSuppressed++
                return
            }
            suppressedBefore = suppressedSinceAccepted.remove(kind) ?: 0L
            lastAcceptedFailureSecond[kind] = second
            writtenPerKind[kind] = writtenSoFar + 1
        }
        val wrapped = JsonValue.Obj(
            mapOf(
                "dir" to JsonValue.Text("fail"),
                "at_mono_ns" to JsonValue.Num(atMonoNs),
                "at_wall_ns" to JsonValue.Num(atWallNs),
                "kind" to JsonValue.Text(kind),
                "n" to JsonValue.Num(n),
                "detail" to (detail?.let { JsonValue.Text(it) } ?: JsonValue.Null),
                "suppressed" to JsonValue.Num(suppressedBefore),
            )
        )
        enqueueLine(Json.encode(wrapped))
    }

    private fun enqueueLine(line: String) {
        if (!running) {
            // Counted, not dropped in silence. A frame the transport wrote after the log
            // was stopped -- teardown releases them in an order, and the link outlives the
            // log by several steps -- incremented nothing at all, so the file could be
            // missing frames and still call itself complete. That is the one thing this
            // class has to be right about.
            synchronized(lock) { droppedNotRunning++ }
            return
        }
        if (!queue.offer(line)) {
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
            // The one failure this class can report about itself: a later write may
            // still succeed (this one failure need not be the drive's last), so it is
            // worth trying to say so in the file it is failing to write, through the
            // same rate-capped path every other failure kind uses. If the disk stays
            // dead this simply fails again next line and is counted the same way.
            offerFailure(
                FailureKinds.LOG_SELF, monoClock(), wallClock(),
                detail = "${e.javaClass.simpleName}: ${e.message}",
            )
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
                failuresSuppressed = failuresSuppressed,
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
        /** Failure occurrences the rate cap held back, across every kind --
         *  the teardown census D12 names, so a fault that fired far faster
         *  than its cap allowed is visible in total even where no single
         *  accepted line's own `suppressed` count says so. */
        val failuresSuppressed: Long,
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

        /**
         * The lifetime cap on [offerFailure] lines per `kind`, beyond the one
         * per second the same method already enforces.
         *
         * `phone_link.refusals`' precedent on the Jetson side, ported: the
         * first ones are the diagnosis, the six hundredth repeats it.
         */
        const val MAX_LINES_PER_KIND = 64

        private const val POLL_MS = 200L
        private const val JOIN_MS = 2_000L
        private const val TAG = "SessionLog"
    }
}
