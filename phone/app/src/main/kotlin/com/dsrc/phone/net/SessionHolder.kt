package com.dsrc.phone.net

import com.dsrc.phone.config.LinkConfig
import com.dsrc.transport.Frame
import com.dsrc.transport.JsonValue
import com.dsrc.transport.Session
import com.dsrc.transport.SessionEnd
import com.dsrc.transport.SessionStats
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * One end of a byte stream, plus the two controls a session needs from it.
 *
 * An interface so the holder's reconnect logic -- which is where the bugs are -- can be
 * driven by a pair of in-memory pipes at test speed, with no socket and no port.
 */
interface Link {
    val input: InputStream
    val output: OutputStream

    /** Read timeout in milliseconds; 0 means block indefinitely. */
    fun readTimeoutMs(value: Int)

    fun close()
}

/**
 * Keeps one session to the Jetson alive for as long as sensing is running.
 *
 * The sensors do not get to know whether the link is up. A capture callback must never
 * block on a socket and must never see an exception from one, so [send] answers false
 * during a gap and the caller counts that as a refusal like any other. The alternative --
 * sensors that reconnect, or that throw -- puts network error handling inside a camera
 * analyser.
 *
 * Reconnection is unconditional and indefinite: the phone is in a moving vehicle, the
 * link drops for reasons that resolve themselves, and a holder that gave up after N
 * attempts would end a drive's collection at the first tunnel. What it must not do is
 * spin, hence the backoff.
 */
class SessionHolder(
    private val config: LinkConfig,
    private val deviceId: String,
    private val monoClock: () -> Long,
    private val wallClock: () -> Long,
    /** Delivered inbound frames, with the reader's own receipt stamps. Called on a
     *  session delivery thread. */
    private val onFrame: (Frame, Long, Long) -> Unit,
    /** Handed each outgoing header, as canonical JSON, for a local recorder. */
    private val onSent: ((String) -> Unit)? = null,
    private val dial: (LinkConfig) -> Link = ::dialTcp,
    /** Sleep, injectable so a test does not spend the backoff. */
    private val sleeper: (Long) -> Unit = { Thread.sleep(it) },
) {

    @Volatile
    private var current: Session? = null

    private val stopped = AtomicBoolean(false)
    private var thread: Thread? = null

    private val attempts = AtomicLong(0)
    private val established = AtomicLong(0)
    private val dialFailures = AtomicLong(0)
    private val sendsWithoutSession = AtomicLong(0)
    private val sessionsEnded = AtomicLong(0)

    @Volatile
    var lastEnd: SessionEnd? = null
        private set

    @Volatile
    var lastError: String? = null
        private set

    /** Whether a handshaken session exists right now. */
    val isUp: Boolean get() = usable(current)

    fun start() {
        // `check`, not `require`: starting twice is a state problem rather than a bad
        // argument, and IllegalStateException is what a caller would catch for it.
        check(thread == null) { "already started" }
        thread = Thread({ loop() }, "dsrc-link").also {
            it.isDaemon = true
            it.start()
        }
    }

    /**
     * Hand a message to the current session, or refuse it.
     *
     * Refusing while the link is down is deliberate and is counted here rather than
     * queued: a queue in front of the session would hold messages whose sequence numbers
     * had not been assigned yet, so a drop from it would leave no gap for the peer to
     * see -- invisible in both sides' accounting, which is the one thing the channel
     * table exists to prevent.
     */
    fun send(
        channel: String,
        extensions: Map<String, JsonValue>,
        payload: ByteArray = ByteArray(0),
        wantsWireStamp: Boolean = false,
    ): Boolean {
        return sendOn(current, channel, extensions, payload, wantsWireStamp)
    }

    /**
     * The send decision, taken against an explicit session.
     *
     * Split out because extracting the *predicate* was not enough: `usable` became testable
     * and both call sites stayed uncovered, so replacing `!usable(session)` with
     * `session == null` here survived nine runs. The defect being guarded is at the call
     * site, not in the helper, and the window it lives in -- between `finish()` clearing
     * `isRunning` and the link thread's `finally` clearing the field -- was caught about
     * once in thirteen runs when driven through the live loop. Passing the session in makes
     * that state constructible instead of raced: a handshaken, closed session is a non-null
     * reference that is not alive.
     */
    /**
     * Send one time-sync ping on the live session, if there is one.
     *
     * The spec makes the phone the initiator -- "The phone initiates and the Jetson
     * only ever answers" -- and `Session` has implemented that half since task 15.
     * Nothing outside the tests ever called it, so no ping was ever sent on a real
     * drive: the Jetson accumulated zero samples, its estimate never formed, and
     * every camera and GPS stamp fell back to arrival-time proxy for the whole
     * session. That run looks healthy -- the proxy is wrong only by the link
     * segment, tens of milliseconds against a 2 s staleness threshold -- which is
     * why it survived a full device test without anyone noticing.
     *
     * False when the link is down, counted the same way a refused send is, because
     * a drive that never syncs should be visible as a number rather than as an
     * absence.
     */
    fun sendTimeSyncPing(exchangeId: Long): Boolean {
        val live = current
        if (!usable(live)) {
            sendsWithoutSession.incrementAndGet()
            return false
        }
        return runCatching { live!!.sendTimeSyncPing(exchangeId) }.getOrDefault(false)
    }

    internal fun sendOn(
        session: Session?,
        channel: String,
        extensions: Map<String, JsonValue>,
        payload: ByteArray = ByteArray(0),
        wantsWireStamp: Boolean = false,
    ): Boolean {
        // Liveness, not merely existence. In that window a send went into a dead session,
        // came back false, and was counted by nothing -- neither here nor in any refusal
        // counter -- which is exactly when a link drops.
        if (!usable(session)) {
            sendsWithoutSession.incrementAndGet()
            return false
        }
        return session!!.send(channel, extensions, payload, wantsWireStamp)
    }

    fun stop() {
        stopped.set(true)
        // Ends the session, which releases the link thread from its wait. Closing the
        // link directly instead would make the reader fail on a closed stream and end
        // the session as a transport error, which is a lie about why it stopped.
        current?.close()
        // And releases it from a backoff sleep, so shutdown is prompt rather than up to
        // maxBackoffMs.
        thread?.interrupt()
    }

    private fun loop() {
        var backoff = config.firstBackoffMs
        while (!stopped.get()) {
            val ended = CountDownLatch(1)
            var link: Link? = null
            try {
                attempts.incrementAndGet()
                val opened = dial(config)
                link = opened
                opened.readTimeoutMs(config.handshakeTimeoutMs)
                val session = Session(
                    input = opened.input,
                    output = opened.output,
                    deviceId = deviceId,
                    role = ROLE,
                    monoClock = monoClock,
                    wallClock = wallClock,
                    onFrame = onFrame,
                    onSent = onSent,
                    onEnd = { end, error ->
                        lastEnd = end
                        lastError = error?.toString()
                        sessionsEnded.incrementAndGet()
                        ended.countDown()
                    },
                )
                // Throws if the peer's hello never arrives or disagrees, so nothing below
                // runs on a half-open session.
                session.start()
                // Cleared only now: the reader thread it just started must block, and a
                // timeout here would end a healthy quiet link as a transport error.
                opened.readTimeoutMs(0)
                current = session
                established.incrementAndGet()
                // A handshake that completed is evidence the peer is there, so the next
                // failure starts its backoff from the bottom again. Carrying the old
                // value forward would leave a link that flaps once stuck at the ceiling
                // for the rest of the drive.
                backoff = config.firstBackoffMs
                ended.await()
            } catch (e: InterruptedException) {
                // stop() interrupting the wait. Not a failure, and it must be caught
                // before the broad clause below or it would be recorded as one.
                Thread.currentThread().interrupt()
            } catch (e: Exception) {
                // Deliberately everything. This thread is the only thing that reconnects,
                // so anything that escapes it ends the link for the rest of the drive
                // with `send` refusing every message and nothing recording why.
                //
                // Narrower clauses were wrong twice over: a handshake refusal arrives as
                // FramingError, which extends Exception rather than IOException *or*
                // RuntimeException, so catching those two let a version mismatch kill the
                // thread outright -- the same shape as a dead writer reporting a healthy
                // session. A version mismatch is also exactly the case a deployed phone
                // meets after the Jetson is updated and it is not.
                dialFailures.incrementAndGet()
                lastError = "${e.javaClass.simpleName}: ${e.message}"
            } finally {
                current = null
                try {
                    link?.close()
                } catch (e: IOException) {
                    // Already failing; a close error adds nothing and must not replace
                    // the reason we got here.
                }
            }

            if (stopped.get() || Thread.currentThread().isInterrupted) break
            if (!sleepBackoff(backoff)) break
            backoff = (backoff * 2).coerceAtMost(config.maxBackoffMs)
        }
        current = null
    }

    /** @return false if the wait was interrupted, meaning stop. */
    private fun sleepBackoff(millis: Long): Boolean =
        try {
            sleeper(millis)
            true
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
            false
        }

    fun stats() = HolderStats(
        attempts = attempts.get(),
        established = established.get(),
        dialFailures = dialFailures.get(),
        sessionsEnded = sessionsEnded.get(),
        sendsWithoutSession = sendsWithoutSession.get(),
        lastEnd = lastEnd,
        lastError = lastError,
        // Only a *live* session's counters. `current` is cleared by the link thread some
        // time after `finish()` flips `isRunning`, so reading it unfiltered reported a dead
        // session's counters as current -- which is the accounting error the test named for
        // this has always described, while passing or failing on whether the link thread
        // had got round to clearing the field yet. It reproduced 3 runs of 3 once an
        // unrelated transport change shifted the timing by a hair.
        //
        // `usable` is the same predicate `sendOn` gates on, so the two agree by
        // construction: if a send would be refused, the counters are not reported either.
        session = current?.takeIf { usable(it) }?.stats(),
    )

    data class HolderStats(
        val attempts: Long,
        val established: Long,
        val dialFailures: Long,
        val sessionsEnded: Long,
        /** Messages refused because there was no session, counted where they were lost. */
        val sendsWithoutSession: Long,
        val lastEnd: SessionEnd?,
        val lastError: String?,
        /** Null while the link is down; the live session's own counters otherwise. */
        val session: SessionStats?,
    ) {
        /**
         * Every attempt either handshook or failed to.
         *
         * Counted at three separate points rather than derived, so the identity can
         * actually be false. One in-flight attempt is legitimately neither yet, which is
         * why this is an inequality: `attempts` is incremented before the dial and the
         * outcome lands after it.
         */
        val accountsForAttempts: Boolean get() = established + dialFailures <= attempts
    }

    companion object {
        /**
         * Whether a session reference can carry a message.
         *
         * Extracted so it can be pinned. Reference *and* liveness, and the second half was
         * caught roughly once in thirteen runs when tested through the live loop, because
         * the premise is a window of microseconds between `finish()` clearing `isRunning`
         * and the link thread's `finally` clearing the field. A 5 ms poll almost always
         * lands after the clear, and once the loop stops reconnecting the window never
         * reopens -- so the test was a lottery with a 1-in-13 ticket.
         *
         * Reducing this to `session != null` was invisible to the whole suite.
         */
        internal fun usable(session: Session?): Boolean = session != null && session.isRunning

        /**
         * The phone's role in the hello.
         *
         * The constant from the transport, not a second copy of the literal. A duplicated
         * "phone" here decided which half of the timebase protocol ran -- `Session`
         * branches on `role == ROLE_PHONE` -- so setting this to "jetson" made the phone
         * answer pings and refuse pongs, inverting the direction the code itself calls
         * "an offset with the sign inverted, a plausible number that is exactly wrong",
         * with the whole suite green.
         */
        const val ROLE = Session.ROLE_PHONE

        /**
         * A real TCP link.
         *
         * `TCP_NODELAY` matters: a 40-byte GPS record would otherwise sit in Nagle's
         * buffer waiting for company, and the keepalive cadence is 1 s. `SO_KEEPALIVE`
         * mirrors the Python side, though the three tuning knobs it sets --
         * `TCP_KEEPIDLE`, `TCP_KEEPINTVL`, `TCP_KEEPCNT` -- have no `java.net.Socket`
         * equivalent on Android, so the OS defaults apply and the protocol's own
         * keepalive is what actually detects a dead peer.
         */
        fun dialTcp(config: LinkConfig): Link {
            val socket = Socket()
            try {
                socket.tcpNoDelay = true
                socket.keepAlive = true
                socket.connect(InetSocketAddress(config.host, config.port), config.connectTimeoutMs)
            } catch (e: IOException) {
                try {
                    socket.close()
                } catch (ignored: IOException) {
                    // The connect failure is the reason; this one is noise.
                }
                throw e
            }
            return object : Link {
                override val input: InputStream = socket.getInputStream()
                override val output: OutputStream = socket.getOutputStream()
                override fun readTimeoutMs(value: Int) {
                    socket.soTimeout = value
                }
                override fun close() = socket.close()
            }
        }
    }
}
