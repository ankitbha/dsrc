package com.dsrc.phone.net

import com.dsrc.phone.config.LinkConfig
import com.dsrc.transport.Channels
import com.dsrc.transport.Frame
import com.dsrc.transport.Framing
import com.dsrc.transport.JsonValue
import com.dsrc.transport.Protocol
import com.dsrc.transport.Session
import com.dsrc.transport.SessionEnd
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * The holder's reconnect logic against a real peer over loopback.
 *
 * Sockets rather than `PipedInputStream`, which was the first attempt and was wrong in a
 * way worth recording: a piped stream fails with "write end dead" once the thread that
 * last wrote to it exits, so a peer whose launcher thread returned killed the session
 * every time and the holder reconnected forever. The test read as a reconnect defect in
 * the holder when the defect was in the harness.
 *
 * Port 0, so the OS picks a free one and parallel runs cannot collide. Not faked either
 * way is the [Session] on both ends -- the handshake is the part the holder gets wrong,
 * and a fake session would agree with whatever the holder happened to do.
 */
class SessionHolderTest {

    private val closers = mutableListOf<() -> Unit>()

    @After
    fun tearDown() {
        // Reverse order: holders before the servers they are dialling, so a reconnect
        // attempt cannot fire at a half-closed listener and log a confusing failure.
        closers.asReversed().forEach { runCatching { it() } }
        closers.clear()
    }

    /** A Jetson-side listener that runs a real session per connection. */
    private inner class PeerServer(
        private val version: Long = Protocol.VERSION.toLong(),
        private val delayHelloMs: Long = 0,
        /** Drop the session this long after the handshake, to make the link flap. */
        private val dropAfterMs: Long = 0,
    ) {
        private val server = ServerSocket(0, 8, InetAddress.getLoopbackAddress())
        val sessions = ConcurrentLinkedQueue<Session>()
        val rolesSeen = ConcurrentLinkedQueue<String>()
        val received = ConcurrentLinkedQueue<Frame>()
        val accepted = AtomicInteger(0)

        val port: Int get() = server.localPort

        init {
            closers += { runCatching { server.close() }; sessions.forEach { runCatching { it.close() } } }
            Thread({ acceptLoop() }, "peer-accept").also {
                it.isDaemon = true
                it.start()
            }
        }

        private fun acceptLoop() {
            while (!server.isClosed) {
                val client = try {
                    server.accept()
                } catch (e: IOException) {
                    return  // closed by tearDown
                }
                accepted.incrementAndGet()
                client.tcpNoDelay = true
                if (version != Protocol.VERSION.toLong()) {
                    writeWrongVersionHello(client)
                    continue
                }
                if (delayHelloMs > 0) Thread.sleep(delayHelloMs)
                runCatching {
                    val session = Session(
                        input = client.getInputStream(),
                        output = client.getOutputStream(),
                        deviceId = "jetson",
                        role = "jetson",
                        monoClock = { System.nanoTime() },
                        wallClock = { System.currentTimeMillis() * 1_000_000L },
                        onFrame = { received.add(it) },
                    )
                    session.start()
                    session.peer?.role?.let { rolesSeen.add(it) }
                    sessions.add(session)
                    if (dropAfterMs > 0) {
                        Thread({
                            Thread.sleep(dropAfterMs)
                            runCatching { session.close() }
                        }, "peer-drop").also { it.isDaemon = true; it.start() }
                    }
                }
            }
        }

        /**
         * A hello no [Session] can produce, so it is written by hand.
         *
         * This is the case a deployed phone meets after the Jetson is updated and it is
         * not, which makes it the one handshake failure guaranteed to happen in the field.
         */
        private fun writeWrongVersionHello(client: Socket) {
            runCatching {
                val header = Framing.header(
                    channel = Channels.CONTROL,
                    sequence = 0,
                    monoNs = 1,
                    wallNs = 2,
                    extensions = mapOf(
                        Session.HELLO to JsonValue.Obj(
                            mapOf(
                                "protocol_version" to JsonValue.Num(version),
                                "device_id" to JsonValue.Text("wrong-version"),
                                "role" to JsonValue.Text("jetson"),
                            )
                        )
                    ),
                    allowReserved = setOf(Session.HELLO),
                )
                client.getOutputStream().write(Framing.encode(header, ByteArray(0)))
                client.getOutputStream().flush()
            }
        }
    }

    /** Records every read-timeout the holder sets, wrapping the production link. */
    private class RecordingLink(private val inner: Link) : Link {
        val timeouts = mutableListOf<Int>()
        override val input: InputStream get() = inner.input
        override val output: OutputStream get() = inner.output
        override fun readTimeoutMs(value: Int) {
            synchronized(timeouts) { timeouts.add(value) }
            inner.readTimeoutMs(value)
        }
        override fun close() = inner.close()
        fun snapshot(): List<Int> = synchronized(timeouts) { timeouts.toList() }
    }

    private fun holder(
        port: Int,
        dial: ((LinkConfig) -> Link)? = null,
        sleeper: (Long) -> Unit = { Thread.sleep(it) },
        onFrame: (Frame) -> Unit = {},
    ): SessionHolder {
        val config = LinkConfig(
            host = "127.0.0.1",
            port = port,
            firstBackoffMs = 5,
            maxBackoffMs = 20,
        )
        val made = SessionHolder(
            config = config,
            deviceId = "phone-test",
            monoClock = { System.nanoTime() },
            wallClock = { System.currentTimeMillis() * 1_000_000L },
            onFrame = onFrame,
            dial = dial ?: SessionHolder::dialTcp,
            sleeper = sleeper,
        )
        closers += { made.stop() }
        return made
    }

    @Test
    fun `a refused dial is retried until one succeeds`() {
        val peer = PeerServer()
        val attempts = AtomicInteger(0)
        val holder = holder(peer.port, dial = { config ->
            // Two scripted refusals, then the real dial. Pointing at a closed port
            // instead would refuse forever and prove only half of this.
            if (attempts.incrementAndGet() < 3) throw IOException("connection refused")
            SessionHolder.dialTcp(config)
        })
        holder.start()

        waitFor { holder.isUp }
        val stats = holder.stats()
        assertEquals("two refusals then one success", 3, stats.attempts)
        assertEquals(1, stats.established)
        assertEquals(2, stats.dialFailures)
        assertTrue(stats.accountsForAttempts)
        assertNotNull(stats.session)
    }

    @Test
    fun `the handshake read timeout is set and then cleared`() {
        val peer = PeerServer()
        var recorder: RecordingLink? = null
        val holder = holder(peer.port, dial = { config ->
            RecordingLink(SessionHolder.dialTcp(config)).also { recorder = it }
        })
        holder.start()
        waitFor { holder.isUp }

        // Both calls, in order. Without the second, a healthy quiet link would die of a
        // socket timeout after handshakeTimeoutMs and be reported as a transport error --
        // at 5 s instead of the spec's stall rule, and under the wrong name.
        assertEquals(listOf(LinkConfig().handshakeTimeoutMs, 0), recorder!!.snapshot())
    }

    @Test
    fun `a message sent while the link is down is refused and counted`() {
        val holder = holder(port = 1, dial = { throw IOException("refused") })
        holder.start()
        waitFor { holder.stats().dialFailures >= 1 }

        assertFalse(holder.send(Channels.GPS, gpsExtensions()))
        assertFalse(holder.isUp)
        assertEquals(1, holder.stats().sendsWithoutSession)
        assertNull("no session, so no session counters", holder.stats().session)
    }

    @Test
    fun `a message sent while the link is up reaches the peer`() {
        val peer = PeerServer()
        val holder = holder(peer.port)
        holder.start()
        waitFor { holder.isUp }

        assertTrue(holder.send(Channels.GPS, gpsExtensions()))
        waitFor { peer.received.any { it.channel == Channels.GPS } }

        val frame = peer.received.first { it.channel == Channels.GPS }
        assertEquals("the first gps message spends sequence 0", 0, frame.sequence)
        assertEquals(0, holder.stats().sendsWithoutSession)
    }

    @Test
    fun `the peer closing brings a new session up`() {
        val peer = PeerServer()
        val holder = holder(peer.port)
        holder.start()
        waitFor { holder.isUp }
        val first = peer.sessions.first()

        // The Jetson restarts.
        first.close()

        waitFor { holder.stats().established >= 2 }
        assertTrue(holder.stats().sessionsEnded >= 1)
        assertTrue("the peer saw two connections", peer.accepted.get() >= 2)
        assertTrue(holder.isUp)
    }

    @Test
    fun `stop ends the session as a local close, not a transport error`() {
        val peer = PeerServer()
        val holder = holder(peer.port)
        holder.start()
        waitFor { holder.isUp }

        holder.stop()
        waitFor { holder.lastEnd != null }
        // Closing the link's streams instead would surface as a read failure on a stream
        // we closed ourselves, naming the cause wrongly in every log that follows.
        assertEquals(SessionEnd.CLOSED_LOCAL, holder.lastEnd)
        assertFalse(holder.isUp)
    }

    @Test
    fun `stop after a local close does not reconnect`() {
        val peer = PeerServer()
        val holder = holder(peer.port)
        holder.start()
        waitFor { holder.isUp }

        holder.stop()
        waitFor { !holder.isUp }
        val attemptsAtStop = holder.stats().attempts
        Thread.sleep(200)   // several backoffs at 5-20 ms
        assertEquals("stopped means stopped", attemptsAtStop, holder.stats().attempts)
    }

    @Test
    fun `stop during a backoff sleep does not wait it out`() {
        val sleeping = CountDownLatch(1)
        val interrupted = CountDownLatch(1)
        // A backoff far longer than any test should take, so waiting it out fails rather
        // than passing slowly. The real Thread.sleep, because an interrupt is the only
        // thing that can end it and a no-op sleeper would prove nothing.
        val holder = SessionHolder(
            config = LinkConfig(host = "127.0.0.1", port = 1, firstBackoffMs = 30_000, maxBackoffMs = 30_000),
            deviceId = "phone-test",
            monoClock = { System.nanoTime() },
            wallClock = { 0 },
            onFrame = {},
            dial = { throw IOException("refused") },
            sleeper = { millis ->
                sleeping.countDown()
                try {
                    Thread.sleep(millis)
                } catch (e: InterruptedException) {
                    interrupted.countDown()
                    throw e
                }
            },
        )
        closers += { holder.stop() }

        holder.start()
        assertTrue("never reached the backoff", sleeping.await(5, TimeUnit.SECONDS))

        val started = System.nanoTime()
        holder.stop()
        assertTrue("the sleep was never interrupted", interrupted.await(5, TimeUnit.SECONDS))
        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        assertTrue("stop waited $elapsedMs ms out of a 30 s backoff", elapsedMs < 2_000)
    }

    @Test
    fun `a peer on the wrong protocol version is a failure, not a connection`() {
        val peer = PeerServer(version = Protocol.VERSION + 1L)
        val holder = holder(peer.port)
        holder.start()

        waitFor { holder.stats().dialFailures >= 1 }
        assertEquals("a version mismatch is not a connection", 0, holder.stats().established)
        assertFalse(holder.isUp)
        assertTrue("the reason must name the version: ${holder.lastError}",
            holder.lastError!!.contains("version"))

        // And it keeps trying, rather than the thread dying with the exception. This is
        // what a narrower catch broke: FramingError extends Exception, not IOException or
        // RuntimeException, so it escaped the loop and the link never came back.
        val after = holder.stats().attempts
        waitFor { holder.stats().attempts > after }
    }

    private fun gpsExtensions(): Map<String, JsonValue> = mapOf(
        "t_capture_mono_ns" to JsonValue.Num(1_000),
        "valid" to JsonValue.Bool(true),
        "lat" to JsonValue.Real(40.7),
        "lon" to JsonValue.Real(-74.0),
        "speed_mps" to JsonValue.Real(13.4),
        "heading_deg" to JsonValue.Real(90.0),
        "fix_quality" to JsonValue.Num(1),
        "num_sats" to JsonValue.Num(9),
        "hdop" to JsonValue.Null,
        "altitude_m" to JsonValue.Real(12.0),
        "utc_epoch_ns" to JsonValue.Num(1_700_000_000_000_000_000L),
    )

    /** Poll for a condition with a deadline, so a failure is a failure and not a hang. */
    private fun waitFor(timeoutMs: Long = 5_000, condition: () -> Boolean) {
        val deadline = System.nanoTime() + timeoutMs * 1_000_000
        while (System.nanoTime() < deadline) {
            if (condition()) return
            Thread.sleep(5)
        }
        throw AssertionError("condition never held within $timeoutMs ms")
    }

    // -- the backoff schedule ------------------------------------------------

    /** The delays the holder asked for, in order. An injected sleeper records them. */
    private fun backoffSchedule(
        config: LinkConfig,
        attempts: Int,
        dial: (LinkConfig) -> Link = { throw IOException("refused") },
    ): List<Long> {
        val asked = java.util.concurrent.CopyOnWriteArrayList<Long>()
        val made = SessionHolder(
            config = config,
            deviceId = "phone-test",
            monoClock = { System.nanoTime() },
            wallClock = { 0 },
            onFrame = {},
            dial = dial,
            sleeper = { millis -> asked.add(millis) },
        )
        closers += { made.stop() }
        made.start()
        val deadline = System.currentTimeMillis() + 5_000
        while (asked.size < attempts && System.currentTimeMillis() < deadline) Thread.sleep(2)
        made.stop()
        return asked.take(attempts)
    }

    @Test
    fun `the backoff doubles and is capped`() {
        // The whole schedule was unpinned. Removing the doubling reconnects every 500 ms
        // for the entire drive when the Jetson is off, against the class's own "what it
        // must not do is spin"; removing the cap puts the wait past an hour after about
        // twenty-four failures.
        val schedule = backoffSchedule(
            LinkConfig(host = "127.0.0.1", port = 1, firstBackoffMs = 10, maxBackoffMs = 80),
            attempts = 6,
        )
        assertEquals(listOf(10L, 20L, 40L, 80L, 80L, 80L), schedule)
    }

    @Test
    fun `a handshake that completed resets the backoff`() {
        // Documented behaviour with no test: without the reset, a link that flaps once is
        // stuck at the ceiling for the rest of the drive.
        // The peer drops the session shortly after the handshake, so the link flaps --
        // which is the case the reset exists for. Without the drop the loop parks on a
        // healthy session and never asks for another delay, so the first version of this
        // test recorded two and proved nothing.
        val peer = PeerServer(dropAfterMs = 30)
        val attempt = AtomicInteger(0)
        val schedule = backoffSchedule(
            LinkConfig(host = "127.0.0.1", port = peer.port, firstBackoffMs = 10, maxBackoffMs = 320),
            attempts = 5,
            dial = { config ->
                // Fail, fail, succeed-then-drop, then fail again. The successful handshake
                // in the middle must send the delay back to the floor.
                when (attempt.incrementAndGet()) {
                    1, 2 -> throw IOException("refused")
                    3 -> SessionHolder.dialTcp(config)
                    else -> throw IOException("refused")
                }
            },
        )
        // 10, 20 climbing; then the third attempt handshakes, so the next wait is 10 again.
        assertEquals(listOf(10L, 20L, 10L, 20L, 40L), schedule)
    }

    // -- what `current` points at -------------------------------------------

    @Test
    fun `a send after the session ends is refused, not routed into a dead session`() {
        // `current` was left pointing at the ended session, so send() called into it
        // instead of counting sendsWithoutSession, and stats().session reported a dead
        // session's counters -- the accounting the docstring says the channel table exists
        // to prevent. No test caught it because the down-link test never establishes a
        // session at all.
        val peer = PeerServer()
        val holder = holder(peer.port, dial = { config ->
            // One connection only: after it ends there is nothing to reconnect to.
            if (peer.accepted.get() == 0) SessionHolder.dialTcp(config)
            else throw IOException("no second connection in this test")
        })
        holder.start()
        waitFor { holder.isUp }
        val before = holder.stats().sendsWithoutSession

        peer.sessions.first().close()
        waitFor { !holder.isUp }

        // Deliberately sent in the window between the session going down and the link
        // thread clearing its reference, because that window is where the accounting was
        // lost: `isRunning` goes false inside finish(), and the field is cleared later.
        assertFalse("a dead session accepted a message", holder.send(Channels.GPS, gpsExtensions()))
        // Sampled here rather than three calls later, which is strictly better but does
        // not make this assertion a pin, and saying so matters more than the tidier claim.
        //
        // Whether it can fail depends on whether the link thread has cleared `current`
        // yet. Under load it had not: this test failed 3 runs of 3 while a transport suite
        // was running beside it, reporting a dead session's counters as current. Idle, it
        // passes 3 of 3 with the filter in `SessionHolder.stats()` deleted, because
        // `current` is already null by the time anything reads it. So the filter is a fix
        // for a race this test *observed* and cannot summon, and the evidence for it is
        // that failure rather than this assertion.
        val inTheWindow = holder.stats()

        assertEquals(
            "the refusal was not counted where it happened: $inTheWindow",
            before + 1,
            inTheWindow.sendsWithoutSession,
        )
        assertNull("a dead session's counters were reported as current", inTheWindow.session)
    }

    @Test
    fun `nothing is sent before the handshake completes`() {
        // `current` is published only after start() returns. Publishing it earlier let a
        // message go out during the handshake window, before the hello -- a protocol
        // violation that `isUp` does not notice, because that gates on isRunning.
        val peer = PeerServer(delayHelloMs = 400)
        val holder = holder(peer.port)
        holder.start()

        // While the peer is deliberately slow to say hello, there is no session to send on.
        //
        // Successes are counted, not refusals. Asserting "some were refused" passes when the
        // early attempts land before the link thread has even dialled and later ones sneak
        // through -- which is exactly what let the mutation survive. Not one may be accepted.
        val deadline = System.currentTimeMillis() + 400
        var accepted = 0
        var attempts = 0
        while (System.currentTimeMillis() < deadline && !holder.isUp) {
            attempts++
            if (holder.send(Channels.GPS, gpsExtensions())) accepted++
            Thread.sleep(5)
        }
        assertTrue("the window was never exercised", attempts > 5)
        assertEquals(
            "$accepted of $attempts messages went out before the handshake completed: ${holder.stats()}",
            0,
            accepted,
        )
        assertEquals(
            "every attempt in the window must be counted as having no session",
            attempts.toLong(),
            holder.stats().sendsWithoutSession,
        )

        // And once the handshake completes, sends are accepted.
        waitFor { holder.isUp }
        assertTrue(holder.send(Channels.GPS, gpsExtensions()))
    }


    @Test
    fun `the phone announces itself as a phone`() {
        // SessionHolder.ROLE decides which half of the timebase protocol the session runs:
        // Session branches on `role == ROLE_PHONE`, so "jetson" here makes the phone answer
        // pings and refuse pongs -- inverting the direction the code itself calls "an
        // offset with the sign inverted, a plausible number that is exactly wrong". Setting
        // it to jetson passed the whole suite, because the peer never looked at the role it
        // was told and so agreed with whatever the holder claimed.
        val peer = PeerServer()
        val holder = holder(peer.port)
        holder.start()
        waitFor { holder.isUp }
        waitFor { peer.rolesSeen.isNotEmpty() }

        assertEquals(
            "the peer was told the wrong role",
            Session.ROLE_PHONE,
            peer.rolesSeen.first(),
        )
        // And the constant is the transport's, not a second copy of the literal.
        assertEquals(Session.ROLE_PHONE, SessionHolder.ROLE)
    }


    @Test
    fun `the attempt identity is a real function of its fields`() {
        // Asserted once with no control, so `get() = true` survived.
        val stats = SessionHolder.HolderStats(
            attempts = 3, established = 1, dialFailures = 2, sessionsEnded = 1,
            sendsWithoutSession = 0, lastEnd = null, lastError = null, session = null,
        )
        assertTrue(stats.accountsForAttempts)
        assertFalse(stats.copy(established = 5).accountsForAttempts)
        assertFalse(stats.copy(dialFailures = 9).accountsForAttempts)
    }


    @Test
    fun `starting twice is refused`() {
        // The guard survived deletion. A second start spawns a second link thread against
        // the same `current` field, so two dial loops race to publish a session and one of
        // them is orphaned -- reconnecting forever with nothing reading it.
        val peer = PeerServer()
        val holder = holder(peer.port)
        holder.start()
        val second = runCatching { holder.start() }
        assertTrue("a second start was allowed", second.isFailure)
        assertTrue(second.exceptionOrNull() is IllegalStateException)
    }


    @Test
    fun `a session reference is usable only while it is alive`() {
        // The liveness half of the send check was caught about once in thirteen runs when
        // tested through the live loop, because its premise is a window of microseconds
        // between finish() clearing isRunning and the link thread clearing the field. A 5 ms
        // poll almost always lands after the clear, and once the scripted dial stops
        // reconnecting the window never reopens -- a test with a 1-in-13 ticket.
        //
        // The predicate is extracted, so no race is needed: a session that has handshaken
        // and then been closed is a non-null reference that is not alive, which is exactly
        // the state the window produces.
        val peer = PeerServer()
        val socket = Socket(InetAddress.getLoopbackAddress(), peer.port).also { closers += { it.close() } }
        val session = Session(
            input = socket.getInputStream(),
            output = socket.getOutputStream(),
            deviceId = "phone-test",
            role = Session.ROLE_PHONE,
            monoClock = { System.nanoTime() },
            wallClock = { 0 },
            onFrame = {},
        )
        session.start()
        assertTrue("a live session must be usable", SessionHolder.usable(session))

        session.close()
        assertFalse("a closed session is not usable", SessionHolder.usable(session))
        // And the reference is still there, which is the whole point: `!= null` is not the
        // question.
        assertFalse(session.isRunning)
        assertFalse("null is not usable either", SessionHolder.usable(null))
    }


    @Test
    fun `the send path refuses a dead session and counts it`() {
        // Extracting the predicate pinned the helper and left both call sites uncovered:
        // replacing `!usable(session)` with `session == null` in send() survived nine runs,
        // and `isUp` had no coverage at all. The decision is taken against an explicit
        // session now, so the state can be built rather than raced -- a handshaken, closed
        // session is a non-null reference that is not alive.
        val peer = PeerServer()
        val socket = Socket(InetAddress.getLoopbackAddress(), peer.port).also { closers += { it.close() } }
        val session = Session(
            input = socket.getInputStream(),
            output = socket.getOutputStream(),
            deviceId = "phone-test",
            role = Session.ROLE_PHONE,
            monoClock = { System.nanoTime() },
            wallClock = { 0 },
            onFrame = {},
        )
        session.start()

        val holder = holder(peer.port, dial = { throw IOException("not dialling in this test") })
        val before = holder.stats().sendsWithoutSession

        // Alive: the send goes through.
        assertTrue(
            "a live session was refused",
            holder.sendOn(session, Channels.GPS, gpsExtensions()),
        )
        assertEquals(before, holder.stats().sendsWithoutSession)

        session.close()
        assertFalse("a dead session was accepted", holder.sendOn(session, Channels.GPS, gpsExtensions()))
        assertEquals(
            "the refusal was not counted where it happened",
            before + 1,
            holder.stats().sendsWithoutSession,
        )
        assertFalse("null was accepted", holder.sendOn(null, Channels.GPS, gpsExtensions()))
        assertEquals(before + 2, holder.stats().sendsWithoutSession)
    }

    @Test
    fun `isUp reports liveness, not merely that a reference exists`() {
        // `isUp` had no coverage: reducing it to `current != null` survived six runs. It is
        // also what makes the older window test land after the field is cleared, so the two
        // were coupled and neither was pinned.
        val peer = PeerServer()
        val holder = holder(peer.port)
        holder.start()
        waitFor { holder.isUp }

        peer.sessions.first().close()
        // Whatever the field holds, `isUp` must be false once the session is not running.
        waitFor { !holder.isUp }
        assertFalse(holder.isUp)
    }

}
