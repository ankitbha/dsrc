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
    private inner class PeerServer(private val version: Long = Protocol.VERSION.toLong()) {
        private val server = ServerSocket(0, 8, InetAddress.getLoopbackAddress())
        val sessions = ConcurrentLinkedQueue<Session>()
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
                    sessions.add(session)
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
}
