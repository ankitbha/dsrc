package com.dsrc.transport

import org.junit.Assume.assumeTrue
import java.io.File
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.test.AfterTest
import kotlin.test.Test

/**
 * The plan's experiments, as measurements rather than assertions.
 *
 * Gated on `-Ddsrc.measure=true` so a thirty-second throughput run does not sit inside the
 * ordinary suite. Nothing here asserts a threshold: a measurement that fails a bound it
 * chose for itself is a test, and the point of these is to produce numbers that did not
 * exist before.
 */
class TransportMeasurementsTest {

    private val enabled = System.getProperty("dsrc.measure") == "true"

    private val sockets = mutableListOf<Socket>()
    private val servers = mutableListOf<ServerSocket>()
    private val sessions = mutableListOf<Session>()

    @AfterTest
    fun cleanup() {
        sessions.forEach { runCatching { it.close() } }
        sockets.forEach { runCatching { it.close() } }
        servers.forEach { runCatching { it.close() } }
        sessions.clear(); sockets.clear(); servers.clear()
    }

    private class Sink {
        val frames = ConcurrentLinkedQueue<Frame>()
    }

    private fun pair(sink: Sink): Pair<Session, Session> {
        val server = ServerSocket(0, 4, InetAddress.getLoopbackAddress()).also { servers.add(it) }
        val client = Socket(InetAddress.getLoopbackAddress(), server.localPort).also { sockets.add(it) }
        val accepted = server.accept().also { sockets.add(it) }
        client.tcpNoDelay = true
        accepted.tcpNoDelay = true

        fun build(socket: Socket, role: String, onFrame: (Frame) -> Unit) = Session(
            input = socket.getInputStream(),
            output = socket.getOutputStream(),
            deviceId = "measure-$role",
            role = role,
            monoClock = { System.nanoTime() },
            wallClock = { System.currentTimeMillis() * 1_000_000 },
            onFrame = onFrame,
        ).also { sessions.add(it) }

        val phone = build(client, "phone") {}
        val jetson = build(accepted, "jetson") { sink.frames.add(it) }
        val a = Thread { phone.start() }
        val b = Thread { jetson.start() }
        a.start(); b.start(); a.join(5_000); b.join(5_000)
        return phone to jetson
    }

    @Test
    fun `experiment 1 - golden vector count`() {
        assumeTrue("set -Ddsrc.measure=true to run", enabled)
        val path = System.getProperty("dsrc.goldenFrames") ?: error("golden vector path not set")
        val text = File(path).readText()
        // Counted from the file rather than restated, so the number cannot drift from it.
        val cases = Regex(""""name"\s*:""").findAll(text).count()
        println("MEASURE golden_cases=$cases")
    }

    @Test
    fun `experiment 3 - loopback throughput and wire cost by channel`() {
        assumeTrue("set -Ddsrc.measure=true to run", enabled)
        val sink = Sink()
        val (phone, _) = pair(sink)

        data class Shape(val channel: String, val payload: Int, val extensions: Map<String, JsonValue>)

        val camera = Shape(
            Channels.CAMERA, 25_700,   // task 18's measured JPEG p50 at 1280x720
            CameraFrameMessage(1, 1, 1280, 720, "jpeg", 85).toExtensions(),
        )
        val gps = Shape(Channels.GPS, 0, GpsRecord.noFix(1).toExtensions())
        // A HERE flow reply for a 9 km corridor. Nothing has measured what one costs on
        // this link: tasks 14 and 16 timed small frames only.
        val here = Shape(
            Channels.HERE, 64 * 1024,
            HereResponse(
                captureMonoNs = 1, requestUrl = "https://data.traffic.hereapi.com/v7/flow?in=corridor:...",
                status = 200, contentType = "application/json", queryLat = 40.7128, queryLon = -74.0060,
                queryRadiusM = 9_000.0, requestMonoNs = 1, responseMonoNs = 2,
            ).toExtensions(),
        )

        for (shape in listOf(gps, camera, here)) {
            val payload = ByteArray(shape.payload) { ((it * 31 + 7) % 256).toByte() }
            // One frame, measured exactly: the encoded size is the wire cost.
            val header = Framing.header(
                channel = shape.channel, sequence = 0, monoNs = System.nanoTime(),
                wallNs = System.currentTimeMillis() * 1_000_000, extensions = shape.extensions,
            )
            val bytes = Framing.encode(header, payload).size
            println("MEASURE wire_bytes channel=${shape.channel} payload=${shape.payload} frame=$bytes")

            // Paced to the channel's own depth, and timed from the first send to the last
            // arrival. The first version divided `accepted` by elapsed time, which on
            // `camera` -- latest_wins at depth one -- reported 68,536 fps and 1,692 MiB/s
            // while exactly **one** frame of two hundred arrived. That is a
            // queue-insertion rate on a channel that threw the rest away, not a
            // throughput, and it could not have gone down however slow the link was.
            val depth = Channels.policy(shape.channel).depth
            val before = sink.frames.size
            val count = if (shape.payload > 10_000) 200 else 2_000
            val start = System.nanoTime()
            var accepted = 0
            repeat(count) {
                // Strict for a depth-one channel: anything in flight would be displaced.
                // A spin, not a sleep. Thread.sleep(0, 200_000) rounds up to a
                // millisecond or more here, so at depth one it cost ~2.6 ms a frame and
                // the 378 fps reported was this loop's granularity, not the transport's
                // rate. Bounded by the send count, so it cannot run away.
                while (phone.outboundPending() >= depth) Thread.onSpinWait()
                if (phone.send(shape.channel, shape.extensions, payload)) accepted++
            }
            val deadline = System.currentTimeMillis() + 30_000
            while (sink.frames.size - before < accepted && System.currentTimeMillis() < deadline) {
                Thread.sleep(1)
            }
            val elapsedNs = System.nanoTime() - start
            val arrived = sink.frames.size - before
            val ours = phone.stats().channels.getValue(shape.channel)
            // Arrivals over elapsed time. Anything else is measuring this loop.
            val perSecond = arrived * 1e9 / elapsedNs
            println(
                "MEASURE throughput channel=${shape.channel} depth=$depth offered=$count " +
                    "accepted=$accepted arrived=$arrived dropped=${ours.dropped} " +
                    "fps=${"%.0f".format(perSecond)} " +
                    "MiB_s=${"%.1f".format(perSecond * bytes / 1_048_576)}"
            )

        }
    }

    @Test
    fun `experiment 4 - queueing latency from the enqueue stamp`() {
        assumeTrue("set -Ddsrc.measure=true to run", enabled)
        val sink = Sink()
        val (phone, _) = pair(sink)

        // What this measures, precisely, because the plan overstated it. `t_mono_ns` is now
        // genuinely the enqueue instant, so `t_mono_ns - t_capture_mono_ns` is the time
        // between the caller stamping a sample and the transport accepting it. That is the
        // pipeline delay -- encode, validate, enqueue.
        //
        // It is NOT the bias task 18's O1 is about. O1 asks how far
        // `t_capture_mono_ns` sits from the shutter, and this subtraction starts *at*
        // t_capture: the shutter is on the far side of it and appears in neither operand.
        // A shutter reading needs a hardware timestamp the camera does not give us.
        for (payloadBytes in listOf(0, 25_700)) {
            val channel = if (payloadBytes == 0) Channels.GPS else Channels.CAMERA
            val payload = ByteArray(payloadBytes) { ((it * 31 + 7) % 256).toByte() }
            val deltas = mutableListOf<Long>()
            val before = sink.frames.size

            repeat(400) {
                // Paced to the depth, so every sample is measured rather than only the ones
                // that survived: unpaced, `camera` yielded 17 usable samples of 400 and a
                // p50 drawn from whichever frames happened not to be displaced.
                //
                // The wait comes *before* the capture stamp. Stamping first put the wait
                // inside the measurement, and camera's p50 came out at 2,995 us -- which
                // was this loop's 200 us sleep granularity on a depth-one queue, reported
                // as if it were the pipeline's cost.
                while (phone.outboundPending() >= Channels.policy(channel).depth) {
                    Thread.onSpinWait()
                }
                val capture = System.nanoTime()
                val extensions = if (channel == Channels.GPS) {
                    GpsRecord.noFix(capture).toExtensions()
                } else {
                    CameraFrameMessage(capture, 1, 1280, 720, "jpeg", 85).toExtensions()
                }
                phone.send(channel, extensions, payload)
            }
            val deadline = System.currentTimeMillis() + 20_000
            while (phone.outboundPending() > 0 && System.currentTimeMillis() < deadline) Thread.sleep(2)
            Thread.sleep(300)

            sink.frames.drop(before).filter { it.channel == channel }.forEach { frame ->
                val mono = (frame.header.entries[Framing.KEY_MONO] as? JsonValue.Num)?.value ?: return@forEach
                val capture = (frame.header.entries[Fields.CAPTURE_KEY] as? JsonValue.Num)?.value ?: return@forEach
                deltas.add(mono - capture)
            }
            if (deltas.isEmpty()) {
                println("MEASURE queueing channel=$channel samples=0")
                continue
            }
            val sorted = deltas.sorted()
            fun pct(p: Double) = sorted[(sorted.size * p).toInt().coerceAtMost(sorted.size - 1)]
            println(
                "MEASURE queueing channel=$channel samples=${sorted.size} " +
                    "p50_us=${pct(0.50) / 1_000} p95_us=${pct(0.95) / 1_000} " +
                    "max_us=${sorted.last() / 1_000} negative=${sorted.count { it < 0 }}"
            )
        }
    }
}
