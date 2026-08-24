package com.dsrc.transport

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TimeSyncTest {

    private val ping = TimeSyncMessage(
        captureMonoNs = 1_000_000_008,
        exchangeId = 17,
        wireMonoNs = 0,
        peerRecvMonoNs = null,
        peerRecvWallNs = null,
        peerWireMonoNs = null,
    )

    private val pong = TimeSyncMessage(
        captureMonoNs = 1_000_000_009,
        exchangeId = 17,
        wireMonoNs = 1_000_000_100,
        peerRecvMonoNs = 2_000_000_050,
        peerRecvWallNs = 1_755_648_000_123_456_789,
        peerWireMonoNs = 1_000_000_020,
    )

    private fun decode(m: TimeSyncMessage) = TimeSyncMessage.fromWire(m.toExtensions(), ByteArray(0))

    // -- the one-type discipline ---------------------------------------------

    @Test
    fun `a ping and a pong are the same type, told apart by nulls`() {
        // One message type, because the channel is the discriminator for everything else
        // on this wire and a second type would need a `kind` field.
        assertTrue(ping.isPing)
        assertTrue(!pong.isPing)
    }

    @Test
    fun `both survive a round trip`() {
        assertEquals(ping, decode(ping))
        assertEquals(pong, decode(pong))
    }

    @Test
    fun `the ping encodes to exactly the recorded header`() {
        val header = Framing.header(Channels.CONTROL, 1, 1_100_000_000, 1_755_648_000_000_000_000, ping.toExtensions(), allowReserved = setOf("t_wire_mono_ns"))
        val recorded = """{"ch":"control","exchange_id":17,"n":0,"seq":1,""" +
            """"t_capture_mono_ns":1000000008,"t_mono_ns":1100000000,"t_peer_recv_mono_ns":null,""" +
            """"t_peer_recv_wall_ns":null,"t_peer_wire_mono_ns":null,""" +
            """"t_wall_ns":1755648000000000000,"t_wire_mono_ns":0}"""
        assertEquals(recorded, Json.encode(Framing.withPayloadLength(header, 0)))
    }

    @Test
    fun `the pong encodes to exactly the recorded header`() {
        val header = Framing.header(Channels.CONTROL, 1, 1_100_000_000, 1_755_648_000_000_000_000, pong.toExtensions(), allowReserved = setOf("t_wire_mono_ns"))
        val recorded = """{"ch":"control","exchange_id":17,"n":0,"seq":1,""" +
            """"t_capture_mono_ns":1000000009,"t_mono_ns":1100000000,"t_peer_recv_mono_ns":2000000050,""" +
            """"t_peer_recv_wall_ns":1755648000123456789,"t_peer_wire_mono_ns":1000000020,""" +
            """"t_wall_ns":1755648000000000000,"t_wire_mono_ns":1000000100}"""
        assertEquals(recorded, Json.encode(Framing.withPayloadLength(header, 0)))
    }

    // -- all or nothing ------------------------------------------------------

    @Test
    fun `a partially filled pong is refused`() {
        // Worse than refusing would be accepting: a consumer would compute an offset from
        // a mixture of set and missing terms, the arithmetic would run, and it would
        // produce a number.
        for (present in listOf(
            TimeSyncMessage.KEY_PEER_RECV_MONO,
            TimeSyncMessage.KEY_PEER_RECV_WALL,
            TimeSyncMessage.KEY_PEER_WIRE,
        )) {
            val partial = ping.toExtensions() + (present to JsonValue.Num(5))
            val error = assertFailsWith<MessageError>("only $present set") {
                TimeSyncMessage.fromWire(partial, ByteArray(0))
            }
            assertEquals(RefusalReason.NULL_NOT_ALLOWED, error.reason)
        }
    }

    @Test
    fun `two of three set is also refused`() {
        val partial = ping.toExtensions() +
            (TimeSyncMessage.KEY_PEER_RECV_MONO to JsonValue.Num(1)) +
            (TimeSyncMessage.KEY_PEER_RECV_WALL to JsonValue.Num(2))
        assertEquals(
            RefusalReason.NULL_NOT_ALLOWED,
            assertFailsWith<MessageError> { TimeSyncMessage.fromWire(partial, ByteArray(0)) }.reason,
        )
    }

    @Test
    fun `an absent peer field is refused, not read as a ping`() {
        // Present-and-null is the only spelling of "not set", so *absent* is
        // `missing_field` -- distinct from the partially-filled pong below, whose unset
        // stamps are present and null and so are `null_not_allowed`. The two reasons exist
        // to separate exactly these cases, which is why the fix for one must not be
        // applied to the other.
        val missing = pong.toExtensions() - TimeSyncMessage.KEY_PEER_WIRE
        assertEquals(
            RefusalReason.MISSING_FIELD,
            assertFailsWith<MessageError> { TimeSyncMessage.fromWire(missing, ByteArray(0)) }.reason,
        )
    }

    @Test
    fun `a payload on the control channel is refused`() {
        assertEquals(
            RefusalReason.UNEXPECTED_PAYLOAD,
            assertFailsWith<MessageError> {
                TimeSyncMessage.fromWire(ping.toExtensions(), byteArrayOf(1))
            }.reason,
        )
    }

    @Test
    fun `a required stamp cannot be null`() {
        for (key in listOf(Fields.CAPTURE_KEY, TimeSyncMessage.KEY_EXCHANGE, TimeSyncMessage.KEY_WIRE)) {
            val nulled = ping.toExtensions() + (key to JsonValue.Null)
            assertEquals(
                RefusalReason.NULL_NOT_ALLOWED,
                assertFailsWith<MessageError>("null $key") {
                    TimeSyncMessage.fromWire(nulled, ByteArray(0))
                }.reason,
                "for $key",
            )
        }
    }

    // -- the responder -------------------------------------------------------

    private fun responder(mono: Long = 5_000, wall: Long = 9_000) =
        TimeSyncResponder(monoClock = { mono }, wallClock = { wall })

    @Test
    fun `a pong echoes the ping's wire stamp, not our own clock`() {
        // Substituting our own clock would replace the initiator's t1 with a value from a
        // different device, and the offset it computed would be wrong by the whole link
        // delay. This is the correction task 15 had to make to its own plan.
        val incoming = ping.copy(wireMonoNs = 1_000_000_020)
        val reply = responder().reply(incoming, recvMonoNs = 2_000_000_050, recvWallNs = 1_755_648_000_123_456_789)!!
        assertEquals(1_000_000_020, reply.peerWireMonoNs)
    }

    @Test
    fun `a pong carries the receive stamps and matches the exchange`() {
        val reply = responder().reply(ping, recvMonoNs = 42, recvWallNs = 99)!!
        assertEquals(17, reply.exchangeId, "a pong must match its ping")
        assertEquals(42, reply.peerRecvMonoNs)
        assertEquals(99, reply.peerRecvWallNs)
        assertTrue(!reply.isPing, "the reply must be a pong")
    }

    @Test
    fun `a pong leaves its own wire stamp to the writer`() {
        // Stamped immediately before the bytes leave, not here: an enqueue stamp includes
        // however long the frame waited behind others, which for a timebase estimate is
        // the dominant error.
        assertEquals(0, responder().reply(ping, 1, 2)!!.wireMonoNs)
    }

    @Test
    fun `a pong is not answered`() {
        // Not an error -- the peer may be answering a ping we never sent -- but replying
        // would put two responders in a loop.
        val responder = responder()
        assertNull(responder.reply(pong, 1, 2))
        assertEquals(1, responder.pingsIgnored)
        assertEquals(0, responder.pongsSent)
    }

    @Test
    fun `the responder counts what it answered`() {
        val responder = responder()
        repeat(3) { responder.reply(ping, 1, 2) }
        assertEquals(3, responder.pongsSent)
        assertEquals(0, responder.pingsIgnored)
    }

    @Test
    fun `the responder carries no state that could change its answer`() {
        // The phone is a responder only: it sees t2 and t3 and has no path to the offset,
        // so it accumulates nothing across exchanges. Asserted behaviourally rather than
        // by reflection -- a reflection check on member *names* would pass for an
        // estimator called something else, and it needs kotlin-reflect at runtime.
        val responder = responder()

        // A hundred exchanges in between must not change the answer to an identical ping.
        val first = responder.reply(ping, recvMonoNs = 100, recvWallNs = 200)!!
        repeat(100) { i ->
            responder.reply(ping.copy(exchangeId = i.toLong(), wireMonoNs = i.toLong()), i.toLong(), i.toLong())
        }
        val later = responder.reply(ping, recvMonoNs = 100, recvWallNs = 200)!!
        assertEquals(first, later, "the reply must depend only on this exchange")
    }

    @Test
    fun `the reply depends on nothing but the ping and the two receive stamps`() {
        // Two independent responders, same inputs, same answer -- so there is no hidden
        // per-instance history feeding the result.
        val a = responder().reply(ping.copy(wireMonoNs = 77), 11, 13)
        val b = responder().reply(ping.copy(wireMonoNs = 77), 11, 13)
        assertEquals(a, b)
    }

    @Test
    fun `a reply round-trips through the wire unchanged`() {
        val reply = responder().reply(ping.copy(wireMonoNs = 7), 11, 13)!!
        assertEquals(reply, TimeSyncMessage.fromWire(reply.toExtensions(), ByteArray(0)))
    }
}
