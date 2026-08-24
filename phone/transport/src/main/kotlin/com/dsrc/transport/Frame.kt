package com.dsrc.transport

import java.io.EOFException
import java.io.InputStream
import java.io.OutputStream

/** A framing violation. Not recoverable: the session must end. */
class FramingError(message: String) : Exception(message)

/**
 * One frame: an opaque payload and the JSON header describing it.
 *
 * The transport assigns no meaning to the payload; `Messages` does that.
 */
class Frame(val header: JsonValue.Obj, val payload: ByteArray) {

    val channel: String
        get() = (header.entries[Framing.KEY_CHANNEL] as? JsonValue.Text)?.value
            ?: throw FramingError("frame has no string ${Framing.KEY_CHANNEL}")

    val sequence: Long
        get() = (header.entries[Framing.KEY_SEQ] as? JsonValue.Num)?.value
            ?: throw FramingError("frame has no integer ${Framing.KEY_SEQ}")
}

/**
 * The wire format from `specs/transport_protocol.md`:
 *
 * ```
 * offset  size  field
 * 0       4     payload_len   uint32 big-endian
 * 4       2     header_len    uint16 big-endian
 * 6       H     header        UTF-8 canonical JSON object
 * 6+H     P     payload       opaque
 * ```
 *
 * Both lengths are checked **before** anything is allocated. A receiver that sizes a
 * buffer from an unvalidated length is a denial of service against itself on the first
 * corrupted prefix, and a corrupted prefix is exactly what a desynchronised stream
 * looks like.
 */
object Framing {

    const val KEY_CHANNEL = "ch"
    const val KEY_SEQ = "seq"
    const val KEY_MONO = "t_mono_ns"
    const val KEY_WALL = "t_wall_ns"
    const val KEY_PAYLOAD_LEN = "n"

    /** Required in every frame, in the order the spec lists them. */
    val REQUIRED_KEYS = listOf(KEY_CHANNEL, KEY_SEQ, KEY_MONO, KEY_WALL, KEY_PAYLOAD_LEN)

    const val PREFIX_BYTES = 6

    /**
     * Reserved for the transport on every channel.
     *
     * Three, matching Python's `RESERVED_EXTENSIONS`. `t_wire_mono_ns` belongs here and
     * was missing: a caller could set it, the frame would go out carrying a
     * caller-controlled value in the field the peer's timebase reads as our departure
     * stamp, and no stamping would have happened. The spec names two under the MUST NOT
     * at one point and calls the wire stamp "a reserved extension" at another; Python
     * implements three and the plan says three.
     */
    val RESERVED_KEYS = listOf("hello", "heartbeat", "t_wire_mono_ns")

    /**
     * Build a header from the transport's own fields plus a message's extensions.
     *
     * Extensions are top-level keys, not a nested object -- that is what makes the
     * header additive, and it is why a receiver must accept and preserve keys it does
     * not recognise. Two are refused here: `hello` and `heartbeat` are reserved for the
     * transport, and a caller's message carrying either would be consumed by the peer as
     * transport traffic rather than delivered, lost with no drop counted and no sequence
     * gap to show it.
     */
    fun header(
        channel: String,
        sequence: Long,
        monoNs: Long,
        wallNs: Long,
        extensions: Map<String, JsonValue> = emptyMap(),
        allowReserved: Set<String> = emptySet(),
    ): JsonValue.Obj {
        for (key in RESERVED_KEYS) {
            if (key in extensions && key !in allowReserved) {
                throw FramingError("'$key' is reserved for the transport")
            }
        }
        for (key in REQUIRED_KEYS) {
            if (key in extensions) throw FramingError("'$key' is owned by the transport")
        }
        return JsonValue.Obj(
            extensions + mapOf(
                KEY_CHANNEL to JsonValue.Text(channel),
                KEY_SEQ to JsonValue.Num(sequence),
                KEY_MONO to JsonValue.Num(monoNs),
                KEY_WALL to JsonValue.Num(wallNs),
            )
        )
    }

    /**
     * Check both size limits without building the frame.
     *
     * For the sender's pre-flight check, where [encode] was being called purely to learn
     * whether the header fits: it allocates prefix + header + payload and copies the
     * payload in, so a 4 MiB camera frame was copied twice on every send -- once to size
     * the header and once to write it. Python sizes at enqueue and reuses the buffer.
     */
    fun checkSizes(header: JsonValue.Obj, payloadSize: Int) {
        val complete = withPayloadLength(header, payloadSize)
        validateHeader(complete, payloadSize)
        val headerSize = Json.encodeToBytes(complete).size
        if (headerSize > Protocol.MAX_HEADER_BYTES) {
            throw FramingError("header is $headerSize bytes, over ${Protocol.MAX_HEADER_BYTES}")
        }
        if (payloadSize > Protocol.MAX_PAYLOAD_BYTES) {
            throw FramingError("payload is $payloadSize bytes, over ${Protocol.MAX_PAYLOAD_BYTES}")
        }
    }

    fun encode(header: JsonValue.Obj, payload: ByteArray): ByteArray {
        // `n` is transport-owned and derived from the payload, so a caller supplies the
        // logical header and the framer fills it in -- which is how the golden vectors
        // record it too. A caller-supplied `n` is honoured only if it agrees, since a
        // disagreement is the desynchronisation the field exists to detect.
        val complete = withPayloadLength(header, payload.size)
        validateHeader(complete, payload.size)
        val headerBytes = Json.encodeToBytes(complete)
        if (headerBytes.size > Protocol.MAX_HEADER_BYTES) {
            throw FramingError("header is ${headerBytes.size} bytes, over ${Protocol.MAX_HEADER_BYTES}")
        }
        if (payload.size > Protocol.MAX_PAYLOAD_BYTES) {
            throw FramingError("payload is ${payload.size} bytes, over ${Protocol.MAX_PAYLOAD_BYTES}")
        }
        val out = ByteArray(PREFIX_BYTES + headerBytes.size + payload.size)
        writeUInt32(out, 0, payload.size)
        writeUInt16(out, 4, headerBytes.size)
        headerBytes.copyInto(out, PREFIX_BYTES)
        payload.copyInto(out, PREFIX_BYTES + headerBytes.size)
        return out
    }

    fun write(header: JsonValue.Obj, payload: ByteArray, out: OutputStream) {
        // Assembled whole, then written once: a frame half on the wire when an
        // exception interrupts it desynchronises the peer permanently, and there is no
        // delimiter for it to resynchronise on.
        out.write(encode(header, payload))
        out.flush()
    }

    /** The header with `n` set from the payload, refusing a caller value that disagrees. */
    fun withPayloadLength(header: JsonValue.Obj, payloadLength: Int): JsonValue.Obj {
        val existing = header.entries[KEY_PAYLOAD_LEN]
        if (existing != null) {
            val declared = (existing as? JsonValue.Num)?.value
                ?: throw FramingError("'$KEY_PAYLOAD_LEN' must be an integer")
            if (declared != payloadLength.toLong()) {
                throw FramingError("'$KEY_PAYLOAD_LEN' is $declared but the payload is $payloadLength bytes")
            }
            return header
        }
        return JsonValue.Obj(header.entries + (KEY_PAYLOAD_LEN to JsonValue.Num(payloadLength.toLong())))
    }

    /**
     * Validate a header the way a receiver would, so a sender never emits what its own
     * decoder would refuse.
     *
     * `n` is deliberately redundant with the binary prefix. A disagreement means the
     * two sides have desynchronised, so it is a protocol error rather than something
     * to reconcile.
     */
    fun validateHeader(header: JsonValue.Obj, payloadLength: Int) {
        for (key in REQUIRED_KEYS) {
            if (key !in header.entries) throw FramingError("header is missing '$key'")
        }
        if (header.entries[KEY_CHANNEL] !is JsonValue.Text) {
            throw FramingError("'$KEY_CHANNEL' must be a string")
        }
        for (key in listOf(KEY_SEQ, KEY_MONO, KEY_WALL, KEY_PAYLOAD_LEN)) {
            if (header.entries[key] !is JsonValue.Num) throw FramingError("'$key' must be an integer")
        }
        val declared = (header.entries[KEY_PAYLOAD_LEN] as JsonValue.Num).value
        if (declared != payloadLength.toLong()) {
            throw FramingError("'$KEY_PAYLOAD_LEN' is $declared but the payload is $payloadLength bytes")
        }
        val channel = (header.entries[KEY_CHANNEL] as JsonValue.Text).value
        if (!Channels.isKnown(channel)) throw FramingError("unknown channel '$channel'")
    }

    /**
     * Read one frame.
     *
     * Reads are capped at [Protocol.MAX_READ_BYTES] so a large frame on a slow link
     * cannot look like a dead peer: the stall timeout is measured on completed reads,
     * not completed frames. At a 4 MiB limit and a 5 s timeout, measuring per frame
     * would end any session on a link under about 839 KB/s, and the session would then
     * reconnect and re-send, so the link would never recover.
     */
    fun read(input: InputStream, onReadProgress: () -> Unit = {}): Frame {
        val prefix = readExactly(input, PREFIX_BYTES, onReadProgress)
        val payloadLength = readUInt32(prefix, 0)
        val headerLength = readUInt16(prefix, 4)

        // Checked before allocating anything, and before a single payload byte is read.
        if (payloadLength > Protocol.MAX_PAYLOAD_BYTES) {
            throw FramingError("payload_len $payloadLength exceeds ${Protocol.MAX_PAYLOAD_BYTES}")
        }
        if (headerLength > Protocol.MAX_HEADER_BYTES) {
            throw FramingError("header_len $headerLength exceeds ${Protocol.MAX_HEADER_BYTES}")
        }
        if (headerLength == 0) throw FramingError("header_len is zero")

        val headerBytes = readExactly(input, headerLength, onReadProgress)
        val decoded = try {
            Json.decodeBytes(headerBytes)
        } catch (e: JsonError) {
            throw FramingError("header is not valid JSON: ${e.message}")
        }
        if (decoded !is JsonValue.Obj) {
            throw FramingError("header is a ${decoded::class.simpleName}, not an object")
        }

        val payload = readExactly(input, payloadLength.toInt(), onReadProgress)
        validateHeader(decoded, payload.size)
        return Frame(decoded, payload)
    }

    private fun readExactly(input: InputStream, length: Int, onReadProgress: () -> Unit): ByteArray {
        val buffer = ByteArray(length)
        var filled = 0
        while (filled < length) {
            // min(chunk, remaining): a frame begins with a 6-byte prefix read and a
            // header read, so a peer keeps a session alive by completing any one read
            // per timeout -- tens of bytes per second, not kilobytes.
            val want = minOf(Protocol.MAX_READ_BYTES, length - filled)
            val read = input.read(buffer, filled, want)
            if (read < 0) {
                // Mid-frame disconnect. A partial frame is never delivered.
                throw EOFException("stream ended $filled/$length bytes into a read")
            }
            filled += read
            if (read > 0) onReadProgress()
        }
        return buffer
    }

    private fun writeUInt32(out: ByteArray, at: Int, value: Int) {
        out[at] = (value ushr 24).toByte()
        out[at + 1] = (value ushr 16).toByte()
        out[at + 2] = (value ushr 8).toByte()
        out[at + 3] = value.toByte()
    }

    private fun writeUInt16(out: ByteArray, at: Int, value: Int) {
        out[at] = (value ushr 8).toByte()
        out[at + 1] = value.toByte()
    }

    /** Read as unsigned, so a length with the top bit set is large rather than negative. */
    internal fun readUInt32(bytes: ByteArray, at: Int): Long =
        (bytes[at].toLong() and 0xFF shl 24) or
            (bytes[at + 1].toLong() and 0xFF shl 16) or
            (bytes[at + 2].toLong() and 0xFF shl 8) or
            (bytes[at + 3].toLong() and 0xFF)

    internal fun readUInt16(bytes: ByteArray, at: Int): Int =
        (bytes[at].toInt() and 0xFF shl 8) or (bytes[at + 1].toInt() and 0xFF)
}
