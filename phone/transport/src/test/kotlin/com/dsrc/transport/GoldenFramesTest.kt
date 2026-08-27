package com.dsrc.transport

import java.io.ByteArrayInputStream
import java.io.File
import java.security.MessageDigest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The cross-language contract, checked against the frozen bytes.
 *
 * `specs/transport_golden_frames.json` records, per case, the logical header, the
 * payload generator, the 6-byte prefix, the encoded header bytes, the total length and a
 * SHA-256 of the whole frame. Every implementation must encode each case to exactly
 * those bytes and decode them back to those fields.
 *
 * Data-driven from the file rather than transcribed: a transcription can contain a typo
 * that agrees with the bug, and a regenerated file has to fail loudly here rather than
 * be silently re-approved. The file is a declared Gradle input, so editing it alone
 * re-runs this.
 */
class GoldenFramesTest {

    private val document: JsonValue.Obj by lazy {
        val path = System.getProperty("dsrc.goldenFrames")
            ?: error("dsrc.goldenFrames is not set; the build must pass the vector path")
        val file = File(path)
        require(file.isFile) { "golden vectors not found at $path" }
        Json.decode(file.readText()) as JsonValue.Obj
    }

    private val cases: List<JsonValue.Obj> by lazy {
        (document.entries.getValue("cases") as JsonValue.Arr).items.map { it as JsonValue.Obj }
    }

    private fun JsonValue.Obj.text(key: String) = (entries.getValue(key) as JsonValue.Text).value
    private fun JsonValue.Obj.num(key: String) = (entries.getValue(key) as JsonValue.Num).value
    private fun JsonValue.Obj.obj(key: String) = entries.getValue(key) as JsonValue.Obj

    /**
     * The wire header for a case.
     *
     * The file nests a case's extension keys under `extensions` for readability; on the
     * wire they are top-level header keys, which is what makes the header additive. `n`
     * is added by the framer, so it is absent here.
     */
    private fun wireHeader(case: JsonValue.Obj): JsonValue.Obj {
        val frame = case.obj("frame")
        val extensions = (frame.entries["extensions"] as? JsonValue.Obj)?.entries ?: emptyMap()
        return JsonValue.Obj(frame.entries.filterKeys { it != "extensions" } + extensions)
    }

    /** `payload[i] = (i * 37 + 11) % 256`, as the file states. */
    private fun payloadFor(spec: JsonValue.Obj): ByteArray {
        assertEquals("pattern", spec.text("kind"), "unknown payload generator")
        val length = spec.num("length").toInt()
        return ByteArray(length) { ((it * 37 + 11) % 256).toByte() }
    }

    private fun hex(bytes: ByteArray) = bytes.joinToString("") { "%02x".format(it) }

    private fun hexToBytes(text: String) =
        ByteArray(text.length / 2) { text.substring(it * 2, it * 2 + 2).toInt(16).toByte() }

    private fun sha256(bytes: ByteArray) = hex(MessageDigest.getInstance("SHA-256").digest(bytes))

    @Test
    fun `the vector file is present, frozen, and the version we implement`() {
        // Guards every assertion below: a missing or truncated file would make them all
        // pass over an empty list.
        assertEquals(Protocol.VERSION.toLong(), document.num("protocol_version"))
        assertTrue(
            (document.entries.getValue("frozen") as JsonValue.Bool).value,
            "the vectors must be marked frozen",
        )
        assertTrue(cases.size >= 18, "only ${cases.size} cases")
    }

    @Test
    fun `every case encodes to exactly the recorded bytes`() {
        val failures = mutableListOf<String>()
        for (case in cases) {
            val name = case.text("name")
            val header = wireHeader(case)
            val payload = payloadFor(case.obj("payload"))
            val encoded = try {
                Framing.encode(header, payload)
            } catch (e: Exception) {
                failures.add("$name: encode threw ${e::class.simpleName}: ${e.message}")
                continue
            }
            val prefix = encoded.copyOfRange(0, Framing.PREFIX_BYTES)
            val headerBytes = encoded.copyOfRange(
                Framing.PREFIX_BYTES,
                Framing.PREFIX_BYTES + case.num("header_len").toInt(),
            )
            if (hex(prefix) != case.text("prefix_hex")) {
                failures.add("$name: prefix ${hex(prefix)} != ${case.text("prefix_hex")}")
            }
            if (hex(headerBytes) != case.text("header_hex")) {
                failures.add("$name: header bytes differ\n  got ${hex(headerBytes)}\n  want ${case.text("header_hex")}")
            }
            if (encoded.size.toLong() != case.num("frame_len")) {
                failures.add("$name: frame_len ${encoded.size} != ${case.num("frame_len")}")
            }
            if (sha256(encoded) != case.text("frame_sha256")) {
                failures.add("$name: sha256 ${sha256(encoded)} != ${case.text("frame_sha256")}")
            }
        }
        assertTrue(
            failures.isEmpty(),
            "${failures.size} of ${cases.size} cases diverged:\n" + failures.joinToString("\n"),
        )
    }

    @Test
    fun `every case decodes back to the recorded fields`() {
        val failures = mutableListOf<String>()
        for (case in cases) {
            val name = case.text("name")
            val header = wireHeader(case)
            val payload = payloadFor(case.obj("payload"))
            val encoded = Framing.encode(header, payload)

            val frame = try {
                Framing.read(ByteArrayInputStream(encoded))
            } catch (e: Exception) {
                failures.add("$name: read threw ${e::class.simpleName}: ${e.message}")
                continue
            }
            // Compared through the canonical encoding, which is the only form in which
            // two headers are meaningfully equal on this wire.
            val expected = Framing.withPayloadLength(header, payload.size)
            if (Json.encode(frame.header) != Json.encode(expected)) {
                failures.add("$name: header round trip differs\n  got ${Json.encode(frame.header)}\n  want ${Json.encode(expected)}")
            }
            if (!frame.payload.contentEquals(payload)) {
                failures.add("$name: payload round trip differs (${frame.payload.size} vs ${payload.size} bytes)")
            }
        }
        assertTrue(failures.isEmpty(), "${failures.size} cases failed to round-trip:\n" + failures.joinToString("\n"))
    }

    @Test
    fun `the recorded header bytes decode to the recorded header`() {
        // The other direction: rather than encoding ours and comparing, take the frozen
        // bytes and check our decoder reads them as the recorded fields. A codec wrong in
        // a self-consistent way passes the encode test and fails this one.
        for (case in cases) {
            val bytes = case.text("header_hex").chunked(2).map { it.toInt(16).toByte() }.toByteArray()
            val decoded = Json.decodeBytes(bytes)
            val expected = Framing.withPayloadLength(
                wireHeader(case),
                case.obj("payload").num("length").toInt(),
            )
            assertEquals(Json.encode(expected), Json.encode(decoded), "case ${case.text("name")}")
        }
    }

    @Test
    fun `every message case decodes through this side's typed decoder`() {
        // The file is described as what keeps the two codecs honest, and until now this
        // side never took it past the framing layer: no `fromWire` was called anywhere in
        // this class, so at the typed-message layer the frozen vectors bound Python alone.
        // Python has the twin of this test; this is the half that was missing.
        //
        // Decoded from the file's own recorded bytes rather than from anything this
        // implementation just produced -- otherwise a bug symmetric across encode and
        // decode passes.
        val framing = setOf("ch", "seq", "t_mono_ns", "t_wall_ns", "n")
        val messageCases = cases.filter { it.text("name").startsWith("message_") }
        assertTrue(messageCases.isNotEmpty(), "no message cases in the vector file")

        for (case in messageCases) {
            val header = Json.decode(
                String(hexToBytes(case.text("header_hex")), Charsets.UTF_8)
            ) as JsonValue.Obj
            val channel = (header.entries.getValue("ch") as JsonValue.Text).value
            val extensions = header.entries.filterKeys { it !in framing }
            val payload = payloadFor(case.obj("payload"))

            // The receive path's own entry point, so this asserts what an arriving frame
            // would actually meet rather than a decoder chosen by the test.
            MessageValidation.check(
                channel = channel,
                extensions = extensions,
                payload = payload,
                checkReservedKeys = false,
            )
        }
    }

    @Test
    fun `the maximum payload case really is at the limit`() {
        // Proof the boundary is exercised rather than merely listed.
        val max = cases.single { it.text("name") == "max_payload" }
        assertEquals(Protocol.MAX_PAYLOAD_BYTES.toLong(), max.obj("payload").num("length"))
    }

    @Test
    fun `a payload one byte over the limit is refused`() {
        val header = JsonValue.Obj(
            mapOf(
                "ch" to JsonValue.Text(Channels.CAMERA),
                "seq" to JsonValue.Num(0),
                "t_mono_ns" to JsonValue.Num(1),
                "t_wall_ns" to JsonValue.Num(2),
                "n" to JsonValue.Num(Protocol.MAX_PAYLOAD_BYTES + 1L),
            )
        )
        val error = runCatching {
            Framing.encode(header, ByteArray(Protocol.MAX_PAYLOAD_BYTES + 1))
        }.exceptionOrNull()
        assertTrue(error is FramingError, "expected FramingError, got $error")
    }
}
