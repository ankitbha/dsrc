package com.dsrc.transport

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class JsonTest {

    private fun obj(vararg pairs: Pair<String, JsonValue>) = JsonValue.Obj(mapOf(*pairs))

    // -- canonical form ------------------------------------------------------

    @Test
    fun `separators carry no spaces`() {
        assertEquals(
            """{"a":1,"b":2}""",
            Json.encode(obj("a" to JsonValue.Num(1), "b" to JsonValue.Num(2))),
        )
    }

    @Test
    fun `keys are sorted regardless of insertion order`() {
        assertEquals(
            """{"a":1,"b":2,"c":3}""",
            Json.encode(obj("c" to JsonValue.Num(3), "a" to JsonValue.Num(1), "b" to JsonValue.Num(2))),
        )
    }

    @Test
    fun `keys are sorted recursively`() {
        // A nested object with unsorted keys is a golden-vector case, so this is not
        // hypothetical.
        val nested = obj(
            "outer" to obj("z" to JsonValue.Num(1), "a" to JsonValue.Num(2)),
            "another" to JsonValue.Num(3),
        )
        assertEquals("""{"another":3,"outer":{"a":2,"z":1}}""", Json.encode(nested))
    }

    @Test
    fun `keys sort by code point, not by utf-16 unit`() {
        // The divergence Kotlin's natural String order would introduce: a surrogate pair
        // begins at 0xD800, which sorts *before* BMP characters at 0xE000 and above in
        // UTF-16 units, while Python sorts by code point and puts astral characters
        // after every BMP one. No golden vector has a non-ASCII key, so nothing else
        // would catch it.
        val astral = "\uD83C\uDF21"        // U+1F321 thermometer, code point 127777
        val bmpHigh = "\uE000"             // private use, code point 57344
        // 0xD83C < 0xE000, so Kotlin's natural order puts the astral key first. That is
        // the divergence: by code point 127777 > 57344, so Python puts it last.
        assertTrue(astral < bmpHigh, "sanity: UTF-16 order puts the astral key first")
        assertTrue(
            Json.compareByCodePoint(bmpHigh, astral) < 0,
            "code point order must put the BMP key first",
        )
        val encoded = Json.encode(obj(astral to JsonValue.Num(1), bmpHigh to JsonValue.Num(2)))
        assertTrue(
            encoded.indexOf(bmpHigh) < encoded.indexOf(astral),
            "encoded as $encoded",
        )
    }

    @Test
    fun `code point comparison agrees with itself on shared prefixes`() {
        assertTrue(Json.compareByCodePoint("a", "ab") < 0)
        assertTrue(Json.compareByCodePoint("ab", "a") > 0)
        assertEquals(0, Json.compareByCodePoint("abc", "abc"))
    }

    // -- strings -------------------------------------------------------------

    @Test
    fun `non-ascii is emitted raw, not escaped`() {
        // ensure_ascii=False. The golden vector carries exactly this string.
        val text = "señal 温度 ±2°C \uD83C\uDF21"
        assertEquals("\"$text\"", Json.encode(JsonValue.Text(text)))
    }

    @Test
    fun `an astral character survives as a surrogate pair`() {
        val thermometer = "\uD83C\uDF21"
        val encoded = Json.encode(JsonValue.Text(thermometer))
        assertEquals(thermometer, (Json.decode(encoded) as JsonValue.Text).value)
        // And as UTF-8 bytes: four for one code point, not two three-byte sequences.
        assertEquals(4, thermometer.toByteArray(Charsets.UTF_8).size)
    }

    @Test
    fun `quote and backslash are escaped`() {
        assertEquals("\"a\\\"b\"", Json.encode(JsonValue.Text("a\"b")))
        assertEquals("\"a\\\\b\"", Json.encode(JsonValue.Text("a\\b")))
    }

    @Test
    fun `the five short control escapes are used`() {
        assertEquals("\"\\b\\f\\n\\r\\t\"", Json.encode(JsonValue.Text("\b\u000c\n\r\t")))
    }

    @Test
    fun `other control characters use lowercase four-digit hex`() {
        assertEquals("\"\\u0000\"", Json.encode(JsonValue.Text("\u0000")))
        assertEquals("\"\\u001f\"", Json.encode(JsonValue.Text("\u001f")))
    }

    @Test
    fun `del is not escaped`() {
        // With ensure_ascii=False Python emits 0x7f raw; escaping it would produce
        // different bytes for the same header.
        assertEquals("\"\u007f\"", Json.encode(JsonValue.Text("\u007f")))
    }

    // -- the integer/float split ---------------------------------------------

    @Test
    fun `an integer beyond two to the fifty-three survives a round trip`() {
        // The trap: a codec parsing every number into a Double returns 9007199254740992
        // here, after which the peer's sequence arithmetic is wrong.
        for (value in listOf(9007199254740993L, -9007199254740993L, Long.MAX_VALUE, Long.MIN_VALUE)) {
            val encoded = Json.encode(JsonValue.Num(value))
            assertEquals(value.toString(), encoded)
            assertEquals(value, (Json.decode(encoded) as JsonValue.Num).value)
        }
    }

    @Test
    fun `an integral float keeps its point-zero through a round trip`() {
        // 35.0 must not become 35: the vectors record it as a float.
        val encoded = Json.encode(JsonValue.Real(35.0))
        assertEquals("35.0", encoded)
        val decoded = Json.decode(encoded)
        assertTrue(decoded is JsonValue.Real, "decoded as ${decoded::class.simpleName}")
        assertEquals("35.0", Json.encode(decoded))
    }

    @Test
    fun `the token's shape decides the type, not the value`() {
        assertTrue(Json.decode("1") is JsonValue.Num)
        assertTrue(Json.decode("1.0") is JsonValue.Real)
        assertTrue(Json.decode("1e2") is JsonValue.Real)
        assertTrue(Json.decode("1E2") is JsonValue.Real)
        assertTrue(Json.decode("-0") is JsonValue.Num)
    }

    @Test
    fun `an integer too large for a long is refused, not widened`() {
        // Widening to a double would lose exactly the precision Num exists to keep.
        assertFailsWith<JsonError> { Json.decode("9223372036854775808") }
    }

    // -- non-finite ----------------------------------------------------------

    @Test
    fun `a non-finite number cannot be encoded`() {
        assertFailsWith<IllegalArgumentException> { Json.encode(JsonValue.Real(Double.NaN)) }
        assertFailsWith<IllegalArgumentException> { Json.encode(JsonValue.Real(Double.POSITIVE_INFINITY)) }
    }

    @Test
    fun `python's bare non-finite tokens are refused on decode`() {
        // Python writes NaN and Infinity as bare tokens unless allow_nan=False; a peer
        // that emitted them must be refused rather than parsed.
        for (token in listOf("NaN", "Infinity", "-Infinity")) {
            assertFailsWith<JsonError> { Json.decode(token) }
        }
    }

    @Test
    fun `a number that overflows to infinity is refused`() {
        assertFailsWith<JsonError> { Json.decode("1e400") }
    }

    // -- structure -----------------------------------------------------------

    @Test
    fun `nulls and booleans round trip`() {
        for (text in listOf("null", "true", "false")) {
            assertEquals(text, Json.encode(Json.decode(text)))
        }
    }

    @Test
    fun `empty containers round trip`() {
        assertEquals("{}", Json.encode(Json.decode("{}")))
        assertEquals("[]", Json.encode(Json.decode("[]")))
    }

    @Test
    fun `array order is preserved`() {
        assertEquals("[3,1,2]", Json.encode(Json.decode("[3,1,2]")))
    }

    @Test
    fun `whitespace between tokens is accepted on decode`() {
        assertEquals("""{"a":1}""", Json.encode(Json.decode(" { \"a\" : 1 } ")))
    }

    @Test
    fun `a duplicate key is refused`() {
        // Python keeps the last one silently, which would make a header's meaning depend
        // on which implementation parsed it.
        assertFailsWith<JsonError> { Json.decode("""{"a":1,"a":2}""") }
    }

    @Test
    fun `malformed input is refused rather than half-parsed`() {
        for (bad in listOf("{", "[", "\"", "{\"a\"}", "{\"a\":}", "[1,]", "tru", "{\"a\":1}x", "-")) {
            assertFailsWith<JsonError>("should refuse: $bad") { Json.decode(bad) }
        }
    }

    @Test
    fun `invalid utf-8 in a header is refused`() {
        // A framing error, not a replacement character that changes the header silently.
        assertFailsWith<JsonError> { Json.decodeBytes(byteArrayOf(0x7B, 0xFF.toByte(), 0x7D)) }
    }


    // -- as strict as CPython ------------------------------------------------

    @Test
    fun `a signed hex escape is refused, not truncated into another character`() {
        // `toIntOrNull(16)` accepts a leading sign, so "\u-041" parsed as -0x41 and
        // toChar() truncated it to U+FFBF -- a different character, silently.
        for (bad in listOf("\"\\u-041\"", "\"\\u+041\"", "\"\\u 041\"", "\"\\uZZZZ\"")) {
            assertFailsWith<JsonError>("should refuse: $bad") { Json.decode(bad) }
        }
    }

    @Test
    fun `a valid hex escape still works`() {
        assertEquals("A", (Json.decode("\"\\u0041\"") as JsonValue.Text).value)
        assertEquals("\u00e9", (Json.decode("\"\\u00e9\"") as JsonValue.Text).value)
    }

    @Test
    fun `an unescaped control character in a string is refused`() {
        // Python's strict parser refuses these, and a raw tab or newline inside a header
        // string is far more likely to be a framing desync than an intended value.
        for (bad in listOf("\"a\tb\"", "\"a\nb\"", "\"a\u0000b\"", "\"a\rb\"")) {
            assertFailsWith<JsonError>("should refuse a raw control char") { Json.decode(bad) }
        }
    }

    @Test
    fun `the escaped forms of those characters are accepted`() {
        assertEquals("a\tb", (Json.decode("\"a\\tb\"") as JsonValue.Text).value)
        assertEquals("a\nb", (Json.decode("\"a\\nb\"") as JsonValue.Text).value)
    }

    @Test
    fun `a leading zero is refused`() {
        // Invalid JSON, and Python enforces it. Accepting it meant the two
        // implementations disagreed about whether a header was even well-formed.
        for (bad in listOf("01", "-01", "00", "007")) {
            assertFailsWith<JsonError>("should refuse $bad") { Json.decode(bad) }
        }
    }

    @Test
    fun `zero itself is fine`() {
        assertEquals(0L, (Json.decode("0") as JsonValue.Num).value)
        assertEquals(0L, (Json.decode("-0") as JsonValue.Num).value)
        assertEquals(0.5, (Json.decode("0.5") as JsonValue.Real).value)
    }

    @Test
    fun `a number with no digits where digits are required is refused`() {
        for (bad in listOf("1.", "-1.", ".5", "1e", "1e+", "1E-")) {
            assertFailsWith<JsonError>("should refuse $bad") { Json.decode(bad) }
        }
    }

    @Test
    fun `an unpaired surrogate cannot be encoded`() {
        // It has no UTF-8 encoding. Left alone it became '?' on the way to bytes, so a
        // header round-tripped to *different* bytes with no error -- where Python raises
        // UnicodeEncodeError.
        assertFailsWith<JsonError> { Json.encode(JsonValue.Text("\uD800")) }
        assertFailsWith<JsonError> { Json.encode(JsonValue.Text("\uDC00")) }
        assertFailsWith<JsonError> { Json.encode(JsonValue.Text("a\uD800b")) }
    }

    @Test
    fun `a properly paired surrogate still encodes`() {
        val thermometer = "\uD83C\uDF21"
        assertEquals("\"$thermometer\"", Json.encode(JsonValue.Text(thermometer)))
    }

    @Test
    fun `an unpaired surrogate arriving as an escape cannot be re-encoded`() {
        // The full wire path: decoding "\ud800" yields a lone surrogate, and re-encoding it
        // must fail rather than quietly produce different bytes.
        val decoded = Json.decode("\"\\ud800\"")
        assertFailsWith<JsonError> { Json.encode(decoded) }
    }

    // -- the golden header ---------------------------------------------------

    @Test
    fun `a full gps header re-encodes to exactly the recorded bytes`() {
        val recorded = """{"altitude_m":35.0,"ch":"gps","fix_quality":1,"hdop":0.9,""" +
            """"heading_deg":91.2,"lat":51.5074,"lon":-0.1278,"n":0,"num_sats":9,"seq":1,""" +
            """"speed_mps":13.4,"t_capture_mono_ns":1000000001,"t_mono_ns":1100000000,""" +
            """"t_wall_ns":1755648000000000000,"utc_epoch_ns":1755648000000000000,"valid":true}"""
        assertEquals(recorded, Json.encode(Json.decode(recorded)))
    }

    @Test
    fun `an all-null gps header re-encodes to exactly the recorded bytes`() {
        val recorded = """{"altitude_m":null,"ch":"gps","fix_quality":0,"hdop":null,""" +
            """"heading_deg":null,"lat":null,"lon":null,"n":0,"num_sats":0,"seq":1,""" +
            """"speed_mps":null,"t_capture_mono_ns":1000000002,"t_mono_ns":1100000000,""" +
            """"t_wall_ns":1755648000000000000,"utc_epoch_ns":null,"valid":false}"""
        assertEquals(recorded, Json.encode(Json.decode(recorded)))
    }

    @Test
    fun `the large-ints header re-encodes to exactly the recorded bytes`() {
        val recorded = """{"big":9223372036854775807,"ch":"gps","n":16,""" +
            """"neg":-9007199254740993,"seq":9007199254740993,""" +
            """"t_mono_ns":9007199254740992,"t_wall_ns":1755648000987654321}"""
        assertEquals(recorded, Json.encode(Json.decode(recorded)))
    }

    @Test
    fun `the non-ascii header re-encodes to exactly the recorded bytes`() {
        val thermometer = "\uD83C\uDF21"
        val recorded = """{"ch":"telemetry","n":3,"note":"señal 温度 ±2°C """ + thermometer +
            """","seq":7,"t_mono_ns":11,"t_wall_ns":13,"unit":"°C"}"""
        assertEquals(recorded, Json.encode(Json.decode(recorded)))
        // And through bytes, which is what actually goes on the wire.
        assertEquals(
            recorded,
            Json.encode(Json.decodeBytes(recorded.toByteArray(Charsets.UTF_8))),
        )
    }

    @Test
    fun `an unsorted header sorts on re-encode`() {
        assertEquals(
            """{"a":1,"z":2}""",
            Json.encode(Json.decode("""{"z":2,"a":1}""")),
        )
    }
}
