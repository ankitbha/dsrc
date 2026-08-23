package com.dsrc.transport

/**
 * The JSON subset a frame header can carry.
 *
 * [JsonLong] and [JsonDouble] are separate types on purpose. The distinction is
 * load-bearing on this wire in both directions: `35.0` must not re-encode as `35`, and
 * `9007199254740993` must not become a double — `specs/transport_golden_frames.json`
 * carries 2^53+1 and `Long.MAX_VALUE`, and a codec that parses every number into a
 * `Double` silently returns 2^53, after which the peer's sequence arithmetic is wrong.
 * That is the reason this module has no JSON dependency: Gson and
 * kotlinx.serialization both collapse the two by default.
 */
sealed interface JsonValue {
    data object Null : JsonValue
    data class Bool(val value: Boolean) : JsonValue
    data class Num(val value: Long) : JsonValue
    data class Real(val value: Double) : JsonValue
    data class Text(val value: String) : JsonValue
    data class Arr(val items: List<JsonValue>) : JsonValue
    data class Obj(val entries: Map<String, JsonValue>) : JsonValue
}

/** A header that does not conform to the canonical form the protocol requires. */
class JsonError(message: String) : IllegalArgumentException(message)

/**
 * Canonical JSON, matching CPython's
 * `json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)`.
 *
 * "Canonical" is not a style choice here: `specs/transport_golden_frames.json` records
 * exact bytes and a SHA-256 per frame, so any disagreement about spacing, key order or
 * escaping makes every frame mismatch.
 */
object Json {

    // ---- encoding ----------------------------------------------------------

    fun encode(value: JsonValue): String = StringBuilder().also { write(value, it) }.toString()

    fun encodeToBytes(value: JsonValue): ByteArray = encode(value).toByteArray(Charsets.UTF_8)

    private fun write(value: JsonValue, out: StringBuilder) {
        when (value) {
            is JsonValue.Null -> out.append("null")
            is JsonValue.Bool -> out.append(if (value.value) "true" else "false")
            is JsonValue.Num -> out.append(value.value.toString())
            is JsonValue.Real -> out.append(Doubles.format(value.value))
            is JsonValue.Text -> writeString(value.value, out)
            is JsonValue.Arr -> {
                out.append('[')
                value.items.forEachIndexed { index, item ->
                    if (index > 0) out.append(',')
                    write(item, out)
                }
                out.append(']')
            }
            is JsonValue.Obj -> {
                out.append('{')
                // Sorted recursively, and by *code point* rather than by Kotlin's
                // natural order. String.compareTo compares UTF-16 units, which puts a
                // surrogate pair (0xD800..0xDFFF) before BMP characters at 0xE000 and
                // above, while Python sorts by code point and puts astral characters
                // after every BMP one. Only a non-ASCII key can tell the two apart, and
                // the golden vectors have none -- so this is another place they agree
                // exactly where it is safe and diverge where it is not.
                value.entries.keys.sortedWith(::compareByCodePoint).forEachIndexed { index, key ->
                    if (index > 0) out.append(',')
                    writeString(key, out)
                    out.append(':')
                    write(value.entries.getValue(key), out)
                }
                out.append('}')
            }
        }
    }

    /** Python's `sort_keys` order: by Unicode code point. */
    internal fun compareByCodePoint(left: String, right: String): Int {
        var i = 0
        var j = 0
        while (i < left.length && j < right.length) {
            val a = left.codePointAt(i)
            val b = right.codePointAt(j)
            if (a != b) return a.compareTo(b)
            i += Character.charCount(a)
            j += Character.charCount(b)
        }
        return (left.length - i).compareTo(right.length - j)
    }

    private fun writeString(value: String, out: StringBuilder) {
        out.append('"')
        for (char in value) {
            when {
                char == '"' -> out.append("\\\"")
                char == '\\' -> out.append("\\\\")
                char == '\b' -> out.append("\\b")
                char == '' -> out.append("\\f")
                char == '\n' -> out.append("\\n")
                char == '\r' -> out.append("\\r")
                char == '\t' -> out.append("\\t")
                // Only C0 controls are escaped. With ensure_ascii=False Python emits
                // everything else raw, including DEL and every non-ASCII character, so a
                // codec that escapes more produces different bytes for the same header.
                char < ' ' -> out.append("\\u%04x".format(char.code))
                else -> out.append(char)
            }
        }
        out.append('"')
    }

    // ---- decoding ----------------------------------------------------------

    fun decode(text: String): JsonValue {
        val parser = Parser(text)
        val value = parser.parseValue()
        parser.skipWhitespace()
        if (!parser.atEnd) throw JsonError("trailing content at offset ${parser.offset}")
        return value
    }

    fun decodeBytes(bytes: ByteArray): JsonValue {
        // Decoding UTF-8 strictly: a malformed sequence must be a framing error, not a
        // replacement character that changes the header's meaning silently.
        val decoder = Charsets.UTF_8.newDecoder()
            .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
            .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
        val text = try {
            decoder.decode(java.nio.ByteBuffer.wrap(bytes)).toString()
        } catch (e: java.nio.charset.CharacterCodingException) {
            throw JsonError("header is not valid UTF-8: ${e.message}")
        }
        return decode(text)
    }

    private class Parser(private val text: String) {
        var offset = 0
            private set

        val atEnd: Boolean get() = offset >= text.length

        fun skipWhitespace() {
            while (offset < text.length && text[offset].isJsonWhitespace()) offset++
        }

        fun parseValue(): JsonValue {
            skipWhitespace()
            if (atEnd) throw JsonError("unexpected end of header")
            return when (val c = text[offset]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> JsonValue.Text(parseString())
                't' -> literal("true", JsonValue.Bool(true))
                'f' -> literal("false", JsonValue.Bool(false))
                'n' -> literal("null", JsonValue.Null)
                else -> if (c == '-' || c in '0'..'9') parseNumber() else {
                    throw JsonError("unexpected '$c' at offset $offset")
                }
            }
        }

        private fun literal(token: String, value: JsonValue): JsonValue {
            if (!text.startsWith(token, offset)) throw JsonError("bad literal at offset $offset")
            offset += token.length
            return value
        }

        private fun parseObject(): JsonValue.Obj {
            offset++ // '{'
            val entries = LinkedHashMap<String, JsonValue>()
            skipWhitespace()
            if (!atEnd && text[offset] == '}') {
                offset++
                return JsonValue.Obj(entries)
            }
            while (true) {
                skipWhitespace()
                if (atEnd || text[offset] != '"') throw JsonError("expected a key at offset $offset")
                val key = parseString()
                if (entries.containsKey(key)) {
                    // Python's parser keeps the last duplicate silently. Refusing is the
                    // safer reading of a header whose meaning would otherwise depend on
                    // which implementation parsed it.
                    throw JsonError("duplicate key '$key' at offset $offset")
                }
                skipWhitespace()
                if (atEnd || text[offset] != ':') throw JsonError("expected ':' at offset $offset")
                offset++
                entries[key] = parseValue()
                skipWhitespace()
                when {
                    atEnd -> throw JsonError("unterminated object")
                    text[offset] == ',' -> offset++
                    text[offset] == '}' -> { offset++; return JsonValue.Obj(entries) }
                    else -> throw JsonError("expected ',' or '}' at offset $offset")
                }
            }
        }

        private fun parseArray(): JsonValue.Arr {
            offset++ // '['
            val items = mutableListOf<JsonValue>()
            skipWhitespace()
            if (!atEnd && text[offset] == ']') {
                offset++
                return JsonValue.Arr(items)
            }
            while (true) {
                items.add(parseValue())
                skipWhitespace()
                when {
                    atEnd -> throw JsonError("unterminated array")
                    text[offset] == ',' -> offset++
                    text[offset] == ']' -> { offset++; return JsonValue.Arr(items) }
                    else -> throw JsonError("expected ',' or ']' at offset $offset")
                }
            }
        }

        private fun parseString(): String {
            offset++ // opening quote
            val out = StringBuilder()
            while (true) {
                if (atEnd) throw JsonError("unterminated string")
                when (val c = text[offset]) {
                    '"' -> { offset++; return out.toString() }
                    '\\' -> {
                        offset++
                        if (atEnd) throw JsonError("unterminated escape")
                        when (val esc = text[offset]) {
                            '"' -> out.append('"')
                            '\\' -> out.append('\\')
                            '/' -> out.append('/')
                            'b' -> out.append('\b')
                            'f' -> out.append('')
                            'n' -> out.append('\n')
                            'r' -> out.append('\r')
                            't' -> out.append('\t')
                            'u' -> {
                                if (offset + 4 >= text.length) throw JsonError("truncated \\u escape")
                                val hex = text.substring(offset + 1, offset + 5)
                                out.append(
                                    hex.toIntOrNull(16)?.toChar()
                                        ?: throw JsonError("bad \\u escape '$hex'")
                                )
                                offset += 4
                            }
                            else -> throw JsonError("unknown escape '\\$esc'")
                        }
                        offset++
                    }
                    else -> { out.append(c); offset++ }
                }
            }
        }

        private fun parseNumber(): JsonValue {
            val start = offset
            if (!atEnd && text[offset] == '-') offset++
            while (!atEnd && text[offset] in '0'..'9') offset++
            var isReal = false
            if (!atEnd && text[offset] == '.') {
                isReal = true
                offset++
                while (!atEnd && text[offset] in '0'..'9') offset++
            }
            if (!atEnd && (text[offset] == 'e' || text[offset] == 'E')) {
                isReal = true
                offset++
                if (!atEnd && (text[offset] == '+' || text[offset] == '-')) offset++
                while (!atEnd && text[offset] in '0'..'9') offset++
            }
            val token = text.substring(start, offset)
            if (token.isEmpty() || token == "-") throw JsonError("bad number at offset $start")

            // The integer/float split is decided by the token's *shape*, exactly as
            // Python decides it, not by whether the value happens to be integral. So
            // `35.0` stays a Real and re-encodes as `35.0`.
            if (isReal) {
                val parsed = token.toDoubleOrNull() ?: throw JsonError("bad number '$token'")
                if (parsed.isNaN() || parsed.isInfinite()) {
                    throw JsonError("non-finite number '$token' on the wire")
                }
                return JsonValue.Real(parsed)
            }
            val parsed = token.toLongOrNull()
                // A value beyond Long is refused rather than widened to a double, which
                // would lose exactly the precision this type exists to keep.
                ?: throw JsonError("integer '$token' does not fit in a 64-bit signed value")
            return JsonValue.Num(parsed)
        }

        private fun Char.isJsonWhitespace() = this == ' ' || this == '\t' || this == '\n' || this == '\r'
    }
}
