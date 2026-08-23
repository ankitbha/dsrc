package com.dsrc.transport

/**
 * Why a message was refused.
 *
 * A closed vocabulary, exactly the second column of the refusal table in
 * `specs/transport_protocol.md`, so an implementation reads off which reason to emit
 * rather than inventing one. One number cannot answer whether four thousand drops were
 * one bad field or four, which is the point of counting them by reason at all.
 */
enum class RefusalReason(val wire: String) {
    MISSING_FIELD("missing_field"),
    WRONG_TYPE("wrong_type"),
    NULL_NOT_ALLOWED("null_not_allowed"),
    NON_FINITE("non_finite"),
    OUT_OF_RANGE("out_of_range"),
    UNKNOWN_VALUE("unknown_value"),
    UNEXPECTED_PAYLOAD("unexpected_payload"),
    RESERVED_KEY("reserved_key"),
    NO_TYPED_MESSAGE("no_typed_message"),
}

/**
 * A message that framed correctly but does not conform.
 *
 * Deliberately not a [FramingError]. The difference is recoverability: a framing error
 * means the byte stream has desynchronised and there is no delimiter to hunt for, so the
 * reader can never find its place again. A message that framed correctly proves the
 * stream is fine — one bad record costs one record, and reconnecting the phone over a
 * single malformed IMU sample at 50 Hz would be far worse than dropping it.
 */
class MessageError(val reason: RefusalReason, message: String) : Exception(message)

/** Field readers that refuse with the right reason rather than the nearest exception. */
object Fields {

    const val CAPTURE_KEY = "t_capture_mono_ns"

    fun require(extensions: Map<String, JsonValue>, key: String): JsonValue =
        extensions[key] ?: throw MessageError(RefusalReason.MISSING_FIELD, "'$key' is absent")

    fun requireInt(extensions: Map<String, JsonValue>, key: String): Long {
        val value = require(extensions, key)
        if (value is JsonValue.Null) {
            throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'$key' is null")
        }
        return (value as? JsonValue.Num)?.value
            ?: throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not an integer")
    }

    fun requireBool(extensions: Map<String, JsonValue>, key: String): Boolean {
        val value = require(extensions, key)
        if (value is JsonValue.Null) {
            throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'$key' is null")
        }
        return (value as? JsonValue.Bool)?.value
            ?: throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not a boolean")
    }

    fun requireString(extensions: Map<String, JsonValue>, key: String): String {
        val value = require(extensions, key)
        if (value is JsonValue.Null) {
            throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'$key' is null")
        }
        return (value as? JsonValue.Text)?.value
            ?: throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not a string")
    }

    /**
     * A required number, accepting an integer where a float is expected.
     *
     * The wire distinguishes the two and the encoder preserves the distinction, but a
     * *reader* of `speed_mps` should not refuse `13` because it arrived without a
     * decimal point — the value is the same real number, and refusing would make the
     * message's acceptance depend on how its producer happened to spell a round value.
     */
    fun requireNumber(extensions: Map<String, JsonValue>, key: String): Double {
        val value = require(extensions, key)
        return when (value) {
            is JsonValue.Real -> value.value
            is JsonValue.Num -> value.value.toDouble()
            is JsonValue.Null -> throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'$key' is null")
            else -> throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not a number")
        }
    }

    /** A nullable number: present-and-null means unavailable, which is the convention. */
    fun optionalNumber(extensions: Map<String, JsonValue>, key: String): Double? {
        val value = require(extensions, key)
        return when (value) {
            is JsonValue.Null -> null
            is JsonValue.Real -> value.value
            is JsonValue.Num -> value.value.toDouble()
            else -> throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not a number or null")
        }
    }

    fun optionalInt(extensions: Map<String, JsonValue>, key: String): Long? {
        val value = require(extensions, key)
        return when (value) {
            is JsonValue.Null -> null
            is JsonValue.Num -> value.value
            else -> throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not an integer or null")
        }
    }

    fun optionalString(extensions: Map<String, JsonValue>, key: String): String? {
        val value = require(extensions, key)
        return when (value) {
            is JsonValue.Null -> null
            is JsonValue.Text -> value.value
            else -> throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not a string or null")
        }
    }

    /**
     * A count: an integer in `[0, 2^63-1]`.
     *
     * Fractions are refused rather than truncated. A fractional count is a bug in the
     * sender, and truncating it would hide that bug behind a plausible number.
     */
    fun requireCount(extensions: Map<String, JsonValue>, key: String): Long {
        val value = require(extensions, key)
        if (value is JsonValue.Real) {
            throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is a count and must not be fractional")
        }
        val count = requireInt(extensions, key)
        if (count < 0) throw MessageError(RefusalReason.OUT_OF_RANGE, "'$key' is $count, below zero")
        return count
    }

    /** A value that must be finite once it is on the wire. */
    fun checkFinite(key: String, value: Double?): Double? {
        if (value != null && (value.isNaN() || value.isInfinite())) {
            // An encoder maps a non-finite value to null before framing, so seeing one
            // here means the peer framed something its own decoder would refuse.
            throw MessageError(RefusalReason.NON_FINITE, "'$key' is $value")
        }
        return value
    }

    /** Reserved transport keys must not appear on a caller's message. */
    fun checkReserved(extensions: Map<String, JsonValue>, allow: Set<String> = emptySet()) {
        val clash = Framing.RESERVED_KEYS.filter { it in extensions && it !in allow }
        if (clash.isNotEmpty()) {
            // A message carrying one would be read as transport traffic by the peer and
            // consumed instead of delivered -- lost with no drop counted and no sequence
            // gap to show it, so invisible in the session summary too.
            throw MessageError(
                RefusalReason.RESERVED_KEY,
                "${clash.sorted().joinToString(", ")} are reserved for the transport",
            )
        }
    }

    /** Channels whose message carries no payload refuse one rather than ignoring it. */
    fun checkNoPayload(payload: ByteArray, channel: String) {
        if (payload.isNotEmpty()) {
            throw MessageError(
                RefusalReason.UNEXPECTED_PAYLOAD,
                "$channel carries no payload but ${payload.size} bytes arrived",
            )
        }
    }

    /** Convert a value that may be NaN into the wire's `null`. */
    fun toWire(value: Double?): JsonValue =
        if (value == null || value.isNaN() || value.isInfinite()) JsonValue.Null else JsonValue.Real(value)

    fun toWire(value: Long?): JsonValue =
        if (value == null) JsonValue.Null else JsonValue.Num(value)

    fun toWire(value: String?): JsonValue =
        if (value == null) JsonValue.Null else JsonValue.Text(value)
}

/**
 * Applies a receiver's rules to an outbound message.
 *
 * The spec makes every refusal a *sender* rule as well: "before a message goes out, it
 * must satisfy the same table." A receiver rule alone leaves the sender free to emit
 * garbage and learn about it as someone else's drop counter, and the Python side runs its
 * typed decoder on every outbound message for exactly this reason.
 */
object OutboundValidation {

    /**
     * Channels whose typed message has no producer yet, so no decoder exists to run.
     *
     * Named explicitly rather than left as a silent gap: `imu`, `here` and `telemetry`
     * land with tasks 20, 21 and 24, and `advisory` and `rate_cmd` are inbound on the
     * phone. The generic checks below still apply to all of them.
     */
    val WITHOUT_A_TYPED_DECODER = setOf(
        Channels.IMU,
        Channels.HERE,
        Channels.TELEMETRY,
        Channels.ADVISORY,
        Channels.RATE_CMD,
        Channels.CAMERA,
    )

    /** Channels whose message carries no payload, from the spec's message table. */
    private val PAYLOAD_BEARING = setOf(Channels.CAMERA, Channels.HERE)

    /**
     * Throws [MessageError] with the reason a receiver would give.
     *
     * @param allowReserved reserved keys this message legitimately carries — the hello and
     *   heartbeat the transport sends itself, and the wire stamp on a timebase message.
     */
    fun check(
        channel: String,
        extensions: Map<String, JsonValue>,
        payload: ByteArray,
        allowReserved: Set<String> = emptySet(),
    ) {
        if (!Channels.isKnown(channel)) {
            throw MessageError(RefusalReason.NO_TYPED_MESSAGE, "unknown channel '$channel'")
        }
        Fields.checkReserved(extensions, allowReserved)

        if (channel !in PAYLOAD_BEARING) Fields.checkNoPayload(payload, channel)

        // Non-finite values are caught here rather than surfacing from the JSON encoder.
        // Doubles.format raises IllegalArgumentException, which is neither MessageError nor
        // FramingError, so it propagated out of send() into the caller -- on the phone,
        // into a sensor callback.
        checkAllFinite(extensions)

        // Then the typed decoder, where one exists: it is the only thing that applies the
        // per-field rules -- ranges, counts, null handling -- and those are most of the
        // refusal table.
        when (channel) {
            Channels.GPS -> GpsRecord.fromWire(extensions, payload)
            Channels.CONTROL ->
                // The hello and heartbeat are transport traffic, not timebase messages.
                if (Session.HELLO !in extensions && Session.HEARTBEAT !in extensions) {
                    TimeSyncMessage.fromWire(extensions, payload)
                }
            else -> Unit
        }
    }

    private fun checkAllFinite(value: Any?) {
        when (value) {
            is Map<*, *> -> value.forEach { (key, entry) ->
                if (entry is JsonValue.Real && (entry.value.isNaN() || entry.value.isInfinite())) {
                    throw MessageError(RefusalReason.NON_FINITE, "'$key' is ${entry.value}")
                }
                checkAllFinite(entry)
            }
            is JsonValue.Obj -> checkAllFinite(value.entries)
            is JsonValue.Arr -> value.items.forEach { checkAllFinite(it) }
            is JsonValue.Real ->
                if (value.value.isNaN() || value.value.isInfinite()) {
                    throw MessageError(RefusalReason.NON_FINITE, "${value.value} on the wire")
                }
            else -> Unit
        }
    }
}
