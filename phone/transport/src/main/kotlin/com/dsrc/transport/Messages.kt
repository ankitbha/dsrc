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

    /**
     * A field that may be absent entirely, not merely null.
     *
     * [optionalNumber] and [optionalString] require the key to be present, which is right
     * for a field that has always existed and is only sometimes unavailable. A field *added*
     * to a shipped protocol is a different case: an older sender does not write it at all,
     * and requiring it would make every one of that sender's messages a `missing_field`
     * refusal. `here` on a rate command is the same shape and the reason the spec calls
     * extensions additive.
     */
    fun absentableNumber(extensions: Map<String, JsonValue>, key: String): Double? {
        val value = extensions[key] ?: return null
        return when (value) {
            is JsonValue.Null -> null
            is JsonValue.Real -> value.value
            is JsonValue.Num -> value.value.toDouble()
            else -> throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not a number or null")
        }
    }

    /** As [absentableNumber], for a string. */
    fun absentableString(extensions: Map<String, JsonValue>, key: String): String? {
        val value = extensions[key] ?: return null
        return when (value) {
            is JsonValue.Null -> null
            is JsonValue.Text -> value.value
            else -> throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not a string or null")
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

    /**
     * A nested object with exactly the named keys present.
     *
     * The spec calls `rates`, `achieved` and `dropped` additive: every listed key must be
     * there, and an unlisted one is carried rather than refused. So a missing key is
     * `missing_field` and an extra key is not an error -- unlike `action`, whose heads are
     * a closed set because an unknown head is a policy the receiver cannot honour.
     */
    fun requireObject(extensions: Map<String, JsonValue>, key: String): Map<String, JsonValue> {
        val value = require(extensions, key)
        if (value is JsonValue.Null) {
            throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'$key' is null")
        }
        val obj = (value as? JsonValue.Obj)
            ?: throw MessageError(RefusalReason.WRONG_TYPE, "'$key' is not an object")
        return obj.entries
    }

    /** A nested object of numbers, keyed exactly as [keys] requires. */
    fun requireNestedNumbers(
        extensions: Map<String, JsonValue>,
        key: String,
        keys: List<String>,
    ): Map<String, Double> {
        val nested = requireObject(extensions, key)
        return keys.associateWith { inner ->
            val value = nested[inner]
                ?: throw MessageError(RefusalReason.MISSING_FIELD, "'$key.$inner' is absent")
            when (value) {
                is JsonValue.Real -> checkFinite("$key.$inner", value.value)!!
                is JsonValue.Num -> value.value.toDouble()
                is JsonValue.Null ->
                    throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'$key.$inner' is null")
                else -> throw MessageError(RefusalReason.WRONG_TYPE, "'$key.$inner' is not a number")
            }
        }
    }

    /** A nested object of counts: integral, non-negative, never fractional. */
    fun requireNestedCounts(
        extensions: Map<String, JsonValue>,
        key: String,
        keys: List<String>,
    ): Map<String, Long> {
        val nested = requireObject(extensions, key)
        return keys.associateWith { inner ->
            val value = nested[inner]
                ?: throw MessageError(RefusalReason.MISSING_FIELD, "'$key.$inner' is absent")
            when (value) {
                // Refused rather than truncated: a fractional count is a bug in the
                // sender, and truncating hides it behind a plausible number.
                is JsonValue.Real ->
                    throw MessageError(
                        RefusalReason.WRONG_TYPE,
                        "'$key.$inner' is a count and must not be fractional",
                    )
                is JsonValue.Num -> {
                    if (value.value < 0) {
                        throw MessageError(
                            RefusalReason.OUT_OF_RANGE,
                            "'$key.$inner' is ${value.value}, below zero",
                        )
                    }
                    value.value
                }
                is JsonValue.Null ->
                    throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'$key.$inner' is null")
                else -> throw MessageError(RefusalReason.WRONG_TYPE, "'$key.$inner' is not an integer")
            }
        }
    }

    /** Wrap a map of numbers for the wire. */
    fun objectOf(values: Map<String, Double>): JsonValue =
        JsonValue.Obj(values.mapValues { (_, v) -> toWire(v) })

    /** Wrap a map of counts for the wire. */
    fun countsOf(values: Map<String, Long>): JsonValue =
        JsonValue.Obj(values.mapValues { (_, v) -> JsonValue.Num(v) as JsonValue })

    /** Wrap a map of strings for the wire. */
    fun stringsOf(values: Map<String, String>): JsonValue =
        JsonValue.Obj(values.mapValues { (_, v) -> JsonValue.Text(v) as JsonValue })

    /** Convert a value that may be NaN into the wire's `null`. */
    fun toWire(value: Double?): JsonValue =
        if (value == null || value.isNaN() || value.isInfinite()) JsonValue.Null else JsonValue.Real(value)

    fun toWire(value: Long?): JsonValue =
        if (value == null) JsonValue.Null else JsonValue.Num(value)

    fun toWire(value: String?): JsonValue =
        if (value == null) JsonValue.Null else JsonValue.Text(value)
}

/**
 * The refusal table, applied in both directions.
 *
 * The spec states every condition as a *receiver* rule -- "message dropped, counted" --
 * and then makes it a sender rule too: "before a message goes out, it must satisfy the
 * same table." Both halves are needed and for different reasons. A receiver rule alone
 * leaves the sender free to emit garbage and learn about it as someone else's drop
 * counter. A sender rule alone leaves the receiver trusting whatever arrives, which is
 * worse: the peer may be an older build, or a different language, or wrong.
 *
 * The receiving half was missing here. `send` ran the decoder and the reader did not, so a
 * malformed frame from the peer was handed to the application unchecked and
 * `inboundRefusals` only ever moved when an application handler happened to throw. A
 * deliberately malformed record from the live Python peer crossed and was counted nowhere.
 */
object MessageValidation {

    /**
     * The typed decoder for every channel, by channel id.
     *
     * A map rather than a `when` with an `else`, and the difference is not style. The
     * `when` had `else -> Unit`, so a channel with no decoder was silently exempt from
     * every per-field rule -- which is most of the refusal table. Alongside it sat a set
     * named `WITHOUT_A_TYPED_DECODER` that documented the exemption and was read by
     * nothing, so it could not go stale in a way anything would notice: documentation
     * posing as logic. With a map, [ALL_CHANNELS_HAVE_A_DECODER] can be asserted, and
     * adding a channel without a decoder fails a test instead of quietly widening the hole.
     */
    private val DECODERS: Map<String, (Map<String, JsonValue>, ByteArray) -> Unit> = mapOf(
        Channels.GPS to { extensions, payload -> GpsRecord.fromWire(extensions, payload) },
        Channels.CAMERA to { extensions, payload -> CameraFrameMessage.fromWire(extensions, payload) },
        Channels.IMU to { extensions, payload -> ImuSample.fromWire(extensions, payload) },
        Channels.HERE to { extensions, payload -> HereResponse.fromWire(extensions, payload) },
        Channels.ADVISORY to { extensions, payload -> AdvisoryMessage.fromWire(extensions, payload) },
        Channels.RATE_CMD to { extensions, payload -> RateCommand.fromWire(extensions, payload) },
        Channels.TELEMETRY to { extensions, payload -> PhoneTelemetry.fromWire(extensions, payload) },
        // No hello/heartbeat exemption here, and there was one. It changed the *location*
        // of a refusal and never its outcome, which is why both halves of it could be
        // deleted with all 250 tests passing. The transport's own traffic does not arrive
        // here: the handshake hello is read by `readPeerHello` before the read loop starts,
        // and a heartbeat is absorbed in `readLoop` and never reaches `deliver`. The one
        // frame the exemption did affect -- a peer re-sending its hello mid-session -- is
        // refused either way, by the same decoder, with the same reason. Outbound it is
        // moot twice over: heartbeats are written straight through `Framing.write`, and a
        // caller that put a reserved key on `control` is refused by the reserved-key rule
        // before any decoder runs.
        Channels.CONTROL to { extensions, payload -> TimeSyncMessage.fromWire(extensions, payload) },
    )

    /** Whether every channel in the table has a decoder. Asserted by a test. */
    val ALL_CHANNELS_HAVE_A_DECODER: Boolean
        get() = Channels.ALL.map { it.id }.toSet() == DECODERS.keys

    /** Channels in the table with no decoder, for a test that names them. */
    val CHANNELS_WITHOUT_A_DECODER: Set<String>
        get() = Channels.ALL.map { it.id }.toSet() - DECODERS.keys

    /** Channels whose message carries no payload, from the spec's message table. */
    private val PAYLOAD_BEARING = setOf(Channels.CAMERA, Channels.HERE)

    /**
     * Throws [MessageError] with the reason a receiver would give.
     *
     * @param allowReserved reserved keys this message legitimately carries -- the hello and
     *   heartbeat the transport sends itself, and the wire stamp on a timebase message.
     */
    /**
     * Apply the table to a frame that arrived.
     *
     * The reserved-key rule is deliberately **not** applied here, and the earlier attempt
     * to allow only the wire stamp was worse than either extreme. It passed
     * `allowReserved = setOf(WIRE_STAMP)` while seven of the eight decoders called
     * `checkReserved` themselves with no allow set, so the allowance was overridden on
     * every channel but `control` -- and the damage ran outbound as well as inbound:
     * `send(gps, ..., wantsWireStamp = true)` passed validation on the caller's map, the
     * writer added the stamp, and the receiving session refused the result as
     * `reserved_key`. A sender emitting what its own decoder refuses is the exact rule
     * the sender check exists to enforce.
     *
     * The rule belongs on the send path. Python agrees structurally: `check_reserved`
     * appears in `MessageRouter.send` and in no `from_wire`. The spec's own keepalive
     * paragraph agrees too -- a reserved key on a data channel "is a caller's message and
     * MUST be delivered" -- though its refusal table lists `reserved_key` as a receiver
     * condition, so the two paragraphs contradict each other. Matching Python is what
     * keeps the link working; the contradiction is recorded in the plan.
     */
    fun checkInbound(frame: Frame) =
        check(frame.channel, frame.header.entries, frame.payload, checkReservedKeys = false)

    fun check(
        channel: String,
        extensions: Map<String, JsonValue>,
        payload: ByteArray,
        allowReserved: Set<String> = emptySet(),
        /** False on the receive path, where the rule does not apply. */
        checkReservedKeys: Boolean = true,
    ) {
        // A FramingError, not a refusal. The spec's framing table is explicit -- "`ch` not
        // in the channel table -> framing error, session ends" -- and the read path already
        // treats it that way. Calling it `no_typed_message` put a framing condition into
        // the refusal vocabulary, where that reason means something narrower.
        if (!Channels.isKnown(channel)) {
            throw FramingError("unknown channel '$channel'")
        }
        if (checkReservedKeys) Fields.checkReserved(extensions, allowReserved)

        if (channel !in PAYLOAD_BEARING) Fields.checkNoPayload(payload, channel)

        // Non-finite values are caught here rather than surfacing from the JSON encoder.
        // Doubles.format raises IllegalArgumentException, which is neither MessageError nor
        // FramingError, so it propagated out of send() into the caller -- on the phone,
        // into a sensor callback.
        checkAllFinite(extensions)

        // Then the typed decoder, which is the only thing that applies the per-field rules
        // -- ranges, counts, null handling -- and those are most of the refusal table.
        val decoder = DECODERS[channel]
            ?: throw MessageError(RefusalReason.NO_TYPED_MESSAGE, "no typed message for '$channel'")
        decoder(extensions, payload)
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
