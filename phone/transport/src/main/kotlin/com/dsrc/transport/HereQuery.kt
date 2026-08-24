package com.dsrc.transport

/**
 * The shape of the HERE traffic query, chosen by the Jetson.
 *
 * The phone composes a URL from this and reads nothing else in it. The spec is explicit
 * that "the HERE query shape and location-referencing mode" is set upstream, because the
 * state that would justify choosing is all on the Jetson — a phone picking its own corridor
 * is a phone originating a sensing decision.
 *
 * [locationRef] is HERE's `locationReferencing` parameter and [in] is its `in` parameter,
 * both passed through verbatim. Neither is parsed here: validating HERE's own query grammar
 * on the phone would mean tracking their API, and getting it wrong would refuse a query the
 * Jetson meant.
 *
 * [lat], [lon] and [radiusM] are not used to build the request. They are what the `here`
 * frame records as the query's location, and the frame's schema requires them. The phone
 * cannot derive them: `in` is an opaque HERE expression that may be a corridor of a hundred
 * points, and picking one would be interpretation. So the Jetson sends the position it
 * wants the record to carry, and the phone echoes it. Writing zeros instead would put three
 * fields on the wire that claim to be a query location and are not one.
 */
data class HereQuery(
    /** HERE's `in` parameter — a corridor, a circle, or whatever else it accepts. */
    val `in`: String,
    /** HERE's `locationReferencing` parameter. */
    val locationRef: String,
    /** What the frame should record as the query's position. Not used to build the request. */
    val lat: Double,
    val lon: Double,
    val radiusM: Double,
) {
    fun toJson(): JsonValue = JsonValue.Obj(
        mapOf(
            "in" to JsonValue.Text(`in`),
            "location_ref" to JsonValue.Text(locationRef),
            "lat" to JsonValue.Real(lat),
            "lon" to JsonValue.Real(lon),
            "radius_m" to JsonValue.Real(radiusM),
        )
    )

    companion object {
        /**
         * Decode the optional `here` object.
         *
         * Absent is null — "this command does not change the query" — which is why the
         * field can be added without a flag day. Present but malformed is a refusal: a
         * command that names a query and gets it wrong is not the same as one that says
         * nothing, and silently ignoring it would leave the phone querying yesterday's
         * corridor while the Jetson believed it had moved.
         */
        fun fromWire(value: JsonValue?): HereQuery? {
            if (value == null || value is JsonValue.Null) return null
            if (value !is JsonValue.Obj) {
                throw MessageError(RefusalReason.WRONG_TYPE, "'here' is not an object")
            }
            val lat = requireNumber(value.entries, "lat")
            val lon = requireNumber(value.entries, "lon")
            if (lat !in -90.0..90.0 || lon !in -180.0..180.0) {
                throw MessageError(RefusalReason.OUT_OF_RANGE, "'here' position is off the globe")
            }
            return HereQuery(
                `in` = requireText(value.entries, "in"),
                locationRef = requireText(value.entries, "location_ref"),
                lat = lat,
                lon = lon,
                radiusM = requireNumber(value.entries, "radius_m"),
            )
        }

        private fun requireNumber(entries: Map<String, JsonValue>, key: String): Double {
            val value = entries[key]
                ?: throw MessageError(RefusalReason.MISSING_FIELD, "'here.$key' is missing")
            if (value is JsonValue.Null) {
                throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'here.$key' is null")
            }
            val number = when (value) {
                is JsonValue.Real -> value.value
                is JsonValue.Num -> value.value.toDouble()
                else -> throw MessageError(RefusalReason.WRONG_TYPE, "'here.$key' is not a number")
            }
            if (!number.isFinite()) {
                throw MessageError(RefusalReason.NON_FINITE, "'here.$key' is not finite")
            }
            return number
        }

        private fun requireText(entries: Map<String, JsonValue>, key: String): String {
            val value = entries[key]
                ?: throw MessageError(RefusalReason.MISSING_FIELD, "'here.$key' is missing")
            if (value is JsonValue.Null) {
                throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'here.$key' is null")
            }
            if (value !is JsonValue.Text) {
                throw MessageError(RefusalReason.WRONG_TYPE, "'here.$key' is not a string")
            }
            if (value.value.isBlank()) {
                throw MessageError(RefusalReason.OUT_OF_RANGE, "'here.$key' is blank")
            }
            return value.value
        }
    }
}
