package com.dsrc.transport

/**
 * One GPS fix, or the absence of one.
 *
 * Every numeric field is nullable except [valid], [fixQuality] and [satellites], per the
 * message table in `specs/transport_protocol.md`. "Unavailable" is present-and-null:
 * never absent, and never a sentinel. Absent would conflate "the sensor said nothing"
 * with "the sender is an older build that never had this field", and the phone app and
 * the Jetson runtime are deployed separately. A sentinel that is a legitimate value in
 * some other unit is a silent-corruption source.
 *
 * [captureMonoNs] and [utcEpochNs] are different clocks by design and must never be
 * differenced: the first is monotonic and meaningful only on the device that produced
 * it, the second is GPS wall time and can step.
 */
data class GpsRecord(
    val captureMonoNs: Long,
    val valid: Boolean,
    val latitude: Double?,
    val longitude: Double?,
    val speedMps: Double?,
    val headingDeg: Double?,
    val fixQuality: Long,
    val satellites: Long,
    val hdop: Double?,
    val altitudeM: Double?,
    val utcEpochNs: Long?,
) {
    fun toExtensions(): Map<String, JsonValue> = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
        KEY_VALID to JsonValue.Bool(valid),
        KEY_LAT to Fields.toWire(latitude),
        KEY_LON to Fields.toWire(longitude),
        KEY_SPEED to Fields.toWire(speedMps),
        KEY_HEADING to Fields.toWire(headingDeg),
        KEY_FIX_QUALITY to JsonValue.Num(fixQuality),
        KEY_SATELLITES to JsonValue.Num(satellites),
        KEY_HDOP to Fields.toWire(hdop),
        KEY_ALTITUDE to Fields.toWire(altitudeM),
        KEY_UTC to Fields.toWire(utcEpochNs),
    )

    companion object {
        const val KEY_VALID = "valid"
        const val KEY_LAT = "lat"
        const val KEY_LON = "lon"
        const val KEY_SPEED = "speed_mps"
        const val KEY_HEADING = "heading_deg"
        const val KEY_FIX_QUALITY = "fix_quality"
        const val KEY_SATELLITES = "num_sats"
        const val KEY_HDOP = "hdop"
        const val KEY_ALTITUDE = "altitude_m"
        const val KEY_UTC = "utc_epoch_ns"

        /** No fix: every numeric field null, and the flag says so. */
        fun noFix(captureMonoNs: Long) = GpsRecord(
            captureMonoNs = captureMonoNs,
            valid = false,
            latitude = null,
            longitude = null,
            speedMps = null,
            headingDeg = null,
            fixQuality = 0,
            satellites = 0,
            hdop = null,
            altitudeM = null,
            utcEpochNs = null,
        )

        fun fromWire(extensions: Map<String, JsonValue>, payload: ByteArray): GpsRecord {
            Fields.checkNoPayload(payload, Channels.GPS)
            Fields.checkReserved(extensions)

            val valid = Fields.requireBool(extensions, KEY_VALID)
            val latitude = Fields.checkFinite(KEY_LAT, Fields.optionalNumber(extensions, KEY_LAT))
            val longitude = Fields.checkFinite(KEY_LON, Fields.optionalNumber(extensions, KEY_LON))

            // Range-checked only while the fix is valid, which is what the spec's table
            // says: an invalid record carries nulls, and a stale coordinate on an
            // invalid record is not the receiver's problem to police.
            if (valid) {
                if (latitude != null && (latitude < -90.0 || latitude > 90.0)) {
                    throw MessageError(RefusalReason.OUT_OF_RANGE, "lat $latitude outside [-90, 90]")
                }
                if (longitude != null && (longitude < -180.0 || longitude > 180.0)) {
                    throw MessageError(RefusalReason.OUT_OF_RANGE, "lon $longitude outside [-180, 180]")
                }
            }

            return GpsRecord(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                valid = valid,
                latitude = latitude,
                longitude = longitude,
                speedMps = Fields.checkFinite(KEY_SPEED, Fields.optionalNumber(extensions, KEY_SPEED)),
                headingDeg = Fields.checkFinite(KEY_HEADING, Fields.optionalNumber(extensions, KEY_HEADING)),
                fixQuality = Fields.requireCount(extensions, KEY_FIX_QUALITY),
                satellites = Fields.requireCount(extensions, KEY_SATELLITES),
                hdop = Fields.checkFinite(KEY_HDOP, Fields.optionalNumber(extensions, KEY_HDOP)),
                altitudeM = Fields.checkFinite(KEY_ALTITUDE, Fields.optionalNumber(extensions, KEY_ALTITUDE)),
                utcEpochNs = Fields.optionalInt(extensions, KEY_UTC),
            )
        }
    }
}
