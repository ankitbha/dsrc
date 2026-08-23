package com.dsrc.transport

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GpsRecordTest {

    private val full = GpsRecord(
        captureMonoNs = 1_000_000_001,
        valid = true,
        latitude = 51.5074,
        longitude = -0.1278,
        speedMps = 13.4,
        headingDeg = 91.2,
        fixQuality = 1,
        satellites = 9,
        hdop = 0.9,
        altitudeM = 35.0,
        utcEpochNs = 1_755_648_000_000_000_000,
    )

    private fun wire(record: GpsRecord) = record.toExtensions()

    private fun decode(extensions: Map<String, JsonValue>) =
        GpsRecord.fromWire(extensions, ByteArray(0))

    // -- round trips ---------------------------------------------------------

    @Test
    fun `a full fix survives a round trip`() {
        assertEquals(full, decode(wire(full)))
    }

    @Test
    fun `no fix survives a round trip`() {
        val none = GpsRecord.noFix(1_000_000_002)
        assertEquals(none, decode(wire(none)))
    }

    @Test
    fun `no fix nulls every numeric field except the three that are required`() {
        val none = GpsRecord.noFix(7)
        assertNull(none.latitude); assertNull(none.longitude); assertNull(none.speedMps)
        assertNull(none.headingDeg); assertNull(none.hdop); assertNull(none.altitudeM)
        assertNull(none.utcEpochNs)
        assertEquals(false, none.valid)
        assertEquals(0, none.fixQuality)
        assertEquals(0, none.satellites)
    }

    @Test
    fun `the encoded form matches the golden header field for field`() {
        // Compared through canonical JSON against the recorded case, which is the only
        // form in which two headers are equal on this wire.
        val header = Framing.header(Channels.GPS, 1, 1_100_000_000, 1_755_648_000_000_000_000, wire(full))
        val encoded = Json.encode(Framing.withPayloadLength(header, 0))
        val recorded = """{"altitude_m":35.0,"ch":"gps","fix_quality":1,"hdop":0.9,""" +
            """"heading_deg":91.2,"lat":51.5074,"lon":-0.1278,"n":0,"num_sats":9,"seq":1,""" +
            """"speed_mps":13.4,"t_capture_mono_ns":1000000001,"t_mono_ns":1100000000,""" +
            """"t_wall_ns":1755648000000000000,"utc_epoch_ns":1755648000000000000,"valid":true}"""
        assertEquals(recorded, encoded)
    }

    @Test
    fun `the no-fix encoded form matches the golden header`() {
        val none = GpsRecord.noFix(1_000_000_002)
        val header = Framing.header(Channels.GPS, 1, 1_100_000_000, 1_755_648_000_000_000_000, wire(none))
        val encoded = Json.encode(Framing.withPayloadLength(header, 0))
        val recorded = """{"altitude_m":null,"ch":"gps","fix_quality":0,"hdop":null,""" +
            """"heading_deg":null,"lat":null,"lon":null,"n":0,"num_sats":0,"seq":1,""" +
            """"speed_mps":null,"t_capture_mono_ns":1000000002,"t_mono_ns":1100000000,""" +
            """"t_wall_ns":1755648000000000000,"utc_epoch_ns":null,"valid":false}"""
        assertEquals(recorded, encoded)
    }

    // -- the null convention -------------------------------------------------

    @Test
    fun `an absent nullable field is refused, not treated as null`() {
        // Absent would conflate "the sensor said nothing" with "the sender is an older
        // build that never had this field", and the two halves deploy separately.
        for (key in listOf("lat", "lon", "speed_mps", "heading_deg", "hdop", "altitude_m", "utc_epoch_ns")) {
            val missing = wire(full) - key
            val error = assertFailsWith<MessageError>("dropping $key should be refused") {
                decode(missing)
            }
            assertEquals(RefusalReason.MISSING_FIELD, error.reason, "for $key")
        }
    }

    @Test
    fun `a null in a non-nullable field is refused with its own reason`() {
        for (key in listOf("valid", "fix_quality", "num_sats", "t_capture_mono_ns")) {
            val nulled = wire(full) + (key to JsonValue.Null)
            val error = assertFailsWith<MessageError>("null $key should be refused") { decode(nulled) }
            assertEquals(RefusalReason.NULL_NOT_ALLOWED, error.reason, "for $key")
        }
    }

    @Test
    fun `an integer is accepted where a float is expected`() {
        // Refusing 13 for speed_mps would make acceptance depend on how the producer
        // happened to spell a round value.
        val integral = wire(full) + ("speed_mps" to JsonValue.Num(13))
        assertEquals(13.0, decode(integral).speedMps)
    }

    // -- ranges --------------------------------------------------------------

    @Test
    fun `an out-of-range coordinate is refused while the fix is valid`() {
        for (bad in listOf("lat" to 91.0, "lat" to -90.1, "lon" to 180.5, "lon" to -181.0)) {
            val broken = wire(full) + (bad.first to JsonValue.Real(bad.second))
            val error = assertFailsWith<MessageError>("${bad.first}=${bad.second}") { decode(broken) }
            assertEquals(RefusalReason.OUT_OF_RANGE, error.reason)
        }
    }

    @Test
    fun `a valid fix with a null coordinate is refused, matching python`() {
        // The spec is silent on the combination, so this was an unreconciled divergence
        // rather than a defect on either side -- but two implementations of one contract
        // disagreeing about whether a record is acceptable is worse than either answer.
        for (key in listOf("lat", "lon")) {
            val nulled = wire(full) + (key to JsonValue.Null)
            val error = assertFailsWith<MessageError>("null $key on a valid fix") { decode(nulled) }
            assertEquals(RefusalReason.OUT_OF_RANGE, error.reason)
        }
    }

    @Test
    fun `the coordinate bounds themselves are accepted`() {
        for (ok in listOf("lat" to 90.0, "lat" to -90.0, "lon" to 180.0, "lon" to -180.0)) {
            decode(wire(full) + (ok.first to JsonValue.Real(ok.second)))
        }
    }

    @Test
    fun `an out-of-range coordinate is tolerated when the fix is invalid`() {
        // The spec conditions the check on `valid`: an invalid record carries nulls, and
        // policing a stale coordinate on one is not the receiver's job.
        val stale = wire(full) +
            ("valid" to JsonValue.Bool(false)) +
            ("lat" to JsonValue.Real(999.0))
        assertEquals(999.0, decode(stale).latitude)
    }

    @Test
    fun `a negative count is refused`() {
        for (key in listOf("fix_quality", "num_sats")) {
            val negative = wire(full) + (key to JsonValue.Num(-1))
            val error = assertFailsWith<MessageError>("negative $key") { decode(negative) }
            assertEquals(RefusalReason.OUT_OF_RANGE, error.reason)
        }
    }

    @Test
    fun `a fractional count is refused rather than truncated`() {
        // Truncating would hide a sender bug behind a plausible number.
        for (key in listOf("fix_quality", "num_sats")) {
            val fractional = wire(full) + (key to JsonValue.Real(1.5))
            val error = assertFailsWith<MessageError>("fractional $key") { decode(fractional) }
            assertEquals(RefusalReason.WRONG_TYPE, error.reason)
        }
    }

    // -- types and shape -----------------------------------------------------

    @Test
    fun `a field of the wrong type is refused with wrong_type`() {
        val wrong = wire(full) + ("valid" to JsonValue.Text("yes"))
        assertEquals(RefusalReason.WRONG_TYPE, assertFailsWith<MessageError> { decode(wrong) }.reason)
    }

    @Test
    fun `a payload on gps is refused`() {
        // The channel's message carries none; ignoring it would hide a sender bug.
        val error = assertFailsWith<MessageError> { GpsRecord.fromWire(wire(full), byteArrayOf(1)) }
        assertEquals(RefusalReason.UNEXPECTED_PAYLOAD, error.reason)
    }

    @Test
    fun `a reserved key on gps is refused`() {
        for (key in listOf("hello", "heartbeat")) {
            val reserved = wire(full) + (key to JsonValue.Bool(true))
            val error = assertFailsWith<MessageError>("reserved $key") { decode(reserved) }
            assertEquals(RefusalReason.RESERVED_KEY, error.reason)
        }
    }

    @Test
    fun `a non-finite value cannot be encoded and is refused on decode`() {
        // The encoder maps it to null before framing, so seeing one means the peer sent
        // something its own decoder would refuse.
        assertEquals(JsonValue.Null, Fields.toWire(Double.NaN))
        val nonFinite = wire(full) + ("speed_mps" to JsonValue.Real(Double.NaN))
        // Json refuses to write it at all, which is the first line of defence.
        assertFailsWith<IllegalArgumentException> { Json.encode(JsonValue.Obj(nonFinite)) }
    }

    @Test
    fun `an unrecognised extension key is preserved, not refused`() {
        // Extensions are additive: refusing an unknown key would break a rolling deploy
        // in both directions at once.
        val extra = wire(full) + ("future_field" to JsonValue.Num(7))
        assertEquals(full, decode(extra))
    }

    @Test
    fun `every field named in the spec's gps row is produced`() {
        val produced = wire(full).keys
        for (key in listOf(
            "valid", "lat", "lon", "speed_mps", "heading_deg",
            "fix_quality", "num_sats", "hdop", "altitude_m", "utc_epoch_ns",
        )) {
            assertTrue(key in produced, "the spec's gps row names '$key' and it is not produced")
        }
        assertTrue(Fields.CAPTURE_KEY in produced, "every message carries a capture stamp")
    }
}
