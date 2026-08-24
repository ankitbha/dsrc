package com.dsrc.transport

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The five messages that did not exist, and the refusal rows that had no implementation.
 *
 * Five of the spec's eight typed messages were missing, and the effect was not a gap in
 * coverage but a gap in *enforcement*: the validator's exemption was `else -> Unit`, so
 * `advisory`, `rate_cmd`, `imu`, `here` and `telemetry` were exempt from every per-field
 * rule, which is most of the refusal table. A stub advisory of `{"k": 1}` travelled.
 */
class TypedMessagesTest {

    private fun imu() = ImuSample(1_000, 0.1, 0.2, 9.8, 0.01, -0.02, 0.003, accuracy = 3)

    private fun here() = HereResponse(
        captureMonoNs = 1_000,
        requestUrl = "https://data.traffic.hereapi.com/v7/flow?in=corridor:...",
        status = 200,
        contentType = "application/json",
        queryLat = 40.7128,
        queryLon = -74.0060,
        queryRadiusM = 9_000.0,
        requestMonoNs = 500,
        responseMonoNs = 900,
    )

    private fun telemetry() = PhoneTelemetry(
        captureMonoNs = 1_000,
        thermalStatus = "nominal",
        thermalHeadroom = 0.42,
        achieved = mapOf("camera_hz" to 4.9, "gps_hz" to 1.0, "imu_hz" to 49.8, "here_hz" to 0.2),
        dropped = mapOf("camera" to 12L, "gps" to 0L, "imu" to 3L, "here" to 0L),
        hereCalls = 41,
        hereErrors = 2,
    )

    private fun advisory() = AdvisoryMessage(
        captureMonoNs = 1_000,
        recSpeedMps = 13.4,
        recSpeedDisplay = 30.0,
        currentSpeedDisplay = 28.0,
        units = "mph",
        headwayTargetS = 2.0,
        laneText = "keep",
        mergeText = "normal",
        trafficText = "clear",
        confidence = 0.87,
        confidenceLabel = "high",
        action = mapOf(
            "desired_speed_bin" to "nominal",
            "desired_headway_bin" to "normal",
            "lane_preference" to "keep",
            "merge_mode" to "normal",
        ),
    )

    private fun rateCmd() = RateCommand(
        captureMonoNs = 1_000,
        rates = mapOf("camera_hz" to 5.0, "gps_hz" to 1.0, "imu_hz" to 50.0, "here_hz" to 0.2),
        trigger = "thermal",
        shadow = false,
    )

    private fun refusalFor(block: () -> Unit): RefusalReason =
        assertFailsWith<MessageError> { block() }.reason

    // -- round trips ---------------------------------------------------------

    @Test
    fun `every message round trips through its own decoder`() {
        assertEquals(imu(), ImuSample.fromWire(imu().toExtensions(), ByteArray(0)))
        assertEquals(telemetry(), PhoneTelemetry.fromWire(telemetry().toExtensions(), ByteArray(0)))
        assertEquals(advisory(), AdvisoryMessage.fromWire(advisory().toExtensions(), ByteArray(0)))
        assertEquals(rateCmd(), RateCommand.fromWire(rateCmd().toExtensions(), ByteArray(0)))
        // `here` carries its body in the payload, so it round trips against one.
        val body = """{"results":[]}""".toByteArray()
        assertEquals(here(), HereResponse.fromWire(here().toExtensions(), body))
    }

    @Test
    fun `the key names match the spec's message table exactly`() {
        // A cross-language contract with a renamed key fails as a missing field on the far
        // side, at run time.
        assertEquals(
            setOf("t_capture_mono_ns", "ax", "ay", "az", "gx", "gy", "gz", "accuracy"),
            imu().toExtensions().keys,
        )
        assertEquals(
            setOf("t_capture_mono_ns", "request_url", "status", "content_type", "query_lat",
                "query_lon", "query_radius_m", "t_request_mono_ns", "t_response_mono_ns"),
            here().toExtensions().keys,
        )
        assertEquals(
            setOf("t_capture_mono_ns", "thermal_status", "thermal_headroom", "achieved",
                "dropped", "here_calls", "here_errors"),
            telemetry().toExtensions().keys,
        )
        assertEquals(
            setOf("t_capture_mono_ns", "rec_speed_mps", "rec_speed_display",
                "current_speed_display", "units", "headway_target_s", "lane_text",
                "merge_text", "traffic_text", "confidence", "confidence_label", "action"),
            advisory().toExtensions().keys,
        )
        assertEquals(
            setOf("t_capture_mono_ns", "rates", "trigger", "shadow"),
            rateCmd().toExtensions().keys,
        )
    }

    @Test
    fun `a nullable field carries null rather than a sentinel`() {
        assertNull(ImuSample.fromWire(imu().copy(accuracy = null).toExtensions(), ByteArray(0)).accuracy)
        assertNull(
            HereResponse.fromWire(here().copy(contentType = null).toExtensions(), ByteArray(0)).contentType,
        )
        assertNull(
            PhoneTelemetry.fromWire(telemetry().copy(thermalHeadroom = null).toExtensions(), ByteArray(0))
                .thermalHeadroom,
        )
    }

    // -- the spec's worked example -------------------------------------------

    @Test
    fun `a zero rate is refused, which is the spec's own worked example`() {
        // "A zero in `rates` is the case that shows why -- it is read as a period, so the
        // field that should have said '10 Hz' instead says 'never', and the failure
        // surfaces on the far side of the link." It was accepted, on both the send and
        // receive paths, because rate_cmd had no decoder at all.
        for (key in RateCommand.RATE_KEYS) {
            val broken = rateCmd().copy(rates = rateCmd().rates + (key to 0.0))
            assertEquals(
                RefusalReason.OUT_OF_RANGE,
                refusalFor { RateCommand.fromWire(broken.toExtensions(), ByteArray(0)) },
                "a zero $key was accepted",
            )
        }
    }

    @Test
    fun `a rate outside the wire's range is refused at both ends`() {
        for (bad in listOf(-1.0, 0.0, 1000.001, 1e9)) {
            val broken = rateCmd().copy(rates = rateCmd().rates + ("gps_hz" to bad))
            assertEquals(
                RefusalReason.OUT_OF_RANGE,
                refusalFor { RateCommand.fromWire(broken.toExtensions(), ByteArray(0)) },
                "$bad Hz was accepted",
            )
        }
        // The boundary is inclusive at the top: 1000 Hz is legal, 0 is not.
        RateCommand.fromWire(
            rateCmd().copy(rates = rateCmd().rates + ("gps_hz" to 1000.0)).toExtensions(),
            ByteArray(0),
        )
    }

    // -- the closed sets -----------------------------------------------------

    @Test
    fun `units outside the three is unknown_value, not a type error`() {
        val broken = advisory().copy(units = "furlongs")
        assertEquals(
            RefusalReason.UNKNOWN_VALUE,
            refusalFor { AdvisoryMessage.fromWire(broken.toExtensions(), ByteArray(0)) },
        )
    }

    @Test
    fun `an action value outside the schema is unknown_value`() {
        val broken = advisory().copy(action = advisory().action + ("merge_mode" to "ram_it"))
        assertEquals(
            RefusalReason.UNKNOWN_VALUE,
            refusalFor { AdvisoryMessage.fromWire(broken.toExtensions(), ByteArray(0)) },
        )
    }

    @Test
    fun `an extra head in action is unknown_value, because the heads are a closed set`() {
        // Unlike `rates` and `achieved`, which are additive: an unknown *head* is a policy
        // this build cannot honour, so accepting it would mean displaying an advisory whose
        // reasoning is partly unread.
        val extensions = advisory().toExtensions().toMutableMap()
        val action = (extensions.getValue("action") as JsonValue.Obj).entries +
            ("desired_altitude_bin" to JsonValue.Text("cruise"))
        extensions["action"] = JsonValue.Obj(action)
        assertEquals(
            RefusalReason.UNKNOWN_VALUE,
            refusalFor { AdvisoryMessage.fromWire(extensions, ByteArray(0)) },
        )
    }

    @Test
    fun `a missing head in action is missing_field`() {
        for (head in AdvisoryMessage.ACTION_HEADS) {
            val extensions = advisory().toExtensions().toMutableMap()
            extensions["action"] =
                JsonValue.Obj((extensions.getValue("action") as JsonValue.Obj).entries - head)
            assertEquals(
                RefusalReason.MISSING_FIELD,
                refusalFor { AdvisoryMessage.fromWire(extensions, ByteArray(0)) },
                "a missing $head gave the wrong reason",
            )
        }
    }

    // -- counts --------------------------------------------------------------

    @Test
    fun `a fractional count is refused rather than truncated`() {
        // Truncating would hide a sender bug behind a plausible number.
        val extensions = telemetry().toExtensions().toMutableMap()
        extensions["dropped"] = JsonValue.Obj(
            (extensions.getValue("dropped") as JsonValue.Obj).entries + ("gps" to JsonValue.Real(1.5)),
        )
        assertEquals(
            RefusalReason.WRONG_TYPE,
            refusalFor { PhoneTelemetry.fromWire(extensions, ByteArray(0)) },
        )
    }

    @Test
    fun `a negative count is out_of_range`() {
        val extensions = telemetry().toExtensions().toMutableMap()
        extensions["dropped"] = JsonValue.Obj(
            (extensions.getValue("dropped") as JsonValue.Obj).entries + ("imu" to JsonValue.Num(-1)),
        )
        assertEquals(
            RefusalReason.OUT_OF_RANGE,
            refusalFor { PhoneTelemetry.fromWire(extensions, ByteArray(0)) },
        )
        assertEquals(
            RefusalReason.OUT_OF_RANGE,
            refusalFor {
                PhoneTelemetry.fromWire(
                    telemetry().copy(hereCalls = -1).toExtensions(), ByteArray(0),
                )
            },
        )
    }

    @Test
    fun `a missing key in a nested object is missing_field`() {
        for (key in RateCommand.RATE_KEYS) {
            val extensions = rateCmd().toExtensions().toMutableMap()
            extensions["rates"] =
                JsonValue.Obj((extensions.getValue("rates") as JsonValue.Obj).entries - key)
            assertEquals(
                RefusalReason.MISSING_FIELD,
                refusalFor { RateCommand.fromWire(extensions, ByteArray(0)) },
                "a missing rates.$key gave the wrong reason",
            )
        }
    }

    // -- the structural property --------------------------------------------

    @Test
    fun `every channel in the table has a typed decoder`() {
        // The point of the map replacing the `when`. With `else -> Unit`, a channel without
        // a decoder was silently exempt from most of the refusal table, and the set that
        // documented the exemption was read by nothing -- so it could not go stale in a way
        // anything would notice.
        assertTrue(
            MessageValidation.ALL_CHANNELS_HAVE_A_DECODER,
            "no decoder for ${MessageValidation.CHANNELS_WITHOUT_A_DECODER}",
        )
    }

    @Test
    fun `every refusal reason but one is reachable on the send path`() {
        val reached = mutableSetOf<RefusalReason>()
        fun attempt(channel: String, extensions: Map<String, JsonValue>, payload: ByteArray = ByteArray(0)) {
            try {
                MessageValidation.check(channel, extensions, payload)
            } catch (e: MessageError) {
                reached.add(e.reason)
            } catch (e: FramingError) {
                // A framing condition, deliberately not a refusal reason.
            }
        }

        attempt(Channels.GPS, GpsRecord.noFix(1).toExtensions() - "valid")
        attempt(Channels.GPS, GpsRecord.noFix(1).toExtensions() + ("valid" to JsonValue.Text("yes")))
        attempt(Channels.GPS, GpsRecord.noFix(1).toExtensions() + ("valid" to JsonValue.Null))
        attempt(Channels.IMU, imu().toExtensions() + ("ax" to JsonValue.Real(Double.NaN)))
        attempt(Channels.RATE_CMD, rateCmd().copy(rates = rateCmd().rates + ("gps_hz" to 0.0)).toExtensions())
        attempt(Channels.ADVISORY, advisory().copy(units = "furlongs").toExtensions())
        attempt(Channels.IMU, imu().toExtensions(), payload = byteArrayOf(1))
        attempt(Channels.GPS, GpsRecord.noFix(1).toExtensions() + ("hello" to JsonValue.Bool(true)))

        // `no_typed_message` is the exception, and it is unreachable *because* every channel
        // now has a decoder. The guard stays as a backstop for a channel added without one,
        // and `ALL_CHANNELS_HAVE_A_DECODER` is what keeps that honest -- so this states the
        // omission rather than quietly leaving one reason untested.
        val expected = RefusalReason.entries.toSet() - RefusalReason.NO_TYPED_MESSAGE
        assertEquals(expected, reached, "unreachable outbound: ${expected - reached}")
    }
}
