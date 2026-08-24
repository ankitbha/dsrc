package com.dsrc.transport

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Verdict-for-verdict against CPython on the inputs that separated the two parsers.
 *
 * The expectations are recorded from running `json.loads`, not guessed. Where this parser
 * is deliberately *stricter* than Python -- integers beyond Long, non-finite tokens,
 * duplicate keys -- that is stated as such rather than hidden, because a divergence you
 * chose is different from one you did not notice.
 */
class DifferentialTest {

    /** true = both accept, false = both reject. */
    private val agreeing = mapOf(
        "0" to true,
        "-0" to true,
        "0.5" to true,
        "1e5" to true,
        "\"\\u0041\"" to true,
        "01" to false,
        "-01" to false,
        "00" to false,
        "007" to false,
        "1." to false,
        "-1." to false,
        ".5" to false,
        "1e" to false,
        "1e+" to false,
        "1E-" to false,
        "\"\\u+041\"" to false,
        "\"\\u-041\"" to false,
        "\"a\tb\"" to false,
        "\"a\nb\"" to false,
    )

    /** Inputs Python accepts and this parser deliberately refuses. */
    private val deliberatelyStricter = mapOf(
        "9223372036854775808" to "beyond Long; widening to a double would lose the precision Num exists to keep",
        "NaN" to "a bare non-finite token; the spec forbids it on the wire",
        "Infinity" to "as above",
        "-Infinity" to "as above",
        "1e400" to "overflows to infinity",
        """{"a":1,"a":2}""" to "a duplicate key would make a header's meaning depend on which parser read it",
    )

    @Test
    fun `the two parsers agree wherever they should`() {
        val disagreements = mutableListOf<String>()
        for ((input, shouldAccept) in agreeing) {
            val accepted = runCatching { Json.decode(input) }.isSuccess
            if (accepted != shouldAccept) {
                disagreements.add("$input: python=${if (shouldAccept) "accept" else "reject"}, we ${if (accepted) "accept" else "reject"}")
            }
        }
        assertTrue(disagreements.isEmpty(), "diverged from CPython on:\n" + disagreements.joinToString("\n"))
    }

    @Test
    fun `the deliberate strictness is actually in force`() {
        // Anti-vacuity: each of these must really be refused, with the reason recorded.
        for ((input, why) in deliberatelyStricter) {
            assertTrue(
                runCatching { Json.decode(input) }.isFailure,
                "$input should be refused ($why)",
            )
        }
    }

    @Test
    fun `a lone surrogate is refused on the way out, where python also refuses`() {
        val decoded = Json.decode("\"\\ud800\"")
        assertTrue(runCatching { Json.encode(decoded) }.isFailure)
    }

    @Test
    fun `the agreeing set actually contains both verdicts`() {
        // A table of all-accepts or all-rejects would pass while testing one direction.
        assertTrue(agreeing.values.any { it }, "no accepting cases")
        assertTrue(agreeing.values.any { !it }, "no rejecting cases")
        assertEquals(19, agreeing.size)
    }

    // -- refusal reasons, message layer --------------------------------------

    /**
     * The reason each side gives for the same fault.
     *
     * Recorded by running `deployment/jetson/transport/messages.py`, not guessed. Round 3
     * found three disagreements here and the spec's refusal table settled each one: the
     * unset stamps in a partial pong are present-and-null so they are `null_not_allowed`;
     * a null where a required object belongs is `null_not_allowed` too, which is the one
     * where Python was wrong and was fixed; and an `action` head outside its closed set is
     * `unknown_value` whatever its JSON type, because the set is what the reader needs to
     * know about.
     *
     * The point of pinning them is that the counters are per-reason on both sides. Two
     * implementations filing the same fault in different buckets makes a summary that
     * cannot be added up, which is the thing per-reason counting exists to provide.
     */
    private fun advisory(action: Map<String, JsonValue>? = null) = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(1),
        "rec_speed_mps" to JsonValue.Real(13.4),
        "rec_speed_display" to JsonValue.Real(30.0),
        "current_speed_display" to JsonValue.Real(28.0),
        "units" to JsonValue.Text("mph"),
        "headway_target_s" to JsonValue.Real(2.0),
        "lane_text" to JsonValue.Text("keep"),
        "merge_text" to JsonValue.Text("normal"),
        "traffic_text" to JsonValue.Text("clear"),
        "confidence" to JsonValue.Real(0.87),
        "confidence_label" to JsonValue.Text("high"),
        "action" to JsonValue.Obj(
            action ?: mapOf(
                "desired_speed_bin" to JsonValue.Text("nominal"),
                "desired_headway_bin" to JsonValue.Text("normal"),
                "lane_preference" to JsonValue.Text("keep"),
                "merge_mode" to JsonValue.Text("normal"),
            )
        ),
    )

    private fun rateCmd() = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(1),
        "rates" to JsonValue.Obj(
            mapOf(
                "camera_hz" to JsonValue.Real(5.0),
                "gps_hz" to JsonValue.Real(1.0),
                "imu_hz" to JsonValue.Real(50.0),
                "here_hz" to JsonValue.Real(0.2),
            )
        ),
        "trigger" to JsonValue.Text("thermal"),
        "shadow" to JsonValue.Bool(false),
    )

    private fun reasonOf(block: () -> Unit): String =
        runCatching { block() }
            .exceptionOrNull()
            .let { (it as? MessageError)?.reason?.wire ?: "ACCEPTED" }

    @Test
    fun `both sides name the same reason for the same fault`() {
        // Expectations from running python3 against messages.py.
        assertEquals(
            "null_not_allowed",
            reasonOf {
                TimeSyncMessage.fromWire(
                    mapOf(
                        Fields.CAPTURE_KEY to JsonValue.Num(1),
                        "exchange_id" to JsonValue.Num(1),
                        Session.WIRE_STAMP to JsonValue.Num(0),
                        "t_peer_recv_mono_ns" to JsonValue.Num(2),
                        "t_peer_recv_wall_ns" to JsonValue.Null,
                        "t_peer_wire_mono_ns" to JsonValue.Null,
                    ),
                    ByteArray(0),
                )
            },
            "a partial pong's unset stamps are present-and-null",
        )

        assertEquals(
            "null_not_allowed",
            reasonOf { RateCommand.fromWire(rateCmd() + ("rates" to JsonValue.Null), ByteArray(0)) },
            "a null where a required object belongs",
        )
        assertEquals(
            "null_not_allowed",
            reasonOf { AdvisoryMessage.fromWire(advisory() + ("action" to JsonValue.Null), ByteArray(0)) },
        )

        assertEquals(
            "unknown_value",
            reasonOf {
                AdvisoryMessage.fromWire(
                    advisory(
                        mapOf(
                            "desired_speed_bin" to JsonValue.Num(5),
                            "desired_headway_bin" to JsonValue.Text("normal"),
                            "lane_preference" to JsonValue.Text("keep"),
                            "merge_mode" to JsonValue.Text("normal"),
                        )
                    ),
                    ByteArray(0),
                )
            },
            "an action head outside its closed set, whatever its JSON type",
        )
    }

    @Test
    fun `the wire contract does not police what the spec never states`() {
        // These were refusals I invented: a zero dimension, a quality outside 1..100, an
        // empty format, a negative frame_id. The spec's message and refusal tables mention
        // none of them and Python accepts all five, so a unilateral receiver rule here
        // refuses what the peer legitimately sends -- the dangerous direction of a
        // cross-language disagreement.
        //
        // A bad *setting* still dies where settings enter, which is SensingConfig.
        val base = CameraFrameMessage(1, 1, 1280, 720, "jpeg", 85).toExtensions()
        for ((label, override) in listOf(
            "quality 0" to ("quality" to JsonValue.Num(0)),
            "quality 101" to ("quality" to JsonValue.Num(101)),
            "width 0" to ("width" to JsonValue.Num(0)),
            "frame_id -1" to ("frame_id" to JsonValue.Num(-1)),
            "empty format" to ("format" to JsonValue.Text("")),
        )) {
            assertEquals(
                "ACCEPTED",
                reasonOf { CameraFrameMessage.fromWire(base + override, ByteArray(0)) },
                "$label is refused here and accepted by python",
            )
        }
    }

}
