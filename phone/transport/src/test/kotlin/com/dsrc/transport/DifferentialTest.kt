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
     * Every refusal reason, compared against Python by *running* Python.
     *
     * The previous version of this asserted Kotlin's reasons and claimed in a docstring
     * that they had been "recorded by running messages.py". A docstring cannot fail, and
     * one of those expectations had never matched: a null `action` was `wrong_type` on the
     * Python side the whole time, because the fix landed in `_nested_object` and
     * `AdvisoryMessage.from_wire` has its own inline check. A test written to prevent drift
     * was itself the drift.
     *
     * `scripts/refusal_reasons.py` prints Python's verdict per named case; this runs it and
     * compares case by case. Adding a case means adding it in both places, and any
     * divergence is a failure rather than a stale comment.
     */
    @Test
    fun `both implementations name the same reason for the same fault`() {
        val theirs = pythonReasons()
        val ours = kotlinReasons()

        assertEquals(theirs.keys, ours.keys, "the two case tables have drifted apart")
        val disagreements = ours.filter { (name, reason) -> theirs[name] != reason }
            .map { (name, reason) -> "$name: kotlin=$reason python=${theirs[name]}" }
        assertTrue(
            disagreements.isEmpty(),
            "the two sides disagree on ${disagreements.size} of ${ours.size} cases:\n" +
                disagreements.joinToString("\n"),
        )
        // A size floor, because two tables shrinking together would otherwise pass; and a
        // distinct-reason floor at what the table actually produces rather than two below it.
        assertTrue(ours.size >= 30, "the case table has shrunk to ${ours.size}")
        assertTrue(
            ours.values.toSet().size >= 7,
            "the table exercises too few distinct reasons: ${ours.values.toSet()}",
        )
        // No case may report a crash: that would match any both-sides-ACCEPTED row.
        assertTrue(
            ours.none { it.value.startsWith("CRASH:") },
            "a decoder crashed rather than refusing: ${ours.filterValues { it.startsWith("CRASH:") }}",
        )
    }

    private fun pythonReasons(): Map<String, String> {
        val root = System.getProperty("dsrc.repoRoot") ?: error("dsrc.repoRoot is not set")
        // The venv interpreter, not whatever `python3` resolves to. On this machine that
        // was 3.14.6 against the project's 3.12.14 -- so the reconciliation was being
        // measured against an interpreter the project does not ship or test with, and
        // nothing recorded which one ran.
        val venv = java.io.File(root, ".venv/bin/python3")
        val interpreter = if (venv.canExecute()) venv.absolutePath else "python3"

        // Streams merged, and the wait comes *first*. Reading stdout to EOF before waitFor
        // meant the timeout could only be reached after the child had already exited: a peer
        // sleeping 40 s made this pass after 41 rather than fail at 30. Merging also removes
        // the deadlock where stderr fills its pipe while nothing is reading it.
        val process = ProcessBuilder(interpreter, "scripts/refusal_reasons.py")
            .directory(java.io.File(root))
            .redirectErrorStream(true)
            .start()
        val collected = StringBuilder()
        val drain = Thread({
            process.inputStream.bufferedReader().forEachLine { collected.appendLine(it) }
        }, "python-drain").also { it.isDaemon = true; it.start() }

        if (!process.waitFor(30, java.util.concurrent.TimeUnit.SECONDS)) {
            process.destroyForcibly()
            error("python did not finish within 30 s using $interpreter")
        }
        drain.join(5_000)
        val out = collected.toString()
        require(process.exitValue() == 0) { "python failed ($interpreter): $out" }
        // The last line, so a warning on stdout cannot break the parse.
        val payload = out.trim().lines().last { it.startsWith("{") }
        val decoded = (Json.decode(payload) as JsonValue.Obj).entries
        return decoded.mapValues { (_, value) -> (value as JsonValue.Text).value }
    }

    /**
     * The reason a decoder gave, or how it failed.
     *
     * A non-MessageError throwable used to fall into "ACCEPTED", so a *crash* matched any
     * case the table expects both sides to accept -- and five of them are exactly that.
     * Making a decoder `error(...)` on zero width survived this test entirely.
     */
    private fun reasonOf(block: () -> Unit): String {
        val thrown = runCatching { block() }.exceptionOrNull() ?: return "ACCEPTED"
        return (thrown as? MessageError)?.reason?.wire
            ?: "CRASH:${thrown.javaClass.simpleName}"
    }

    private fun kotlinReasons(): Map<String, String> {
        val gps = GpsRecord.noFix(1).toExtensions()
        val camera = CameraFrameMessage(1, 1, 1280, 720, "jpeg", 85).toExtensions()
        val action = mapOf(
            "desired_speed_bin" to JsonValue.Text("nominal"),
            "desired_headway_bin" to JsonValue.Text("normal"),
            "lane_preference" to JsonValue.Text("keep"),
            "merge_mode" to JsonValue.Text("normal"),
        )
        fun advisory(head: Map<String, JsonValue> = action) = mapOf(
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
            "action" to JsonValue.Obj(head),
        )
        val rates = mapOf(
            "camera_hz" to JsonValue.Real(5.0), "gps_hz" to JsonValue.Real(1.0),
            "imu_hz" to JsonValue.Real(50.0), "here_hz" to JsonValue.Real(0.2),
        )
        fun rateCmd(r: Map<String, JsonValue> = rates) = mapOf(
            Fields.CAPTURE_KEY to JsonValue.Num(1),
            "rates" to JsonValue.Obj(r),
            "trigger" to JsonValue.Text("thermal"),
            "shadow" to JsonValue.Bool(false),
        )
        val ping = mapOf(
            Fields.CAPTURE_KEY to JsonValue.Num(1),
            "exchange_id" to JsonValue.Num(1),
            Session.WIRE_STAMP to JsonValue.Num(0),
            "t_peer_recv_mono_ns" to JsonValue.Null,
            "t_peer_recv_wall_ns" to JsonValue.Null,
            "t_peer_wire_mono_ns" to JsonValue.Null,
        )
        val empty = ByteArray(0)

        return mapOf(
            "gps count is null" to reasonOf { GpsRecord.fromWire(gps + ("fix_quality" to JsonValue.Null), empty) },
            "gps capture stamp is null" to reasonOf { GpsRecord.fromWire(gps + (Fields.CAPTURE_KEY to JsonValue.Null), empty) },
            "camera frame id is null" to reasonOf { CameraFrameMessage.fromWire(camera + ("frame_id" to JsonValue.Null), empty) },
            "camera format is null" to reasonOf { CameraFrameMessage.fromWire(camera + ("format" to JsonValue.Null), empty) },
            "advisory units is null" to reasonOf { AdvisoryMessage.fromWire(advisory() + ("units" to JsonValue.Null), empty) },
            "rate_cmd trigger is null" to reasonOf { RateCommand.fromWire(rateCmd() + ("trigger" to JsonValue.Null), empty) },
            "gps required bool is null" to reasonOf { GpsRecord.fromWire(gps + ("valid" to JsonValue.Null), empty) },
            "gps valid fix with null coordinates" to reasonOf { GpsRecord.fromWire(gps + ("valid" to JsonValue.Bool(true)), empty) },
            "gps negative count" to reasonOf { GpsRecord.fromWire(gps + ("num_sats" to JsonValue.Num(-1)), empty) },
            "gps fractional count" to reasonOf { GpsRecord.fromWire(gps + ("num_sats" to JsonValue.Real(1.5)), empty) },
            "camera quality zero" to reasonOf { CameraFrameMessage.fromWire(camera + ("quality" to JsonValue.Num(0)), empty) },
            "camera quality over one hundred" to reasonOf { CameraFrameMessage.fromWire(camera + ("quality" to JsonValue.Num(101)), empty) },
            "camera zero width" to reasonOf { CameraFrameMessage.fromWire(camera + ("width" to JsonValue.Num(0)), empty) },
            "camera negative frame id" to reasonOf { CameraFrameMessage.fromWire(camera + ("frame_id" to JsonValue.Num(-1)), empty) },
            "camera empty format" to reasonOf { CameraFrameMessage.fromWire(camera + ("format" to JsonValue.Text("")), empty) },
            "advisory action is null" to reasonOf { AdvisoryMessage.fromWire(advisory() + ("action" to JsonValue.Null), empty) },
            "advisory action is not an object" to reasonOf { AdvisoryMessage.fromWire(advisory() + ("action" to JsonValue.Num(5)), empty) },
            "advisory action head is an integer" to reasonOf {
                AdvisoryMessage.fromWire(advisory(action + ("desired_speed_bin" to JsonValue.Num(5))), empty)
            },
            "advisory action head outside the set" to reasonOf {
                AdvisoryMessage.fromWire(advisory(action + ("merge_mode" to JsonValue.Text("ram_it"))), empty)
            },
            "advisory action missing a head" to reasonOf {
                AdvisoryMessage.fromWire(advisory(action - "merge_mode"), empty)
            },
            "advisory units outside the three" to reasonOf {
                AdvisoryMessage.fromWire(advisory() + ("units" to JsonValue.Text("furlongs")), empty)
            },
            "rate_cmd rates is null" to reasonOf { RateCommand.fromWire(rateCmd() + ("rates" to JsonValue.Null), empty) },
            "rate_cmd zero rate" to reasonOf { RateCommand.fromWire(rateCmd(rates + ("gps_hz" to JsonValue.Real(0.0))), empty) },
            "rate_cmd rate above the ceiling" to reasonOf {
                RateCommand.fromWire(rateCmd(rates + ("gps_hz" to JsonValue.Real(1000.001))), empty)
            },
            "rate_cmd missing a rate key" to reasonOf { RateCommand.fromWire(rateCmd(rates - "imu_hz"), empty) },
            "control partial pong" to reasonOf {
                TimeSyncMessage.fromWire(ping + ("t_peer_recv_mono_ns" to JsonValue.Num(2)), empty)
            },
            "control absent peer field" to reasonOf { TimeSyncMessage.fromWire(ping - "t_peer_wire_mono_ns", empty) },
            "control negative exchange id" to reasonOf { TimeSyncMessage.fromWire(ping + ("exchange_id" to JsonValue.Num(-5)), empty) },
            "control negative wire stamp" to reasonOf { TimeSyncMessage.fromWire(ping + (Session.WIRE_STAMP to JsonValue.Num(-7)), empty) },
            "control payload on a channel that carries none" to reasonOf { TimeSyncMessage.fromWire(ping, byteArrayOf(1)) },
        )
    }

}
