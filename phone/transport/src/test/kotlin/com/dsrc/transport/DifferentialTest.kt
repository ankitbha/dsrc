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

        // Crashes first, and the order is the point. A one-sided crash is caught by the
        // disagreement check below either way -- but it is reported as "the two sides
        // disagree on 1 of 32 cases", which sends a reader looking for a refusal-table
        // divergence that is not there. Checked here, the same mutant reads as what it is.
        // `theirs` is included because it was not: the assertion covered Kotlin only, and a
        // Python decoder that blew up used to take the whole script down, which reads as
        // "python failed to run" rather than naming the input.
        // Verdict and input shape are compared separately, so a mismatch says which it is.
        // Together in one string they would report an input drift as a refusal-table
        // disagreement, which is the wrong diagnosis for the wrong reader.
        fun verdict(value: String) = value.substringBefore('|')
        fun shape(value: String) = value.substringAfter('|', "")

        val crashed = (ours + theirs).filterValues { verdict(it).startsWith("CRASH:") }
        assertTrue(crashed.isEmpty(), "a decoder crashed rather than refusing: $crashed")

        val differentInputs = ours.filter { (name, value) -> shape(theirs.getValue(name)) != shape(value) }
            .map { (name, value) -> "$name:\n  kotlin ${shape(value)}\n  python ${shape(theirs.getValue(name))}" }
        assertTrue(
            differentInputs.isEmpty(),
            "the two sides agree on the reason while asking different questions:\n" +
                differentInputs.joinToString("\n"),
        )

        // Recorded, not papered over. These four are the multi-fault rows above, where the
        // two decoders genuinely disagree because they check fields in different orders.
        // Listing them means a NEW divergence still fails, and fixing one of these also
        // fails -- forcing the list to shrink deliberately rather than rotting into a
        // permanent exemption. The resolution is a precedence rule ("report the most
        // structural fault present", so absent beats null beats wrong-typed beats
        // out-of-range) applied on both sides, which needs both decoders to evaluate every
        // field rather than throw on the first, and is a change worth making on purpose.
        val knownOrderDivergences = setOf(
            "two faults: gps negative count and wrong-typed speed",
            "two faults: gps null capture stamp and wrong-typed utc",
            "two faults: camera missing height and null frame id",
            "two faults: control negative exchange id and partial pong",
        )
        val disagreements = ours
            .filter { (name, value) -> verdict(theirs.getValue(name)) != verdict(value) }
            .filterKeys { it !in knownOrderDivergences }
            .map { (name, value) -> "$name: kotlin=${verdict(value)} python=${verdict(theirs.getValue(name))}" }
        val agreedAfterAll = knownOrderDivergences
            .filter { verdict(theirs.getValue(it)) == verdict(ours.getValue(it)) }
        assertTrue(
            agreedAfterAll.isEmpty(),
            "these were recorded as check-order divergences and now agree; remove them from " +
                "knownOrderDivergences: $agreedAfterAll",
        )
        assertTrue(
            disagreements.isEmpty(),
            "the two sides disagree on ${disagreements.size} of ${ours.size} cases:\n" +
                disagreements.joinToString("\n"),
        )
        // A size floor, because two tables shrinking together would otherwise pass; and a
        // distinct-*refusal* floor. The previous count was over all values, so "ACCEPTED"
        // and any "CRASH:" spelling counted as reasons: a floor of 7 was really 6 of the
        // nine the table defines, and a decoder that started crashing would have pushed the
        // number *up*.
        assertTrue(ours.size >= 30, "the case table has shrunk to ${ours.size}")
        val reasons = ours.values.map { verdict(it) }
            .filter { it != "ACCEPTED" && !it.startsWith("CRASH:") }.toSet()
        // Seven of the nine, and the two absences are structural rather than gaps.
        // `reserved_key` is a *sender* rule -- it lives in MessageValidation.check, which
        // this table does not call, because it compares decoders. `no_typed_message` is
        // raised when a channel has no decoder, and every channel has one (asserted by
        // ALL_CHANNELS_HAVE_A_DECODER), so it cannot occur while that holds. Naming them
        // beats a floor of 7 that looks like an arbitrary number.
        assertTrue(
            reasons.size >= 7,
            "the table exercises too few distinct refusal reasons: $reasons",
        )

    }

    private fun pythonReasons(): Map<String, String> {
        val root = System.getProperty("dsrc.repoRoot") ?: error("dsrc.repoRoot is not set")
        // The venv interpreter, not whatever `python3` resolves to. On this machine that
        // was 3.14.6 against the project's 3.12.14 -- so the reconciliation was being
        // measured against an interpreter the project does not ship or test with, and
        // nothing recorded which one ran.
        val venv = java.io.File(root, ".venv/bin/python3")
        // Recorded on the way past, not only when something fails. The fallback below is
        // silent by nature: on a machine without the venv this test still goes green, and
        // a green reconciliation that does not say which interpreter produced it is a
        // measurement with no provenance. The version comes from the interpreter itself
        // rather than from a constant, so it cannot drift from what actually ran.
        val interpreter = if (venv.canExecute()) {
            venv.absolutePath
        } else {
            println(
                "MEASURE python_interpreter=fallback path=python3 -- .venv is absent, so " +
                    "this reconciliation is against an interpreter the project does not ship",
            )
            "python3"
        }

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
        // The version, from the run itself. The comment above this method says the whole
        // point was that "nothing recorded which one ran", and until now nothing did on the
        // path where it worked.
        val version = out.lines().firstOrNull { it.startsWith("VERSION ") }?.removePrefix("VERSION ")
        println("MEASURE python_interpreter=$interpreter version=${version ?: "unreported"}")
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

    /**
     * One case: its verdict, and a fingerprint of the inputs that produced it.
     *
     * Comparing the case *names* proves only that both sides have a row called the same
     * thing. Round 6 asked for the inputs to be compared too, and most of that concern is
     * already covered from an unexpected direction: if one side's inputs change and the
     * other's do not, the two verdicts diverge and the disagreement check catches it. What
     * survives is a *reason-preserving* edit on one side only -- nulling a different field
     * that refuses for the same reason -- where both sides still answer `null_not_allowed`
     * about two different questions.
     *
     * So the fingerprint is structural rather than a digest of the values. A value digest
     * would have to agree with Python's float formatting exactly, and it cannot even be
     * computed for the two `non_finite` cases: canonical JSON refuses to encode a NaN on
     * both sides, which is the property those cases exist to check. Keys and types are
     * enough to catch a case that stops exercising the field it is named for, and they
     * cost nothing to keep in step.
     */
    private fun case(
        extensions: Map<String, JsonValue>,
        payload: ByteArray,
        decode: (Map<String, JsonValue>, ByteArray) -> Unit,
    ): String = "${reasonOf { decode(extensions, payload) }}|${shapeOf(extensions, payload)}"

    /** `key:type` for every key, sorted and recursive, plus the payload length. */
    private fun shapeOf(extensions: Map<String, JsonValue>, payload: ByteArray): String =
        objectShape(extensions) + "#${payload.size}"

    private fun objectShape(entries: Map<String, JsonValue>): String =
        entries.entries.map { (key, value) -> "$key:${typeName(value)}" }.sorted().joinToString(",")

    /**
     * A type vocabulary both languages can spell the same way.
     *
     * `JsonValue.Num` and Python's `int` are the same thing on this wire and neither name
     * is portable, so the comparison uses neither.
     */
    private fun typeName(value: JsonValue): String = when (value) {
        // Values, not just types, for everything both languages spell identically. Round 8
        // showed why: with `key:type` alone, two same-typed siblings are interchangeable,
        // so moving "ram_it" from `merge_mode` to `lane_preference` on one side only was
        // still invisible -- which is verbatim the drift the previous commit claimed to
        // have closed. Recursion bought key-set drift and nothing else.
        //
        // Reals carry a rendering rather than a value because the two runtimes need not
        // agree on how a double prints. Both use shortest-round-trip for the magnitudes in
        // this table, and if that ever stops being true every row fails at once rather than
        // one row failing quietly -- which is the right way for this to break.
        is JsonValue.Null -> "null"
        is JsonValue.Bool -> "bool=${value.value}"
        is JsonValue.Num -> "int=${value.value}"
        is JsonValue.Real -> if (value.value.isFinite()) "real=${value.value}" else "real=nonfinite"
        is JsonValue.Text -> "str=${value.value}"
        // Recursive, and round 7 is why. Emitting a bare "obj" hid the contents of
        // `action` and `rates`, so six of the thirty-two rows -- the three advisory
        // action heads and the three rate keys -- were still open to exactly the drift
        // this guard was added for. Moving `lane_preference: "ram_it"` to `merge_mode`
        // on one side only was reason-preserving *and* shape-preserving, and the suite
        // stayed green. A positive control one level up did fail, which is how it was
        // clear the mechanism worked and the depth did not.
        is JsonValue.Obj -> "obj(${objectShape(value.entries)})"
        is JsonValue.Arr -> "arr[${value.items.joinToString(",") { typeName(it) }}]"
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
            // non_finite was the one reason in this table's reach that no case exercised,
            // and the floor did not notice because "ACCEPTED" was counted as a reason. It
            // has to be built here rather than parsed from a header: a bare NaN on the wire
            // is a framing error on both sides now, so a decoder only sees one from an
            // in-process caller.
            "gps speed is not finite" to case(gps + ("speed_mps" to JsonValue.Real(Double.NaN)), empty) { e, p -> GpsRecord.fromWire(e, p) },
            "gps altitude is infinite" to case(gps + ("altitude_m" to JsonValue.Real(Double.POSITIVE_INFINITY)), empty) { e, p -> GpsRecord.fromWire(e, p) },
            // The one entry of the shape vocabulary nothing exercised, so the two sides
            // could have spelled a list differently with nothing to say so. Extensions are
            // additive, so an unknown key is preserved rather than refused.
            "gps unknown key holding a list is accepted by both" to case(
                gps + ("future_field" to JsonValue.Arr(listOf(JsonValue.Num(1), JsonValue.Text("two"), JsonValue.Null))),
                empty,
            ) { e, p -> GpsRecord.fromWire(e, p) },
            "gps count is null" to case(gps + ("fix_quality" to JsonValue.Null), empty) { e, p -> GpsRecord.fromWire(e, p) },
            "gps capture stamp is null" to case(gps + (Fields.CAPTURE_KEY to JsonValue.Null), empty) { e, p -> GpsRecord.fromWire(e, p) },
            "camera frame id is null" to case(camera + ("frame_id" to JsonValue.Null), empty) { e, p -> CameraFrameMessage.fromWire(e, p) },
            "camera format is null" to case(camera + ("format" to JsonValue.Null), empty) { e, p -> CameraFrameMessage.fromWire(e, p) },
            "advisory units is null" to case(advisory() + ("units" to JsonValue.Null), empty) { e, p -> AdvisoryMessage.fromWire(e, p) },
            "rate_cmd trigger is null" to case(rateCmd() + ("trigger" to JsonValue.Null), empty) { e, p -> RateCommand.fromWire(e, p) },
            "gps required bool is null" to case(gps + ("valid" to JsonValue.Null), empty) { e, p -> GpsRecord.fromWire(e, p) },
            "gps valid fix with null coordinates" to case(gps + ("valid" to JsonValue.Bool(true)), empty) { e, p -> GpsRecord.fromWire(e, p) },
            "gps negative count" to case(gps + ("num_sats" to JsonValue.Num(-1)), empty) { e, p -> GpsRecord.fromWire(e, p) },
            "gps fractional count" to case(gps + ("num_sats" to JsonValue.Real(1.5)), empty) { e, p -> GpsRecord.fromWire(e, p) },
            "camera quality zero is accepted by both" to case(camera + ("quality" to JsonValue.Num(0)), empty) { e, p -> CameraFrameMessage.fromWire(e, p) },
            "camera quality over one hundred is accepted by both" to case(camera + ("quality" to JsonValue.Num(101)), empty) { e, p -> CameraFrameMessage.fromWire(e, p) },
            "camera zero width is accepted by both" to case(camera + ("width" to JsonValue.Num(0)), empty) { e, p -> CameraFrameMessage.fromWire(e, p) },
            "camera negative frame id is accepted by both" to case(camera + ("frame_id" to JsonValue.Num(-1)), empty) { e, p -> CameraFrameMessage.fromWire(e, p) },
            "camera empty format is accepted by both" to case(camera + ("format" to JsonValue.Text("")), empty) { e, p -> CameraFrameMessage.fromWire(e, p) },
            "advisory action is null" to case(advisory() + ("action" to JsonValue.Null), empty) { e, p -> AdvisoryMessage.fromWire(e, p) },
            "advisory action is not an object" to case(advisory() + ("action" to JsonValue.Num(5)), empty) { e, p -> AdvisoryMessage.fromWire(e, p) },
            "advisory action head is an integer" to case(advisory(action + ("desired_speed_bin" to JsonValue.Num(5))), empty) { e, p -> AdvisoryMessage.fromWire(e, p) },
            "advisory action head outside the set" to case(advisory(action + ("merge_mode" to JsonValue.Text("ram_it"))), empty) { e, p -> AdvisoryMessage.fromWire(e, p) },
            "advisory action missing a head" to case(advisory(action - "merge_mode"), empty) { e, p -> AdvisoryMessage.fromWire(e, p) },
            "advisory units outside the three" to case(advisory() + ("units" to JsonValue.Text("furlongs")), empty) { e, p -> AdvisoryMessage.fromWire(e, p) },
            "rate_cmd rates is null" to case(rateCmd() + ("rates" to JsonValue.Null), empty) { e, p -> RateCommand.fromWire(e, p) },
            "rate_cmd zero rate" to case(rateCmd(rates + ("gps_hz" to JsonValue.Real(0.0))), empty) { e, p -> RateCommand.fromWire(e, p) },
            "rate_cmd rate above the ceiling" to case(rateCmd(rates + ("gps_hz" to JsonValue.Real(1000.001))), empty) { e, p -> RateCommand.fromWire(e, p) },
            "rate_cmd missing a rate key" to case(rateCmd(rates - "imu_hz"), empty) { e, p -> RateCommand.fromWire(e, p) },
            "control partial pong" to case(ping + ("t_peer_recv_mono_ns" to JsonValue.Num(2)), empty) { e, p -> TimeSyncMessage.fromWire(e, p) },
            "control absent peer field" to case(ping - "t_peer_wire_mono_ns", empty) { e, p -> TimeSyncMessage.fromWire(e, p) },
            "control negative exchange id" to case(ping + ("exchange_id" to JsonValue.Num(-5)), empty) { e, p -> TimeSyncMessage.fromWire(e, p) },
            "control negative wire stamp" to case(ping + (Session.WIRE_STAMP to JsonValue.Num(-7)), empty) { e, p -> TimeSyncMessage.fromWire(e, p) },
            // -- two faults at once ------------------------------------------------
            //
            // Every row above injects exactly one fault, so the ORDER in which each
            // decoder applies its rules has never been compared. It differs: Python
            // checks utc_epoch_ns before the constructor and the counts ahead of
            // lat/speed, while this side checks the capture stamp first and speed/heading
            // ahead of the counts. One systematic mapping bug in a phone build produces
            // multi-fault records, and then inboundRefusals and errors_by_reason name
            // different causes for the same frames -- which is what per-reason counting
            // exists to prevent.
            "two faults: gps negative count and wrong-typed speed" to case(
                gps + ("num_sats" to JsonValue.Num(-1)) + ("speed_mps" to JsonValue.Text("x")),
                empty,
            ) { e, p -> GpsRecord.fromWire(e, p) },
            "two faults: gps null capture stamp and wrong-typed utc" to case(
                gps + (Fields.CAPTURE_KEY to JsonValue.Null) + ("utc_epoch_ns" to JsonValue.Text("x")),
                empty,
            ) { e, p -> GpsRecord.fromWire(e, p) },
            "two faults: camera missing height and null frame id" to case(
                camera - "height" + ("frame_id" to JsonValue.Null),
                empty,
            ) { e, p -> CameraFrameMessage.fromWire(e, p) },
            "two faults: control negative exchange id and partial pong" to case(
                ping + ("exchange_id" to JsonValue.Num(-5)) +
                    (TimeSyncMessage.KEY_PEER_RECV_MONO to JsonValue.Num(5)),
                empty,
            ) { e, p -> TimeSyncMessage.fromWire(e, p) },
            "control payload on a channel that carries none" to case(ping, byteArrayOf(1)) { e, p -> TimeSyncMessage.fromWire(e, p) },
        )
    }

}
