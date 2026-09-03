"""Re-apply every mutation used as a pin, and check each is still caught.

A mutation test proves something on the day it is run and not after. The reason
this exists is a case where it stopped being true: a test written for the
Failed-from-RUNNING route was driven by a throwing status listener, and a later,
unrelated fix -- containing listener exceptions -- closed the door that trigger
used. The test kept passing. Mutating the teardown away left all 51 instrumented
tests green, and nothing announced that the pin had lapsed.

So each entry below names a defect that a specific test is supposed to catch. Run
it after landing a batch of fixes, not only after landing one.

    python3 scripts/remutate.py

An anchor that no longer matches is reported as a failure rather than skipped:
the code moved, and whether the pin survived the move is exactly the open
question. Every mutation is restored in a `finally`, including on Ctrl-C -- but
this edits files in place, so do not run it with uncommitted work you would mind
losing, and never point it at a tree a validator is reading.
"""
import pathlib, shutil, subprocess, sys
from xml.etree import ElementTree

ROOT = pathlib.Path(".")
GRADLE = ["env", "JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home",
          "./phone/gradlew", "-p", "phone"]

# Deliberately absent: "deliver decodes a control frame twice"
# (`if (frame.channel != Channels.CONTROL)` -> `if (true)`).
#
# It has no behavioural observable, and I looked for one rather than assuming. Both
# routes refuse the same frames with the same reasons, because checkInbound's control
# entry *is* TimeSyncMessage.fromWire. The two things checkInbound would add back --
# checkAllFinite over the whole extension map, and the payload rule -- are respectively
# unreachable inbound (both parsers refuse every non-finite literal at the framing layer,
# so no frame carrying one can arrive) and already enforced by fromWire itself.
#
# What restoring the double decode actually does is make handleTimeSync's own
# `catch (e: MessageError)` dead, because checkInbound's refusal fires first. So it is
# guarded here, indirectly but genuinely: the entry below named "timebase: a decode
# failure is passed through, not refused" starts SURVIVING the moment the double decode
# comes back. An entry that asserts nothing directly is worse than a note saying which
# entry covers it.
MUTATIONS = [
    ("provenance: a computed class counts as evidence, so an all-substituted drive answers",
     "deployment/jetson/eval_run.py",
     "        if classes_present & PROVENANCE_PRIMARY_EVIDENCE:",
     "        if classes_present - (provenance.SUBSTITUTED | {provenance.SOURCE_DERIVED_EMPTY}):",
     "python"),
    ("summary: the failures axis drops its unreadable-source census",
     "deployment/jetson/eval_run.py",
     "        if unreadable > 0:\n            not_evaluable_sources[source_name] = unreadable",
     "        if False:\n            not_evaluable_sources[source_name] = unreadable",
     "python"),
    ("summary: an axis points at a section whether or not it was rendered",
     "deployment/jetson/eval_run.py",
     '        f" See {section}." if rendered_sections is None or section in rendered_sections',
     '        f" See {section}." if True',
     "python"),
    ("inbound: a delivery failure is filed as delivered",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     '            lastDeliveryFailure = "${t.javaClass.name}: ${t.message}"\n            deliveryFailures.incrementAndGet()\n            inbound.countFailed(frame.channel)',
     '            lastDeliveryFailure = "${t.javaClass.name}: ${t.message}"\n            deliveryFailures.incrementAndGet()\n            inbound.countDelivered(frame.channel)',
     "transport"),
    # One entry, not two. The other spelling of this deleted a declaration that is still
    # referenced below it, so it never compiled -- and a build that fails to compile exits
    # non-zero, which the old exit-code scoring counted as CAUGHT. It measured the Kotlin
    # compiler and reported a pin.
    ("stats: the per-channel map is read inline, after the totals",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "            inboundChannels = inboundChannels,",
     "            inboundChannels = inbound.counters(),",
     "transport"),
    # Replaced the write-ordering entry. That one was inherently probabilistic -- the
    # window it opened is nanoseconds wide, so it reported SURVIVED one run and CAUGHT the
    # next with nothing changed, and a harness cannot tell a lapsed pin from an unlucky
    # one. The two fields are one volatile record now, so there is no ordering left to get
    # wrong; what is worth pinning is that the count is reported at all.
    ("stats: an outbound framing refusal is counted but never reported",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "            outboundFramingRefusals = framing?.count ?: 0,",
     "            outboundFramingRefusals = 0,",
     "transport"),
    ("inbound: depth() returns the total across channels",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Queues.kt",
     "    fun depth(channel: String): Long = synchronized(lock) { queues.getValue(channel).size.toLong() }",
     "    fun depth(channel: String): Long = synchronized(lock) { queues.values.sumOf { it.size.toLong() } }",
     "transport"),
    ("timebase: a decode failure is passed through, not refused",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "        } catch (e: MessageError) {\n            countInboundRefusal(frame.channel, e.reason.wire)\n            return TimeSyncOutcome.REFUSED\n        }",
     "        } catch (e: MessageError) {\n            return TimeSyncOutcome.NOT_OURS\n        }",
     "transport"),
    ("timebase: the wrong-direction pong gets a decode reason",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "            countInboundRefusal(frame.channel, RefusalReason.UNKNOWN_VALUE.wire)\n            return TimeSyncOutcome.REFUSED\n        }\n        // Wire-stamped",
     "            countInboundRefusal(frame.channel, RefusalReason.WRONG_TYPE.wire)\n            return TimeSyncOutcome.REFUSED\n        }\n        // Wire-stamped",
     "transport"),
    ("camera: a rejected submit escapes offer",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/CameraPipeline.kt",
     "            abandoned.incrementAndGet()\n            return false\n        }\n        return true",
     "            throw t\n        }\n        return true",
     "app"),
    ("frame: hashCode is a constant",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/CapturedFrame.kt",
     "    override fun hashCode(): Int {",
     "    override fun hashCode(): Int {\n        if (true) return 0",
     "app"),
    ("status: listener delivery is unguarded",
     "phone/app/src/main/kotlin/com/dsrc/phone/SensingStatus.kt",
     "        try {\n            listener.onState(state)\n        } catch (t: Throwable) {",
     "        try {\n            listener.onState(state)\n        } catch (t: NoSuchElementException) {",
     "app"),
    ("imu: an unpaired accelerometer event is not counted",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPipeline.kt",
     "        unpaired++\n    }",
     "    }",
     "app"),
    ("imu: the gyro age keeps the last instead of the max",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPipeline.kt",
     "gyroAgeMaxNs = maxOf(gyroAgeMaxNs, reading.gyroAgeNs)",
     "gyroAgeMaxNs = reading.gyroAgeNs",
     "app"),
    ("imu: an accelerometer axis is transposed on the way to the sample",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPipeline.kt",
     "            az = reading.az,", "            az = reading.ay,",
     "app"),
    ("imu: a gyro axis is transposed on the way to the reading",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "                gx = gx, gy = gy, gz = gz,", "                gx = gx, gy = gz, gz = gz,",
     "app"),
    ("log: a line offered after stop vanishes silently",
     "phone/app/src/main/kotlin/com/dsrc/phone/log/SessionLog.kt",
     "            synchronized(lock) { droppedNotRunning++ }\n            return", "            return", "app"),
    ("log: the hello is not recorded",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "            recordSent(header, 0)\n        } catch (e: Exception) {", "        } catch (e: Exception) {", "transport"),
    ("log: the heartbeat is not recorded",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "        heartbeatsSent.incrementAndGet()\n        recordSent(header, 0)",
     "        heartbeatsSent.incrementAndGet()", "transport"),
    ("log: a full queue blocks the caller instead of dropping",
     "phone/app/src/main/kotlin/com/dsrc/phone/log/SessionLog.kt",
     "        if (!queue.offer(line)) {",
     "        if (!queue.offer(headerJson)) { queue.put(headerJson) } else if (false) {", "app"),
    # Re-anchored: the three call sites were folded into one `recordSent(header, payloadSize)`,
    # and the old anchor named `message.payload.size`, so this had been skipped silently.
    ("log: the recorded header is not the one that was sent",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "        runCatching { record(Json.encode(Framing.withPayloadLength(header, payloadSize))) }",
     '        runCatching { record("{}") }', "transport"),
    ("telemetry: the api-30 guard is removed",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalReader.kt",
     "        if (sdkInt < android.os.Build.VERSION_CODES.R) return Headroom(null, REASON_API_TOO_OLD)",
     "", "app"),
    ("telemetry: a NaN headroom reaches the wire",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalReader.kt",
     "        if (!value.isFinite()) return Headroom(null, REASON_NOT_A_NUMBER)", "", "app"),
    ("telemetry: achieved is a raw delta rather than a rate",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/TelemetryReporter.kt",
     "            if (delta <= 0L) 0.0 else delta * 1e9 / elapsedNs", "            delta.toDouble()", "app"),
    ("advisory: one arriving after the stop is taken",
     "phone/app/src/main/kotlin/com/dsrc/phone/ui/AdvisoryHolder.kt",
     "        if (!accepting) {", "        if (false) {", "app"),
    ("advisory: a stale one stays on the display",
     "phone/app/src/main/kotlin/com/dsrc/phone/ui/AdvisoryHolder.kt",
     "        if (nowNs - arrivedAtNs > maxAgeNs) {", "        if (false) {", "app"),
    ("advisory: the age is taken from the sender's capture stamp",
     "phone/app/src/main/kotlin/com/dsrc/phone/ui/AdvisoryHolder.kt",
     "        arrivedAtNs = nowNs", "        arrivedAtNs = advisory.captureMonoNs", "app"),
    ("here: the api key reaches the recorded url",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/HereClient.kt",
     '                if (key != null) add("apiKey=${encode(key)}")',
     '                add("apiKey=${encode(key ?: "")}")', "app"),
    ("here: a call is made before a query is commanded",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/HerePipeline.kt",
     "                client == null || query == null -> {", "                false -> {", "app"),
    ("here: the recorded position becomes zeros",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/HerePipeline.kt",
     "            queryLat = target.lat,", "            queryLat = 0.0,", "app"),
    ("imu: the gyro stream is not gated on the timebase",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "        if (!checkTimebase(captureNs, appNowNs, monoNowNs)) {\n            refusedWrongTimebase.incrementAndGet()\n            return\n        }",
     "", "app"),
    ("imu: the delivery bound moves",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "        const val MAX_PLAUSIBLE_DELIVERY_NS = 2_000_000_000L",
     "        const val MAX_PLAUSIBLE_DELIVERY_NS = 1_000_000_000L", "app"),
    # Split in two. `checkTimebase` is called from both onGyro and onAccelerometer, and the
    # single anchor matched both -- so this mutated whichever came first (the gyro) and the
    # accelerometer's gate had no pin of its own at all.
    ("imu: the gyro timebase gate is skipped entirely",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "        if (!checkTimebase(captureNs, appNowNs, monoNowNs)) {\n"
     "            refusedWrongTimebase.incrementAndGet()\n"
     "            return\n"
     "        }",
     "        if (false) {\n"
     "            refusedWrongTimebase.incrementAndGet()\n"
     "            return\n"
     "        }",
     "app"),
    ("imu: the accelerometer timebase gate is skipped entirely",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "            return ImuOutcome.WrongTimebase\n        }",
     "            @Suppress(\"UNREACHABLE_CODE\")\n            if (false) return ImuOutcome.WrongTimebase\n        }",
     "app"),
    ("imu: the moot branch uses the delivery bound again",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "if (clockGapNs <= MAX_TOLERABLE_CLOCK_GAP_NS) {",
     "if (clockGapNs <= maxDeliveryNs) {",
     "app"),
    ("imu: the first gyro reading is kept, not the latest",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "        gyroNs = captureNs\n        hasGyro = true",
     "        if (!hasGyro) gyroNs = captureNs\n        hasGyro = true",
     "app"),
    ("imu: a negative gyro age keeps its sign",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "                gyroAgeNs = if (age < 0) -age else age,",
     "                gyroAgeNs = age,",
     "app"),
    ("imu: the monotonic baseline advances only on accepted",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPipeline.kt",
     "        val previous = lastCaptureNs.getAndSet(reading.captureMonoNs)",
     "        val previous = lastCaptureNs.get()",
     "app"),
    ("frames: json accepts bare NaN again",
     "deployment/jetson/transport/frames.py",
     "            parse_constant=_reject_json_constant,",
     "",
     "python"),
    ("messages: require_str loses its null clause",
     "deployment/jetson/transport/messages.py",
     '    if value is None:\n        raise MessageError(f"{field} must not be null", REASON_NULL_NOT_ALLOWED)\n    if not isinstance(value, str):',
     '    if not isinstance(value, str):',
     "python"),
    # Task 30. The first two are the round-1 and round-2 defects themselves: both
    # shipped, both were signed off by a test that named the behaviour, and both
    # made a live drive's log read as a pure-shadow one -- which is the reading
    # task 35 scores from.
    ("shadow: reference_rates_hold reads the flip it came FROM",
     "deployment/jetson/policy/shadow_mode.py",
     '"reference_rates_hold": not self._ever_live,',
     '"reference_rates_hold": not any(f.was == LIVE for f in self._flips),',
     "python"),
    ("shadow: the absent list is emitted unconditionally",
     "deployment/jetson/policy/shadow_mode.py",
     '                "structurally_absent":\n                    [] if self._born_live else list(ABSENT_IN_PURE_SHADOW),',
     '                "structurally_absent": list(ABSENT_IN_PURE_SHADOW),',
     "python"),
    ("shadow: a holder constructed live does not count as ever live",
     "deployment/jetson/policy/shadow_mode.py",
     "self._ever_live = mode == LIVE",
     "self._ever_live = False",
     "python"),
    # Mutated to a no-op assignment rather than deleted. Deleting the two lines
    # orphaned the nested `_feed_possible_from` guard beneath them, so the module
    # failed to import and the "catch" was the Python parser -- the same thing the
    # BUILD_ERROR path exists to stop on the Kotlin side, and the harness said so by
    # naming a module instead of a test as the catcher.
    ("shadow: flipping INTO live is not recorded",
     "deployment/jetson/policy/shadow_mode.py",
     "                self._ever_live = True",
     "                self._ever_live = self._ever_live",
     "python"),
    # The list is a claim about the controller's inputs. Adding one that a shadow
    # drive plainly HAS -- the camera runs at reference rates precisely because
    # nothing is applied -- passed the check that compared it to itself.
    ("shadow: an input that IS present is declared absent",
     "deployment/jetson/policy/shadow_mode.py",
     'ABSENT_IN_PURE_SHADOW = (\n    "feed_congestion",',
     'ABSENT_IN_PURE_SHADOW = (\n    "camera_density_bin",\n    "feed_congestion",',
     "python"),
    # All three locks, because all three survived the test that was supposed to
    # cover them.
    #
    # This block used to carry a note excluding the append/assignment reorder as an
    # equivalent mutant, on the grounds that "`was` is identical either way". That was
    # true only of the particular rewrite I tested, which captured `was` into a local
    # first. The natural reorder does not: it reads `self._mode` after the assignment
    # and writes `was == now` on every flip, destroying the only field that says which
    # side the drive started on. It is below, as an entry.
    ("shadow: the mode getter drops its lock",
     "deployment/jetson/policy/shadow_mode.py",
     "    @property\n    def mode(self) -> str:\n        with self._lock:\n            return self._mode",
     "    @property\n    def mode(self) -> str:\n        return self._mode",
     "python"),
    ("shadow: to_record drops its lock",
     "deployment/jetson/policy/shadow_mode.py",
     "    def to_record(self) -> dict[str, Any]:\n        with self._lock:\n            return {",
     "    def to_record(self) -> dict[str, Any]:\n        if True:\n            return {",
     "python"),
    ("shadow: flip_to drops its lock",
     "deployment/jetson/policy/shadow_mode.py",
     "        with self._lock:\n            if mode == self._mode:",
     "        if True:\n            if mode == self._mode:",
     "python"),
    ("shadow: a flip records the mode it went TO",
     "deployment/jetson/policy/shadow_mode.py",
     "            self._flips.append(Flip(at_mono=self._now(), was=self._mode, now=mode))\n            self._mode = mode",
     "            self._mode = mode\n            self._flips.append(Flip(at_mono=self._now(), was=self._mode, now=mode))",
     "python"),
    # The mixed drive. `[]` for a drive promoted mid-way says nothing is missing from
    # a log whose leading segment had no feed at all, which is the unsafe direction:
    # task 35 would credit a policy on a rule that could not have fired there.
    ("shadow: a mid-drive promotion reports nothing absent",
     "deployment/jetson/policy/shadow_mode.py",
     "[] if self._born_live else list(ABSENT_IN_PURE_SHADOW),",
     "[] if self._ever_live else list(ABSENT_IN_PURE_SHADOW),",
     "python"),
    ("shadow: the feed boundary re-dates on every promotion",
     "deployment/jetson/policy/shadow_mode.py",
     "                if self._feed_possible_from is None:\n                    self._feed_possible_from = self._flips[-1].at_mono",
     "                self._feed_possible_from = self._flips[-1].at_mono",
     "python"),
    # The query is the sole mechanism by which the feed comes to exist, so a
    # `command_for` that drops it produces a pure-shadow log in LIVE mode. The test
    # that looked like it covered this compared two references to one object.
    ("shadow: command_for drops the here query",
     "deployment/jetson/policy/shadow_mode.py",
     "        here=decision.here_query,",
     "        here=None,",
     "python"),
    ("shadow: command_for drops the capture stamp",
     "deployment/jetson/policy/shadow_mode.py",
     "        t_capture_mono_ns=t_capture_mono_ns,",
     "        t_capture_mono_ns=0,",
     "python"),
    ("shadow: is_live always says no",
     "deployment/jetson/policy/shadow_mode.py",
     "        return self.mode == LIVE",
     "        return False",
     "python"),
    # Task 31. The first two are the same defect from both ends: a redial that
    # reconnected the link to objects the run was not holding, and a camera that
    # ended the drive 5 ms after the phone hung up, before the redial was looked for.
    ("link: a rebind builds new sensors instead of rebinding the held ones",
     "deployment/jetson/sensors/phone_link.py",
     "            if not all((self.camera.rebind(self.router, self.adapter),\n                        self.gps.rebind(self.router, self.adapter))):\n                return False",
     "            self.camera = PhoneCameraStream(self.router, self.adapter).start()\n            self.gps = PhoneGpsReader(self.router, self.adapter).start()",
     "python"),
    ("link: a lost session ends the stream even though a redial is expected",
     "deployment/jetson/sensors/phone_link.py",
     "        self.camera.expect_redial(True)",
     "        self.camera.expect_redial(False)",
     "python"),
    # A new device inherits nothing the old one said.
    ("link: the new phone inherits the old phone's temperature",
     "deployment/jetson/sensors/phone_link.py",
     "            self._telemetry = None",
     "            pass",
     "python"),
    ("link: the new phone inherits the old phone's road",
     "deployment/jetson/sensors/phone_link.py",
     "            self.here = HereFeed()\n            self.here_failure = None",
     "            pass",
     "python"),
    ("link: the finished session's record is overwritten",
     "deployment/jetson/sensors/phone_link.py",
     "            self.sessions.append(self._session_record())",
     "            pass",
     "python"),
    ("link: giving up on the redial is not recorded",
     "deployment/jetson/sensors/phone_link.py",
     '                self.supervisor_ended = self.supervisor_ended or (\n                    "stopped" if self._stop.is_set()\n                    else f"gave_up_after_{self.rebind_timeout_s:g}s"\n                )',
     "                pass",
     "python"),
    ("link: a supervisor is started per session",
     "deployment/jetson/sensors/phone_link.py",
     "        if self._supervisor is None:",
     "        if True:",
     "python"),
    # The loop. Each of these left the suite green while the module's central
    # property -- a command does not go down every tick -- failed on a real drive.
    ("loop: a missing query counts as a move",
     "deployment/jetson/policy/sensing_loop.py",
     "        if query is None or self._last_query is None:\n            # Gaining or losing a fix changes the command's content, so it is already\n            # covered by \"changed\" -- and calling it a move would report a distance\n            # from a position that does not exist.\n            return False",
     "        if query is None or self._last_query is None:\n            return True",
     "python"),
    ("loop: the feed never reaches the controller",
     "deployment/jetson/policy/sensing_loop.py",
     "        feed_congestion=None if feed is None else feed.downstream_congestion,",
     "        feed_congestion=None,",
     "python"),
    ("loop: the heartbeat fires one tick late, forever",
     "deployment/jetson/policy/sensing_loop.py",
     "if self._last_sent_at is not None and now - self._last_sent_at >= self.heartbeat_s:",
     "if self._last_sent_at is not None and now - self._last_sent_at > self.heartbeat_s:",
     "python"),
    # Task 31 round 2. The first is the round-1 fix's own defect: preserving the
    # camera's identity across a rebind, which is what lets the run survive a
    # redial, carried the previous phone's frame-id high-water mark with it.
    # Two entries, not one. Written as a single two-line anchor it could only be
    # deleted whole -- and deleting just `_latest = None` restores the defect in
    # full, because the stale frame is re-served and raises the mark right back.
    # A pin whose granularity is coarser than the defect is not a pin.
    ("source: a rebind keeps the previous phone's frame high-water mark",
     "deployment/jetson/sensors/phone_source.py",
     "            self._last_consumed_id = -1",
     "            pass",
     "python"),
    ("source: a rebind re-serves the previous phone's last frame",
     "deployment/jetson/sensors/phone_source.py",
     "            self._latest = None\n",
     "",
     "python"),
    ("source: a rebind keeps the previous phone's decode failures",
     "deployment/jetson/sensors/phone_source.py",
     "        self._drop_counter = 0\n        self.decode_failures = 0",
     "        self._drop_counter = 0",
     "python"),
    ("source: a rebind keeps the previous session's dropped frames",
     "deployment/jetson/sensors/phone_source.py",
     "        self._drop_counter = 0\n        self.decode_failures = 0",
     "        self.decode_failures = 0",
     "python"),
    # The gps reader had no _on_rebound at all -- the camera's defect in the sensor
    # the observation, the HERE query and the V2V beacon are all derived from.
    ("source: the new phone inherits the old phone's position",
     "deployment/jetson/sensors/phone_source.py",
     "        with self._lock:\n            self._fix = GpsFix()",
     "        pass",
     "python"),
    ("source: the new phone inherits the old phone's parse count",
     "deployment/jetson/sensors/phone_source.py",
     "        self.diagnostics.sentences_parsed = 0\n        self.diagnostics.last_error = None",
     "        pass",
     "python"),
    # A sensor that refused to rebind must not look rebound.
    ("link: a failed rebind is recorded as a clean redial",
     "deployment/jetson/sensors/phone_link.py",
     "            if not all((self.camera.rebind(self.router, self.adapter),\n                        self.gps.rebind(self.router, self.adapter))):\n                return False",
     "            self.camera.rebind(self.router, self.adapter)\n            self.gps.rebind(self.router, self.adapter)",
     "python"),
    ("link: pings and telemetry are counted across sessions",
     "deployment/jetson/sensors/phone_link.py",
     "            self.telemetry_received = 0\n            self.pings_answered = 0",
     "            pass",
     "python"),
    ("link: a clean stop does not say the supervisor stopped",
     "deployment/jetson/sensors/phone_link.py",
     '        self.supervisor_ended = self.supervisor_ended or "stopped"',
     "        pass",
     "python"),
    ("source: a rebind keeps the previous phone's message count",
     "deployment/jetson/sensors/phone_source.py",
     '"""Forget what belonged to the previous peer. Default: the message count."""\n        self.messages_received = 0',
     '"""Forget what belonged to the previous peer. Default: the message count."""',
     "python"),
    # The feed had no route into the pipeline at all, so the entire HERE ingestion
    # path terminated in a log record and DISAGREEMENT was dead on every drive.
    ("pipeline: the traffic reading never reaches the observation builder",
     "deployment/jetson/pipeline.py",
     "            vehicles, gps, time.monotonic(), peers, feed",
     "            vehicles, gps, time.monotonic(), peers",
     "python"),
    # Deliberately absent: the rebind-entry ordering, and the worker's `finally:
    # stop.set()`. Reverting the first makes its test FLAKY rather than failing --
    # a pin that reports SURVIVED on the runs where the race does not land is worse
    # than none. The second is inside a closure in `run_live` with no test harness;
    # `test_no_undefined_names` covers the NameError class and nothing covers this.
    # Joint round 1. Both sit on a SEAM, which is why three rounds of per-task
    # validation each missed them: the producer's contract and the consumer's
    # assumption were each self-consistent.
    #
    # `at()` answers from geometry against the position it is handed, so an old fix
    # relocates the answer rather than degrading it -- while every other consumer of
    # that same fix refuses it. Three entries, because the gate has three halves and
    # a coarse anchor is how a pin comes to be satisfied by the defect it names.
    ("here: the fix's own age is not checked at all",
     "deployment/jetson/sensors/here_feed.py",
     "        if (fix_age_s + fix_bound_s > self._max_fix_age_s\n                or fix_age_s < -fix_bound_s):",
     "        if False:",
     "python"),
    ("here: a fix from this clock's future is fresh",
     "deployment/jetson/sensors/here_feed.py",
     "                or fix_age_s < -fix_bound_s):",
     "                or False):",
     "python"),
    ("here: the timebase bound is not charged against the fix age",
     "deployment/jetson/sensors/here_feed.py",
     "        if gps.timebase is not None and gps.timebase.bound_s is not None:\n            fix_bound_s = float(gps.timebase.bound_s)",
     "        pass",
     "python"),
    ("here: a stale fix is reported as an absent one",
     "deployment/jetson/sensors/here_feed.py",
     "            return FlowReading(outcome=Outcome.STALE_FIX, **provenance,\n                               detail=f\"fix is {fix_age_s:.1f}s old\")",
     "            return FlowReading(outcome=Outcome.UNUSABLE_FIX, **provenance,\n                               detail=f\"fix is {fix_age_s:.1f}s old\")",
     "python"),
    # A gap in the tick stream credited as dwell time. Nothing resets the controller
    # on a redial, and the worker stops calling it for as long as the rebind takes.
    ("controller: a stall between two evidence ticks counts as dwell",
     "deployment/jetson/policy/sensing_controller.py",
     "            if self._raised_since is None or gapped:",
     "            if self._raised_since is None:",
     "python"),
    ("controller: the evidence gap is bounded by the dwell instead of the cadence",
     "deployment/jetson/policy/sensing_controller.py",
     'MAX_EVIDENCE_GAP_S = 2.0 / (IDLE_RATES["camera_hz"] * min(THERMAL_SCALE.values()))',
     "MAX_EVIDENCE_GAP_S = RAISE_DWELL_S",
     "python"),
    # Joint round 3: the failure paths. Every one of these left a broken run writing
    # a record indistinguishable from a working one.
    ("run_demo: a drive whose camera stops producing never ends",
     "deployment/jetson/run_demo.py",
     "            if deadline and time.monotonic() >= deadline:\n                break\n            frame = camera.wait_for_fresh(timeout=1.0)",
     "            frame = camera.wait_for_fresh(timeout=1.0)",
     "python"),
    ("link: messages the channel threw away are not in the record",
     "deployment/jetson/sensors/phone_link.py",
     '                    "dropped_outbound": stats.dropped_outbound,',
     '                    "dropped_outbound": 0,',
     "python"),
    ("link: the decoder's refusals are not in the record",
     "deployment/jetson/sensors/phone_link.py",
     '            record["messages"] = self.router.to_record()',
     "            pass",
     "python"),
    ("link: nothing drains the imu channel",
     "deployment/jetson/sensors/phone_link.py",
     "            self._imu = (message, receipt.t_recv_mono_ns / 1e9)\n            self.imu_received += 1",
     "            pass",
     "python"),
    ("link: the imu record implies something consumes it",
     "deployment/jetson/sensors/phone_link.py",
     '                "feeds_the_controller": False,',
     '                "feeds_the_controller": True,',
     "python"),
    # Joint round 4: concurrency. Two senders on one channel is the NORMAL
    # arrangement here -- the heartbeat timer and the ping responder share CONTROL.
    ("session: the sequence is drawn outside the lock that orders the queue",
     "deployment/jetson/transport/session.py",
     "        with self._send_order[channel]:",
     "        if True:",
     "python"),
    # A reader that outlived its join does not linger, it follows the rebind onto the
    # next session: two readers on one channel, compounding per redial.
    ("link: a rebind proceeds over a reader that would not stop",
     "deployment/jetson/sensors/phone_link.py",
     "                if worker.is_alive():\n                    stopped = False",
     "                pass",
     "python"),
    ("link: the refusal to rebind over a live reader is not recorded",
     "deployment/jetson/sensors/phone_link.py",
     '                self.supervisor_ended = "readers_would_not_stop"',
     "                pass",
     "python"),
    # The report and its arrival time as two stores. A read between them yields a
    # status with a None age, and the controller's staleness gate SKIPS a None.
    ("link: the telemetry report and its age can be read apart",
     "deployment/jetson/sensors/phone_link.py",
     "    @property\n    def telemetry_at_mono(self) -> float | None:\n        \"\"\"When that report arrived. Never None while `telemetry` is not.\"\"\"\n        held = self._telemetry\n        return None if held is None else held[1]",
     "    @property\n    def telemetry_at_mono(self) -> float | None:\n        return None",
     "python"),
    # Joint round 5: the numeric core. Both geometry entries are behavioural --
    # the feed answered about the wrong piece of road, or refused the right one.
    ("here: distance is measured to the vertices, not to the shape",
     "deployment/jetson/sensors/here_feed.py",
     "        for start, end in zip(self.points, self.points[1:]):\n            best = min(best, point_to_segment_m(lat, lon, *start, *end))",
     "        pass",
     "python"),
    ("here: the link ahead is chosen by distance alone, ignoring the corridor",
     "deployment/jetson/sensors/here_feed.py",
     "        _, distance, offset_deg, nearest = min(ahead, key=lambda c: (c[0], c[1]))",
     "        _, distance, offset_deg, nearest = min(ahead, key=lambda c: c[1])",
     "python"),
    ("geo: the segment projection is not clamped to the segment",
     "deployment/jetson/geo.py",
     "    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)",
     "    t = t",
     "python"),
    # The provenance was read off the sample count, which cannot see the guard below
    # it, so a neutral fallback was tagged `derived` on half the ticks of a brake.
    ("builder: the acceleration provenance ignores the window guard",
     "deployment/jetson/perception/observation_builder.py",
     '        src["ego_acceleration"] = (\n'
     '            provenance.SOURCE_DERIVED if accel_derived else provenance.SOURCE_FALLBACK_NEUTRAL\n'
     '        )',
     '        src["ego_acceleration"] = provenance.SOURCE_DERIVED',
     "python"),
    ("builder: a window too short to fit a slope is reported as derived",
     "deployment/jetson/perception/observation_builder.py",
     "        if t[-1] - t[0] < 0.3:\n            return 0.0, False",
     "        if t[-1] - t[0] < 0.3:\n            return 0.0, True",
     "python"),
    # Braking. Every acceleration in the controller's tests was positive.
    # Task 34 moved this comparison out of `decide` and into the `RuleCheck` it
    # builds for `Trigger.EVENT`; the anchor follows it there. Still the same
    # property: dropping `abs()` reads only acceleration as an event and stops
    # braking from ever firing it.
    ("controller: only acceleration is an event, not braking",
     "deployment/jetson/policy/sensing_controller.py",
     "            event_fired = abs(value) >= EVENT_ACCEL_MPS2",
     "            event_fired = value >= EVENT_ACCEL_MPS2",
     "python"),
    # The published bound, as opposed to the field the estimator computed.
    ("timebase: the published one-way bound is halved on the way out",
     "deployment/jetson/transport/timebase.py",
     "            bound_ns=estimate.rtt_min_ns + drift_ns,",
     "            bound_ns=estimate.rtt_min_ns // 2 + drift_ns,",
     "python"),
    # Both halves of the builder's freshness predicate.
    ("builder: a bound too wide to resolve no longer refuses the fix",
     "deployment/jetson/perception/observation_builder.py",
     "        timebase_unresolved = uncertainty_s > cfg.gps_stale_after_s * cfg.max_bound_fraction",
     "        timebase_unresolved = False",
     "python"),
    ("builder: the bound is not charged against the staleness window",
     "deployment/jetson/perception/observation_builder.py",
     "            and gps_age + uncertainty_s <= cfg.gps_stale_after_s",
     "            and gps_age <= cfg.gps_stale_after_s",
     "python"),
    # Joint round 6: what the phone does with a command, and what reads the logs.
    # Forces the UNMEASURABLE branch (`shortfall is None`, nothing to
    # compare against) to read complete. That is a different fact from a
    # genuinely truncated log, which has its own pin immediately below on
    # the `False` branch.
    ("eval: an unmeasurable log is analysed as a complete run",
     "deployment/jetson/eval_run.py",
     "        log_complete = None",
     "        log_complete = True",
     "python"),
    ("eval: a truncated log is analysed as a complete run",
     "deployment/jetson/eval_run.py",
     "        log_complete = False",
     "        log_complete = True",
     "python"),
    ("eval: unparseable lines are skipped without being counted",
     "deployment/jetson/eval_run.py",
     "                unparseable += 1\n                continue",
     "                continue",
     "python"),
    # Two pins on one line, not one: `overall_pass` folds in two
    # independent terms (`log_complete` and `coverage_untrustworthy`), and a
    # single pin could not tell one of them ceasing to matter apart from the
    # other. Each drops only its own term and leaves the rest in place.
    ("eval: a short log does not fail the run",
     "deployment/jetson/eval_run.py",
     '        "overall_pass": bool(overall and integrity["log_complete"] and not coverage_untrustworthy),',
     '        "overall_pass": bool(overall and not coverage_untrustworthy),',
     "python"),
    ("eval: an untrustworthy tick coverage does not fail the run",
     "deployment/jetson/eval_run.py",
     '        "overall_pass": bool(overall and integrity["log_complete"] and not coverage_untrustworthy),',
     '        "overall_pass": bool(overall and integrity["log_complete"]),',
     "python"),
    # `analyze()` reads `tick_id` through `.get` so a tick without one
    # degrades. A direct subscript raises three hundred lines before
    # `_tick_coverage`'s own decline for a missing `tick_id` is reached,
    # which puts that decline out of reach on every drive and fixture.
    ("eval_run: a tick with no tick_id crashes analyze() instead of degrading",
     "deployment/jetson/eval_run.py",
     '        tick_id = t.get("tick_id")',
     '        tick_id = t["tick_id"]',
     "python"),
    # Two of `_tick_id_trust_reason`'s three checks. The third
    # (`if b <= a: ... "a restart"`) is deliberately left unpinned: removing
    # it is an equivalent mutant, since the duplicate check already returns
    # on `b == a` and no input distinguishes the two.
    ("eval_run: a null tick_id reaches the ordering check instead of declining",
     "deployment/jetson/eval_run.py",
     "    if any(i is None for i in ids):\n"
     '        return "some ticks carry no tick_id"',
     "    if False:\n"
     '        return "some ticks carry no tick_id"',
     "python"),
    ("eval_run: a repeated tick_id declines with the restart's reason instead of its own",
     "deployment/jetson/eval_run.py",
     "    if len(set(ids)) != len(ids):\n"
     '        return "tick_id repeats at least once"',
     "    if False:\n"
     '        return "tick_id repeats at least once"',
     "python"),
    ("eval: the log is not compared against the count the run reported",
     "deployment/jetson/eval_run.py",
     "        shortfall = expected_ticks - len(ticks)",
     "        shortfall = 0",
     "python"),
    # Joint round 7: claims against behaviour, and what grows over a long drive.
    # The tick log is the only per-tick collection with no cap -- 5,251 bytes a
    # record, 567 MB/hour at 30 fps -- and its writer had no guard at all.
    # The writer's catch is deliberately `BaseException`. Narrowing it back
    # to the two anticipated types is what kills the writer silently on
    # anything else, which is the defect this pin names.
    ("logger: a failed write kills the writer silently",
     "deployment/jetson/logio/metadata_logger.py",
     "            except BaseException as exc:",
     "            except (OSError, ValueError) as exc:",
     "python"),
    # The record already popped off the queue when the write raised is not
    # `close()`'s own `qsize()` to find -- by the time `close()` runs it is
    # no longer queued -- so only this pin covers it.
    ("logger: a record already popped off the queue when the write fails is not counted",
     "deployment/jetson/logio/metadata_logger.py",
     "                self.writer_failure = f\"{type(exc).__name__}: {exc}\"\n"
     "                self.dropped_records += 1\n"
     "                return",
     "                self.writer_failure = f\"{type(exc).__name__}: {exc}\"\n"
     "                return",
     "python"),
    ("logger: close blocks on a queue nothing is draining",
     "deployment/jetson/logio/metadata_logger.py",
     "        if self._thread.is_alive():\n            try:\n                self._queue.put(None, timeout=2.0)",
     "        if True:\n            try:\n                self._queue.put(None)",
     "python"),
    ("logger: records still queued when the writer stopped are not counted",
     "deployment/jetson/logio/metadata_logger.py",
     "        self.dropped_records += self._queue.qsize()",
     "        pass",
     "python"),
    # The term without which the inbound account cannot balance.
    ("link: the record cannot say whether a message was dropped or uncollected",
     "deployment/jetson/sensors/phone_link.py",
     '                    "abandoned_inbound": stats.abandoned_inbound,',
     "",
     "python"),
    ("link: the leaked handshake workers are absent from the summary",
     "deployment/jetson/sensors/phone_link.py",
     '            "handshake_workers_leaked": self._listener.handshake_workers_leaked,',
     "",
     "python"),
    ("link: the refusal list grows without bound",
     "deployment/jetson/sensors/phone_link.py",
     "        if len(self.refusals) < self.MAX_REFUSALS:",
     "        if True:",
     "python"),
    # Joint round 8: the sim contract and the V2V beacon, the last two untouched
    # areas. The beacon had no test in this directory at all.
    ("beacon: a stale fix is broadcast as a current position",
     "deployment/jetson/v2v/beacon.py",
     "            if fix is not None and fix.valid and not self._fix_is_stale(fix):",
     "            if fix is not None and fix.valid:",
     "python"),
    ("beacon: a fix from this clock's future is treated as fresh",
     "deployment/jetson/v2v/beacon.py",
     "        return abs(time.monotonic() - fix.t_mono) > self.max_fix_age_s",
     "        return (time.monotonic() - fix.t_mono) > self.max_fix_age_s",
     "python"),
    ("beacon: peers are ranged against a stale ego fix",
     "deployment/jetson/v2v/beacon.py",
     "        if ego is None or not ego.valid or self._fix_is_stale(ego):",
     "        if ego is None or not ego.valid:",
     "python"),
    ("beacon: expired peers survive when our own fix is gone",
     "deployment/jetson/v2v/beacon.py",
     "        self._expire_peers(now)",
     "        pass",
     "python"),
    # The observation vector against the distribution the policy was trained on.
    ("builder: the queue is reported as derived over an empty population",
     "deployment/jetson/perception/observation_builder.py",
     '            "local_queue_estimate": (\n'
     '                provenance.SOURCE_DERIVED if abs_speeds\n'
     '                else provenance.SOURCE_DERIVED_EMPTY if not in_range\n'
     '                else provenance.SOURCE_FALLBACK_NEUTRAL\n'
     '            ),',
     '            "local_queue_estimate": provenance.SOURCE_DERIVED,',
     "python"),
    ("builder: the etiquette flag claims a derived segment density",
     "deployment/jetson/perception/observation_builder.py",
     '            "uncongested_low_speed_flag": provenance.SOURCE_APPROXIMATED,',
     '            "uncongested_low_speed_flag": provenance.SOURCE_DERIVED,',
     "python"),
    ("builder: the av density divides by a literal instead of the admission range",
     "deployment/jetson/perception/observation_builder.py",
     "            av_density = av_count / max((2.0 * cfg.peer_range_m) / 1000.0, 1e-9)",
     "            av_density = av_count / max((2.0 * 150.0) / 1000.0, 1e-9)",
     "python"),
    # A reorder leaves the dimension at 39 and puts every value in the wrong slot.
    ("actor: the bundle is guarded on its dimension alone",
     "deployment/jetson/policy/actor_runtime.py",
     "        if exported is not None and exported != current:",
     "        if False:",
     "python"),
    # Task 32 round 1. The record has to distinguish a run whose bytes crossed the
    # tailnet from one whose bytes crossed USB, and it did not.
    ("tailnet: a loopback session is reported as having crossed the tailnet",
     "deployment/jetson/tailnet.py",
     "        if host in (peer.get(\"addresses\") or []):",
     "        if True:",
     "python"),
    ("tailnet: an unavailable status still claims a path",
     "deployment/jetson/tailnet.py",
     '    if not status.get("available"):',
     "    if False:",
     "python"),
    # The anchor here was the old one-line disambiguation and it stopped matching when
    # that became a loop, so the harness SKIPPED it for several rounds. A skipped entry
    # prints differently from a caught one and is not a pin either way.
    ("tailnet: two peers sharing a hostname collapse into one",
     "deployment/jetson/tailnet.py",
     "            name = candidate",
     "            pass",
     "python"),
    ("tailnet: a hostname collision is not recorded",
     "deployment/jetson/tailnet.py",
     "            collisions.append(name)",
     "            pass",
     "python"),
    ("link: the session's own peer address is dropped from the record",
     "deployment/jetson/sensors/phone_link.py",
     '            record["peer"] = stats.peer',
     "            pass",
     "python"),
    # Task 32 round 2. Four of the six were in round 1's own fix.
    ("tailnet: a relay region is attributed to a direct connection",
     "deployment/jetson/tailnet.py",
     '                "relay": peer.get("relay") if path == "relay" else "",',
     '                "relay": peer.get("relay"),',
     "python"),
    ("tailnet: a tailnet address whose peer went offline reads as a usb run",
     "deployment/jetson/tailnet.py",
     "    known = (status.get(\"known_addresses\") or {}).get(host)",
     "    known = None",
     "python"),
    # Replaced the two entries that mutated `_in_tailnet_range`. That function is gone:
    # membership is answered from the peer list, because 100.64.0.0/10 is the shared
    # CGNAT block rather than a Tailscale allocation, so a range test called every
    # address in it a peer -- including 100.100.100.100, Tailscale's own resolver.
    # Restoring a range test is therefore the mutation worth making.
    ("tailnet: membership is guessed from the address instead of the peer list",
     "deployment/jetson/tailnet.py",
     "    known = (status.get(\"known_addresses\") or {}).get(host)",
     "    known = \"a peer\" if host.startswith(\"100.\") else None",
     "python"),
    # The loop that disambiguates two peers sharing one hostname. Not a wrong answer:
    # the pre-fix form does not return at all, in a function called from a run's
    # teardown, so the pinning test runs it on a thread with a deadline.
    ("tailnet: the disambiguating loop tests one name forever",
     "deployment/jetson/tailnet.py",
     "            attempt = 0\n            while candidate in peers:\n                attempt += 1\n                candidate = f\"{name} [{suffix} #{attempt}]\"",
     "            while candidate in peers:\n                candidate = f\"{name} [{suffix} #{len(peers)}]\"",
     "python"),
    # Deleted, not moved: `suffix = ... else "?"` is now an equivalent mutant. It lost a
    # peer only while the disambiguating loop could not advance; with the counter in
    # place, seven peers sharing one hostname and no addresses all survive under either
    # suffix (checked, rather than argued: 7 of 7 both ways, named `[#1]..[#6]` and
    # `[? #1]..[? #6]`). The property it was written for is pinned by the entry above,
    # which mutates the loop itself.
    ("tailnet: a collision is counted once per peer instead of once per name",
     "deployment/jetson/tailnet.py",
     "            if name not in collisions:\n                collisions.append(name)",
     "            collisions.append(name)",
     "python"),
    ("link: the record does not say which session it describes",
     "deployment/jetson/sensors/phone_link.py",
     '            record["session_id"] = stats.session_id',
     "            pass",
     "python"),
    # Having the field is not the same as the field being consistent. `to_record`
    # read `self.session` once per block, and the redial supervisor replaces it
    # between reads, so one record could carry two session ids and two handsets'
    # counters with nothing in it to say which was which.
    # Anchored with the comment line above it. `_session_record` carries the same
    # call, so the bare line matches twice and an ambiguous anchor pins nothing.
    ("link: one record is assembled from two separate session reads",
     "deployment/jetson/sensors/phone_link.py",
     '            # What the transport did, as opposed to what the readers made of it.\n            "wire": self._wire_record(session),',
     '            # What the transport did, as opposed to what the readers made of it.\n            "wire": self._wire_record(),',
     "python"),
    # Found by reading the first real run's record rather than by a test: the control
    # channel showed 241 received against 121 delivered, with dropped and abandoned
    # both zero, on a run that lost nothing. The transport eats its own keepalives --
    # `_record_inbound` counts one in `received` and returns before the queue -- and
    # the record published every term of the inbound account except that one.
    ("link: a consumed heartbeat is left out of the inbound account",
     "deployment/jetson/sensors/phone_link.py",
     '            record["heartbeats_received"] = stats.heartbeats_received',
     '            record["heartbeats_received"] = 0',
     "python"),
    ("link: the rebind snapshot is assembled from two separate session reads",
     "deployment/jetson/sensors/phone_link.py",
     '            "imu_received": self.imu_received,\n            "wire": self._wire_record(session),',
     '            "imu_received": self.imu_received,\n            "wire": self._wire_record(),',
     "python"),
    # Found by running the Jetson's own selfcheck on the Jetson with its own
    # receiver attached, indoors. The reader thread ended on the first RMC sentence
    # carrying a time and no date, which is what a searching receiver sends.
    ("gps: the datetime property is read through a guard that cannot catch it",
     "deployment/jetson/sensors/gps_reader.py",
     "                stamp = None\n                if (getattr(msg, \"datestamp\", None) is not None\n                        and getattr(msg, \"timestamp\", None) is not None):\n                    try:\n                        stamp = msg.datetime\n                    except (TypeError, ValueError):\n                        stamp = None",
     '                stamp = getattr(msg, "datetime", None)',
     "python"),
    ("gps: one sentence the reader cannot handle ends the reader",
     "deployment/jetson/sensors/gps_reader.py",
     "                self.diagnostics.ingest_errors += 1",
     "                pass",
     "python"),
    # Task 29's rule, pinned from task 30's side: the whole "a shadow drive cannot
    # reach DISAGREEMENT" claim rests on this returning False for a missing feed.
    # The report table's own instance of this task's rule. `capture` is a point in
    # time carrying `ms: 0.0`; counting it as a duration put a stage that took no time
    # into the table on the first real run.
    ("eval: an instant is averaged in as a zero-length stage",
     "deployment/jetson/eval_run.py",
     '            if basis == "instant":\n                # A point in time carries `ms: 0.0` as a placeholder, not a duration of\n                # zero. Averaging it in reports a stage that took no time, which is the\n                # one thing this table exists to make impossible.\n                continue',
     "            if False:\n                continue",
     "python"),
    # Deleting the `continue` alone is a no-op: an absent entry's `ms` is None, so the
    # guard below skips it anyway. Two independent guards protect this, and a mutation
    # that removes one of them proves nothing -- so the mutation makes the absent branch
    # contribute the zero it exists to prevent.
    ("eval: an absent stage contributes a zero to the table",
     "deployment/jetson/eval_run.py",
     '                slot["absent_reasons"][why] = slot["absent_reasons"].get(why, 0) + 1\n                continue',
     '                slot["absent_reasons"][why] = slot["absent_reasons"].get(why, 0) + 1\n                slot["values"].append(0.0)\n                continue',
     "python"),
    ("controller: a missing feed counts as disagreement",
     "deployment/jetson/policy/sensing_controller.py",
     "    if feed_congestion is None or camera_density_bin is None:\n        return False",
     "    if feed_congestion is None or camera_density_bin is None:\n        return True",
     "python"),

    # Task 33 round 1. The offline join and the two estimators it reads.
    #
    # The live adapter and the offline join have to refuse the same estimates for
    # the same reasons, and the only way to check that is to persist the verdict
    # the live gate already computes -- both the round-trip and the one-way sides
    # of it, since only one of the two carried it before this round.
    ("timebase: a one-way estimate's usability is not persisted",
     "deployment/jetson/transport/timebase.py",
     '            "offset_samples": 0 if estimate is None else estimate.offset_samples,\n'
     '            "usable": reason is None,\n'
     '            "why_not_usable": reason,\n'
     '        }',
     '            "offset_samples": 0 if estimate is None else estimate.offset_samples,\n'
     '        }',
     "python"),
    ("run_demo: usable/why_not_usable are not read off the estimator",
     "deployment/jetson/run_demo.py",
     '            "usable": estimator.usable,\n            "why_not_usable": estimator.why_not_usable(),',
     '            "usable": True,\n            "why_not_usable": None,',
     "python"),
    ("eval: return converts against an estimate the live adapter would have refused",
     "deployment/jetson/eval_run.py",
     '        if e.get("source") == source\n'
     '        and e.get("session_id") == session_id\n'
     '        and _was_usable(e)\n'
     '    ]',
     '        if e.get("source") == source\n'
     '        and e.get("session_id") == session_id\n'
     '    ]',
     "python"),
    ("eval: an old log with too few samples is taken as usable",
     "deployment/jetson/eval_run.py",
     '    if record.get("offset_samples", 0) < MIN_OFFSET_SAMPLES:\n'
     '        return False',
     '    if record.get("offset_samples", 0) < MIN_OFFSET_SAMPLES:\n'
     '        pass',
     "python"),
    # `rtt_min_ns` is a property of the estimate, written into every persisted line --
    # unlike staleness, it survives in an old-format one just as well as a new one, so
    # the ceiling applies to both.
    ("eval: an old log round-trip line above the RTT ceiling is taken as usable",
     "deployment/jetson/eval_run.py",
     '    rtt = record.get("rtt_min_ns")\n'
     '    return rtt is None or rtt <= MAX_ACCEPTABLE_RTT_NS',
     '    return True',
     "python"),
    # The ceiling is a round-trip concept -- on a one-way line `rtt_min_ns` is a delay
    # spread, and `OneWayEstimator` has no ceiling clause on it at all.
    ("eval: the round-trip RTT ceiling is applied to a one-way line too",
     "deployment/jetson/eval_run.py",
     '    if record.get("source") != "round_trip":\n'
     '        return True',
     '    pass',
     "python"),
    # Both estimators are rebuilt whole on every redial, so `estimate_id` restarts
    # at 1 on the new session -- wall time alone cannot tell a stale estimate from
    # the previous peer apart from a current one, which is what `session_id` is for.
    ("eval: return converts against an estimate from a different session",
     "deployment/jetson/eval_run.py",
     '        if e.get("source") == source\n'
     '        and e.get("session_id") == session_id\n'
     '        and _was_usable(e)',
     '        if e.get("source") == source\n'
     '        and _was_usable(e)',
     "python"),
    ("run_demo: the session id is not carried onto the timebase estimate line",
     "deployment/jetson/run_demo.py",
     '            "session_id": session_id,',
     '            "session_id": None,',
     "python"),
    # `tick_session_id` is the one place `phone.session.session_id` is read per tick --
    # `run_live` used to read it twice, once for the tick record and once inside
    # `_log_timebase_estimates`, and a rebind between the two reads could put a
    # different session's id on each. Pinned on the helper directly, which needs none
    # of the detector/policy bundle/config/camera/GPS machinery a full run does; the
    # call sites in `run_live` just pass its one result on.
    ("run_demo: tick_session_id ignores which session the phone actually holds",
     "deployment/jetson/run_demo.py",
     "    if phone is None or phone.session is None:\n        return None\n    return phone.session.session_id",
     "    return None",
     "python"),
    # One helper, so `pipeline.Tick.to_record()` and `policy.sensing_loop` name the
    # same tick with the same integer. Each call site pinned separately: mutating
    # only one of the two away from the shared helper is exactly the way this broke
    # the first time, and a pin on the helper alone would not have caught it.
    ("time_sync: capture_stamp_ns truncates instead of rounding",
     "deployment/jetson/sensors/time_sync.py",
     "    return int(round(t_capture_mono * 1e9))",
     "    return int(t_capture_mono * 1e9)",
     "python"),
    ("pipeline: the tick record's capture stamp bypasses the shared conversion",
     "deployment/jetson/pipeline.py",
     '            "t_capture_mono_ns": capture_stamp_ns(self.t_capture_mono),',
     '            "t_capture_mono_ns": int(self.t_capture_mono * 1e9),',
     "python"),
    ("sensing_loop: the outbound capture stamp bypasses the shared conversion",
     "deployment/jetson/policy/sensing_loop.py",
     "        capture_ns = capture_stamp_ns(tick.t_capture_mono)",
     "        capture_ns = int(tick.t_capture_mono * 1e9)",
     "python"),
    # `AdvisoryHolder.accept` replaces `latest` unconditionally and counts nothing
    # about what it replaced, so "the advisory expired" was never the only
    # explanation for an absent render stage, and on a healthy link -- ticking
    # every frame against a 250 ms UI poll -- it was rarely even the likely one.
    ("eval: render's absent reason names a mechanism that did not occur",
     "deployment/jetson/eval_run.py",
     '    if shown_record is None:\n'
     '        return StageTiming.absent(\n'
     '            clock="phone", reason="no advisory_shown line for this capture stamp"\n'
     '        )',
     '    if shown_record is None:\n'
     '        return StageTiming.absent(\n'
     '            clock="phone", reason="advisory expired before current() returned it"\n'
     '        )',
     "python"),
    # `shown` and `expired` are not a partition of `received` -- an advisory `current()`
    # already returned once can still go on to age out, so a derived
    # received - shown - expired goes negative on that sequence. Counted at the moment of
    # replacement instead, in `accept`.
    ("advisory: superseded counts a replacement that had already been shown",
     "phone/app/src/main/kotlin/com/dsrc/phone/ui/AdvisoryHolder.kt",
     "        if (latest != null && shownAtNs == null) superseded++",
     "        if (latest != null) superseded++",
     "app"),
    # The direction and the sign of the one offline conversion nothing had pinned a
    # value for. A test that only checks `basis == "converted"` and that the fields
    # are non-None passes both of these just as well as the correct arithmetic.
    ("eval: the return conversion's sign is flipped",
     "deployment/jetson/eval_run.py",
     "        return_ms = (converted.t_remote_mono_ns - wire_ns) / 1e6",
     "        return_ms = (wire_ns - converted.t_remote_mono_ns) / 1e6",
     "python"),
    ("eval: return converts in the wrong direction",
     "deployment/jetson/eval_run.py",
     "        converted = estimate.convert_to_local(recv_ns)",
     "        converted = estimate.convert_to_remote(recv_ns)",
     "python"),
    # With the source filter gone, a round-trip request is answered by whichever
    # estimate is nearest in wall time regardless of source, and is still stamped
    # `source: "round_trip"` -- a one-way number, with a one-way bound, presented
    # as though it were bounded by half a round trip.
    ("eval: return's source filter is dropped",
     "deployment/jetson/eval_run.py",
     '    candidates = [\n'
     '        e for e in timebase_estimates\n'
     '        if e.get("source") == source\n'
     '        and e.get("session_id") == session_id\n'
     '        and _was_usable(e)\n'
     '    ]',
     '    candidates = [\n'
     '        e for e in timebase_estimates\n'
     '        if e.get("session_id") == session_id\n'
     '        and _was_usable(e)\n'
     '    ]',
     "python"),
    ("eval: the round-trip/one-way preference order is reversed",
     "deployment/jetson/eval_run.py",
     '    for source in ("round_trip", "one_way"):',
     '    for source in ("one_way", "round_trip"):',
     "python"),
    # No path reads `last_timings` before `build()` has run, so this pair is latent
    # rather than live -- but a caller that ever did read it early would get a
    # number indistinguishable from a real zero-length fuse instead of a missing
    # value, which is exactly the "cannot distinguish failure from success" defect
    # this whole log format exists to close.
    ("builder: fuse_ms starts at a placeholder zero instead of absent",
     "deployment/jetson/perception/observation_builder.py",
     "        self.last_timings: dict[str, float] = {}",
     '        self.last_timings: dict[str, float] = {"fuse_ms": 0.0}',
     "python"),
    ("pipeline: a missing fuse timing is reported as a measured zero",
     "deployment/jetson/pipeline.py",
     '        fuse_ms = self.builder.last_timings.get("fuse_ms")\n'
     '        stages["fuse"] = (\n'
     '            StageTiming.absent(clock="jetson", reason="builder recorded no fuse timing this tick")\n'
     '            if fuse_ms is None else StageTiming.measured(fuse_ms, clock="jetson")\n'
     '        )',
     '        stages["fuse"] = StageTiming.measured(\n'
     '            self.builder.last_timings.get("fuse_ms", 0.0), clock="jetson"\n'
     '        )',
     "python"),
    ("phone_source: a phone-side span with stamps out of order is reported as measured",
     "deployment/jetson/sensors/phone_source.py",
     '    if end_ns < start_ns:\n'
     '        return StageTiming.absent(clock="phone", reason=OUT_OF_ORDER_REASON)\n'
     '    return StageTiming.measured((end_ns - start_ns) / 1e6, clock="phone")',
     '    return StageTiming.measured((end_ns - start_ns) / 1e6, clock="phone")',
     "python"),
    # Deliberately absent: "run_phone_drive: the matched-a-real-frame count is
    # recomputed instead of using the stamp actually sent"
    # (`scripts/run_phone_drive.py:209`, `sent_stamps.append(int(tick.t_capture_mono *
    # 1e9))` -> the fixed `outcome.command.t_capture_mono_ns`).
    #
    # `scripts/` carries no test suite -- the lint gates exclude it and this harness
    # runs pytest against `deployment/jetson/tests/` only -- so a mutation there can
    # never be CAUGHT and would sit as a permanent, silent SURVIVED. That is a worse
    # record than no entry: it reads as a lapsed pin when it never was one. The script
    # is exercised by hand (its own docstring: "python3 scripts/run_phone_drive.py"),
    # and the fixed line reads the integer the router actually sent rather than
    # recomputing it, which is what removed the drift in the first place.
    ("pong: the carried receipt stamp is a fresh clock reading, not the reader's own",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "                recvMonoNs = message.recvMonoNs,",
     "                recvMonoNs = monoClock(),",
     "transport"),
    ("deliver: the delivered receipt stamps are fresh clock readings, not the reader's own",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "            onFrame(frame, message.recvMonoNs, message.recvWallNs)",
     "            onFrame(frame, monoClock(), wallClock())",
     "transport"),

    # Task 34. Attribution's three-state vocabulary, and the identity and thermal
    # bookkeeping it depends on.
    #
    # A rule missing its input used to be indistinguishable from one that ran and
    # found nothing -- the whole reason `not_evaluable` exists as its own status
    # rather than folding into `quiet`. Pinned on `disagreement`'s not_evaluable
    # return, the one already covered from the other side by the
    # "controller: a missing feed counts as disagreement" entry above.
    # The one rule whose not-evaluable branch had no assertion. Unreachable today --
    # the observation builder substitutes a neutral float before the controller looks --
    # but task 36's provenance work makes this the live path for a dead accelerometer,
    # and a `quiet` status beside a named missing input reads as "the sensor was read
    # and the road was calm".
    ("controller: the event rule reports quiet when it has no acceleration to read",
     "deployment/jetson/policy/sensing_controller.py",
     '                status=RULE_NOT_EVALUABLE, missing=("ego_acceleration",),',
     '                status=RULE_QUIET, missing=("ego_acceleration",),',
     "python"),
    ("controller: a missing rule's input reports quiet instead of not_evaluable",
     "deployment/jetson/policy/sensing_controller.py",
     "        return RuleCheck(status=RULE_NOT_EVALUABLE, missing=missing, evidence=evidence)",
     "        return RuleCheck(status=RULE_QUIET, missing=missing, evidence=evidence)",
     "python"),
    # `rules_fired` is supposed to be derived from the same `checks` dict the
    # attribution record carries, so the two cannot drift apart. Loosening the
    # filter to "not not_evaluable" still looks like a status check but starts
    # counting every quiet rule as fired the moment nothing in a tick is missing --
    # caught on a fully quiet decision, where the true list is empty.
    ("controller: rules_fired counts a quiet rule as fired",
     "deployment/jetson/policy/sensing_controller.py",
     "        fired = [rule for rule in RULES if checks[rule].status == RULE_FIRED]",
     "        fired = [rule for rule in RULES if checks[rule].status != RULE_NOT_EVALUABLE]",
     "python"),
    # A skin threshold that strictly lowers the scale below what the status alone
    # reached is supposed to claim the cause. Dropping the reassignment leaves the
    # scale correctly cut but the cause still naming the status -- a record that
    # states the right number for the wrong reason, on a phone whose status word
    # never moved while its skin warmed 5.4 C under load.
    ("controller: a skin-driven backoff still names the status as its cause",
     "deployment/jetson/policy/sensing_controller.py",
     '                if THERMAL_SCALE["severe"] < scale:\n'
     "                    cause = THERMAL_CAUSE_SKIN_HOT\n"
     '                scale = min(scale, THERMAL_SCALE["severe"])',
     '                scale = min(scale, THERMAL_SCALE["severe"])',
     "python"),

    # Task 34 round 2. The attribution record itself was unpinned: every test read
    # `decision.attribution` off the dataclass, and nothing asserted the shape
    # `Decision.to_record()` actually emits -- the one artifact a tick log carries
    # forward. These three close that.
    ("controller: the attribution record is dropped from Decision.to_record()",
     "deployment/jetson/policy/sensing_controller.py",
     # Anchored on the one line, not on the block that used to close the dict. A
     # field added after `attribution` moved the brace, so the three-line anchor
     # stopped matching and the entry printed SKIP while pinning nothing.
     '            "attribution": self.attribution.to_record(),\n',
     "",
     "python"),
    ("controller: the first decision is always reported as first",
     "deployment/jetson/policy/sensing_controller.py",
     "        first_decision = self._last is None",
     "        first_decision = True",
     "python"),
    ("controller: level_sensitive is always reported as false",
     "deployment/jetson/policy/sensing_controller.py",
     '                "level_sensitive": IDLE_RATES[key] != ACTIVE_RATES[key],',
     '                "level_sensitive": False,',
     "python"),
    # `rules` is transformed through `RuleCheck.to_record()` on the way out, unlike
    # `gates` and `per_sensor` which the record carries by reference -- so a filter
    # applied here changes what the record says without changing the object a test
    # might check instead.
    ("controller: quiet and not-evaluable rules are dropped from the emitted rules block",
     "deployment/jetson/policy/sensing_controller.py",
     '            "rules": {name: check.to_record() for name, check in self.rules.items()},',
     '            "rules": {name: check.to_record() for name, check in self.rules.items()'
     ' if check.status == RULE_FIRED},',
     "python"),
    ("controller: wants_more in the gates record is always reported as false",
     "deployment/jetson/policy/sensing_controller.py",
     '            "wants_more": wants_more,',
     '            "wants_more": False,',
     "python"),

    # Task 35. A shadow decision log that says what it can and cannot score --
    # the exact inputs a decision was made from, the instant it was made, and
    # a witnessed (not assumed) full-rate reference, plus the offline scorer
    # that replays a log and refuses before it misleads.
    #
    # `Inputs.to_record` is the replay substrate `score_shadow.py` rests on, and
    # the temptation this pins against is copying the evidence path's own
    # rounding (`RuleCheck.to_record`, sensing_controller.py:188) onto it. A
    # value rounded to four places can flip a threshold comparison a replayed
    # decision never actually made.
    ("controller: Inputs.to_record rounds policy_margin to four places",
     "deployment/jetson/policy/sensing_controller.py",
     '            "policy_margin": self.policy_margin,',
     '            "policy_margin": round(self.policy_margin, 4)'
     ' if isinstance(self.policy_margin, float) else self.policy_margin,',
     "python"),
    # `decided_at_mono` has to be the exact instant the gates above it compared,
    # not a nearby one -- dwell, hold, bridge and gap all compare differences of
    # it, so a replay fed anything else is a different drive at exactly the
    # ticks that straddle a boundary. This substitutes the previous decision's
    # instant, which is wrong in the same way a read taken earlier than the
    # controller's own is wrong: caught by the scripted mixed-drive replay
    # diverging on state that depends on elapsed time (dwell, hold, gap).
    ("controller: decided_at_mono is the previous decision's instant, not this one's",
     "deployment/jetson/policy/sensing_controller.py",
     "            decided_at_mono=now,",
     "            decided_at_mono=self._last_at if self._last_at is not None else now,",
     "python"),
    # The plan's original mutation for this field, previously argued not to
    # exist because sensing_controller.py has no direct handle on "the
    # loop's" clock read. It exists: reading the clock again here, after the
    # gates above already compared the first read, is an instant a few
    # microseconds later than the one they used -- caught by a clock that
    # advances on every read, which a clock fixed until `.advance()` is
    # called cannot exercise.
    ("controller: decided_at_mono is a fresh clock read, not the one the gates compared",
     "deployment/jetson/policy/sensing_controller.py",
     "            decided_at_mono=now,",
     "            decided_at_mono=self._now(),",
     "python"),
    # The reference block has to say a phone was never heard from, not that it
    # reported zero -- those are different drives, and a candidate scored
    # against a manufactured "achieved nothing" reading is scored against data
    # that does not exist. Caught by the absence test, which requires all three
    # fields null together rather than a zeroed achieved map.
    # Re-anchored twice now: `at_mono` moved this line once, and task 39's
    # `here_calls`/`here_errors` (added to the same dict literal, on both
    # branches) moved it again.
    ("sensing_loop: the reference block reports 0.0 achieved when the phone never reported",
     "deployment/jetson/policy/sensing_loop.py",
     '        return {"achieved": None, "dropped": None, "age_s": None, "at_mono": None,\n'
     '                "here_calls": None, "here_errors": None, "absent": "no_telemetry"}',
     '        return {"achieved": {key: 0.0 for key in RATE_KEYS},'
     ' "dropped": {key: 0 for key in DROP_KEYS},\n'
     '                "here_calls": None, "here_errors": None, "absent": "no_telemetry"}',
     "python"),
    # The defect class this task exists for, reproduced directly: folding a
    # rule's not-evaluable ticks into "agree" is exactly "this candidate was
    # never given the inputs to decide on" reported as "this candidate agreed" --
    # caught by the denominator test, which requires the not-evaluable count to
    # come out beside the agree/differ counts rather than inside them.
    ("score_shadow: a not_evaluable tick is counted into agree",
     "deployment/jetson/score_shadow.py",
     '            if incumbent_status == RULE_NOT_EVALUABLE:\n'
     '                per_rule[rule]["not_evaluable"] += 1',
     '            if incumbent_status == RULE_NOT_EVALUABLE:\n'
     '                per_rule[rule]["agree"] += 1',
     "python"),
    # `rules_never_exercised` has to name a rule that was structurally absent
    # from every tick's inputs, not one that simply never fired. A rule that is
    # `quiet` on every tick describes a calm road; a rule that is
    # `not_evaluable` on every tick describes an instrument the drive never
    # had. Reporting the first as the second is the exact confusion task 34
    # closed, reopened here in the scorer. Caught by the pure-shadow candidate
    # test, which drives ticks where every rule other than `source_disagreement`
    # is genuinely quiet (not absent) and would wrongly qualify under this
    # mutation.
    ("score_shadow: rules_never_exercised is computed from quiet instead of not_evaluable",
     "deployment/jetson/score_shadow.py",
     "statuses.count(RULE_NOT_EVALUABLE) != total",
     "statuses.count(RULE_QUIET) != total",
     "python"),
    # `activity` was uncovered: nothing named it, so both of the following
    # survived the full suite. Caught by the drive with a known, unequal
    # active/idle split and a tick where thermal backs off alone.
    ("score_shadow: activity counts idle ticks as active",
     "deployment/jetson/score_shadow.py",
     '        if candidate_r["attribution"]["gates"]["level"] == "active":',
     '        if candidate_r["attribution"]["gates"]["level"] == "idle":',
     "python"),
    ("score_shadow: RAISE_RULES counts a thermal backoff as a raise",
     "deployment/jetson/score_shadow.py",
     "RAISE_RULES = (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT)",
     "RAISE_RULES = (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT, Trigger.THERMAL)",
     "python"),

    # Task 35 round 2. `reports` and the fresh/stale split it and three
    # other fields depend on.
    #
    # An age recomputed from `now` on every tick cannot by itself reveal
    # that the underlying report did not change, so a tick rate at or below
    # the telemetry rate makes the age sequence flat rather than
    # decreasing -- the exact drive `write_run`'s own default produces.
    ("score_shadow: reports counts ticks whose age happened to decrease, not distinct arrivals",
     "deployment/jetson/score_shadow.py",
     '    reports = len({r["at_mono"] for r in known_age})',
     '    reports = 1 + sum(1 for i in range(1, len(ages)) if ages[i] < ages[i - 1]) if ages else 0',
     "python"),
    # Not reachable through `reference_from` today -- `telemetry_age_s`
    # comes off the Jetson's own monotonic clock -- but this is now the
    # fifth predicate in the repo answering "is this report too old", and
    # the only one that would have disagreed with the other four.
    # `max()` and the mean propagate a NaN to the whole field, and `json.dumps` writes
    # it as a bare `NaN` that strict parsers refuse -- so one unusable age would cost a
    # reader both numbers and the ability to load the file.
    ("score_shadow: a non-finite age is averaged into the reported ages",
     "deployment/jetson/score_shadow.py",
     '    ages = [r["age_s"] for r in known_age if math.isfinite(r["age_s"])]',
     '    ages = [r["age_s"] for r in known_age]',
     "python"),
    ("score_shadow: the stale predicate disagrees with the controller on a non-finite or negative-beyond-bound age",
     "deployment/jetson/score_shadow.py",
     "    return not math.isfinite(age_s) or abs(age_s) > MAX_TELEMETRY_AGE_S",
     "    return age_s > MAX_TELEMETRY_AGE_S",
     "python"),
    # The four fields below all read `fresh`/`stale`/`known_age` correctly in
    # today's code; each survived the full suite before this round because
    # every existing witness drive re-reports on every tick, so `stale` is
    # always empty and the excluded and unexcluded computations coincide.
    # Caught by a drive whose fresh and stale reports carry different
    # numbers.
    ("score_shadow: ticks_stale treats the bound itself as stale",
     "deployment/jetson/score_shadow.py",
     "    return not math.isfinite(age_s) or abs(age_s) > MAX_TELEMETRY_AGE_S",
     "    return not math.isfinite(age_s) or abs(age_s) >= MAX_TELEMETRY_AGE_S",
     "python"),
    ("score_shadow: achieved_mean is taken over every report, stale included",
     "deployment/jetson/score_shadow.py",
     '    achieved_mean = (\n        {key: _mean([r["achieved"][key] for r in fresh]) for key in RATE_KEYS}\n        if fresh else None\n    )',
     '    achieved_mean = (\n        {key: _mean([r["achieved"][key] for r in known_age]) for key in RATE_KEYS}\n        if known_age else None\n    )',
     "python"),
    ("score_shadow: dropped_final is the last report regardless of staleness",
     "deployment/jetson/score_shadow.py",
     '    dropped_final = fresh[-1]["dropped"] if fresh else None',
     '    dropped_final = known_age[-1]["dropped"] if known_age else None',
     "python"),
    ("score_shadow: age_s_max reports the mean instead",
     "deployment/jetson/score_shadow.py",
     '        "age_s_max": max(ages) if ages else None,',
     '        "age_s_max": _mean(ages),',
     "python"),

    # The activity drive never supplied a feed, so `source_disagreement` was
    # `not_evaluable` on every one of its ticks and this membership could
    # never be exercised. Caught by a stretch of ticks where the feed
    # disagrees with the camera and nothing else fires.
    ("score_shadow: RAISE_RULES drops source_disagreement",
     "deployment/jetson/score_shadow.py",
     "RAISE_RULES = (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT)",
     "RAISE_RULES = (Trigger.EVENT, Trigger.NARROW_MARGIN)",
     "python"),

    # The segment split: which ticks each witness reads, and what an empty
    # one renders as.
    ("score_shadow: the contaminated witness is computed over the reference ticks",
     "deployment/jetson/score_shadow.py",
     '            _reference_witness(contaminated_ticks) if contaminated_ticks else None',
     '            _reference_witness(reference_ticks) if contaminated_ticks else None',
     "python"),
    ("score_shadow: the reference witness is computed over the whole drive again",
     "deployment/jetson/score_shadow.py",
     '        "reference_witness": _reference_witness(reference_ticks) if reference_ticks else None,',
     '        "reference_witness": _reference_witness(sensing_ticks) if reference_ticks else None,',
     "python"),
    ("score_shadow: an empty contaminated segment gets a zeroed block instead of null",
     "deployment/jetson/score_shadow.py",
     '            _reference_witness(contaminated_ticks) if contaminated_ticks else None\n        ),',
     '            _reference_witness(contaminated_ticks)\n        ),',
     "python"),
    # The other half of the same symmetry: an empty REFERENCE segment must
    # render the same way an empty contaminated one already does.
    ("score_shadow: an empty reference segment gets a zeroed block instead of null",
     "deployment/jetson/score_shadow.py",
     '        "reference_witness": _reference_witness(reference_ticks) if reference_ticks else None,',
     '        "reference_witness": _reference_witness(reference_ticks),',
     "python"),
    ("score_shadow: render_table assumes reference_witness is never null",
     "deployment/jetson/score_shadow.py",
     '    rw = result["reference_witness"]\n    if rw is not None:',
     '    rw = result["reference_witness"]\n    if True:',
     "python"),

    ("score_shadow: a missing metadata.jsonl is not refused by name",
     "deployment/jetson/score_shadow.py",
     "    if not metadata_path.exists():",
     "    if False:",
     "python"),
    ("score_shadow: the candidate shape check inspects only the first record",
     "deployment/jetson/score_shadow.py",
     "    if records and not all(_has_valid_attribution(r) for r in records):",
     "    if records and not _has_valid_attribution(records[0]):",
     "python"),

    # `summary.json`'s own recorded tick count is never checked against what the
    # log actually replays. A log truncated after a clean `close()` states a
    # tick count beside a file that is short of it, with `unparseable_lines` and
    # `replay_identity` both reading clean, because a record `close()` never
    # wrote leaves no trace for either of those to find. Caught by a drive whose
    # log is shorter than the tick count in its own summary.
    ("score_shadow: the recorded tick count is not compared against what was scored",
     "deployment/jetson/score_shadow.py",
     '    return {"ticks_recorded": recorded, "ticks_scored": scored, "ticks_missing": recorded - scored}',
     '    return {"ticks_recorded": recorded, "ticks_scored": scored, "ticks_missing": 0}',
     "python"),
    # `born_live` asks whether the FIRST tick was live. A drive promoted from
    # shadow to live partway through has a leading segment that never had a
    # feed in it, and only a first-tick reading can tell that segment apart
    # from one that was live throughout -- reading it as "was any tick ever
    # live" reports nothing absent from a segment that structurally had
    # nothing. Caught by a drive with no `summary.json` that starts shadow and
    # is promoted partway.
    ("score_shadow: born_live is read as ever_live",
     "deployment/jetson/score_shadow.py",
     '    born_live = bool(sensing_ticks) and sensing_ticks[0]["sensing"]["shadow"] is False',
     '    born_live = any(t["sensing"]["shadow"] is False for t in sensing_ticks)',
     "python"),
    # `first_live_tick_id` exists so a reader can find the tick in the log, not
    # to restate how many ticks preceded it -- the two only coincide on a log
    # whose ids happen to equal their own position. Caught by a drive whose
    # logged ids do not start at 0.
    ("score_shadow: first_live_tick_id reports the tick's position instead of its logged id",
     "deployment/jetson/score_shadow.py",
     '            first_live_tick_id = t.get("tick_id")',
     "            first_live_tick_id = i",
     "python"),
    # `reports` counts distinct arrivals a tick observed, including one that is
    # already stale on every tick that ever sees it -- excluding stale reports
    # here would undercount arrivals by exactly the ones that never arrived
    # fresh. Caught by a drive whose second report only ever arrives after a
    # stall long enough that it is stale the instant it is first observed.
    ("score_shadow: reports excludes an arrival that was only ever observed stale",
     "deployment/jetson/score_shadow.py",
     '    reports = len({r["at_mono"] for r in known_age})\n'
     '    fresh = [r for r in known_age if not _is_stale_report(r["age_s"])]\n'
     '    stale = [r for r in known_age if _is_stale_report(r["age_s"])]',
     '    fresh = [r for r in known_age if not _is_stale_report(r["age_s"])]\n'
     '    stale = [r for r in known_age if _is_stale_report(r["age_s"])]\n'
     '    reports = len({r["at_mono"] for r in fresh})',
     "python"),

    # Task 36. Per-tick field provenance: a substituted acceleration told
    # apart from a measured calm one, a stale speed window refused rather
    # than reported as a fresh slope, an empty detection set named rather
    # than read as a measurement, and a schema refusal that replaces a
    # traceback on a pre-task-36 log.
    ("sensing_loop: a substituted acceleration is passed through instead of nulled",
     "deployment/jetson/policy/sensing_loop.py",
     '        ego_acceleration=(\n'
     '            None if provenance.is_substituted(ego_acceleration_source)\n'
     '            else obs.get("ego_acceleration")\n'
     '        ),',
     '        ego_acceleration=obs.get("ego_acceleration"),',
     "python"),
    # Caught by the 1.9 s / 2.1 s boundary test: without this, a slope fitted
    # before a GPS dropout is still reported `derived` for as long as the
    # dropout lasts, because the window's own sample count and span cannot
    # see how stale the fix behind it has gone. Re-anchored from an earlier
    # version that measured staleness off the last appended sample's own
    # clock, which is what let a receiver that keeps returning the same
    # fix -- still `gps_fresh` by its own age test -- look freshly appended
    # for a further `gps_stale_after_s` after it had actually gone stale;
    # taking this tick's own `gps_fresh` verdict instead closed that gap.
    ("observation_builder: the speed window's own staleness is never checked",
     "deployment/jetson/perception/observation_builder.py",
     '        if not gps_fresh:\n'
     '            return 0.0, False\n'
     '        t = t - t.mean()',
     '        t = t - t.mean()',
     "python"),
    # Caught by the zero-in-range-tracks test: reverts `local_density_bin` to
    # unconditional `derived`, so an empty detection set is indistinguishable
    # from a measured light road again -- the defect the disagreement rule's
    # over-report was named, not removed, because of (D4).
    ("observation_builder: local_density_bin is derived even from an empty detection set",
     "deployment/jetson/perception/observation_builder.py",
     '            src["local_density_bin"] = provenance.SOURCE_DERIVED_EMPTY\n'
     '        else:',
     '            src["local_density_bin"] = provenance.SOURCE_DERIVED\n'
     '        else:',
     "python"),
    # Caught by the parametrized `is_substituted` closure test on the
    # `unattributed` member: a field the builder forgot to tag would then be
    # decided on as if it were measured, rather than failing the rule safe.
    ("provenance: is_substituted no longer treats unattributed as a substitution",
     "deployment/jetson/perception/provenance.py",
     "    return source in SUBSTITUTED",
     "    return source in SUBSTITUTED and source != SOURCE_UNATTRIBUTED",
     "python"),
    # This pins the deliberate non-change (D4), which is the decision here
    # most likely to be "fixed" by a later reader: under shipped constants
    # the disagreement rule fires iff the camera detected nothing, so gating
    # `derived_empty` as a substitution deletes the rule's only firing path.
    # That deletion is NOT what this mutation is caught by, though: it
    # cannot be, because `is_substituted` has exactly one production
    # caller -- `ego_acceleration_source`, in `inputs_from` -- and that
    # field is never `derived_empty`. `camera_density_bin` is never passed
    # through `is_substituted` at all. So this mutation cannot reach the
    # disagreement rule or any other decision; what actually catches it is
    # the vocabulary-closure test on the constant itself
    # (`SOURCE_DERIVED_EMPTY not in SUBSTITUTED`), which guards the
    # constant rather than any rule's behaviour.
    ("provenance: derived_empty is added to SUBSTITUTED",
     "deployment/jetson/perception/provenance.py",
     "SUBSTITUTED = frozenset({\n"
     "    SOURCE_FALLBACK_NEUTRAL,\n"
     "    SOURCE_STATIC_CONFIG,\n"
     "    SOURCE_SIM_PARITY,\n"
     "    SOURCE_UNATTRIBUTED,\n"
     "})",
     "SUBSTITUTED = frozenset({\n"
     "    SOURCE_FALLBACK_NEUTRAL,\n"
     "    SOURCE_STATIC_CONFIG,\n"
     "    SOURCE_SIM_PARITY,\n"
     "    SOURCE_UNATTRIBUTED,\n"
     "    SOURCE_DERIVED_EMPTY,\n"
     "})",
     "python"),
    # Caught by the schema-refusal test: without this check, a pre-task-36
    # log's 13-key `decision_inputs` reaches `Inputs.from_record` inside
    # `_replay_incumbent` and raises `ValueError` instead of being refused by
    # name -- the traceback this refusal exists to replace with a named exit.
    ("score_shadow: the decision_inputs schema check never refuses anything",
     "deployment/jetson/score_shadow.py",
     "        if missing or unknown:",
     "        if False:",
     "python"),

    # Task 36 validation round.
    #
    # A 39-key map missing one real encoder slot and carrying one bogus name
    # in its place has the right COUNT, so comparing counts reports coverage
    # that is not there. Caught by the bogus-name test.
    ("observation_builder: covers_encoder compares a count instead of the field names",
     "deployment/jetson/perception/observation_builder.py",
     "        return set(field_sources) == set(sim_contract.encoded_slot_names())",
     "        return len(field_sources) == len(sim_contract.encoded_slot_names())",
     "python"),
    # The nested `cooperation.*` entries are supposed to be the SAME class as
    # their own flat field, not a neighbour's -- `merge_pressure` is the one
    # of the three that stays `fallback_neutral` even with peers present, so
    # it is the only one of the three whose class actually differs from the
    # other two on a peers-present tick. Caught by the peers-present test.
    ("observation_builder: cooperation.merge_pressure copies segment_target_speed's class",
     "deployment/jetson/perception/observation_builder.py",
     '        src["cooperation.merge_pressure"] = src["merge_pressure"]',
     '        src["cooperation.merge_pressure"] = src["segment_target_speed"]',
     "python"),
    # `inputs_from`'s own class for a field the map has no entry for at all,
    # hardcoded to `measured` instead of read off the builder -- a wrong-key
    # or dropped-lookup regression that a fixed constant across every test
    # using the shared grounded fixture cannot distinguish from correct.
    # Caught by the real-builder round-trip test, which varies the fix
    # between fresh and stale and checks the source moves with it.
    ("sensing_loop: ego_speed_source is hardcoded instead of read off field_sources",
     "deployment/jetson/policy/sensing_loop.py",
     '    ego_speed_source = src.get("ego_speed", provenance.SOURCE_UNATTRIBUTED)',
     "    ego_speed_source = provenance.SOURCE_MEASURED",
     "python"),
    # The defect class this task exists for, reproduced directly in the one
    # place a wrong key would matter most: `camera_density_bin_source` reads
    # `ego_speed`'s class instead of `local_density_bin`'s, so a fresh fix
    # with no vehicles would claim `measured` where the truth is
    # `derived_empty`. Caught by the real-builder round-trip test built on
    # exactly that tick shape.
    ("sensing_loop: camera_density_bin_source reads the wrong obs key",
     "deployment/jetson/policy/sensing_loop.py",
     '    camera_density_bin_source = src.get("local_density_bin", provenance.SOURCE_UNATTRIBUTED)',
     '    camera_density_bin_source = src.get("ego_speed", provenance.SOURCE_UNATTRIBUTED)',
     "python"),
    # D14's whole contribution -- a liveness bound the disagreement rule can
    # cite beside its claim -- never reaches the controller if this is
    # dropped. Caught by the aging-out test, which requires a non-null age
    # after a detection has come and gone.
    ("sensing_loop: camera_last_detection_age_s never reaches Inputs",
     "deployment/jetson/policy/sensing_loop.py",
     '        camera_last_detection_age_s=diagnostics.get("last_detection_age_s"),',
     "        camera_last_detection_age_s=None,",
     "python"),
    # The operator otherwise gets a refusal with no names -- D10 chose a
    # schema-derived refusal specifically so it could say which keys, and on
    # which tick. Caught by the render test asserting the missing/unknown
    # key names actually appear in the printed table.
    ("score_shadow: render_table drops the schema detail on a refusal",
     "deployment/jetson/score_shadow.py",
     '        schema = result.get("schema")\n        if schema is not None:',
     '        schema = result.get("schema")\n        if False:',
     "python"),
    # A continuous evidence value (`camera_last_detection_age_s`) is a
    # little different on nearly every tick of a real drive, so bucketing it
    # by exact value the same way a categorical one is produces one bucket
    # per tick rather than a reason. Caught by the summary test, which
    # requires a min/median/max entry instead of five one-tick buckets.
    ("score_shadow: the why map buckets a continuous evidence value instead of summarising it",
     "deployment/jetson/score_shadow.py",
     "            if present and len(numeric) == len(present):",
     "            if False:",
     "python"),
    # The schema check has to hold on EVERY tick of the log, not only the
    # first -- a log can drift shape partway through. Caught by a corrupted
    # tick placed after the first one.
    ("score_shadow: the decision_inputs schema check only inspects the first tick",
     "deployment/jetson/score_shadow.py",
     '    for t in sensing_ticks:\n'
     '        present = set(t["sensing"]["decision_inputs"])',
     '    for t in sensing_ticks[:1]:\n'
     '        present = set(t["sensing"]["decision_inputs"])',
     "python"),
    # `_input_provenance` is supposed to answer three separate questions, one
    # per `INPUT_SOURCE_FIELDS` entry -- reading the acceleration's own key
    # regardless of which field is being counted would report the same
    # class for `ego_speed` and `camera_density_bin` as for
    # `ego_acceleration`. Caught by a run whose three fields are given
    # deliberately distinct classes.
    ("score_shadow: _input_provenance reads ego_acceleration_source for every field",
     "deployment/jetson/score_shadow.py",
     '            source = decision_inputs.get(f"{field_name}_source")',
     '            source = decision_inputs.get("ego_acceleration_source")',
     "python"),
    # `by_source`/`fields_by_source` pool every tick's `field_sources`
    # regardless of its size, so a run whose maps are not uniform needs to
    # say so rather than reporting the first tick's size as if it applied
    # throughout. Caught by a run built from a mix of a 1-key and a 39-key
    # fixture.
    ("eval_run: provenance_fields_mixed is never detected",
     "deployment/jetson/eval_run.py",
     "    provenance_fields_mixed = len(provenance_field_sizes) > 1",
     "    provenance_fields_mixed = False",
     "python"),

    # Task 36 round 2 (a re-audit of the round-1 fixes above).
    #
    # The span guard in `_speed_slope` only compares the window's first and
    # last timestamps, so it cannot see a hole in the middle of it. GPS
    # returning fresh on the first tick after a real dropout used to feed
    # the slope a window mixing pre-dropout and post-dropout samples, and
    # the fitted slope across the gap was reported `derived` -- a real
    # measurement of an interval containing none. Caught by the dropout
    # test, which drives a genuine `gps.valid=False` gap (not merely an
    # aging held fix) and checks the tick GPS returns on.
    ("observation_builder: the speed window is not cleared on a non-fresh tick",
     "deployment/jetson/perception/observation_builder.py",
     '            self._ego.speed_samples.clear()\n',
     "",
     "python"),
    # The same defect `ObservationBuilder._covers_encoder` was written to
    # catch (a map with the right COUNT of keys but the wrong NAMES reads as
    # complete), left in the one surface an operator actually reads. Caught
    # by the same-size name-swap test: 39 keys, one real slot deleted and a
    # made-up name added in its place.
    ("eval_run: covers_encoder compares a count instead of the field names",
     "deployment/jetson/eval_run.py",
     "        else provenance_field_names == set(sim_contract.encoded_slot_names())\n",
     "        else provenance_fields == sim_contract.local_obs_dim()\n",
     "python"),
    # Distinct from the mutation above: this drops the mixed-size
    # short-circuit itself rather than the name comparison it guards, so a
    # run whose tick-to-tick provenance map size varies could read as
    # complete whenever the union of every name ever seen happens to equal
    # the full 39-name set -- which it does whenever an incomplete tick's
    # keys are a subset of a later, complete tick's keys. Caught by the
    # mixed-run tests in both fixture orders (the un-reordered one alone
    # left this branch unexercised under the previous, count-based
    # comparison: the first tick's size was already unequal to 39 with no
    # help from this clause).
    ("eval_run: the mixed-size short-circuit is dropped from covers_encoder",
     "deployment/jetson/eval_run.py",
     "    covers_encoder = (\n"
     "        None if provenance_fields is None\n"
     "        else False if provenance_fields_mixed\n"
     "        else provenance_field_names == set(sim_contract.encoded_slot_names())\n"
     "    )\n",
     "    covers_encoder = (\n"
     "        None if provenance_fields is None\n"
     "        else provenance_field_names == set(sim_contract.encoded_slot_names())\n"
     "    )\n",
     "python"),
    # `provenance_fields` is supposed to be the FIRST tick's map size (what
    # `render_markdown` calls it: "first tick has {pf}"), not the last one
    # to be seen. Caught by the mixed-run test's own size assertion, which
    # needs a first tick and a last tick of different sizes to tell the two
    # apart at all.
    ("eval_run: provenance_fields reports the last tick's size instead of the first",
     "deployment/jetson/eval_run.py",
     "        provenance_field_sizes.add(len(field_sources))\n"
     "        if provenance_fields is None:\n"
     "            provenance_fields = len(field_sources)\n",
     "        provenance_field_sizes.add(len(field_sources))\n"
     "        provenance_fields = len(field_sources)\n",
     "python"),
    # A tick whose `field_sources` map is empty used to be skipped entirely
    # when sizing the run, so a run half carrying no provenance and half
    # carrying the full map read as uniform (one size seen) and complete.
    # Caught by the empty-then-full test.
    ("eval_run: an empty field_sources tick is excluded from the mixture",
     "deployment/jetson/eval_run.py",
     "        field_sources = t.get(\"field_sources\") or {}\n"
     "        provenance_field_sizes.add(len(field_sources))\n"
     "        if provenance_fields is None:\n"
     "            provenance_fields = len(field_sources)\n"
     "        provenance_field_names.update(field_sources)\n",
     "        field_sources = t.get(\"field_sources\") or {}\n"
     "        if field_sources:\n"
     "            provenance_field_sizes.add(len(field_sources))\n"
     "            if provenance_fields is None:\n"
     "                provenance_fields = len(field_sources)\n"
     "        provenance_field_names.update(field_sources)\n",
     "python"),

    # Task 36 round 3: the renderer, not the provenance plumbing task 36 itself
    # added. `pctl` already computes min/p50/p95/max for every distribution in
    # this file, including missingness; the markdown line read only the mean,
    # so a bimodal (or trimodal) run reported a percentage none of its ticks
    # actually had. Caught by the trimodal fixture, whose assertions on the
    # rendered line do not hold if the spread or the distinct-value count it
    # names is dropped.
    ("eval_run: encoder-field missingness renders no spread beside the mean",
     "deployment/jetson/eval_run.py",
     '    lines.append(\n'
     '        f"- encoder-field missingness: mean {m[\'mean\']:.1%}"\n'
     '        + (f" of {pf} provenance-tagged fields" if pf is not None else "")\n'
     '        + f" ({spread})"\n'
     '    )\n',
     '    lines.append(\n'
     '        f"- encoder-field missingness: mean {m[\'mean\']:.1%}"\n'
     '        + (f" of {pf} provenance-tagged fields" if pf is not None else "")\n'
     '    )\n',
     "python"),

    # Task 37: thermal and throttle-event log for both devices.
    ("thermal: an unreadable zone reports 0.0 instead of absent",
     "deployment/jetson/sensors/thermal.py",
     "            return THERMAL_BASIS_ABSENT, zones_reason",
     "            self._last_sample_mono = now\n"
     "            self._last_selected_celsius = 0.0\n"
     "            return THERMAL_BASIS_MEASURED, None",
     "python"),
    ("thermal: a stale reading is reported as measured",
     "deployment/jetson/sensors/thermal.py",
     "            basis = THERMAL_BASIS_MEASURED if age <= 2 * self._interval_s else THERMAL_BASIS_STALE",
     "            basis = THERMAL_BASIS_MEASURED",
     "python"),
    ("thermal: the freshness bound is a typed 2.0 rather than 2 x interval",
     "deployment/jetson/sensors/thermal.py",
     "age <= 2 * self._interval_s",
     "age <= 2.0",
     "python"),
    ("thermal: an unobservable event stream is reported as quiet",
     "deployment/jetson/sensors/thermal.py",
     "            return _tick_event_record(\n"
     "                RULE_NOT_EVALUABLE, 0, None, (MISSING_COOLING_STATE,),\n"
     "                passes_attempted=self._cooling_passes_attempted,\n"
     "                passes_readable=self._cooling_passes_readable,\n"
     "            )",
     "            return _tick_event_record(\n"
     "                RULE_QUIET, 0, None, (),\n"
     "                passes_attempted=self._cooling_passes_attempted,\n"
     "                passes_readable=self._cooling_passes_readable,\n"
     "            )",
     "python"),
    # Round 2 re-audit: a real, already-logged transition was being reported as
    # `not_evaluable`/`count: 0` whenever some other cooling device on the same
    # pass -- or a later pass -- never gave a reading (C2). The fix keeps
    # `fired` and the real count whenever anything actually fired, carrying the
    # incompleteness in `missing`/the pass counters instead of erasing it.
    ("thermal: a fired jetson count is reported as not_evaluable when observation is incomplete",
     "deployment/jetson/sensors/thermal.py",
     "        if self._jetson_event_count > 0:\n"
     "            if self._cooling_fully_readable():\n"
     "                return _tick_event_record(RULE_FIRED, self._jetson_event_count, self._jetson_last_event)",
     "        if self._jetson_event_count > 0 and self._cooling_fully_readable():\n"
     "            if self._cooling_fully_readable():\n"
     "                return _tick_event_record(RULE_FIRED, self._jetson_event_count, self._jetson_last_event)",
     "python"),
    ("thermal: the summary's fired jetson count is reported as not_evaluable when observation is incomplete",
     "deployment/jetson/sensors/thermal.py",
     "        if self._jetson_event_count > 0:\n"
     "            if self._cooling_fully_readable():\n"
     "                return _summary_event_record(\n"
     "                    RULE_FIRED, self._jetson_event_count, (), by_unit=self._jetson_events_by_unit,\n"
     "                )",
     "        if self._jetson_event_count > 0 and self._cooling_fully_readable():\n"
     "            if self._cooling_fully_readable():\n"
     "                return _summary_event_record(\n"
     "                    RULE_FIRED, self._jetson_event_count, (), by_unit=self._jetson_events_by_unit,\n"
     "                )",
     "python"),
    ("thermal: the jetson event count is derived by subtraction rather than counted",
     "deployment/jetson/sensors/thermal.py",
     "                    self._jetson_event_count += 1",
     "                    self._jetson_event_count = self._samples - self._basis_counts[THERMAL_BASIS_MEASURED]",
     "python"),
    ("thermal: at_mono is read fresh rather than the phone's own report instant",
     "deployment/jetson/sensors/thermal.py",
     '            "skin_temp_absent": getattr(telemetry, "skin_temp_absent", None),\n'
     '            "at_mono": telemetry_at,\n'
     '            "absent": None,\n'
     "        }",
     '            "skin_temp_absent": getattr(telemetry, "skin_temp_absent", None),\n'
     '            "at_mono": self._now(),\n'
     '            "absent": None,\n'
     "        }",
     "python"),
    ("phone: the reported status comes from the listener, not the poll",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/TelemetryReporter.kt",
     "                thermalStatus = reading.thermalStatus,",
     "                thermalStatus = reading.lastTransitionTo ?: reading.thermalStatus,",
     "app"),
    ("phone: a headroom absence collapses two distinct causes into one reason",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalReader.kt",
     "        if (value < 0.0 || value > MAX_PLAUSIBLE_HEADROOM) return Headroom(null, REASON_OUT_OF_BAND)",
     "        if (value < 0.0 || value > MAX_PLAUSIBLE_HEADROOM) return Headroom(null, REASON_NOT_A_NUMBER)",
     "app"),
    ("phone: the watcher is nulled but not unregistered",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalStatusWatcher.kt",
     "        if (registered) {\n            unregister(listener)\n            registered = false\n        }",
     "        if (registered) {\n            registered = false\n        }",
     "app"),

    # Task 37 round 2: the four confirmed critical findings.
    ("thermal: a phone redial's report is copied instead of accumulated",
     "deployment/jetson/sensors/thermal.py",
     '                at_ns = getattr(telemetry, "thermal_change_at_mono_ns", None)\n'
     "                self._phone_event_count += delta\n"
     '                self._phone_last_event = {"at_mono": now, "from": frm, "to": to}',
     '                at_ns = getattr(telemetry, "thermal_change_at_mono_ns", None)\n'
     "                self._phone_event_count = changes\n"
     '                self._phone_last_event = {"at_mono": now, "from": frm, "to": to}',
     "python"),
    ("thermal: the phone event baseline is not cleared when telemetry goes absent",
     "deployment/jetson/sensors/thermal.py",
     "        if telemetry is None:\n"
     "            self._prev_status_changes = None\n"
     "            return\n"
     '        changes = getattr(telemetry, "thermal_status_changes", None)',
     "        if telemetry is None:\n"
     "            return\n"
     '        changes = getattr(telemetry, "thermal_status_changes", None)',
     "python"),
    # Round 2 re-audit: the gate above turned out to discard a real, genuine
    # rise whenever the descriptors did not survive the report (C1), so it was
    # removed rather than kept as a mutation target -- a rise with no
    # descriptors is now the behaviour, not the defect. The three entries
    # below replace it: discarding the rise is still a defect, just a
    # different one (the count/event, not the gate), and the gate's `or` moved
    # to a narrower distinction worth its own pin.
    ("thermal: a rise with absent descriptors is discarded instead of counted",
     "deployment/jetson/sensors/thermal.py",
     "                self._phone_event_count += delta\n",
     "                self._phone_event_count += delta if (frm is not None or to is not None) else 0\n",
     "python"),
    ("thermal: a half-described transition is folded into the without-descriptors count",
     "deployment/jetson/sensors/thermal.py",
     "                if frm is None and to is None:\n"
     "                    self._phone_count_without_descriptors += delta",
     "                if frm is None or to is None:\n"
     "                    self._phone_count_without_descriptors += delta",
     "python"),
    ("thermal: a multi-transition gap is not recorded",
     "deployment/jetson/sensors/thermal.py",
     "                if delta > 1:\n"
     "                    self._phone_gap_events += 1",
     "                if delta > 1:\n"
     "                    pass",
     "python"),
    ("thermal: cooling readable once is reported the same as readable throughout",
     "deployment/jetson/sensors/thermal.py",
     "        return (\n"
     "            self._cooling_passes_attempted > 0\n"
     "            and self._cooling_passes_attempted == self._cooling_passes_readable\n"
     "        )",
     "        return self._cooling_passes_readable > 0",
     "python"),
    ("thermal: a cooling device that lists but will not read is dropped, not named missing",
     "deployment/jetson/sensors/thermal.py",
     "            if raw is None:\n"
     "                missing.append(name)\n"
     "                continue",
     "            if raw is None:\n"
     "                continue",
     "python"),
    ("thermal: an unparseable cooling state is dropped, not named missing",
     "deployment/jetson/sensors/thermal.py",
     "            except ValueError:\n"
     "                missing.append(name)\n"
     "                continue",
     "            except ValueError:\n"
     "                continue",
     "python"),
    # Round 2 re-audit: a device whose own `type` would not read fell into
    # neither `states` nor `missing` (M1), which a caller cannot tell apart
    # from a root with no `cooling_device*` entries at all -- fixed by naming
    # it in `missing` too, keyed by directory name.
    ("thermal: a cooling device whose type will not read is filed as nothing attempted",
     "deployment/jetson/sensors/thermal.py",
     "            if name is None:\n"
     "                missing.append(entry.name)\n"
     "                continue",
     "            if name is None:\n"
     "                continue",
     "python"),
    # Round 2 re-audit: the dedup this fixed (M1) was also checking `name in
    # missing`, so a device sharing a type name with one that had already
    # failed to read was skipped outright rather than given its own attempt
    # (m9) -- regressed by this same round, since the pre-round-1 dedup only
    # checked `states`.
    ("thermal: a device sharing a failed name's type is never given its own attempt",
     "deployment/jetson/sensors/thermal.py",
     "            if name in states:\n"
     "                continue",
     "            if name in states or name in missing:\n"
     "                continue",
     "python"),
    # Round 2 re-audit (m12): a device missing on one pass replaced the whole
    # map instead of merging into it, so it had nothing to diff against once
    # it started reading again -- a real transition spanning the gap went
    # unnoticed. Unpinned before this: every existing test that exercises a
    # missed pass never brought the device back afterward.
    ("thermal: a device missing on one pass loses its history instead of keeping it",
     "deployment/jetson/sensors/thermal.py",
     "        self._prev_cooling = {**(self._prev_cooling or {}), **states}",
     "        self._prev_cooling = dict(states)",
     "python"),
    ("eval_run: the per-tick thermal basis is computed and never rendered",
     "deployment/jetson/eval_run.py",
     '    ticks_by_basis = thermal.get("ticks_by_basis") or {}\n'
     "    if ticks_by_basis:\n"
     "        # The real signal for a sampler thread that died partway through the\n"
     "        # run (see `thermal_result`'s docstring): `sample_gaps_s` cannot show\n"
     "        # it, because a gap exists only between samples that were written.\n"
     '        parts = ", ".join(f"{basis} {n}" for basis, n in sorted(ticks_by_basis.items()))\n'
     '        lines.append(f"- jetson thermal seen by ticks: {parts}")\n'
     "    return lines",
     "    return lines",
     "python"),
    # Anchor updated for task 38's dataclass rewrite of `load_records`, which
    # introduced a `record_type` local read once per line rather than calling
    # `record.get("type")` again in every branch. The mutation itself --
    # dropping the thermal_event branch entirely -- is unchanged.
    ("eval_run: load_records drops the thermal_event branch",
     "deployment/jetson/eval_run.py",
     '            elif record_type == "thermal_event":\n'
     "                thermal_events.append(record)",
     "",
     "python"),

    # Task 37 round 2: the MAJOR findings.
    ("thermal: no phone at all reports quiet instead of not_evaluable",
     "deployment/jetson/sensors/thermal.py",
     "        if telemetry is None:\n"
     "            return _tick_event_record(RULE_NOT_EVALUABLE, 0, None, (MISSING_TELEMETRY,))",
     "        if telemetry is None:\n"
     "            return _tick_event_record(RULE_QUIET, 0, None, ())",
     "python"),
    ("thermal: an older phone build reports quiet instead of not_evaluable",
     "deployment/jetson/sensors/thermal.py",
     '        if getattr(telemetry, "thermal_status_changes", None) is None:\n'
     "            return _tick_event_record(RULE_NOT_EVALUABLE, 0, None, (MISSING_STATUS_CHANGES,))",
     '        if getattr(telemetry, "thermal_status_changes", None) is None:\n'
     "            return _tick_event_record(RULE_QUIET, 0, None, ())",
     "python"),
    ("run_demo: the thermal sampler is never constructed",
     "deployment/jetson/run_demo.py",
     '        if config["logio"]["thermal"]:',
     '        if False and config["logio"]["thermal"]:',
     "python"),
    ("run_demo: the tick's thermal block reads sensing.reference instead of the sampler",
     "deployment/jetson/run_demo.py",
     "                if thermal_sampler is not None:\n"
     '                    record["thermal"] = thermal_sampler.latest()',
     "                if thermal_sampler is not None:\n"
     '                    record["thermal"] = (\n'
     "                        outcome.to_record().get(\"reference\") if outcome is not None\n"
     "                        else thermal_sampler.latest()\n"
     "                    )",
     "python"),
    # Round 2 re-audit: `run_live`'s own stop-then-summarize logic moved into
    # `_thermal_summary` so it can be driven directly rather than pinned by
    # where its two calls sit in the source text (M2) -- the three anchors
    # below moved with it.
    ("run_demo: the drive summary never gets a thermal block",
     "deployment/jetson/run_demo.py",
     '        thermal_summary = _thermal_summary(thermal_sampler, stats_sampler)\n'
     "        if thermal_summary is not None:\n"
     '            summary["thermal"] = thermal_summary',
     '        thermal_summary = _thermal_summary(thermal_sampler, stats_sampler)\n'
     "        if thermal_summary is not None:\n"
     '            summary["_thermal_disabled"] = thermal_summary',
     "python"),
    ("run_demo: the thermal sampler is never stopped",
     "deployment/jetson/run_demo.py",
     "    if thermal_sampler is None:\n"
     "        return None\n"
     "    thermal_sampler.stop()\n"
     "    return thermal_sampler.to_record(\n"
     "        jtop_available=stats_sampler.available if stats_sampler is not None else None\n"
     "    )",
     "    if thermal_sampler is None:\n"
     "        return None\n"
     "    return thermal_sampler.to_record(\n"
     "        jtop_available=stats_sampler.available if stats_sampler is not None else None\n"
     "    )",
     "python"),
    ("run_demo: the thermal sampler is stopped after its record is read",
     "deployment/jetson/run_demo.py",
     "    if thermal_sampler is None:\n"
     "        return None\n"
     "    thermal_sampler.stop()\n"
     "    return thermal_sampler.to_record(\n"
     "        jtop_available=stats_sampler.available if stats_sampler is not None else None\n"
     "    )",
     "    if thermal_sampler is None:\n"
     "        return None\n"
     "    result = thermal_sampler.to_record(\n"
     "        jtop_available=stats_sampler.available if stats_sampler is not None else None\n"
     "    )\n"
     "    thermal_sampler.stop()\n"
     "    return result",
     "python"),
    # Round 3 re-audit: the comment's own promise -- "must not take the
    # pipeline down with it" -- was false. Killing the thread on its first bad
    # pass took the whole thermal log down for the rest of the drive,
    # including the phone's own record, which never touches sysfs at all. The
    # fix isolates each device's own read (the two entries below) and keeps
    # the loop itself running past whatever still reaches this handler.
    ("thermal: the sampler loop kills the thread on any failed pass",
     "deployment/jetson/sensors/thermal.py",
     "            except Exception:\n"
     "                # Either device's own read is already isolated inside\n"
     "                # `sample_once`, so reaching here means something else failed\n"
     "                # -- the sink, say. Either way, this one pass is skipped and\n"
     "                # the loop keeps running at the same pace, exactly as if it\n"
     "                # had not been attempted: `sampler_stopped` is reserved for a\n"
     "                # sampler that was actually asked to stop or never started,\n"
     "                # not for one that lost a pass.\n"
     "                pass",
     "            except Exception:\n"
     "                self._running = False\n"
     "                return",
     "python"),
    # Both pins anchor on the same three-line `_safe_call` invocation with
    # different mutations; each anchor still resolves exactly once on its
    # own.
    ("thermal: a jetson zone-read failure is not isolated from the rest of the pass",
     "deployment/jetson/sensors/thermal.py",
     "        census, zones_missing, zones_reason = _safe_call(\n"
     "            self._jetson.read_zones, ({}, (), ABSENT_READ_ERROR)\n"
     "        )",
     "        census, zones_missing, zones_reason = self._jetson.read_zones()",
     "python"),
    ("thermal: a jetson cooling-read failure is not isolated from the rest of the pass",
     "deployment/jetson/sensors/thermal.py",
     "        cooling, cooling_missing = _safe_call(self._jetson.read_cooling, ({}, ()))",
     "        cooling, cooling_missing = self._jetson.read_cooling()",
     "python"),
    ("thermal: a jetson read failure is reported as no zone readable instead of its own reason",
     "deployment/jetson/sensors/thermal.py",
     "        census, zones_missing, zones_reason = _safe_call(\n"
     "            self._jetson.read_zones, ({}, (), ABSENT_READ_ERROR)\n"
     "        )",
     "        census, zones_missing, zones_reason = _safe_call(\n"
     "            self._jetson.read_zones, ({}, (), ABSENT_NO_ZONE_READABLE)\n"
     "        )",
     "python"),
    ("thermal: the wait after a pass does not subtract its own duration",
     "deployment/jetson/sensors/thermal.py",
     "            elapsed = time.monotonic() - pass_started\n"
     "            self._stop_event.wait(max(0.0, self._interval_s - elapsed))",
     "            self._stop_event.wait(self._interval_s)",
     "python"),
    ("thermal: the quiet jetson tick record carries no pass-count evidence",
     "deployment/jetson/sensors/thermal.py",
     "        # `quiet` is a claim of full observation, exactly like `not_evaluable`\n"
     "        # is a claim of its absence -- both need the same pass counters as\n"
     "        # evidence, or the claim can only be taken on faith.\n"
     "        return _tick_event_record(\n"
     "            RULE_QUIET, 0, None,\n"
     "            passes_attempted=self._cooling_passes_attempted,\n"
     "            passes_readable=self._cooling_passes_readable,\n"
     "        )",
     "        # `quiet` is a claim of full observation, exactly like `not_evaluable`\n"
     "        # is a claim of its absence -- both need the same pass counters as\n"
     "        # evidence, or the claim can only be taken on faith.\n"
     "        return _tick_event_record(RULE_QUIET, 0, None)",
     "python"),
    ("thermal: the quiet jetson summary record carries no pass-count evidence",
     "deployment/jetson/sensors/thermal.py",
     "        return _summary_event_record(\n"
     "            RULE_QUIET, 0, (), by_unit=self._jetson_events_by_unit,\n"
     "            passes_attempted=self._cooling_passes_attempted,\n"
     "            passes_readable=self._cooling_passes_readable,\n"
     "        )",
     "        return _summary_event_record(RULE_QUIET, 0, (), by_unit=self._jetson_events_by_unit)",
     "python"),
    ("eval_run: the quiet jetson line drops its pass counters",
     "deployment/jetson/eval_run.py",
     "            elif device == \"jetson\":\n"
     "                # `quiet` is a claim of full observation, so it carries the\n"
     "                # same pass counters `not_evaluable` does -- a reader can\n"
     "                # check \"readable throughout\" instead of taking it on faith.\n"
     "                passes = (\n"
     "                    f\" ({ev.get('passes_readable')} of {ev.get('passes_attempted')} passes fully readable)\"\n"
     "                    if ev.get(\"passes_attempted\") else \"\"\n"
     "                )\n"
     "                lines.append(f\"- throttle events, jetson: quiet -- cooling devices readable throughout, \"\n"
     "                             f\"{count} transitions{passes}\")",
     "            elif device == \"jetson\":\n"
     "                lines.append(f\"- throttle events, jetson: quiet -- cooling devices readable throughout, \"\n"
     "                             f\"{count} transitions\")",
     "python"),
    ("thermal: _read_trimmed does not catch a would-block TypeError",
     "deployment/jetson/sensors/thermal.py",
     "    except (OSError, TypeError):\n"
     "        return None",
     "    except OSError:\n"
     "        return None",
     "python"),
    ("thermal: the hottest-zone fallback picks the coolest zone instead",
     "deployment/jetson/sensors/thermal.py",
     "            self._selected_zone = max(census, key=census.get)",
     "            self._selected_zone = min(census, key=census.get)",
     "python"),
    ("thermal: samples counts only the measured passes",
     "deployment/jetson/sensors/thermal.py",
     "            self._process_phone_events(telemetry, now)\n"
     "            self._accumulate_phone_summary(telemetry, telemetry_at, now)\n"
     "            self._samples += 1",
     "            self._process_phone_events(telemetry, now)\n"
     "            self._accumulate_phone_summary(telemetry, telemetry_at, now)\n"
     "            if jetson_basis == THERMAL_BASIS_MEASURED:\n"
     "                self._samples += 1",
     "python"),
    ("thermal: an absent pass is counted as measured",
     "deployment/jetson/sensors/thermal.py",
     "        if zones_reason is not None:\n"
     "            self._last_zones_reason = zones_reason\n"
     "            self._basis_counts[THERMAL_BASIS_ABSENT] += 1",
     "        if zones_reason is not None:\n"
     "            self._last_zones_reason = zones_reason\n"
     "            self._basis_counts[THERMAL_BASIS_MEASURED] += 1",
     "python"),
    ("thermal: a zone-read absence reason is never accumulated",
     "deployment/jetson/sensors/thermal.py",
     "            self._basis_counts[THERMAL_BASIS_ABSENT] += 1\n"
     "            self._absent_reasons[zones_reason] = self._absent_reasons.get(zones_reason, 0) + 1\n"
     "            return THERMAL_BASIS_ABSENT, zones_reason",
     "            self._basis_counts[THERMAL_BASIS_ABSENT] += 1\n"
     "            return THERMAL_BASIS_ABSENT, zones_reason",
     "python"),
    ("thermal: zones_seen and per_zone_max_c are never accumulated",
     "deployment/jetson/sensors/thermal.py",
     "        self._zones_seen.update(census)\n"
     "        for name, value in census.items():\n"
     "            self._per_zone_max[name] = max(self._per_zone_max.get(name, value), value)",
     "        pass",
     "python"),
    ("thermal: the phone's own sample count is never accumulated",
     "deployment/jetson/sensors/thermal.py",
     "        self._phone_samples += 1\n"
     '        status = getattr(telemetry, "thermal_status", None)',
     '        status = getattr(telemetry, "thermal_status", None)',
     "python"),
    ("thermal: a phone absence is never counted in the summary",
     "deployment/jetson/sensors/thermal.py",
     "        if telemetry is None:\n"
     "            self._phone_absent_counts[ABSENT_NO_TELEMETRY] = (\n"
     "                self._phone_absent_counts.get(ABSENT_NO_TELEMETRY, 0) + 1\n"
     "            )\n"
     "            return\n"
     "        self._phone_samples += 1",
     "        if telemetry is None:\n"
     "            return\n"
     "        self._phone_samples += 1",
     "python"),
    ("thermal: the headroom-absence counters are never accumulated",
     "deployment/jetson/sensors/thermal.py",
     '        headroom = getattr(telemetry, "thermal_headroom", None)\n'
     "        if headroom is None:\n"
     "            # A null headroom with no stated reason is still a null headroom --\n"
     '            # counting it nowhere would make "always answered" and "never\n'
     '            # answered and never said why" the same empty dict.\n'
     '            reason = getattr(telemetry, "thermal_headroom_absent", None) or HEADROOM_ABSENT_UNSPECIFIED\n'
     "            self._phone_headroom_absent_counts[reason] = self._phone_headroom_absent_counts.get(reason, 0) + 1",
     "",
     "python"),
    ("thermal: p50 and p95 are swapped",
     "deployment/jetson/sensors/thermal.py",
     '        "p50": at(0.50), "p95": at(0.95), "max": ordered[-1],',
     '        "p50": at(0.95), "p95": at(0.50), "max": ordered[-1],',
     "python"),
    ("thermal: the summary event record's empty missing list is dropped",
     "deployment/jetson/sensors/thermal.py",
     '    record: dict[str, Any] = {"status": status, "count": count, "missing": list(missing)}',
     '    record: dict[str, Any] = {"status": status, "count": count}\n'
     "    if missing:\n"
     '        record["missing"] = list(missing)',
     "python"),
    ("thermal: the disappearing held zone is reported as no zone readable",
     "deployment/jetson/sensors/thermal.py",
     "            self._last_zones_reason = ABSENT_ZONE_DISAPPEARED\n"
     "            self._basis_counts[THERMAL_BASIS_ABSENT] += 1\n"
     "            self._absent_reasons[ABSENT_ZONE_DISAPPEARED] = (\n"
     "                self._absent_reasons.get(ABSENT_ZONE_DISAPPEARED, 0) + 1\n"
     "            )\n"
     "            return THERMAL_BASIS_ABSENT, ABSENT_ZONE_DISAPPEARED",
     "            self._last_zones_reason = ABSENT_NO_ZONE_READABLE\n"
     "            self._basis_counts[THERMAL_BASIS_ABSENT] += 1\n"
     "            self._absent_reasons[ABSENT_NO_ZONE_READABLE] = (\n"
     "                self._absent_reasons.get(ABSENT_NO_ZONE_READABLE, 0) + 1\n"
     "            )\n"
     "            return THERMAL_BASIS_ABSENT, ABSENT_NO_ZONE_READABLE",
     "python"),
    ("eval_run: the phone status line names only the modal status",
     "deployment/jetson/eval_run.py",
     "        if status_counts:\n"
     "            # Every status seen, not only the modal one -- a build that spent\n"
     "            # 78 of 178 reports `severe` must not render as if it never left\n"
     "            # `nominal`.\n"
     "            breakdown = \", \".join(\n"
     '                f"{status} {n}" for status, n in sorted(status_counts.items(), key=lambda kv: -kv[1])\n'
     "            )\n"
     '            skin = phone.get("skin_temp_c")\n'
     "            skin_str = (\n"
     "                f\"; skin {phone.get('skin_zone')} p50 {skin['p50']:.1f} C, max {skin['max']:.1f} C\"\n"
     '                if skin else ""\n'
     "            )\n"
     '            lines.append(f"- phone: {breakdown} of {n_phone} reports{skin_str}")',
     "        if status_counts:\n"
     "            dominant = max(status_counts, key=status_counts.get)\n"
     '            skin = phone.get("skin_temp_c")\n'
     "            skin_str = (\n"
     "                f\"; skin {phone.get('skin_zone')} p50 {skin['p50']:.1f} C, max {skin['max']:.1f} C\"\n"
     '                if skin else ""\n'
     "            )\n"
     '            lines.append(f"- phone: {dominant} on {status_counts[dominant]} of {n_phone} reports{skin_str}")',
     "python"),
    ("phone: an older build's frame collapses to zero instead of null",
     "phone/transport/src/main/kotlin/com/dsrc/transport/PhoneTelemetry.kt",
     '                thermalStatusChanges = Fields.absentableInt(extensions, "thermal_status_changes"),',
     '                thermalStatusChanges = Fields.absentableInt(extensions, "thermal_status_changes") ?: 0,',
     "transport"),
    # Round 2 re-audit (m12): the encode half of the same distinction -- a
    # `null` count re-encoded as a reported zero on the way out, rather than
    # omitted the way a genuine absence is everywhere else on this frame.
    # Unpinned before this: the existing "decodes to null not zero" test built
    # its "older build" frame by stripping the key from an already-encoded
    # map by hand, so `toExtensions()`'s own null handling was never actually
    # called with a null value.
    ("phone: toExtensions re-encodes a null thermal_status_changes as zero",
     "phone/transport/src/main/kotlin/com/dsrc/transport/PhoneTelemetry.kt",
     '        thermalStatusChanges?.let { put("thermal_status_changes", JsonValue.Num(it)) }',
     '        put("thermal_status_changes", JsonValue.Num(thermalStatusChanges ?: 0))',
     "transport"),

    # Task 37 round 2: MINOR findings.
    ("thermal: the two percentile conventions in the report disagree",
     "deployment/jetson/sensors/thermal.py",
     "    def at(fraction: float) -> float:\n"
     "        # `n == 1` needs no special case: `rank` is then always 0, `lo` and\n"
     "        # `hi` both 0, and the interpolation term is zero -- the general\n"
     "        # formula already returns `ordered[0]`.\n"
     "        rank = fraction * (n - 1)\n"
     "        lo = int(rank)\n"
     "        hi = min(lo + 1, n - 1)\n"
     "        return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)",
     "    def at(fraction: float) -> float:\n"
     "        return ordered[min(n - 1, int(fraction * n))]",
     "python"),
    # Round 2 re-audit: `basis_counts["stale"]` was initialised alongside
    # `measured`/`absent` but incremented nowhere -- `stale` is a
    # `ThermalReading.basis` value `_latest_jetson` assigns only when read,
    # never while a sample is taken -- so it read as a measurement of
    # something that never happens (m7).
    ("thermal: basis_counts carries a stale entry that can never move",
     "deployment/jetson/sensors/thermal.py",
     '        self._basis_counts: dict[str, int] = {THERMAL_BASIS_MEASURED: 0, THERMAL_BASIS_ABSENT: 0}',
     "        self._basis_counts: dict[str, int] = {b: 0 for b in THERMAL_BASES}",
     "python"),
    ("eval_run: a thermal section with no summary prints nothing",
     "deployment/jetson/eval_run.py",
     "    if not thermal:\n"
     "        return []\n"
     '    lines = ["", "## Thermal", ""]\n'
     '    s = thermal.get("summary")',
     '    if not thermal or not thermal.get("summary"):\n'
     "        return []\n"
     '    lines = ["", "## Thermal", ""]\n'
     '    s = thermal.get("summary")',
     "python"),
    # Round 2 re-audit: `.get("passes_attempted")` guards the line but
    # `ev['passes_readable']`/`ev['passes_attempted']` index it, so a record
    # with the first key and not the second raised `KeyError` instead of
    # degrading (m8) -- every other read in this function uses `.get`.
    ("eval_run: a not_evaluable record missing passes_readable raises instead of degrading",
     "deployment/jetson/eval_run.py",
     "            if status == RULE_NOT_EVALUABLE:\n"
     '                missing = ", ".join(ev.get("missing") or []) or "a reason this record does not carry"\n'
     "                passes = (\n"
     "                    f\" ({ev.get('passes_readable')} of {ev.get('passes_attempted')} passes fully readable)\"\n"
     '                    if ev.get("passes_attempted") else ""\n'
     "                )",
     "            if status == RULE_NOT_EVALUABLE:\n"
     '                missing = ", ".join(ev.get("missing") or []) or "a reason this record does not carry"\n'
     "                passes = (\n"
     "                    f\" ({ev['passes_readable']} of {ev['passes_attempted']} passes fully readable)\"\n"
     '                    if ev.get("passes_attempted") else ""\n'
     "                )",
     "python"),
    # Round 2 re-audit: C2's fix reaches this rendering too -- a `fired`
    # record can now carry `missing` when the observation behind it was
    # incomplete, and the line must say so rather than reading as a complete
    # picture.
    ("eval_run: the fired line drops its incompleteness note",
     "deployment/jetson/eval_run.py",
     "            elif status == RULE_FIRED:\n"
     "                # `missing` can accompany `fired` too (a real transition observed\n"
     "                # while some cooling device on the same pass, or a later pass,\n"
     "                # never gave a reading) -- named here so the count is not read as\n"
     "                # a complete picture when it is not.\n"
     "                passes = (\n"
     "                    f\" ({ev.get('passes_readable')} of {ev.get('passes_attempted')} passes fully readable)\"\n"
     '                    if ev.get("missing") else ""\n'
     "                )\n"
     '                lines.append(f"- throttle events, {device}: fired -- {count} transitions{passes}")',
     "            elif status == RULE_FIRED:\n"
     '                lines.append(f"- throttle events, {device}: fired -- {count} transitions")',
     "python"),
    ("thermal: the millidegree threshold is a strict inequality",
     "deployment/jetson/sensors/thermal.py",
     "        celsius = number / 1000.0 if abs(number) >= 1000 else float(number)",
     "        celsius = number / 1000.0 if abs(number) > 1000 else float(number)",
     "python"),
    ("phone: the millidegree threshold is a strict inequality",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalZones.kt",
     "            val celsius = if (kotlin.math.abs(number) >= 1000L) number / 1000.0 else number.toDouble()",
     "            val celsius = if (kotlin.math.abs(number) > 1000L) number / 1000.0 else number.toDouble()",
     "app"),
    ("phone: the thermal watcher's default executor is never shut down",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalStatusWatcher.kt",
     "    fun stop() {\n"
     "        if (registered) {\n"
     "            unregister(listener)\n"
     "            registered = false\n"
     "        }\n"
     "        ownExecutor?.shutdown()\n"
     "    }",
     "    fun stop() {\n"
     "        if (registered) {\n"
     "            unregister(listener)\n"
     "            registered = false\n"
     "        }\n"
     "    }",
     "app"),
    # Round 2 re-audit (m4/m5): the KDoc claimed an injected executor is left
    # running and only this class's own default one is shut down, but the code
    # shut down *any* `ExecutorService` passed in, own or injected -- decided
    # in favour of the KDoc's ownership rule (a shared pool a caller hands in
    # is the caller's to shut down), which needed an actual ownership flag
    # rather than a type check to implement.
    ("phone: stop shuts down a caller-supplied executor too",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalStatusWatcher.kt",
     "    fun stop() {\n"
     "        if (registered) {\n"
     "            unregister(listener)\n"
     "            registered = false\n"
     "        }\n"
     "        ownExecutor?.shutdown()\n"
     "    }",
     "    fun stop() {\n"
     "        if (registered) {\n"
     "            unregister(listener)\n"
     "            registered = false\n"
     "        }\n"
     "        (activeExecutor as? ExecutorService)?.shutdown()\n"
     "    }",
     "app"),
    ("phone: the thermal watcher's own executor is only shut down when it was started",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalStatusWatcher.kt",
     "    fun stop() {\n"
     "        if (registered) {\n"
     "            unregister(listener)\n"
     "            registered = false\n"
     "        }\n"
     "        ownExecutor?.shutdown()\n"
     "    }",
     "    fun stop() {\n"
     "        if (registered) {\n"
     "            unregister(listener)\n"
     "            registered = false\n"
     "            ownExecutor?.shutdown()\n"
     "        }\n"
     "    }",
     "app"),
    # Round 2 re-audit (C1 root cause): `changesCount` and `lastTransition`
    # were two separate lock acquisitions with a binder call
    # (`power.currentThermalStatus`) between them on the caller's side, so the
    # listener firing in that window could move the count and the transition
    # out of step. `snapshot()` reads both under one lock acquisition instead.
    ("phone: snapshot drops the last transition",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalStatusWatcher.kt",
     "    fun snapshot(): Snapshot = synchronized(lock) { Snapshot(changes, last) }",
     "    fun snapshot(): Snapshot = synchronized(lock) { Snapshot(changes, null) }",
     "app"),

    # Task 38: failure event log.
    ("failures: an unreadable source is reported as quiet",
     "deployment/jetson/logio/failure_log.py",
     "                elif st.passes_attempted == 0 or st.passes_attempted != st.passes_readable:\n                    status = RULE_NOT_EVALUABLE",
     "                elif st.passes_attempted == 0 or st.passes_attempted != st.passes_readable:\n                    status = RULE_QUIET",
     "python"),
    ("failures: a source that lost its instrument closes as recovered",
     "deployment/jetson/logio/failure_log.py",
     "            st.last_missing = snap.missing or MISSING_NO_SOURCE\n            if st.open_episode is not None:\n                self._close_episode(st, source, now, t_wall, OUTCOME_UNOBSERVABLE)\n            return",
     "            st.last_missing = snap.missing or MISSING_NO_SOURCE\n            if st.open_episode is not None:\n                self._close_episode(st, source, now, t_wall, OUTCOME_RECOVERED)\n            return",
     "python"),
    ("failures: a session-scoped counter is diffed across a redial",
     "deployment/jetson/logio/failure_log.py",
     "                if st.open_episode is not None:\n                    self._close_episode(st, source, now, t_wall, OUTCOME_UNOBSERVABLE)\n                st.baseline_total = 0\n                st.baseline_by_reason = {}",
     "                if st.open_episode is not None:\n                    self._close_episode(st, source, now, t_wall, OUTCOME_UNOBSERVABLE)",
     "python"),
    ("failures: a counter that went backwards is clamped to zero",
     "deployment/jetson/logio/failure_log.py",
     "            delta = total - st.baseline_total",
     "            delta = max(0, total - st.baseline_total)",
     "python"),
    ("failures: the scan record is written only when something is open",
     "deployment/jetson/logio/failure_log.py",
     "        if self._sink is not None:\n            self._sink.write(record)",
     '        if self._sink is not None and record["open"]:\n            self._sink.write(record)',
     "python"),
    ("failures: the still window is a typed 3.0 rather than 3 x interval",
     "deployment/jetson/logio/failure_log.py",
     "        self._close_after_s = quiet_passes_to_close * interval_s",
     "        self._close_after_s = 3.0",
     "python"),
    ("failures: episode n is overwritten per pass instead of accumulated",
     "deployment/jetson/logio/failure_log.py",
     "                ep.n += delta",
     "                ep.n = delta",
     "python"),
    ("failures: the tick block reports a stale scan as measured",
     "deployment/jetson/logio/failure_log.py",
     "            basis = FAILURE_BASIS_MEASURED if age <= 2 * self._interval_s else FAILURE_BASIS_STALE",
     "            basis = FAILURE_BASIS_MEASURED",
     "python"),
    ("failures: the tick-loop exception is swallowed",
     "deployment/jetson/run_demo.py",
     "                failures.note_pipeline_exception(exc)\n            raise",
     "                failures.note_pipeline_exception(exc)",
     "python"),
    ("failures: an emitted reason is not checked against its source's vocabulary",
     "deployment/jetson/logio/failure_log.py",
     'def _here_refused_reason_valid(reason: str) -> bool:\n    base = reason.split(":", 1)[0]\n    return base in _OUTCOME_MEMBERS',
     "def _here_refused_reason_valid(reason: str) -> bool:\n    return True",
     "python"),
    ("eval_run: the failure section is computed and not rendered",
     "deployment/jetson/eval_run.py",
     "    if not failures:\n        return []",
     "    if True:\n        return []",
     "python"),
    ("eval_run: the record lists are swapped in the dataclass",
     "deployment/jetson/eval_run.py",
     "        failure_scans=failure_scans, failure_events=failure_events,",
     "        failure_scans=thermal_samples, failure_events=failure_events,",
     "python"),
    ("phone: a suppressed failure is not counted",
     "phone/app/src/main/kotlin/com/dsrc/phone/log/SessionLog.kt",
     "                suppressedSinceAccepted[kind] = (suppressedSinceAccepted[kind] ?: 0L) + 1L",
     "                Unit",
     "app"),
    ("phone: the failure line displaces a header line",
     "phone/app/src/main/kotlin/com/dsrc/phone/log/SessionLog.kt",
     "            if (writtenSoFar >= MAX_LINES_PER_KIND || lastAcceptedFailureSecond[kind] == second) {",
     "            if (false) {",
     "app"),

    # Task 38, round 2: the failure sampler's own `run_live` wiring, on the
    # same precedent as the thermal sampler's pins above -- M3 found five of
    # six wiring points reachable by no test, this being the mechanism.
    ("run_demo: the failure sampler is never constructed",
     "deployment/jetson/run_demo.py",
     '        if config["logio"]["failures"]:',
     '        if False and config["logio"]["failures"]:',
     "python"),
    ("run_demo: the tick's failures block is never written",
     "deployment/jetson/run_demo.py",
     '                if failures is not None:\n'
     '                    record["failures"] = failures.latest()',
     '                if failures is not None:\n'
     '                    record["_failures_disabled"] = failures.latest()',
     "python"),
    ("run_demo: the drive summary never gets a failures block",
     "deployment/jetson/run_demo.py",
     '            failures.stop()\n'
     '            summary["failures"] = failures.to_record()',
     '            failures.stop()\n'
     '            summary["_failures_disabled"] = failures.to_record()',
     "python"),
    ("run_demo: log_health.json is never written",
     "deployment/jetson/run_demo.py",
     "            logger.close()\n"
     "            _write_log_health(logger, logger.run_dir)",
     "            logger.close()",
     "python"),

    # Task 38's validation round: the 14 findings the validator confirmed by
    # reproducing them, each pinned so the specific defect cannot come back.
    # The property this pin guards: every call is credited to `run_total`
    # and to `by_reason_total` unconditionally, before the open-episode
    # branch below is looked at.
    ("failures: a lone blind tick is never credited to the source's own total",
     "deployment/jetson/logio/failure_log.py",
     "            st.run_total += 1\n"
     '            st.by_reason_total["no_frame"] = st.by_reason_total.get("no_frame", 0) + 1\n'
     "            if st.first_t_mono is None:\n"
     "                st.first_t_mono = now\n"
     "            st.last_t_mono = now\n\n"
     "            if st.open_episode is not None:\n"
     "                st.open_episode.n += 1\n"
     "                st.open_episode.last_t_mono = now\n"
     "            else:",
     "            if st.open_episode is not None:\n"
     "                st.run_total += 1\n"
     '                st.by_reason_total["no_frame"] = st.by_reason_total.get("no_frame", 0) + 1\n'
     "                st.last_t_mono = now\n"
     "                st.open_episode.n += 1\n"
     "                st.open_episode.last_t_mono = now\n"
     "            else:",
     "python"),
    ("failures: camera.dropped_unconsumed is always run-scoped",
     "deployment/jetson/logio/failure_log.py",
     'camera_dropped_scope = "session" if hasattr(camera, "decode_failures") else "run"',
     'camera_dropped_scope = "run"',
     "python"),
    ("failures: a backwards step overwrites the previous one instead of accumulating",
     "deployment/jetson/logio/failure_log.py",
     "                self._backwards.setdefault(source.name, []).append(\n"
     '                    {"from": st.baseline_total, "to": total, "t_mono": now}\n'
     "                )",
     "                self._backwards[source.name] = [\n"
     '                    {"from": st.baseline_total, "to": total, "t_mono": now}\n'
     "                ]",
     "python"),
    ("failures: an episode past the cap is opened again on every later movement pass",
     "deployment/jetson/logio/failure_log.py",
     "        kept = self._episode_count(st) < MAX_EPISODES_PER_SOURCE",
     "        kept = True",
     "python"),
    ("failures: a suppressed episode's occurrences are not counted anywhere",
     "deployment/jetson/logio/failure_log.py",
     "        if not episode.kept:\n"
     "            # Its occurrences still count -- toward `suppressed`, not toward\n"
     "            # a kept episode's own total -- but it never had a record and\n"
     "            # never counted as one of the source's `episodes`.\n"
     "            st.suppressed_total += episode.n\n"
     "            return",
     "        if not episode.kept:\n"
     "            return",
     "python"),
    ("failures: a raising accessor stops the whole pass rather than costing one source",
     "deployment/jetson/logio/failure_log.py",
     "        try:\n"
     "            snap = source.read(ctx)\n"
     "        except Exception:\n"
     "            # Every source after this one in the registry still gets its own\n"
     "            # pass -- an accessor's own bug costs this one source a reading,\n"
     "            # not the rest of the scan.\n"
     "            snap = SourceSnapshot(readable=False, missing=MISSING_ACCESSOR_RAISED)",
     "        snap = source.read(ctx)\n"
     "        if False:\n"
     "            snap = SourceSnapshot(readable=False, missing=MISSING_ACCESSOR_RAISED)",
     "python"),
    ("failures: a truncated detail is never counted",
     "deployment/jetson/logio/failure_log.py",
     "        if snap.detail_truncated:\n"
     "            st.truncated_details += 1",
     "        if False:\n"
     "            st.truncated_details += 1",
     "python"),
    ("eval_run: a suppressed occurrence is dropped from the phone's own total",
     "deployment/jetson/eval_run.py",
     'occurrences = int(record.get("n", 1)) + int(record.get("suppressed", 0))',
     'occurrences = int(record.get("n", 1))',
     "python"),
    ("eval_run: a still-open episode reports first_pass_n as its final occurrence count",
     "deployment/jetson/eval_run.py",
     '            "n": close_record.get("n") if close_record else None,',
     '            "n": close_record.get("n") if close_record else open_record.get("first_pass_n"),',
     "python"),
    ("eval_run: a phone log is dropped when the jetson log predates the failure event log",
     "deployment/jetson/eval_run.py",
     "    if phone_log_path is not None:\n"
     "        if failures is None:",
     "    if phone_log_path is not None and failures is not None:\n"
     "        if failures is None:",
     "python"),
    ("phone: a line dropped by the queue still burns its rate-cap slot",
     "phone/app/src/main/kotlin/com/dsrc/phone/log/SessionLog.kt",
     "            if (!enqueueLine(Json.encode(wrapped))) {\n"
     "                return\n"
     "            }\n"
     "            suppressedSinceAccepted.remove(kind)",
     "            enqueueLine(Json.encode(wrapped))\n"
     "            suppressedSinceAccepted.remove(kind)",
     "app"),
    ("app: the peer ending the session is not reported as a failure",
     "phone/app/src/main/kotlin/com/dsrc/phone/net/SessionHolder.kt",
     "                        sessionsEnded.incrementAndGet()\n"
     "                        onFailure(FailureKinds.LINK_SESSION_ENDED, monoClock(), wallClock(), end.toString())",
     "                        sessionsEnded.incrementAndGet()",
     "app"),
    ("phone: a write failure is not reported as log.self",
     "phone/app/src/main/kotlin/com/dsrc/phone/log/SessionLog.kt",
     "            offerFailure(\n"
     "                FailureKinds.LOG_SELF, monoClock(), wallClock(),\n"
     '                detail = "${e.javaClass.simpleName}: ${e.message}",\n'
     "            )",
     "            Unit",
     "app"),

    # -- this round's fixes (C1, M1, M2, M3, m1, m2, M4) -----------------------

    ("eval_run: a backwards-step list is read as a single step, not iterated",
     "deployment/jetson/eval_run.py",
     "    backwards = s.get(\"counter_went_backwards\") or {}\n"
     "    for name, steps in backwards.items():\n"
     "        for step in steps:\n"
     "            lines.append(\n"
     "                f\"- {name}: counter went backwards, {step.get('from')} -> {step.get('to')} \"\n"
     "                \"(the counter is right; this record is the one that is wrong)\"\n"
     "            )",
     "    backwards = s.get(\"counter_went_backwards\") or {}\n"
     "    for name, step in backwards.items():\n"
     "        lines.append(\n"
     "            f\"- {name}: counter went backwards, {step.get('from')} -> {step.get('to')} \"\n"
     "            \"(the counter is right; this record is the one that is wrong)\"\n"
     "        )",
     "python"),
    ("failures: a second pipeline exception while one episode is open is not counted",
     "deployment/jetson/logio/failure_log.py",
     "            if st.open_episode is None:\n"
     "                self._open_episode(\n"
     "                    st, source, now, t_wall, reason, 1, None,\n"
     "                    SourceSnapshot(readable=True, detail=detail, detail_truncated=truncated),\n"
     "                )\n"
     "            else:\n"
     "                st.open_episode.n += 1\n"
     "                st.open_episode.last_t_mono = now",
     "            if st.open_episode is None:\n"
     "                self._open_episode(\n"
     "                    st, source, now, t_wall, reason, 1, None,\n"
     "                    SourceSnapshot(readable=True, detail=detail, detail_truncated=truncated),\n"
     "                )",
     "python"),
    ("failures: camera.dropped_unconsumed never attaches a session id, so a redial diffs it",
     "deployment/jetson/logio/failure_log.py",
     "    snap = _fixed(\"unconsumed\", int(ctx.camera.dropped_frames))\n"
     "    return SourceSnapshot(readable=True, by_reason=snap.by_reason, session_id=ctx.pass_session_id)",
     "    return _fixed(\"unconsumed\", int(ctx.camera.dropped_frames))",
     "python"),
    ("failures: pipeline.exception is declared non-cumulative",
     "deployment/jetson/logio/failure_log.py",
     "        Source(\"pipeline.exception\", _read_pipeline_exception, None, None,\n"
     "               \"run\", True, True, \"jetson\"),",
     "        Source(\"pipeline.exception\", _read_pipeline_exception, None, None,\n"
     "               \"run\", False, True, \"jetson\"),",
     "python"),
    ("failures: MISSING silently drops a declared reason",
     "deployment/jetson/logio/failure_log.py",
     "MISSING = frozenset({\n"
     "    MISSING_NO_PHONE, MISSING_NO_SESSION, MISSING_NO_TELEMETRY,\n"
     "    MISSING_SESSION_MOVED, MISSING_NO_SOURCE, MISSING_ACCESSOR_RAISED,\n"
     "})",
     "MISSING = frozenset({\n"
     "    MISSING_NO_PHONE, MISSING_NO_SESSION, MISSING_NO_TELEMETRY,\n"
     "    MISSING_SESSION_MOVED, MISSING_NO_SOURCE,\n"
     "})",
     "python"),
    ("run_demo: the tick-loop exception is recorded but never reaches the caller",
     "deployment/jetson/run_demo.py",
     "            if failures is not None:\n"
     "                failures.note_pipeline_exception(exc)\n"
     "            raise",
     "            if failures is not None:\n"
     "                failures.note_pipeline_exception(exc)\n"
     "            if True:\n"
     "                return\n"
     "            raise",
     "python"),
    ("run_demo: a blind tick is never counted",
     "deployment/jetson/run_demo.py",
     "            if frame is None:\n"
     "                if failures is not None:\n"
     "                    failures.note_no_frame(end_of_stream=camera.end_of_stream)\n"
     "                if camera.end_of_stream:",
     "            if frame is None:\n"
     "                if 0:\n"
     "                    if failures is not None:\n"
     "                        failures.note_no_frame(end_of_stream=camera.end_of_stream)\n"
     "                if camera.end_of_stream:",
     "python"),
    ("app: no IMU hardware is never reported as a failure",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuSource.kt",
     "            Log.w(TAG, \"no IMU: accelerometer=$accelerometer gyroscope=$gyroscope\")\n"
     "            onFailure(FailureKinds.IMU_NO_HARDWARE, \"accelerometer=$accelerometer gyroscope=$gyroscope\")\n"
     "            return",
     "            Log.w(TAG, \"no IMU: accelerometer=$accelerometer gyroscope=$gyroscope\")\n"
     "            return",
     "app"),

    # -- task 38 re-audit: the hardware drive's own findings (D1-D7) -----------

    ("failures: a pseudo-source's episode is closed by the generic quiet streak",
     "deployment/jetson/logio/failure_log.py",
     "        if source.name in PSEUDO_SOURCES:",
     "        if source.name in ():",
     "python"),
    ("failures: missing is cleared the moment a source becomes readable again",
     "deployment/jetson/logio/failure_log.py",
     "        st.last_readable = True\n"
     "        # `last_missing` is not reset here: it names the reason the most\n"
     "        # recent UNREADABLE pass gave, not the current pass. A source read\n"
     "        # as `not_evaluable` at some point mid-run and readable again by\n"
     "        # teardown must still report what was missing -- clearing it on\n"
     "        # every readable pass left every such row with an empty `missing`.\n"
     "        st.passes_readable += 1",
     "        st.last_readable = True\n"
     "        st.last_missing = None\n"
     "        st.passes_readable += 1",
     "python"),
    ("eval_run: the not-evaluable line names the readable count, not the unreadable one",
     "deployment/jetson/eval_run.py",
     "            passes_attempted = row.get(\"passes_attempted\", 0)\n"
     "            passes_unreadable = passes_attempted - row.get(\"passes_readable\", 0)\n"
     "            lines.append(\n"
     "                f\"- {name}: NOT EVALUABLE on {passes_unreadable} of \"",
     "            passes_attempted = row.get(\"passes_attempted\", 0)\n"
     "            passes_unreadable = row.get(\"passes_readable\", 0)\n"
     "            lines.append(\n"
     "                f\"- {name}: NOT EVALUABLE on {passes_unreadable} of \"",
     "python"),
    ("failures: phone.dropped reads a stale telemetry snapshot through a down session",
     "deployment/jetson/logio/failure_log.py",
     "    if _session_stats_or_none(phone) is None:\n"
     "        # `PhoneLink.telemetry` is cleared on a rebind (`_rebind`), not on\n"
     "        # session loss -- so while the link is down, it still holds the last\n"
     "        # report received before the outage. Reading it without this gate\n"
     "        # reports the phone's drop counters as current, unchanged, for the\n"
     "        # whole outage, which is exactly the recovery `link.down` exists to\n"
     "        # let every phone-side source refuse to claim.\n"
     "        return SourceSnapshot(readable=False, missing=MISSING_NO_SESSION)\n"
     "    telemetry = phone.telemetry\n"
     "    if telemetry is None:\n"
     "        return SourceSnapshot(readable=False, missing=MISSING_NO_TELEMETRY)\n"
     "    dropped = telemetry.dropped or {}",
     "    telemetry = phone.telemetry\n"
     "    if telemetry is None:\n"
     "        return SourceSnapshot(readable=False, missing=MISSING_NO_TELEMETRY)\n"
     "    dropped = telemetry.dropped or {}",
     "python"),
    ("failures: phone.here_errors reads a stale telemetry snapshot through a down session",
     "deployment/jetson/logio/failure_log.py",
     "    if _session_stats_or_none(phone) is None:\n"
     "        # See `_read_phone_dropped`: the same stale-snapshot hazard applies\n"
     "        # to `here_errors`, read off the same `telemetry` object.\n"
     "        return SourceSnapshot(readable=False, missing=MISSING_NO_SESSION)\n"
     "    telemetry = phone.telemetry\n"
     "    if telemetry is None:\n"
     "        return SourceSnapshot(readable=False, missing=MISSING_NO_TELEMETRY)\n"
     "    total = int(telemetry.here_errors)",
     "    telemetry = phone.telemetry\n"
     "    if telemetry is None:\n"
     "        return SourceSnapshot(readable=False, missing=MISSING_NO_TELEMETRY)\n"
     "    total = int(telemetry.here_errors)",
     "python"),
    ("failures: by_reason_total credits only the dominant reason, not every reason that moved",
     "deployment/jetson/logio/failure_log.py",
     "            if reason_deltas:\n"
     "                for key, value in reason_deltas.items():\n"
     "                    st.by_reason_total[key] = st.by_reason_total.get(key, 0) + value\n"
     "            else:\n"
     "                st.by_reason_total[reason] = st.by_reason_total.get(reason, 0) + delta",
     "            st.by_reason_total[reason] = st.by_reason_total.get(reason, 0) + delta",
     "python"),
    ("eval_run: the episode count and the closed-episode total are always reported as one number",
     "deployment/jetson/eval_run.py",
     "        if len(episodes) == closed_total:",
     "        if True:",
     "python"),
    # `note_frame` opens with the identical `st = ...; source = ...` pair,
    # so a two-line anchor matches in both places. The anchor carries
    # `st.run_total += 1` as well -- `note_no_frame`'s own next statement,
    # which `note_frame` does not share -- to stay unique to the call this
    # pin is about.
    ("failures: camera.blind_ticks.passes_attempted also counts direct notifications",
     "deployment/jetson/logio/failure_log.py",
     "            st = self._state[\"camera.blind_ticks\"]\n"
     "            source = self._by_name[\"camera.blind_ticks\"]\n"
     "\n"
     "            st.run_total += 1",
     "            st = self._state[\"camera.blind_ticks\"]\n"
     "            st.passes_attempted += 1\n"
     "            st.passes_readable += 1\n"
     "            st.last_readable = True\n"
     "            st.last_readable_t_mono = now\n"
     "            source = self._by_name[\"camera.blind_ticks\"]\n"
     "\n"
     "            st.run_total += 1",
     "python"),
    ("failures: pipeline.exception.passes_attempted also counts direct notifications",
     "deployment/jetson/logio/failure_log.py",
     "            st = self._state[\"pipeline.exception\"]\n"
     "            st.run_total += 1",
     "            st = self._state[\"pipeline.exception\"]\n"
     "            st.passes_attempted += 1\n"
     "            st.passes_readable += 1\n"
     "            st.last_readable = True\n"
     "            st.run_total += 1",
     "python"),

    # Task 39. Session summary: seven instrument axes (answered of attempted,
    # plus a reason census when they differ, never a ratio or a health word),
    # nine cross-record reconciliations, and the two new fields on
    # `reference_from`. Every entry below was verified against a mutated
    # mirror before being added here, not merely written down.
    ("summary: a zero-answering axis is dropped from the headline list",
     "deployment/jetson/eval_run.py",
     '        axis["unbuildable"] is None\n'
     '        and axis["attempted"] not in (None, 0)\n'
     '        and axis["answered"] == axis["attempted"]',
     '        axis["unbuildable"] is None\n'
     '        and axis["attempted"] is not None\n'
     '        and axis["answered"] == axis["attempted"]',
     "python"),
    ("summary: an axis's census is derived by subtraction",
     "deployment/jetson/eval_run.py",
     '    census, violations = _census_and_violations(counts, THERMAL_FAILURES_VOCABULARY)\n'
     '    return AxisResult(\n'
     '        axis="thermal", attempted=attempted, answered=answered,\n'
     '        attempted_is="ticks carrying a thermal block",\n'
     '        answered_is="ticks whose thermal.jetson.basis is measured",\n'
     '        unanswered_by_reason=census,',
     '    census, violations = _census_and_violations(counts, THERMAL_FAILURES_VOCABULARY)\n'
     '    return AxisResult(\n'
     '        axis="thermal", attempted=attempted, answered=answered,\n'
     '        attempted_is="ticks carrying a thermal block",\n'
     '        answered_is="ticks whose thermal.jetson.basis is measured",\n'
     '        unanswered_by_reason={"unanswered": attempted - answered},',
     "python"),
    ("sensing_loop: a phone never heard from reports zero API calls",
     "deployment/jetson/policy/sensing_loop.py",
     '                "here_calls": None, "here_errors": None, "absent": "no_telemetry"}',
     '                "here_calls": 0, "here_errors": None, "absent": "no_telemetry"}',
     "python"),
    ("summary: achieved is averaged over ticks rather than reports",
     "deployment/jetson/eval_run.py",
     '            by_at_mono[ref["at_mono"]] = ref',
     '            by_at_mono[id(ref)] = ref',
     "python"),
    ("summary: percentiles are reported for a modality slower than the window",
     "deployment/jetson/eval_run.py",
     "            elif lambda_per_window < 1.0:",
     "            elif False:",
     "python"),
    ("summary: achieved is compared to commanded on a shadow drive",
     "deployment/jetson/eval_run.py",
     '    if not ever_live:\n'
     '        return False, f"mode shadow on {ticks} of {ticks} decisions"',
     '    if False:\n'
     '        return False, f"mode shadow on {ticks} of {ticks} decisions"',
     "python"),
    ("summary: a HERE counter that restarts at zero is differenced across the redial",
     "deployment/jetson/eval_run.py",
     '        bucket = by_session.setdefault(t.get("session_id"), {"calls": [], "errors": []})',
     '        bucket = by_session.setdefault(None, {"calls": [], "errors": []})',
     "python"),
    ("summary: the blind-tick reconciliation is not checked",
     "deployment/jetson/eval_run.py",
     "    if total == blind_ticks:",
     "    if True:",
     "python"),
    ("summary: a missing summary.json is reported as a held reconciliation",
     "deployment/jetson/eval_run.py",
     '            Reconciliation("blind_ticks_matches_camera_source", "unavailable", reason),',
     '            Reconciliation("blind_ticks_matches_camera_source", "held"),',
     "python"),
    ("eval_run: a zero-tick drive produces no report",
     "deployment/jetson/eval_run.py",
     "    if not loaded.ticks:",
     "    if False:",
     "python"),
    ("eval_run: the Overall line drops its axis clause",
     "deployment/jetson/eval_run.py",
     "    if session is not None:\n"
     '        overall_line += f" -- {_overall_clause(session)}"',
     "    if session is not None:\n"
     "        pass  # clause dropped",
     "python"),
    ("summary: a vocabulary violation is absorbed into a known key",
     "deployment/jetson/eval_run.py",
     '            reason = jetson.get("reason") or "unstated"\n'
     "            counts[reason] = counts.get(reason, 0) + 1",
     '            reason = jetson.get("reason") or "unstated"\n'
     "            if reason not in THERMAL_FAILURES_VOCABULARY:\n"
     "                reason = sorted(THERMAL_FAILURES_VOCABULARY)[0]\n"
     "            counts[reason] = counts.get(reason, 0) + 1",
     "python"),

    # Task 39's validation round (C1/C2 criticals; M1-M7 majors; the MINOR and
    # TEST GAPS items). Every entry below was verified against a mutated
    # mirror before being added here, matching the process the entries above
    # it describe.
    ("summary: a source-row reconciliation holds when there are no rows to compare",
     "deployment/jetson/eval_run.py",
     '    sources = failures_summary.get("sources") or {}\n'
     "    if not sources:\n"
     '        return Reconciliation(name, "unavailable", "no source rows to compare",\n'
     '                               compared=0, compared_is="rows_compared")',
     '    sources = failures_summary.get("sources") or {}\n'
     "    if False:\n"
     '        return Reconciliation(name, "unavailable", "no source rows to compare",\n'
     '                               compared=0, compared_is="rows_compared")',
     "python"),
    ("summary: blind_ticks holds when neither number is actually present",
     "deployment/jetson/eval_run.py",
     '    total = None if row is None else row.get("total")\n'
     "    if total is None or blind_ticks is None:",
     '    total = None if row is None else row.get("total")\n'
     "    if False:",
     "python"),
    ("summary: events_written holds when there are no source rows to sum",
     "deployment/jetson/eval_run.py",
     '    sources = failures_summary.get("sources") or {}\n'
     "    if not sources:\n"
     "        return Reconciliation(\n"
     '            "events_written_matches_open_and_close_records", "unavailable",\n'
     '            "no source rows to sum events_written over", compared=0, compared_is="rows_compared",\n'
     "        )",
     '    sources = failures_summary.get("sources") or {}\n'
     "    if False:\n"
     "        return Reconciliation(\n"
     '            "events_written_matches_open_and_close_records", "unavailable",\n'
     '            "no source rows to sum events_written over", compared=0, compared_is="rows_compared",\n'
     "        )",
     "python"),
    ("summary: triggers_match_summary holds when there are no sensing ticks",
     "deployment/jetson/eval_run.py",
     "    sensing_ticks_n = len(_sensing_ticks(loaded.ticks))\n"
     "    if sensing_ticks_n == 0:",
     "    sensing_ticks_n = len(_sensing_ticks(loaded.ticks))\n"
     "    if False:",
     "python"),
    ("summary: here_calls_ge_responses_received holds when no tick carries here_calls",
     "deployment/jetson/eval_run.py",
     "    if responses is None or calls_total is None:",
     "    if responses is None:",
     "python"),
    ("api_calls: no_telemetry is keyed on here_calls rather than reference.absent",
     "deployment/jetson/eval_run.py",
     '        if ref.get("absent") is not None:\n'
     "            counts[REFERENCE_NO_TELEMETRY] = counts.get(REFERENCE_NO_TELEMETRY, 0) + 1\n"
     '        elif ref.get("here_calls") is None:',
     '        if ref.get("here_calls") is None:\n'
     "            counts[REFERENCE_NO_TELEMETRY] = counts.get(REFERENCE_NO_TELEMETRY, 0) + 1\n"
     '        elif ref.get("here_calls") is None:',
     "python"),
    ("api_calls: a shape violation is merged into no_telemetry instead of censused separately",
     "deployment/jetson/eval_run.py",
     "counts[API_CALLS_FIELD_NOT_RECORDED] = counts.get(API_CALLS_FIELD_NOT_RECORDED, 0) + 1",
     "counts[REFERENCE_NO_TELEMETRY] = counts.get(REFERENCE_NO_TELEMETRY, 0) + 1",
     "python"),
    ("summary: the Overall line folds unbuildable axes into did-not-answer",
     "deployment/jetson/eval_run.py",
     "    did_not = len(axes) - fully_answered - unbuildable",
     "    did_not = len(axes) - fully_answered",
     "python"),
    ("triggers: the not-evaluable-by-rule census is dropped from the axis record",
     "deployment/jetson/eval_run.py",
     "            for name in RULES:\n"
     '                if rules[name].get("status") == RULE_NOT_EVALUABLE:\n'
     "                    not_evaluable_by_rule[name] = not_evaluable_by_rule.get(name, 0) + 1",
     "            pass",
     "python"),
    ("summary: ## Sensing is never appended to report.md",
     "deployment/jetson/eval_run.py",
     "    if session is not None:\n"
     '        lines += _sensing_lines(session.get("sensing"))\n'
     "    join = r.get(\"phone_join\")",
     "    if False:\n"
     '        lines += _sensing_lines(session.get("sensing"))\n'
     "    join = r.get(\"phone_join\")",
     "python"),
    ("summary: a HERE counter that decreases within a session is silently clamped",
     "deployment/jetson/eval_run.py",
     "        if last_calls - first_calls < 0:\n"
     "            backwards_sessions.append(session_id)",
     "        if False:\n"
     "            backwards_sessions.append(session_id)",
     "python"),
    ("summary: a drive that never recorded here_calls reports a measured zero",
     "deployment/jetson/eval_run.py",
     '        "calls_total": None if not_measured else calls_total,',
     '        "calls_total": calls_total,',
     "python"),
    ("summary: zero_calls_because fires without two observations to support it",
     "deployment/jetson/eval_run.py",
     '        and any(row["observations"] >= 2 for row in here["by_session"])',
     "        and True",
     "python"),
    ("summary: a failed reconciliation is keyed on its check name, not its axis",
     "deployment/jetson/eval_run.py",
     '    axis = _RECONCILIATION_AXIS.get(rec["name"], rec["name"])',
     '    axis = rec["name"]',
     "python"),
    ("summary: the zero-tick report never names a dropped log",
     "deployment/jetson/eval_run.py",
     "    if log_health is not None:\n"
     '        lines += _log_health_lines({"log_health": log_health})',
     "    if False:\n"
     '        lines += _log_health_lines({"log_health": log_health})',
     "python"),
    ("summary: inputs.metadata_jsonl is a literal True",
     "deployment/jetson/eval_run.py",
     '            "metadata_jsonl": metadata_jsonl_present,',
     '            "metadata_jsonl": True,',
     "python"),
    ("summary: the rates axis headline claims a noun (reports) its count is not purely made of",
     "deployment/jetson/eval_run.py",
     '"rates": "telemetry observations"',
     '"rates": "reports"',
     "python"),
    ("summary: lambda_per_window is computed and then discarded",
     "deployment/jetson/eval_run.py",
     '            entry["lambda_per_window"] = lambda_per_window',
     "            pass",
     "python"),
    ("summary: the quantisation at lambda == 1.0 is not named",
     "deployment/jetson/eval_run.py",
     "                if lambda_per_window == 1.0:",
     "                if False:",
     "python"),
    ("summary: clamped_ticks and thermal_scaled_ticks are rendered nowhere",
     "deployment/jetson/eval_run.py",
     '        if row["clamped_ticks"] or row["thermal_scaled_ticks"]:',
     "        if False:",
     "python"),
    ("sensing_result: a null summary[phone] block raises AttributeError",
     "deployment/jetson/eval_run.py",
     '    here["responses_received"] = (\n'
     '        ((summary or {}).get("phone") or {}).get("here") or {}\n'
     '    ).get("responses_received")',
     '    here["responses_received"] = ((summary or {}).get("phone", {}).get("here", {}) or {}).get(\n'
     '        "responses_received"\n'
     "    )",
     "python"),
    ("rates axis: a malformed sensing block raises KeyError instead of degrading",
     "deployment/jetson/eval_run.py",
     '        ref = t["sensing"].get("reference") or {}\n'
     '        if ref.get("absent") is not None:\n'
     "            no_telemetry_ticks += 1",
     '        ref = t["sensing"]["reference"]\n'
     '        if ref.get("absent") is not None:\n'
     "            no_telemetry_ticks += 1",
     "python"),
    ("sensing_result: a malformed sensing block raises KeyError instead of degrading",
     "deployment/jetson/eval_run.py",
     "        raw = [\n"
     "            (\n"
     '                t["sensing"].get("decided_at_mono"),\n'
     '                (t["sensing"].get("rates") or {}).get(key),\n'
     '                (t["sensing"].get("rates") or {}).get("camera_hz"),\n'
     "            )\n"
     "            for t in sensing_ticks\n"
     "        ]",
     "        raw = [\n"
     '            (t["sensing"]["decided_at_mono"], t["sensing"]["rates"][key],\n'
     '             (t["sensing"].get("rates") or {}).get("camera_hz"))\n'
     "            for t in sensing_ticks\n"
     "        ]",
     "python"),
    ("reference shape reconciliation: a pre-task-39 log fails rather than reads unavailable",
     "deployment/jetson/eval_run.py",
     '            "reference_absent_iff_fields_null", "unavailable",\n            HERE_CALLS_PREDATES_TASK_39,',
     '            "reference_absent_iff_fields_null", "failed",\n            HERE_CALLS_PREDATES_TASK_39,',
     "python"),
]

RESULTS = {
    "transport": [ROOT / "phone/transport/build/test-results"],
    "app": [ROOT / "phone/app/build/test-results"],
    "python": [ROOT / "build/pytest-results"],
}


def failing_tests(kind):
    """Names of the tests that failed, the total testcases seen, and whether
    every JUnit XML file found parsed cleanly.

    Parsed as XML, not by regex over the attributes. Gradle writes `name` first and
    pytest writes `classname` first, so a pattern that fixes the order silently matches
    nothing on one of the two -- which is how the first version of this reported both
    Python pins as SURVIVED while they were being caught perfectly well. A harness that
    reports a false SURVIVED is the same failure as one that reports a false CAUGHT.

    No XML at all, an XML file truncated mid-write (a killed run, a full
    disk), and a genuine zero-failures result all produce the same empty
    name list, and an empty list on its own reads as SURVIVED -- the verdict
    a real pass gets. The third element returned here is False for either of
    the first two, so the caller can tell "nothing failed" apart from
    "nothing was observed".
    """
    names = []
    total = 0
    found_xml = False
    parse_failed = False
    for base in RESULTS[kind]:
        for report in base.rglob("*.xml"):
            found_xml = True
            try:
                root = ElementTree.parse(report).getroot()
            except ElementTree.ParseError:
                parse_failed = True
                continue
            for case in root.iter("testcase"):
                total += 1
                if case.find("failure") is None and case.find("error") is None:
                    continue
                cls = (case.get("classname") or "").rsplit(".", 1)[-1]
                names.append(f"{cls}.{case.get('name')}" if cls else str(case.get("name")))
    return names, total, found_xml and not parse_failed


BUILD_ERROR = ["<the mutation did not compile>"]

#: A run that did not settle anything about a test outcome: the pytest
#: process exited with a code other than 0 or 1 (2 interrupted, 3 internal
#: error, 4 usage error, 5 no tests collected), or wrote no usable JUnit XML
#: at all, or collected a testcase count that disagrees with what a clean
#: tree collects. An empty failing_tests() result under any of those is
#: evidence nothing was OBSERVED, not evidence nothing failed -- reporting
#: it as SURVIVED is the false-negative half of the asymmetry
#: `failing_tests`'s own docstring names, closed here rather than there
#: because it takes a returncode and a testcase count neither `run()`'s
#: Gradle arm nor `failing_tests` alone has both of.
INCONCLUSIVE = ["<inconclusive: the run did not produce a trustworthy result>"]

#: pytest's own exit codes that say anything about a TEST outcome. Every
#: other code (2, 3, 4, 5) means the run itself did not complete as a test
#: run.
PYTEST_VERDICT_RETURNCODES = frozenset({0, 1})

#: Testcases a clean, unmutated Python suite collects today. A mutated run
#: collecting a different count did not exercise the same suite it is being
#: scored against -- a partial file copy or a conftest import that silently
#: drops a whole module both trivially report zero failures, because most
#: of the suite never ran at all. Update this when the suite's own test
#: count changes; a run reporting a different count is not proof of drift,
#: only that it needs checking against `pytest --collect-only` before being
#: trusted either way.
EXPECTED_PYTHON_TESTCASES = 2060


def is_collection_error(names):
    """Whether the run failed to import a module rather than failing a test.

    pytest files a collection error as a testcase whose `name` is the file path, so a
    mutation that leaves the source unparseable shows up as every test in that module
    "failing" -- and a harness that only asks whether something failed reports a pin.
    That is the Python spelling of the compiler-measuring failure BUILD_ERROR exists
    for, and one entry here was doing it: deleting a two-line block orphaned the
    nested `if` beneath it.
    """
    return any(name.endswith(".py") or name.endswith(".py]") for name in names)


def run(kind):
    """Run the suite and report *which* tests failed, not merely that something did.

    Keying on the process exit code was not enough, and the reason is the whole
    argument for this script existing. A run reported SURVIVED for the outbound
    framing-refusal pin while the same mutation, applied by hand and run against the
    same test, failed 5 times out of 5. Either verdict might have been the true one and
    the harness could not say which -- so a tool built because pins lapse silently was
    itself producing a verdict nobody could check.

    Naming the failing test settles it. A mutation that is caught says which assertion
    caught it, and one that is "caught" by an unrelated failure is now visible as such
    instead of counting as a pass. That settles a false CAUGHT; a false SURVIVED
    needs its own check. A pytest process that never started a test run at all -- a
    bad flag, a crash, an OOM kill, a conftest import failure, an XML truncated by a
    killed run, a partial tree copy that collects 18 testcases instead of the whole
    suite -- returns the same empty `failing_tests()` list a real pass does. The
    subprocess returncode and the collected testcase count are both checked here so
    those are reported INCONCLUSIVE rather than SURVIVED.
    """
    for base in RESULTS[kind]:
        if base.exists():
            shutil.rmtree(base)          # or a previous run's failures count as this one's
    if kind == "python":
        result = subprocess.run(
            [".venv/bin/python3", "-m", "pytest", "-q", "deployment/jetson/tests/",
             "-p", "no:cacheprovider", f"--junit-xml={RESULTS['python'][0]}/results.xml"],
            capture_output=True, text=True,
        )
        if result.returncode not in PYTEST_VERDICT_RETURNCODES:
            return INCONCLUSIVE
    else:
        target = ":transport:test" if kind == "transport" else ":app:test"
        result = subprocess.run(GRADLE + [target, "--rerun-tasks"], capture_output=True, text=True)
        if "e: file://" in result.stdout or "e: file://" in result.stderr:
            return BUILD_ERROR
    names, total, usable = failing_tests(kind)
    if not usable:
        return INCONCLUSIVE
    if kind == "python" and total != EXPECTED_PYTHON_TESTCASES:
        return INCONCLUSIVE
    return names

# A killed run used to leave its mutation in the tree. The `finally` below restores on an
# exception and on Ctrl-C, and not on a SIGKILL -- and one of those left
# `UNKNOWN_VALUE -> WRONG_TYPE` applied to Session.kt, which then surfaced as two unrelated
# tests failing in a later run. Chasing that as a regression is exactly the wrong trail.
#
# So the pristine text goes to a sidecar first. If it exists on startup the previous run
# died mid-mutation, and this restores it before doing anything else rather than mutating
# an already-mutated file.
SIDECAR = pathlib.Path(".remutate-restore")

if SIDECAR.exists():
    saved = SIDECAR.read_text().split("\n", 1)
    target = ROOT / saved[0]
    print(f"  a previous run died mid-mutation; restoring {saved[0]}")
    target.write_text(saved[1])
    SIDECAR.unlink()

# Optional kind filter: `python3 scripts/remutate.py python` runs only the Python
# entries. The docstring says to run this after landing a batch of fixes, and a batch
# is almost always one kind -- rebuilding both Gradle suites per mutation to check a
# Python change is why it was reached for less often than it should have been.
# And an optional name filter, `--name=SUBSTR`, repeatable. Re-running one batch of
# fixes against 95 Python entries costs about an hour and a half of full pytest runs
# to check four of them, which is why the entries touched by a round were often left
# unverified. Kind and name compose: both must match.
ARGV = sys.argv[1:]
NAME_FILTERS = [a.split("=", 1)[1] for a in ARGV if a.startswith("--name=")]
WANTED = [a for a in ARGV if not a.startswith("--name=")] or None
if WANTED:
    unknown = [k for k in WANTED if k not in RESULTS]
    if unknown:
        sys.exit(f"unknown kind(s) {unknown}; known: {sorted(RESULTS)}")

survived = []
for name, rel, old, new, kind in MUTATIONS:
    if WANTED and kind not in WANTED:
        continue
    if NAME_FILTERS and not any(f in name for f in NAME_FILTERS):
        continue
    path = ROOT / rel
    keep = path.read_text()
    if keep.count(old) > 1:
        # Whichever site comes first is a property of the file's layout, not of anything
        # this registry chose; the device harness had an entry mutating the wrong sensor
        # under the other one's name for exactly this reason.
        print(f"  ANCHOR AMBIGUOUS   {name} ({keep.count(old)} sites)")
        survived.append(name + f" [ambiguous anchor, {keep.count(old)} sites]")
        continue
    if old not in keep:
        print(f"  SKIP  {name}  (anchor not found -- the code moved)")
        survived.append(name + " [anchor moved]")
        continue
    mutated = keep.replace(old, new, 1)
    SIDECAR.write_text(f"{rel}\n{keep}")
    try:
        path.write_text(mutated)
        failed = run(kind)
    finally:
        path.write_text(keep)
        SIDECAR.unlink(missing_ok=True)
    if failed and failed is not BUILD_ERROR and failed is not INCONCLUSIVE and is_collection_error(failed):
        failed = BUILD_ERROR
    if failed == BUILD_ERROR:
        # Distinct from both verdicts. A mutation that does not compile -- or, in
        # Python, does not import -- proves nothing about any test, and reporting it
        # as CAUGHT is how one of these entries came to measure the compiler for weeks.
        survived.append(name + " [did not build/import]")
        print(f"  DID NOT BUILD      {name}")
    elif failed == INCONCLUSIVE:
        # A third state, distinct from both verdicts for the same reason BUILD_ERROR
        # is: the run did not produce a trustworthy answer either way, so it counts
        # against the exit code (a clean run is not one with an inconclusive result
        # sitting in it) without being reported as a caught OR a survived mutation.
        survived.append(name + " [inconclusive -- the run did not produce a trustworthy result]")
        print(f"  INCONCLUSIVE       {name}")
    elif failed:
        print(f"  CAUGHT ({len(failed)})         {name}")
        print(f"                     by {failed[0]}")
    else:
        survived.append(name)
        print(f"  *** SURVIVED ***   {name}")

print()
print("survived:", len(survived), survived if survived else "(none)")
sys.exit(1 if survived else 0)
