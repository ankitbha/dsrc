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
     "        if (sdkInt < android.os.Build.VERSION_CODES.R) return null", "", "app"),
    ("telemetry: a NaN headroom reaches the wire",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ThermalReader.kt",
     "        if (!value.isFinite()) return null", "", "app"),
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
     '    if not isinstance(value, str):\n        raise MessageError(f"{field} is {type(value).__name__}, expected str", REASON_WRONG_TYPE)',
     '    if value is None or not isinstance(value, str):\n        raise MessageError(f"{field} is {type(value).__name__}, expected str", REASON_WRONG_TYPE)',
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
     '        src["ego_acceleration"] = "derived" if accel_derived else "fallback_neutral"',
     '        src["ego_acceleration"] = "derived"',
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
    ("eval: a truncated log is analysed as a complete run",
     "deployment/jetson/eval_run.py",
     '        "log_complete": bool(unparseable == 0 and (shortfall is None or shortfall == 0)),',
     '        "log_complete": True,',
     "python"),
    ("eval: unparseable lines are skipped without being counted",
     "deployment/jetson/eval_run.py",
     "                unparseable += 1\n                continue",
     "                continue",
     "python"),
    ("eval: a short log does not fail the run",
     "deployment/jetson/eval_run.py",
     '        "overall_pass": overall and integrity["log_complete"],',
     '        "overall_pass": overall,',
     "python"),
    ("eval: the log is not compared against the count the run reported",
     "deployment/jetson/eval_run.py",
     "        shortfall = expected_ticks - len(ticks)",
     "        shortfall = 0",
     "python"),
    # Joint round 7: claims against behaviour, and what grows over a long drive.
    # The tick log is the only per-tick collection with no cap -- 5,251 bytes a
    # record, 567 MB/hour at 30 fps -- and its writer had no guard at all.
    ("logger: a failed write kills the writer silently",
     "deployment/jetson/logio/metadata_logger.py",
     "            except (OSError, ValueError) as exc:",
     "            except ZeroDivisionError as exc:",
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
     '            "local_queue_estimate": "derived" if abs_speeds else "fallback_neutral",',
     '            "local_queue_estimate": "derived",',
     "python"),
    ("builder: the etiquette flag claims a derived segment density",
     "deployment/jetson/perception/observation_builder.py",
     '            "uncongested_low_speed_flag": "approximated",',
     '            "uncongested_low_speed_flag": "derived",',
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
     "            checks[Trigger.EVENT] = RuleCheck(status=RULE_NOT_EVALUABLE,",
     "            checks[Trigger.EVENT] = RuleCheck(status=RULE_QUIET,",
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
    # The reference block has to say a phone was never heard from, not that it
    # reported zero -- those are different drives, and a candidate scored
    # against a manufactured "achieved nothing" reading is scored against data
    # that does not exist. Caught by the absence test, which requires all three
    # fields null together rather than a zeroed achieved map.
    ("sensing_loop: the reference block reports 0.0 achieved when the phone never reported",
     "deployment/jetson/policy/sensing_loop.py",
     'return {"achieved": None, "dropped": None, "age_s": None, "absent": "no_telemetry"}',
     'return {"achieved": {key: 0.0 for key in RATE_KEYS},'
     ' "dropped": {key: 0 for key in DROP_KEYS}, "age_s": None, "absent": "no_telemetry"}',
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
]

RESULTS = {
    "transport": [ROOT / "phone/transport/build/test-results"],
    "app": [ROOT / "phone/app/build/test-results"],
    "python": [ROOT / "build/pytest-results"],
}


def failing_tests(kind):
    """Names of the tests that failed, from the JUnit XML the run just wrote."""
    # Parsed as XML, not by regex over the attributes. Gradle writes `name` first and
    # pytest writes `classname` first, so a pattern that fixes the order silently matches
    # nothing on one of the two -- which is how the first version of this reported both
    # Python pins as SURVIVED while they were being caught perfectly well. A harness that
    # reports a false SURVIVED is the same failure as one that reports a false CAUGHT.
    names = []
    for base in RESULTS[kind]:
        for report in base.rglob("*.xml"):
            try:
                root = ElementTree.parse(report).getroot()
            except ElementTree.ParseError:
                continue
            for case in root.iter("testcase"):
                if case.find("failure") is None and case.find("error") is None:
                    continue
                cls = (case.get("classname") or "").rsplit(".", 1)[-1]
                names.append(f"{cls}.{case.get('name')}" if cls else str(case.get("name")))
    return names


BUILD_ERROR = ["<the mutation did not compile>"]


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
    instead of counting as a pass.
    """
    for base in RESULTS[kind]:
        if base.exists():
            shutil.rmtree(base)          # or a previous run's failures count as this one's
    if kind == "python":
        subprocess.run(
            [".venv/bin/python3", "-m", "pytest", "-q", "deployment/jetson/tests/",
             "-p", "no:cacheprovider", f"--junit-xml={RESULTS['python'][0]}/results.xml"],
            capture_output=True, text=True,
        )
    else:
        target = ":transport:test" if kind == "transport" else ":app:test"
        result = subprocess.run(GRADLE + [target, "--rerun-tasks"], capture_output=True, text=True)
        if "e: file://" in result.stdout or "e: file://" in result.stderr:
            return BUILD_ERROR
    return failing_tests(kind)

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
    # For require_str the null clause must also go, or the mutation is a no-op.
    mutated = keep.replace(old, new, 1)
    if "require_str" in name:
        mutated = mutated.replace(
            '    if value is None:\n        raise MessageError(f"{field} must not be null", REASON_NULL_NOT_ALLOWED)\n    if value is None or not isinstance(value, str):',
            '    if value is None or not isinstance(value, str):', 1)
    SIDECAR.write_text(f"{rel}\n{keep}")
    try:
        path.write_text(mutated)
        failed = run(kind)
    finally:
        path.write_text(keep)
        SIDECAR.unlink(missing_ok=True)
    if failed and failed is not BUILD_ERROR and is_collection_error(failed):
        failed = BUILD_ERROR
    if failed == BUILD_ERROR:
        # Distinct from both verdicts. A mutation that does not compile -- or, in
        # Python, does not import -- proves nothing about any test, and reporting it
        # as CAUGHT is how one of these entries came to measure the compiler for weeks.
        survived.append(name + " [did not build/import]")
        print(f"  DID NOT BUILD      {name}")
    elif failed:
        print(f"  CAUGHT ({len(failed)})         {name}")
        print(f"                     by {failed[0]}")
    else:
        survived.append(name)
        print(f"  *** SURVIVED ***   {name}")

print()
print("survived:", len(survived), survived if survived else "(none)")
sys.exit(1 if survived else 0)
