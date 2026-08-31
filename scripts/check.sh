#!/bin/zsh
# Everything that has to be green, in the order that fails fastest.
#
# `lintDebug` is in here because of task 24: getThermalHeadroom is API 30 and the app's
# minSdk is 29, so on an Android 10 handset the call raised NoSuchMethodError inside a
# lambda whose caller swallowed it, and the entire telemetry stream -- thermal status
# included -- produced nothing for a whole drive with no log line. Lint had the answer and
# nothing ran lint. It is one `NewApi` error away from happening again on any platform call
# added later, and the instrumented suite cannot see it: the emulator is API 31.
set -e
cd "$(dirname "$0")/.."
export JAVA_HOME=$(/usr/libexec/java_home -v 17)

echo "== lint =="
# :app only. `:transport` is a plain Kotlin JVM module with no Android plugin, so it has no
# lint task -- and asking for one fails the whole run before anything is checked.
./phone/gradlew -p phone :app:lintDebug

echo "== jvm =="
./phone/gradlew -p phone :app:test :transport:test

echo "== python =="
.venv/bin/python3 -m pytest -q deployment/jetson/tests/ -p no:cacheprovider

# Two passes, and the split is not cosmetic. `ImuWireTest` measures a delivered
# sample rate, and delivered rate is bounded by what the device can sustain rather
# than by the rate commanded. Run after the camera tests it measured 47.0 samples/s
# following a raise, against baselines of 51.0 and 49.7 samples/s -- no increase --
# while the same test run alone measured 60.7 samples/s against a baseline of about
# 51 samples/s. The two baselines differed by 2.6 per cent and the handset's thermal
# readings were normal, so the device was steady and the throughput ceiling was the
# device's, not the command's.
#
# The assertion is left intact. Every way of making it pass inside one suite either
# weakens it or removes its ability to fail: skipping when the raise produces no
# increase would make the test unable to catch the defect it exists for. So the
# measurement is given a quiet device instead, at the cost of one extra install.
IMU_TESTS=com.dsrc.phone.sensors.ImuWireTest

echo "== instrumented (everything but the rate measurement) =="
python3 scripts/with_device.py -- env JAVA_HOME="$JAVA_HOME" \
    ./phone/gradlew -p phone :app:connectedDebugAndroidTest --rerun-tasks \
    -Pandroid.testInstrumentationRunnerArguments.notClass=$IMU_TESTS

echo "== instrumented (the rate measurement, on a quiet device) =="
python3 scripts/with_device.py -- env JAVA_HOME="$JAVA_HOME" \
    ./phone/gradlew -p phone :app:connectedDebugAndroidTest --rerun-tasks \
    -Pandroid.testInstrumentationRunnerArguments.class=$IMU_TESTS

echo "== all green =="
