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

echo "== instrumented =="
python3 scripts/with_device.py -- env JAVA_HOME="$JAVA_HOME" \
    ./phone/gradlew -p phone :app:connectedDebugAndroidTest --rerun-tasks

echo "== all green =="
