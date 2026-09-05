"""Unit and sanity tests for the USB backend (task 40), none of which need a
phone.

`AdbReverse`'s subprocess call is injectable, so every failure path below --
`adb` missing, `adb` refusing, a mapping vanishing mid-run, two devices with
no serial to disambiguate -- is exercised with a fake in place of a real
device. `UsbAcceptor`'s own byte-stream behaviour is proved once here against
a real loopback socket (with the `adb reverse` side faked out): the
conformance suites in `test_transport_backend_contract.py` and
`test_transport_acceptor_contract.py` are what prove it against a real phone,
gated on one being attached.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time

import pytest

from transport.connection import ConnectionClosed
from transport.tcp import TcpAcceptor, dial
from transport.usb import (
    AdbError,
    AdbReverse,
    ReverseSpec,
    UsbAcceptor,
    adb_version,
    attached_serial,
)


# -- fakes --------------------------------------------------------------


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRun:
    """Records every `adb` invocation and answers per subcommand.

    Each subcommand's answer is a `FakeCompleted` (or an exception to raise)
    looked up by the first non-`-s`/serial token, so a test configures only
    the subcommands it cares about.
    """

    def __init__(self):
        self.calls: list[list[str]] = []
        self.answers: dict[str, FakeCompleted | BaseException] = {}
        self.default = FakeCompleted(0, "", "")

    def set(self, subcommand: str, answer) -> None:
        self.answers[subcommand] = answer

    def __call__(self, args, *, timeout):
        self.calls.append(list(args))
        # args is ["adb", "-s", serial, <subcommand>, ...] or ["adb", <subcommand>, ...]
        rest = args[1:]
        if rest[:1] == ["-s"]:
            rest = rest[2:]
        key = rest[0] if rest else ""
        if key == "reverse" and len(rest) > 1 and rest[1] == "--remove":
            key = "reverse --remove"
        elif key == "reverse" and len(rest) > 1 and rest[1] == "--list":
            key = "reverse --list"
        answer = self.answers.get(key, self.default)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class RecordingReverse:
    """A drop-in for `AdbReverse` that records calls instead of shelling out.

    Used to test `UsbAcceptor`'s orchestration -- when it re-verifies, when it
    re-establishes, the order it tears down in -- independent of `AdbReverse`'s
    own subprocess handling, which is tested separately below.
    """

    def __init__(self, serial: str = "ZY227VV4XC", *, verify_sequence=None):
        self._serial = serial
        self.establish_calls: list[ReverseSpec] = []
        self.remove_calls: list[ReverseSpec] = []
        self.verify_calls = 0
        self._verify_sequence = list(verify_sequence) if verify_sequence is not None else None
        self.raise_on_remove: BaseException | None = None
        #: Raised by `establish()` on every call AFTER the one at
        #: construction -- i.e. only re-establishment attempts, mirroring a
        #: replug: the mapping went missing (`verify` said so) and putting
        #: it back also fails for a beat while the device re-enumerates.
        self.raise_on_reestablish: BaseException | None = None
        #: `UsbAcceptor.__init__` calls `sweep()` unconditionally, before
        #: `establish()` -- recorded so a test can assert it was asked to,
        #: and configurable so a test can simulate finding (and reporting)
        #: a stray.
        self.sweep_calls: list[int] = []
        self.sweep_return = 0

    @property
    def serial(self) -> str:
        return self._serial

    def establish(self, spec: ReverseSpec) -> None:
        if self.establish_calls and self.raise_on_reestablish is not None:
            self.establish_calls.append(spec)
            raise self.raise_on_reestablish
        self.establish_calls.append(spec)

    def sweep(self, device_port: int) -> int:
        self.sweep_calls.append(device_port)
        return self.sweep_return

    def verify(self, spec: ReverseSpec) -> bool:
        self.verify_calls += 1
        if self._verify_sequence is not None:
            return self._verify_sequence.pop(0)
        return True

    def remove(self, spec: ReverseSpec) -> None:
        self.remove_calls.append(spec)
        if self.raise_on_remove is not None:
            raise self.raise_on_remove

    def transport_id(self) -> str | None:
        return "1"


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# -- AdbReverse: serial is required, never inferred ----------------------


def test_serial_required_raises_before_any_subprocess_call():
    fake = FakeRun()
    with pytest.raises(ValueError):
        AdbReverse("", run=fake)
    assert fake.calls == []


def test_usb_acceptor_requires_a_serial():
    fake = FakeRun()
    with pytest.raises(ValueError):
        UsbAcceptor("", port=0, run=fake)


# -- AdbReverse: the failure paths, none of which need a phone -----------


def test_adb_not_on_path_raises_adb_error():
    fake = FakeRun()
    fake.set("reverse", FileNotFoundError("no such file: adb"))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    with pytest.raises(AdbError, match="not on PATH"):
        reverse.establish(ReverseSpec(47811, 47811))


def test_adb_returning_nonzero_raises_adb_error_with_stderr():
    fake = FakeRun()
    fake.set("reverse", FakeCompleted(1, "", "error: more than one device/emulator"))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    with pytest.raises(AdbError, match="more than one device"):
        reverse.establish(ReverseSpec(47811, 47811))


def test_adb_timeout_raises_adb_error():
    fake = FakeRun()
    fake.set("reverse", subprocess.TimeoutExpired(cmd="adb", timeout=10.0))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    with pytest.raises(AdbError, match="timed out"):
        reverse.establish(ReverseSpec(47811, 47811))


def test_remove_is_tolerant_of_a_device_already_unplugged():
    """`adb reverse --remove` against an absent device fails; teardown must
    not raise out of it regardless."""
    fake = FakeRun()
    fake.set("reverse --remove", FakeCompleted(1, "", "error: device offline"))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    reverse.remove(ReverseSpec(47811, 47811))  # must not raise


def test_remove_is_tolerant_of_adb_not_on_path():
    fake = FakeRun()
    fake.set("reverse --remove", FileNotFoundError("no adb"))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    reverse.remove(ReverseSpec(47811, 47811))  # must not raise


# -- AdbReverse: list / verify / transport_id / version -------------------


def test_list_returns_every_mapping_the_scoped_call_reports():
    """`-s <serial>` routes the request to that device's adbd server-side --
    it is not a client-side filter -- so every line the scoped call prints
    belongs to this serial and none is re-filtered out."""
    fake = FakeRun()
    fake.set(
        "reverse --list",
        FakeCompleted(0, "ZY227VV4XC tcp:47811 tcp:47811\nUsbFfs tcp:9999 tcp:8888\n", ""),
    )
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.list() == [("tcp:47811", "tcp:47811"), ("tcp:9999", "tcp:8888")]


def test_list_does_not_require_the_first_column_to_be_the_serial():
    """Measured on real hardware (ZY227VV4XC): `adb -s ZY227VV4XC reverse
    --list` printed `UsbFfs tcp:47811 tcp:47811` -- the transport name, not
    the serial. A `list()` that required `parts[0] == serial` never found
    its own mapping and reported every timeout as one it had to
    re-establish."""
    fake = FakeRun()
    fake.set("reverse --list", FakeCompleted(0, "UsbFfs tcp:47811 tcp:47811\n", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.list() == [("tcp:47811", "tcp:47811")]
    assert reverse.verify(ReverseSpec(47811, 47811)) is True


# -- B9 (validation round 2): verify() must check BOTH halves of the pair --


def test_verify_is_false_when_the_device_port_matches_but_the_local_port_does_not():
    """The exact defect: a mapping whose device port matches but whose
    local port points somewhere else is not this mapping, and the old
    device-port-only check read it as healthy -- R2's failure, invisible
    to the mechanism built to detect it."""
    fake = FakeRun()
    fake.set("reverse --list", FakeCompleted(0, "UsbFfs tcp:47811 tcp:9999\n", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.verify(ReverseSpec(device_port=47811, local_port=47811)) is False


def test_verify_is_true_only_when_both_halves_match():
    fake = FakeRun()
    fake.set("reverse --list", FakeCompleted(0, "UsbFfs tcp:47811 tcp:9999\n", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.verify(ReverseSpec(device_port=47811, local_port=9999)) is True


# -- AdbReverse.sweep() (C5/B9/B12, validation round 2) --------------------


def test_sweep_removes_every_mapping_for_the_device_port_and_counts_them():
    fake = FakeRun()
    fake.set("reverse --list", FakeCompleted(0, "UsbFfs tcp:47811 tcp:9999\n", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    removed = reverse.sweep(47811)
    assert removed == 1
    remove_calls = [c for c in fake.calls if c[3:5] == ["reverse", "--remove"]]
    assert remove_calls == [["adb", "-s", "ZY227VV4XC", "reverse", "--remove", "tcp:47811"]]


def test_sweep_ignores_mappings_for_a_different_device_port():
    fake = FakeRun()
    fake.set("reverse --list", FakeCompleted(0, "UsbFfs tcp:9999 tcp:9999\n", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.sweep(47811) == 0
    assert not any(c[3:5] == ["reverse", "--remove"] for c in fake.calls)


def test_sweep_is_zero_when_nothing_is_registered():
    fake = FakeRun()
    fake.set("reverse --list", FakeCompleted(0, "", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.sweep(47811) == 0


def test_list_is_empty_not_an_error_when_adb_is_unreachable():
    fake = FakeRun()
    fake.set("reverse --list", FileNotFoundError("no adb"))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.list() == []


def test_verify_true_when_mapping_present():
    fake = FakeRun()
    fake.set("reverse --list", FakeCompleted(0, "ZY227VV4XC tcp:47811 tcp:47811\n", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.verify(ReverseSpec(47811, 47811)) is True


def test_verify_false_when_mapping_absent():
    fake = FakeRun()
    fake.set("reverse --list", FakeCompleted(0, "", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.verify(ReverseSpec(47811, 47811)) is False


def test_transport_id_parses_the_matching_serial_only():
    fake = FakeRun()
    fake.set(
        "devices",
        FakeCompleted(
            0,
            "List of devices attached\n"
            "ZY227VV4XC     device usb:1-2.2 product:sofia_retail transport_id:1\n"
            "emulator-5554  device product:sdk_gphone transport_id:7\n",
            "",
        ),
    )
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.transport_id() == "1"


def test_transport_id_is_none_when_serial_absent():
    fake = FakeRun()
    fake.set("devices", FakeCompleted(0, "List of devices attached\n", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.transport_id() is None


def test_adb_version_reads_first_line():
    fake = FakeRun()
    fake.set("version", FakeCompleted(0, "Android Debug Bridge version 1.0.41\nOther\n", ""))
    assert adb_version(run=fake) == "Android Debug Bridge version 1.0.41"


def test_adb_version_is_none_when_adb_missing():
    fake = FakeRun()
    fake.set("version", FileNotFoundError("no adb"))
    assert adb_version(run=fake) is None


# -- attached_serial: disambiguation, the two-device case -----------------


def test_attached_serial_none_when_no_device():
    fake = FakeRun()
    fake.set("devices", FakeCompleted(0, "List of devices attached\n", ""))
    serial, reason = attached_serial(run=fake)
    assert serial is None
    assert "no device" in reason


def test_attached_serial_ambiguous_with_two_devices():
    fake = FakeRun()
    fake.set(
        "devices",
        FakeCompleted(
            0,
            "List of devices attached\n"
            "ZY227VV4XC     device usb:1-2.2\n"
            "OTHERSERIAL99  device usb:1-2.3\n",
            "",
        ),
    )
    serial, reason = attached_serial(run=fake)
    assert serial is None
    assert "ambiguous" in reason


def test_attached_serial_picks_the_one_device():
    fake = FakeRun()
    fake.set(
        "devices",
        FakeCompleted(0, "List of devices attached\nZY227VV4XC     device usb:1-2.2\n", ""),
    )
    serial, reason = attached_serial(run=fake)
    assert serial == "ZY227VV4XC"
    assert reason == "ok"


def test_attached_serial_excludes_unauthorized_and_offline():
    fake = FakeRun()
    fake.set(
        "devices",
        FakeCompleted(
            0,
            "List of devices attached\n"
            "ZY227VV4XC     unauthorized usb:1-2.2\n",
            "",
        ),
    )
    serial, reason = attached_serial(run=fake)
    assert serial is None
    assert "no device" in reason


# -- UsbAcceptor: mapping missing at construction --------------------------


def test_construction_raises_and_does_not_leak_the_socket_when_reverse_fails():
    port = free_port()
    fake = FakeRun()
    fake.set("reverse", FakeCompleted(1, "", "error: device offline"))
    with pytest.raises(AdbError):
        UsbAcceptor("ZY227VV4XC", port=port, run=fake)
    # If the TcpAcceptor's socket had leaked, binding the same port again
    # would fail with "address already in use".
    second = TcpAcceptor("127.0.0.1", port)
    second.close()


def test_construction_sweeps_before_establishing():
    """C5/B9/B12: a stray mapping for this device port -- left by an
    earlier drive that was killed, crashed, or raised before its own
    teardown -- must be gone before the fresh mapping goes up, not after."""
    reverse = RecordingReverse()
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    try:
        assert reverse.sweep_calls == [acceptor.port]
        assert acceptor.reverses_swept == 0
    finally:
        acceptor.close()


def test_construction_reports_a_swept_stray():
    reverse = RecordingReverse()
    reverse.sweep_return = 1
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    try:
        assert acceptor.reverses_swept == 1
    finally:
        acceptor.close()


def test_construction_really_removes_a_stray_left_by_a_leaked_run():
    """End to end against a real AdbReverse (fake subprocess): a mapping
    for the exact port we are about to bind, already registered -- as a
    leaked run would leave it -- is gone by the time construction returns,
    and the fresh one is the only one left."""
    port = free_port()
    fake = FakeRun()
    fake.set(
        "reverse --list",
        FakeCompleted(0, f"UsbFfs tcp:{port} tcp:{port}\n", ""),
    )
    acceptor = UsbAcceptor("ZY227VV4XC", port=port, run=fake)
    try:
        assert acceptor.reverses_swept == 1
        remove_calls = [c for c in fake.calls if c[3:5] == ["reverse", "--remove"]]
        assert remove_calls == [["adb", "-s", "ZY227VV4XC", "reverse", "--remove", f"tcp:{port}"]]
        establish_calls = [c for c in fake.calls if c[3] == "reverse" and c[4] != "--remove" and c[4] != "--list"]
        assert len(establish_calls) == 1
        # Swept before established: the remove call comes first in the
        # recorded order.
        remove_index = fake.calls.index(remove_calls[0])
        establish_index = fake.calls.index(establish_calls[0])
        assert remove_index < establish_index
    finally:
        acceptor.close()


# -- UsbAcceptor: re-verification and re-establishment on timeout ---------


def test_accept_reestablishes_a_mapping_found_missing_on_timeout():
    reverse = RecordingReverse(verify_sequence=[False])
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    try:
        assert acceptor.accept(timeout=0.05) is None
        assert acceptor.reverses_reestablished == 1
        # Once at construction, once at re-establishment.
        assert len(reverse.establish_calls) == 2
    finally:
        acceptor.close()


def test_accept_does_not_reestablish_a_mapping_still_present():
    reverse = RecordingReverse(verify_sequence=[True, True])
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    try:
        assert acceptor.accept(timeout=0.05) is None
        assert acceptor.accept(timeout=0.05) is None
        assert acceptor.reverses_reestablished == 0
        assert len(reverse.establish_calls) == 1
    finally:
        acceptor.close()


def test_accept_does_not_reverify_after_close():
    reverse = RecordingReverse(verify_sequence=[False])
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    acceptor.close()
    assert reverse.verify_calls == 0
    with pytest.raises(ConnectionClosed):
        acceptor.accept(timeout=0.05)
    assert reverse.verify_calls == 0


# -- A3 (validation round 1): a re-establishment that itself fails --------


def test_accept_catches_a_reestablish_failure_counts_it_and_returns_none():
    """A `replug` is D5's expected in-car event: the mapping goes missing
    (`verify` says so) and `establish()` raises `AdbError` ("device not
    found") for the seconds the device takes to re-enumerate. Before the
    fix this propagated out of `accept()`; the only caller
    (`endpoint.py`'s accept loop) catches only `ConnectionClosed` and
    `OSError` and returns on anything else, ending the listener thread for
    the rest of the drive with the socket still bound."""
    from transport.usb import AdbError

    reverse = RecordingReverse(verify_sequence=[False])
    reverse.raise_on_reestablish = AdbError("device 'ZY227VV4XC' not found")
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    try:
        result = acceptor.accept(timeout=0.05)
        assert result is None, "a failed reestablish must still read as a timeout, not raise"
        assert acceptor.reverse_reestablish_failures == 1
        assert acceptor.reverses_reestablished == 0
        # The attempt was made and counted as an attempt on the fake too.
        assert len(reverse.establish_calls) == 2
    finally:
        acceptor.close()


def test_accept_keeps_polling_across_repeated_reestablish_failures():
    """Not a one-shot catch: a replug can take several poll cycles to clear,
    and each failed attempt in that stretch must leave the acceptor able to
    try again on the next poll rather than latching into a dead state."""
    from transport.usb import AdbError

    reverse = RecordingReverse(verify_sequence=[False, False, False])
    reverse.raise_on_reestablish = AdbError("device not found")
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    try:
        for _ in range(3):
            assert acceptor.accept(timeout=0.02) is None
        assert acceptor.reverse_reestablish_failures == 3
    finally:
        acceptor.close()


# -- UsbAcceptor: teardown order and idempotence --------------------------


def test_close_and_accept_do_not_race_the_reverse_mapping():
    """B1: `close()` used to race `accept()`'s re-verify-and-reestablish
    step and re-establish the mapping `close()` had just removed, violating
    task 40's own "shows none after the process exits". Reproduced
    deterministically with a slow `remove()`: `close()` is made to block
    inside it on a background thread; `accept()` is given a SHORT socket
    timeout so it reaches its own re-verify step -- and, pre-fix, finishes
    it -- well before `remove()` is allowed to return. The check that
    actually discriminates the two implementations happens BEFORE
    `remove()` is released: unlocked, `accept()` completes a second
    `establish()` while `close()` is still mid-`remove()`; locked, it is
    still blocked waiting for the same lock `close()` holds.

    (An earlier version of this test gave `accept()` a socket timeout
    close to how long `remove()` was held, so `self._tcp.close()` --
    reached only after `remove()` returns -- usually raced ahead and
    unblocked `accept()`'s own socket wait with `ConnectionClosed` before
    the mapping race it meant to expose ever ran; it passed against the
    unfixed code for the wrong reason. `establish_calls`, not
    `accept_thread.is_alive()`, is what the fix actually changes.)
    """
    remove_started = threading.Event()
    remove_may_proceed = threading.Event()

    class SlowRemoveReverse(RecordingReverse):
        def remove(self, spec):
            remove_started.set()
            assert remove_may_proceed.wait(timeout=5.0), "test setup: never released"
            super().remove(spec)

    reverse = SlowRemoveReverse(verify_sequence=[False] * 10)
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    assert len(reverse.establish_calls) == 1  # construction only, so far

    close_thread = threading.Thread(target=acceptor.close)
    close_thread.start()
    assert remove_started.wait(timeout=5.0), "close() never reached remove()"

    accept_result = []

    def call_accept():
        try:
            # Short: this must return (a genuine timeout, nothing is
            # dialing in) and reach the re-verify step well inside the
            # window `remove()` is held open below, or the race this test
            # exists to force never happens.
            accept_result.append(acceptor.accept(timeout=0.02))
        except ConnectionClosed:
            accept_result.append("closed")

    accept_thread = threading.Thread(target=call_accept)
    accept_thread.start()
    # NOT a join: the fixed code is SUPPOSED to leave accept_thread blocked
    # here (on the lock close_thread holds), so waiting for it to finish
    # would wait for the very thing under test. A plain sleep, generous
    # against the 0.02s socket timeout plus an immediate (delay-free)
    # establish() call, is the unlocked code's whole window to finish in.
    time.sleep(0.2)

    # The discriminating check, taken while `close()` is STILL mid-`remove()`
    # (it cannot have reached `remove_calls.append` yet -- `remove_may_proceed`
    # has not been set, and `self._tcp` is therefore still open, so nothing
    # here raced `self._tcp.close()` either). Unlocked, `accept()` already
    # re-established the mapping by now; locked, it is parked behind the
    # same lock `close()` holds and has not touched `establish()` again at
    # all.
    assert len(reverse.remove_calls) == 0, "test setup: close() finished too early to check"
    assert len(reverse.establish_calls) == 1, (
        "accept() re-established the mapping while close() was still removing it"
    )
    assert accept_thread.is_alive(), (
        "accept() finished without ever blocking on close()'s lock -- either "
        "the fix regressed, or this reproduction stopped forcing the race"
    )

    remove_may_proceed.set()
    close_thread.join(timeout=5.0)
    assert not close_thread.is_alive()

    assert len(reverse.remove_calls) == 1
    assert len(reverse.establish_calls) == 1


def test_close_removes_the_reverse_then_closes_the_tcp_acceptor():
    reverse = RecordingReverse()
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    acceptor.close()
    assert len(reverse.remove_calls) == 1
    with pytest.raises(ConnectionClosed):
        acceptor.accept(timeout=0.05)


def test_close_is_idempotent():
    reverse = RecordingReverse()
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    acceptor.close()
    acceptor.close()
    acceptor.close()
    assert len(reverse.remove_calls) == 1


# -- UsbAcceptor: usb_record() --------------------------------------------


def test_usb_record_shape():
    reverse = RecordingReverse(serial="ZY227VV4XC")
    fake_run = FakeRun()
    fake_run.set("version", FakeCompleted(0, "Android Debug Bridge version 1.0.41\n", ""))
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse, run=fake_run)
    try:
        record = acceptor.usb_record()
        assert record["serial"] == "ZY227VV4XC"
        assert record["transport_id"] == "1"
        assert record["reverse_spec"] == f"tcp:{acceptor.port} tcp:{acceptor.port}"
        assert record["adb_version"] == "Android Debug Bridge version 1.0.41"
        assert record["reverses_reestablished"] == 0
        assert record["reverse_reestablish_failures"] == 0
        assert record["reverses_swept"] == 0
        assert record["address"] == ["127.0.0.1", acceptor.port]
    finally:
        acceptor.close()


# -- Sanity: UsbAcceptor really is a working byte-stream acceptor ---------
#
# The `adb reverse` side is faked out (RecordingReverse); the TCP side is
# real. This is the smoke test that proves the composition works end to end
# without a phone -- the conformance suites below prove the real device path.


def test_usb_acceptor_end_to_end_with_a_real_socket_and_a_fake_reverse():
    reverse = RecordingReverse()
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    try:
        client = dial("127.0.0.1", acceptor.port)
        server = acceptor.accept(timeout=5.0)
        assert server is not None
        client.send_all(b"hello-over-usb")
        assert server.recv_exact(len(b"hello-over-usb")) == b"hello-over-usb"
        client.close()
        with pytest.raises(ConnectionClosed):
            server.recv_exact(1)
        server.close()
    finally:
        acceptor.close()


def test_usb_acceptor_close_releases_a_waiting_accept():
    reverse = RecordingReverse()
    acceptor = UsbAcceptor("ZY227VV4XC", port=0, reverse=reverse)
    released = threading.Event()

    def wait_to_accept():
        try:
            acceptor.accept(timeout=5.0)
        except BaseException:
            pass
        released.set()

    waiter = threading.Thread(target=wait_to_accept, daemon=True)
    waiter.start()
    time.sleep(0.1)
    assert not released.is_set()
    started = time.monotonic()
    acceptor.close()
    assert released.wait(timeout=5.0)
    assert time.monotonic() - started < 1.0
