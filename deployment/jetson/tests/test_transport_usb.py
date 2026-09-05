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

    @property
    def serial(self) -> str:
        return self._serial

    def establish(self, spec: ReverseSpec) -> None:
        self.establish_calls.append(spec)

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
        FakeCompleted(0, "ZY227VV4XC tcp:47811 tcp:47811\ntcp:9999 tcp:9999\n", ""),
    )
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.list() == ["tcp:47811", "tcp:9999"]


def test_list_does_not_require_the_first_column_to_be_the_serial():
    """Measured on real hardware (ZY227VV4XC): `adb -s ZY227VV4XC reverse
    --list` printed `UsbFfs tcp:47811 tcp:47811` -- the transport name, not
    the serial. A `list()` that required `parts[0] == serial` never found
    its own mapping and reported every timeout as one it had to
    re-establish."""
    fake = FakeRun()
    fake.set("reverse --list", FakeCompleted(0, "UsbFfs tcp:47811 tcp:47811\n", ""))
    reverse = AdbReverse("ZY227VV4XC", run=fake)
    assert reverse.list() == ["tcp:47811"]
    assert reverse.verify(ReverseSpec(47811, 47811)) is True


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


# -- UsbAcceptor: teardown order and idempotence --------------------------


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
