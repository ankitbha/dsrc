"""USB transport backend: an `Acceptor` built on `adb reverse`, not a new byte
stream.

The in-car path is `adb reverse`, which makes the phone's device-local
`127.0.0.1:<port>` resolve to a port on this machine, tunnelled over USB by
`adbd`. Both ends are ordinary TCP sockets once that mapping exists, so
`TcpConnection` already satisfies the `ByteConnection` contract on this
kernel -- proved in `test_transport_backend_contract.py` -- and nothing here
constructs a new one.

What USB adds is one piece of external state the network backend does not
have: the reverse mapping itself. It disappears on a replug or an
`adb kill-server`, and its absence presents to the phone as ECONNREFUSED --
indistinguishable, from the phone's side, from a Jetson that never started
listening. `UsbAcceptor` owns that mapping's lifetime: it establishes the
mapping on construction, re-verifies it whenever `accept()` times out, and
removes it on `close()`.

Binding `127.0.0.1` rather than `0.0.0.0` is deliberate (see the module
docstring's task-40 plan, decision D2): the `adb` server runs on this same
machine and dials its own loopback, so loopback is sufficient, and binding
`0.0.0.0` while Tailscale is up would leave both paths live at once, making
"which path did this run use" a question rather than an impossibility --
`tailnet.path_for_address` exists to answer exactly that question from the
accepted socket's remote address.

Nothing above the `Acceptor` seam changes: not `session.py`, not
`frames.py`, not `messages.py`, not `handshake.py`, not `timebase.py`, and
not either spec file.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Callable

from transport.tcp import DEFAULT_PORT, TcpAcceptor, TcpConnection

#: Bounded, so a wedged `adb` server cannot hang a run that is only trying to
#: establish or verify a mapping.
DEFAULT_ADB_TIMEOUT_S = 10.0

#: `subprocess.run` shape, injectable so the failure paths below are
#: unit-testable without a phone: a fake with this signature can return
#: whatever exit code and output a test wants, or raise the exception a
#: missing `adb` binary raises.
RunAdb = Callable[..., subprocess.CompletedProcess]


def _default_run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


class AdbError(Exception):
    """`adb` refused, timed out, or is not on `PATH`."""


@dataclass(frozen=True)
class ReverseSpec:
    """One `adb reverse <device> <local>` mapping.

    Both fields are usually equal -- see decision D4, `47811` on both sides --
    but are named separately because `adb reverse` itself takes two arguments
    and a mapping is not fully described by one port alone.
    """

    device_port: int
    local_port: int

    def device_arg(self) -> str:
        return f"tcp:{self.device_port}"

    def local_arg(self) -> str:
        return f"tcp:{self.local_port}"

    def as_string(self) -> str:
        return f"{self.device_arg()} {self.local_arg()}"


def adb_version(*, run: RunAdb = _default_run, timeout_s: float = DEFAULT_ADB_TIMEOUT_S) -> str | None:
    """`adb version`'s first line, or None when it could not be read.

    Not per-serial -- `adb version` reports the client binary, which is one
    fact about the host running it, not about any one attached device.
    """
    try:
        result = run(["adb", "version"], timeout=timeout_s)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    return first_line or None


class AdbReverse:
    """Owns one serial's `adb reverse` mappings.

    Every call shells `adb -s <serial> reverse ...` with a bounded timeout.
    The serial is required and never inferred: `adb` with more than one
    device attached and no `-s` picks none and fails ambiguously, and a
    caller that let that surface as "no device" would misdiagnose two
    attached phones as zero.
    """

    def __init__(
        self,
        serial: str,
        *,
        timeout_s: float = DEFAULT_ADB_TIMEOUT_S,
        run: RunAdb = _default_run,
    ) -> None:
        if not serial:
            raise ValueError(
                "serial is required: adb with more than one device attached is "
                "ambiguous and silently picks none"
            )
        self._serial = serial
        self._timeout_s = timeout_s
        self._run = run

    @property
    def serial(self) -> str:
        return self._serial

    def _adb(self, *args: str) -> subprocess.CompletedProcess:
        try:
            return self._run(["adb", "-s", self._serial, *args], timeout=self._timeout_s)
        except FileNotFoundError as exc:
            raise AdbError(f"adb is not on PATH: {exc}") from None
        except subprocess.TimeoutExpired as exc:
            raise AdbError(
                f"adb -s {self._serial} {' '.join(args)} timed out after {self._timeout_s}s"
            ) from exc
        except OSError as exc:
            raise AdbError(f"adb -s {self._serial} {' '.join(args)} failed: {exc}") from exc

    def establish(self, spec: ReverseSpec) -> None:
        """Create or replace the mapping. Raises AdbError on any failure."""
        result = self._adb("reverse", spec.device_arg(), spec.local_arg())
        if result.returncode != 0:
            raise AdbError(
                f"adb reverse {spec.as_string()} failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )

    def remove(self, spec: ReverseSpec) -> None:
        """Idempotent, and tolerant of a device already gone.

        `adb reverse --remove` against a device that unplugged mid-run fails,
        and a teardown path that raised on that would leave `UsbAcceptor.close()`
        unable to complete -- the one place a raise is worse than a silent no-op.
        """
        try:
            self._adb("reverse", "--remove", spec.device_arg())
        except AdbError:
            pass

    def list(self) -> list[str]:
        """This serial's registered device-port mappings, as `tcp:N` strings.

        Empty when `adb` itself could not be reached, not just when there are
        no mappings -- a caller checking `spec.device_arg() in reverse.list()`
        then reads "not present" either way, which is the conservative answer:
        an unreachable `adb` should look like a missing mapping, not a
        confirmed one.
        """
        try:
            result = self._adb("reverse", "--list")
        except AdbError:
            return []
        if result.returncode != 0:
            return []
        mappings = []
        for line in result.stdout.splitlines():
            parts = line.split()
            # "<serial> <device-port> <local-port>"
            if len(parts) >= 2 and parts[0] == self._serial:
                mappings.append(parts[1])
        return mappings

    def verify(self, spec: ReverseSpec) -> bool:
        return spec.device_arg() in self.list()

    _TRANSPORT_ID_RE = re.compile(r"transport_id:(\d+)")

    def transport_id(self) -> str | None:
        """This serial's `transport_id` off `adb devices -l`, or None."""
        try:
            result = self._adb("devices", "-l")
        except AdbError:
            return None
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if not line.startswith(self._serial):
                continue
            match = self._TRANSPORT_ID_RE.search(line)
            if match:
                return match.group(1)
        return None


class UsbAcceptor:
    """Backend-side listener: `Acceptor` over a loopback `TcpAcceptor` plus the
    `adb reverse` mapping that makes the phone able to reach it.

    Connections handed back are plain `TcpConnection` instances -- the
    `ByteConnection` contract is satisfied by code already proved on this
    kernel, and nothing here re-implements it.
    """

    def __init__(
        self,
        serial: str,
        port: int = DEFAULT_PORT,
        device_port: int | None = None,
        *,
        reverse: AdbReverse | None = None,
        run: RunAdb = _default_run,
    ) -> None:
        self._reverse = reverse if reverse is not None else AdbReverse(serial, run=run)
        self._run = run
        # Bound first, so `port=0` resolves to a real local port before the
        # reverse spec is built -- the same "resolved after bind" convention
        # `TcpAcceptor.address` documents for itself.
        self._tcp = TcpAcceptor("127.0.0.1", port)
        try:
            self._spec = ReverseSpec(
                device_port=device_port if device_port is not None else self._tcp.port,
                local_port=self._tcp.port,
            )
            self._reverse.establish(self._spec)
        except BaseException:
            # The acceptor bound a socket above; if the reverse cannot be
            # established this object is never usable, and it must not leak
            # the socket while failing to exist.
            self._tcp.close()
            raise
        self.reverses_reestablished = 0
        self._closed = False

    @property
    def serial(self) -> str:
        return self._reverse.serial

    @property
    def address(self) -> tuple[str, int]:
        return self._tcp.address

    @property
    def host(self) -> str:
        return self._tcp.host

    @property
    def port(self) -> int:
        return self._tcp.port

    def accept(self, timeout: float | None = None) -> TcpConnection | None:
        """Delegates to the wrapped `TcpAcceptor`.

        Before returning `None` on a timeout, re-verifies the reverse mapping
        is still listed. A mapping found missing -- cleared by a replug or an
        `adb kill-server` -- is re-established and counted, rather than left
        for the phone to retry against forever with nothing on this side ever
        noticing.

        A closed acceptor never reaches the check below: `TcpAcceptor.accept`
        raises `ConnectionClosed` rather than returning `None` once closed, so
        `connection is None` here only happens on a genuine timeout.
        """
        connection = self._tcp.accept(timeout=timeout)
        if connection is None and not self._reverse.verify(self._spec):
            self._reverse.establish(self._spec)
            self.reverses_reestablished += 1
        return connection

    def close(self) -> None:
        """Idempotent. Removes the reverse, then closes the `TcpAcceptor`.

        In that order: removing the reverse first means a client cannot dial
        in during the window between the two, and tolerant of a device
        already unplugged -- `AdbReverse.remove` itself swallows that.
        """
        if self._closed:
            return
        self._closed = True
        self._reverse.remove(self._spec)
        self._tcp.close()

    def usb_record(self) -> dict[str, object]:
        """What a run record needs to say which cable it used.

        Task 32's lesson was that a record which cannot name its own path is
        not evidence; `path_for_address` already reports `127.0.0.1` as USB,
        which says which path, not which cable or how often it dropped.
        """
        return {
            "serial": self.serial,
            "transport_id": self._reverse.transport_id(),
            "reverse_spec": self._spec.as_string(),
            "adb_version": adb_version(run=self._run),
            "reverses_reestablished": self.reverses_reestablished,
            "address": list(self.address),
        }


_DEVICES_LINE_RE = re.compile(r"^(\S+)\s+(\S+)")


def attached_serial(
    *, run: RunAdb = _default_run, timeout_s: float = DEFAULT_ADB_TIMEOUT_S
) -> tuple[str | None, str]:
    """Exactly one attached device's serial, or None with the reason.

    Used to gate the conformance suites and any bench script on a real
    phone being present, rather than failing with an ambiguous `adb` error
    when zero or more than one is attached.
    """
    try:
        result = run(["adb", "devices", "-l"], timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"adb devices -l failed: {exc}"
    if result.returncode != 0:
        return None, f"adb devices -l exited {result.returncode}"
    serials = []
    for line in result.stdout.splitlines()[1:]:
        match = _DEVICES_LINE_RE.match(line)
        if match and match.group(2) == "device":
            serials.append(match.group(1))
    if not serials:
        return None, "no device attached (adb devices -l lists none in state 'device')"
    if len(serials) > 1:
        return None, f"{len(serials)} devices attached; ambiguous without an explicit serial"
    return serials[0], "ok"
