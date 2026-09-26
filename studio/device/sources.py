"""Where bytes come from. The Link reads any ByteSource the same way; the serial
port and the synthetic demo device are two implementations."""
from __future__ import annotations

from typing import Protocol

from .protocol import BAUD


class SourceError(Exception):
    """Opening or reading failed; the message is shown to the user as is."""


class ByteSource(Protocol):
    name: str

    def open(self) -> None: ...

    def read(self) -> bytes:
        """Return what is available (may be b""); block at most a few tens of ms."""
        ...

    def close(self) -> None: ...


class SerialSource:
    def __init__(self, port: str) -> None:
        self.name = port
        self._ser = None

    def open(self) -> None:
        import serial
        try:
            self._ser = serial.Serial(self.name, BAUD, timeout=0.05)
        except (OSError, serial.SerialException) as e:
            text = str(e)
            if "PermissionError" in text or "Access is denied" in text or "거부" in text:
                raise SourceError("다른 프로그램이 포트를 사용 중입니다 (벤더 뷰어, sscom, 다른 EMG Studio 등). "
                                  "그 프로그램을 닫고 다시 연결하세요.") from e
            raise SourceError(text) from e

    def read(self) -> bytes:
        import serial
        try:
            # read what is waiting; a large blocking read returns in ~1 s bursts on Windows
            return self._ser.read(max(1, self._ser.in_waiting))
        except (OSError, serial.SerialException) as e:
            raise SourceError(str(e)) from e

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None


def find_ch340() -> list[str]:
    """COM ports of the armband's USB dongle (CH340, VID 1A86 / PID 7523)."""
    from serial.tools import list_ports
    return [p.device for p in list_ports.comports() if (p.vid, p.pid) == (0x1A86, 0x7523)]


def list_ports() -> list[tuple[str, str]]:
    from serial.tools import list_ports as lp
    return [(p.device, p.description) for p in lp.comports()]
