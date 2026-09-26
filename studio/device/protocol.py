"""What the armband sends, as data. Both the parser and the byte console read the
frame layout from here, so a protocol change is made in one place.

Two output modes, switched only by the armband button (NOTES §12):
  HEX   98-byte frames, 50 frames/s, 10 samples x 8 ch of 8-bit EMG + IMU + battery
  ASCII one text line per sample, 8 space-separated 12-bit integers, ~67 Hz
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

CH = 8
BAUD = 115200
FRAME_LEN = 98
HDR = b"\xAA\xAA\x5F"
TAIL = 0x55
SAMPLES_PER_FRAME = 10
FRAME_MS = 20

# (name, first byte, end byte) of a HEX frame
FRAME_FIELDS = (
    ("header", 0, 3), ("timestamp", 3, 7), ("acc", 7, 10), ("gyro", 10, 13), ("angle", 13, 16),
    ("emg", 16, 96), ("battery", 96, 97), ("tail", 97, 98),
)
FIELD = {name: slice(lo, hi) for name, lo, hi in FRAME_FIELDS}
IMU_NAMES = ["AccX", "AccY", "AccZ", "GyX", "GyY", "GyZ", "Pitch", "Roll", "Yaw"]


@dataclass(frozen=True)
class ModeSpec:
    name: str
    fs: float        # EMG samples per second
    full: int        # largest sample value
    center: int      # value at rest
    bits: int
    bps: int         # bytes per second the link should carry


MODES = {
    "hex": ModeSpec("hex", 500.0, 255, 127, 8, FRAME_LEN * 50),
    "ascii": ModeSpec("ascii", 67.1, 4095, 2048, 12, 2675),   # rate measured in NOTES §12
}


def spec_for(mode: str | None) -> ModeSpec:
    """The spec of `mode`, or HEX (the power-on mode) while no mode is known yet."""
    return MODES.get(mode or "hex")


class LinkState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
