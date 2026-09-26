"""Put EMG samples on the perf_counter clock the camera worker stamps frames with.

HEX frames carry a device timestamp (+20 ms per frame). Serial and Bluetooth
deliver frames in bursts, so arrival time is a poor sample time. Instead the
device clock is shifted onto the host clock by the smallest arrival-minus-device
offset seen: the least-delayed frame. Whether the stamp marks a frame's first or
last sample is unknown, which leaves about +-20 ms of uncertainty (NOTES §26).
"""
from __future__ import annotations

import numpy as np

from .protocol import SAMPLES_PER_FRAME

SAMPLE_S = 0.002
DRIFT_PER_FRAME = 2e-6   # lets the offset follow slow crystal drift upwards
RESTART_S = 1.0          # a jump larger than this means the device clock restarted


class ClockAligner:
    def __init__(self) -> None:
        self.offset: float | None = None

    def reset(self) -> None:
        self.offset = None

    def stamp(self, ts_ms: int, arrival: float) -> np.ndarray:
        """perf_counter times (s) of the 10 samples in the frame stamped `ts_ms`."""
        off = arrival - ts_ms / 1000.0
        if self.offset is None or off < self.offset or off - self.offset > RESTART_S:
            self.offset = off
        else:
            self.offset += DRIFT_PER_FRAME
        last = ts_ms / 1000.0 + self.offset
        return last - (SAMPLES_PER_FRAME - 1 - np.arange(SAMPLES_PER_FRAME)) * SAMPLE_S
