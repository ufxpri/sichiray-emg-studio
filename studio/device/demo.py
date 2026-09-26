"""A synthetic armband. It emits real HEX frames or ASCII lines, so the demo runs
through the same parsers, statistics and windows as the device.

Channels carry deliberate faults so the noise table has something to find:
  all   a small 60 Hz mains hum (common to every channel)
  ch2   slow electrode movement (0.5-3 Hz wander)
  ch6   poor contact: almost no muscle signal, higher noise floor
  ch8   a charger-like 9.77 Hz sawtooth line (NOTES §18)
Muscle activity cycles every 8 s: rest, fist (all), rest, 'index' (ch3-4), rest.
Occasionally a frame is dropped or stray bytes are inserted.
"""
from __future__ import annotations

import time

import numpy as np

from .protocol import CH, FRAME_MS, MODES, SAMPLES_PER_FRAME, TAIL


class DemoSource:
    GAIN = np.array([1.0, 0.8, 1.2, 1.1, 1.6, 0.15, 0.7, 0.9])

    def __init__(self, mode: str = "hex") -> None:
        from scipy.signal import butter
        self.name = "데모"
        self.mode = mode
        self.rng = np.random.default_rng(1)
        self.n = 0              # 500 Hz sample counter
        self.ts = 0
        self.frame_no = 0
        self.batt = 87.0
        self.sos = butter(4, [30, 150], "bandpass", fs=500, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, CH))
        self.wander = np.zeros(CH)
        self._next = 0.0

    # ------------------------------------------------------------ ByteSource
    def open(self) -> None:
        self._next = time.perf_counter()

    def read(self) -> bytes:
        """One frame (or line), paced to real time."""
        wait = self._next - time.perf_counter()
        if wait > 0:
            time.sleep(min(wait, 0.02))
            if self._next > time.perf_counter():
                return b""
        self._next += self.period
        return self.step()

    def close(self) -> None:
        pass

    # ------------------------------------------------------------ device behaviour
    @property
    def period(self) -> float:
        return FRAME_MS / 1000 if self.mode == "hex" else 1 / MODES["ascii"].fs

    def press_mode_button(self) -> None:
        """What the armband button does: double click = ASCII, single = HEX."""
        self.mode = "ascii" if self.mode == "hex" else "hex"

    def step(self) -> bytes:
        if self.mode == "hex":
            return self._frame()
        x = self._synth(max(1, round(500 / MODES["ascii"].fs)))[-1]
        v = np.clip(np.round(2048 + x * 16), 0, 4095).astype(int)
        return (" ".join(str(i) for i in v) + "\r\n").encode()

    def _synth(self, k: int) -> np.ndarray:
        """k samples at 500 Hz, in 8-bit counts around 0."""
        from scipy.signal import sosfilt
        t = (self.n + np.arange(k)) / 500.0
        self.n += k
        cyc = t % 8.0
        fist = ((cyc > 2) & (cyc < 4)).astype(float)
        index = ((cyc > 5) & (cyc < 7)).astype(float)
        env = fist[:, None] * self.GAIN[None, :] * 14
        env[:, 2:4] += index[:, None] * 12
        mus, self.zi = sosfilt(self.sos, self.rng.standard_normal((k, CH)), axis=0, zi=self.zi)
        x = mus * env * 2.2 + 0.35 * self.rng.standard_normal((k, CH))
        x += 0.8 * np.sin(2 * np.pi * 60 * t)[:, None]
        x[:, 5] += 1.2 * self.rng.standard_normal(k)
        self.wander = 0.995 * self.wander + 0.25 * self.rng.standard_normal(CH)
        x[:, 1] += self.wander[1] * 3 + 5 * np.sin(2 * np.pi * 1.3 * t) * (np.sin(2 * np.pi * 0.07 * t) > 0.3)
        x[:, 7] += 7 * ((t * 9.77) % 1.0 - 0.5)
        return x

    def _frame(self) -> bytes:
        x = self._synth(SAMPLES_PER_FRAME)
        self.frame_no += 1
        self.ts += FRAME_MS
        if self.frame_no % 600 == 0:   # a lost frame: timestamp jumps, bytes never arrive
            self.ts += FRAME_MS
        emg = np.clip(np.round(127 + x), 0, 255).astype(np.uint8)
        t = self.n / 500.0
        imu = [127 + int(20 * np.sin(t * 0.9 + i)) for i in range(6)]
        imu += [127 + int(40 * np.sin(t * 0.7)), 127 + int(30 * np.sin(t * 0.4)), 127]
        self.batt = max(0.0, self.batt - FRAME_MS / 1000 / 60)
        body = bytes([0xAA, 0xAA, 0x5F]) + self.ts.to_bytes(4, "big") + bytes(imu) \
            + emg.tobytes() + bytes([int(self.batt), TAIL])
        if self.frame_no % 450 == 0:   # line noise: a few stray bytes before the frame
            body = bytes([0x13, 0xAA, 0x00]) + body
        return body
