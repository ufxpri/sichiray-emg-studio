"""Link health counters. LinkStats is mutated only under Link.lock; the UI reads a
frozen LinkSnapshot."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .protocol import FRAME_MS, SAMPLES_PER_FRAME, LinkState


@dataclass(frozen=True)
class LinkSnapshot:
    source: str
    state: LinkState
    error: str
    mode: str | None
    gen: int
    uptime: float
    idle: float | None
    bytes_total: int
    bps: float
    sps: float
    frames_ok: int
    resync_bytes: int
    lost_frames: int
    irregular_ts: int
    last_ts: int | None
    ts_mean: float | None
    ts_min: int | None
    ts_max: int | None
    lines_ok: int
    lines_bad: int
    battery: int | None
    imu: tuple[int, ...] | None
    mode_changes: int
    battery_history: tuple[tuple[float, int], ...]
    rate_history: tuple[tuple[float, float, float], ...]   # (uptime s, bytes/s, samples/s)


def _rate(win: deque, now: float, span: float = 1.0) -> float:
    while win and now - win[0][0] > span:
        win.popleft()
    return sum(n for _, n in win) / span


class LinkStats:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t0 = time.monotonic()
        self.bytes_total = 0
        self.rx_win: deque[tuple[float, int]] = deque()
        self.smp_win: deque[tuple[float, int]] = deque()
        self.last_rx = 0.0
        self.rate_hist: deque[tuple[float, float, float]] = deque(maxlen=3600)
        self._hist_t = 0.0
        self.batt_hist: deque[tuple[float, int]] = deque(maxlen=20000)
        self.battery: int | None = None
        self.imu: tuple[int, ...] | None = None
        self.mode_changes = 0
        self.reset_mode()

    def reset_mode(self) -> None:
        self.frames_ok = 0
        self.resync_bytes = 0
        self.lost_frames = 0
        self.irregular_ts = 0
        self.last_ts: int | None = None
        self.ts_deltas: deque[int] = deque(maxlen=250)
        self.lines_ok = 0
        self.lines_bad = 0

    # ------------------------------------------------------------ recording
    def record_rx(self, n: int, now: float) -> None:
        self.bytes_total += n
        self.rx_win.append((now, n))
        self.last_rx = now
        if now - self._hist_t >= 1.0:                 # one history point per second
            self._hist_t = now
            self.rate_hist.append((now - self.t0, _rate(self.rx_win, now), _rate(self.smp_win, now)))

    def record_samples(self, n: int, now: float) -> None:
        self.smp_win.append((now, n))

    def record_frame(self, ts: int, imu: tuple[int, ...], battery: int, now: float) -> None:
        self.frames_ok += 1
        if self.last_ts is not None:
            d = ts - self.last_ts
            self.ts_deltas.append(d)
            if d != FRAME_MS:
                if d > FRAME_MS and d % FRAME_MS == 0:
                    self.lost_frames += d // FRAME_MS - 1
                else:
                    self.irregular_ts += 1
        self.last_ts = ts
        self.record_samples(SAMPLES_PER_FRAME, now)
        self.imu = imu
        if battery != self.battery or not self.batt_hist or now - self.t0 - self.batt_hist[-1][0] > 5:
            self.batt_hist.append((now - self.t0, battery))
        self.battery = battery

    def record_lines(self, ok: int, bad: int, now: float) -> None:
        self.lines_ok += ok
        self.lines_bad += bad
        if ok:
            self.record_samples(ok, now)

    def record_resync(self, n: int) -> None:
        self.resync_bytes += n

    def record_mode_change(self) -> None:
        self.reset_mode()
        self.mode_changes += 1

    # ------------------------------------------------------------ reading
    def snapshot(self, *, source: str, state: LinkState, error: str, mode: str | None, gen: int) -> LinkSnapshot:
        now = time.monotonic()
        d = list(self.ts_deltas)
        return LinkSnapshot(
            source=source, state=state, error=error, mode=mode, gen=gen, uptime=now - self.t0,
            idle=(now - self.last_rx) if self.last_rx else None, bytes_total=self.bytes_total,
            bps=_rate(self.rx_win, now), sps=_rate(self.smp_win, now), frames_ok=self.frames_ok,
            resync_bytes=self.resync_bytes, lost_frames=self.lost_frames, irregular_ts=self.irregular_ts,
            last_ts=self.last_ts, ts_mean=float(np.mean(d)) if d else None,
            ts_min=min(d) if d else None, ts_max=max(d) if d else None, lines_ok=self.lines_ok,
            lines_bad=self.lines_bad, battery=self.battery, imu=self.imu, mode_changes=self.mode_changes,
            battery_history=tuple(self.batt_hist), rate_history=tuple(self.rate_hist),
        )
