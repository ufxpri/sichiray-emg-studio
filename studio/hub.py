"""Hub: one Link, one Pipeline, one clock. Windows connect to `ticked` and read rings.

Rings kept here (all aligned row for row, all in logical channel order):
  pre    raw minus mode centre, rotated       (input to the filters)
  filt   after the filter chain
  env    envelope
  norm   envelope normalised by the rest/MVC calibration
  cen    raw minus mode centre, physical order   (what the pose learner stores)
  t      perf_counter seconds of each row
Raw bytes-as-received values live in link.emg (physical channel order).
"""
from __future__ import annotations

import os
import time
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from .camera import CameraLink
from .device import CH, MODES, Link, Ring
from .pipeline import Pipeline, Settings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("EMG_DATA_DIR", os.path.join(HERE, "data"))
SETTINGS_PATH = os.path.join(DATA_DIR, "studio_settings.json")
TICK_HZ = 30


class Hub(QObject):
    ticked = Signal()
    settings_changed = Signal()
    calib_progress = Signal(str, float)   # kind, 0..1
    calib_done = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.link = Link()
        self.camera = CameraLink()
        self.settings = Settings.load(SETTINGS_PATH)
        self.pipe = Pipeline(self.settings)
        cap = 60 * 500
        self.pre, self.filt, self.env, self.norm = (Ring(cap, CH) for _ in range(4))
        self.cen = Ring(cap, CH)   # raw minus mode centre, physical order: unfiltered input kept for training
        self.t = Ring(cap, 1)   # perf_counter seconds of each row, same clock as camera frames
        self.gen = -1
        self.pos = 0
        self._cal: dict | None = None
        self.rate_hist: deque[tuple[float, float, float]] = deque(maxlen=3600)  # (t, B/s, samples/s)
        self._rate_t = 0.0
        from .pose import PoseLearner
        self.pose = PoseLearner(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000 // TICK_HZ)

    # ------------------------------------------------------------ properties
    @property
    def mode(self) -> str | None:
        return self.link.mode

    @property
    def spec(self) -> dict:
        return MODES.get(self.link.mode or "hex")

    @property
    def fs(self) -> float:
        return self.spec["fs"]

    # ------------------------------------------------------------ settings
    def apply(self, settings: Settings | None = None) -> None:
        """Rebuild the chain from (possibly edited) settings and persist them."""
        if settings is not None:
            self.settings = settings
        self.pipe.configure(self.settings, self.fs, self.spec["center"])
        self.settings.save(SETTINGS_PATH)
        self.settings_changed.emit()

    # ------------------------------------------------------------ clock
    def _tick(self) -> None:
        if self.link.gen != self.gen:
            self.gen = self.link.gen
            self.pos = 0
            for r in (self.pre, self.filt, self.env, self.norm, self.cen, self.t):
                r.clear()
            self.pipe.configure(self.settings, self.fs, self.spec["center"])
            self.settings_changed.emit()
        with self.link.lock:   # samples and their times are written together under this lock
            raw, new_pos = self.link.emg.since(self.pos)
            tt, _ = self.link.tsamp.since(self.pos)
        self.pos = new_pos
        n = min(len(raw), len(tt))
        raw, tt = raw[len(raw) - n:], tt[len(tt) - n:]
        if len(raw):
            self.t.write(tt)
            self.cen.write(raw - self.pipe.center)
            y, e, n = self.pipe.process(raw)
            self.pre.write(self.pipe.centred(raw))
            self.filt.write(y)
            self.env.write(e)
            self.norm.write(n)
            self._calibrate(y, e)
        self.pose.tick()
        now = time.monotonic()
        if now - self._rate_t >= 1.0 and self.link.status == "연결됨":
            self._rate_t = now
            snap = self.link.snapshot()
            if self.rate_hist and snap["uptime"] < self.rate_hist[-1][0]:
                self.rate_hist.clear()   # reconnected: the clock restarted
            self.rate_hist.append((snap["uptime"], snap["bps"], snap["sps"]))
        self.ticked.emit()

    # ------------------------------------------------------------ calibration
    def start_calibration(self, kind: str, seconds: float = 3.0) -> None:
        """kind 'rest' or 'mvc'. Collects `seconds` of envelope, then stores it."""
        self._cal = dict(kind=kind, need=int(seconds * self.fs), env=[], filt=[])

    def clear_calibration(self) -> None:
        self.settings.rest = self.settings.mvc = None
        self.apply()

    def _calibrate(self, y: np.ndarray, e: np.ndarray) -> None:
        c = self._cal
        if not c:
            return
        c["env"].append(e)
        c["filt"].append(y)
        got = sum(len(a) for a in c["env"])
        self.calib_progress.emit(c["kind"], min(1.0, got / c["need"]))
        if got < c["need"]:
            return
        env = np.vstack(c["env"])
        skip = min(len(env) // 4, int(0.3 * self.fs))   # let the envelope settle
        env = env[skip:]
        if c["kind"] == "rest":
            self.settings.rest = [float(v) for v in env.mean(axis=0)]
        else:
            self.settings.mvc = [float(v) for v in np.percentile(env, 95, axis=0)]
        self._cal = None
        self.apply()
        self.calib_done.emit(c["kind"])

    def rest_rms(self) -> np.ndarray | None:
        """Resting RMS of the filtered signal, derived from the rest envelope.
        For a zero-mean Gaussian, mean(|x|) = RMS * sqrt(2/pi)."""
        if not self.settings.rest:
            return None
        return np.array(self.settings.rest) / np.sqrt(2 / np.pi)
