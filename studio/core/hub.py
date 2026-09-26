"""Hub: assembles the parts and runs the 30 Hz clock. Windows read through its query
methods and react to its signals; they never reach into the Link or the Pipeline.

Each tick: new samples from the Link -> Pipeline -> StreamStore -> Calibrator,
then the pose learner, then `ticked` for the windows.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from .. import config
from ..device.link import Link
from ..device.protocol import ModeSpec, spec_for
from ..device.stats import LinkSnapshot
from ..hand.camera_link import CameraLink
from ..pose.learner import PoseLearner
from .calibration import Calibrator
from .pipeline import Pipeline
from .settings import Calibration, FilterSpec, Settings
from .streams import StreamStore


class Hub(QObject):
    ticked = Signal()
    settings_changed = Signal()
    calib_progress = Signal(str, float)   # kind, 0..1
    calib_done = Signal(str)

    def __init__(self, link: Link | None = None, camera: CameraLink | None = None,
                 settings_path: str = config.SETTINGS_PATH) -> None:
        super().__init__()
        self.link = link or Link()
        self.camera = camera or CameraLink()
        self._settings_path = settings_path
        self._settings = Settings.load(settings_path)
        self._pipe = Pipeline(self._settings)
        self.streams = StreamStore()
        self._cal = Calibrator()
        self._gen = -1
        self._pos = 0
        self.pose = PoseLearner(emg=self, frames=self.camera)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.start(1000 // config.TICK_HZ)

    # ------------------------------------------------------------ state
    @property
    def mode(self) -> str | None:
        return self.link.mode

    @property
    def spec(self) -> ModeSpec:
        return spec_for(self.link.mode)

    @property
    def fs(self) -> float:
        return self.spec.fs

    @property
    def gen(self) -> int:
        return self._gen

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def filter(self) -> FilterSpec:
        return self._settings.filter

    # ------------------------------------------------------------ settings
    def apply(self, settings: Settings) -> None:
        """The one way settings change: rebuild the chain, persist, notify."""
        self._settings = settings
        self._pipe.configure(settings, self.spec)
        settings.save(self._settings_path)
        self.settings_changed.emit()

    def set_filter(self, spec: FilterSpec) -> None:
        self.apply(replace(self._settings, filter=spec))

    def filter_notes(self) -> list[str]:
        return list(self._pipe.notes)

    def filter_response(self) -> tuple[np.ndarray, np.ndarray]:
        return self._pipe.response()

    # ------------------------------------------------------------ calibration
    def start_calibration(self, kind: str, seconds: float = 3.0) -> None:
        self._cal.start(kind, seconds, self.fs)

    def clear_calibration(self) -> None:
        self.apply(replace(self._settings, calib=Calibration()))

    def rest_rms(self) -> np.ndarray | None:
        return self._settings.calib.rest_rms()

    # ------------------------------------------------------------ queries for windows
    def raw_last(self, n: int) -> np.ndarray:
        """Values exactly as received, physical order."""
        return self.link.emg.last(n)

    def raw_logical_last(self, n: int) -> np.ndarray:
        return self.filter.to_logical(self.link.emg.last(n))

    def imu_last(self, n: int) -> np.ndarray:
        return self.link.imu.last(n)

    def bytes_since(self, pos: int) -> tuple[bytes, int]:
        return self.link.bytes.since(pos)

    def bytes_total(self) -> int:
        return self.link.bytes.total

    def snapshot(self) -> LinkSnapshot:
        return self.link.snapshot()

    # ------------------------------------------------------------ EmgSource (for the pose learner)
    def recent_raw(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        return self.streams.recent_raw(n)

    def recent_filtered(self, n: int) -> np.ndarray:
        return self.streams.filt.last(n)

    # ------------------------------------------------------------ clock
    def tick(self) -> None:
        gen, raw, t, self._pos = self.link.read_since(self._gen, self._pos)
        if gen != self._gen:                       # new mode or connection: new scale and rate
            self._gen = gen
            self.streams.clear()
            self._pipe.configure(self._settings, self.spec)
            self.settings_changed.emit()
        if len(raw):
            pre, y, env, norm = self._pipe.process(raw)
            self.streams.write(t, self._pipe.centred(raw), pre, y, env, norm)
            self._feed_calibration(env)
        self.pose.tick()
        self.ticked.emit()

    def _feed_calibration(self, env: np.ndarray) -> None:
        if not self._cal.active:
            return
        kind = self._cal.kind
        progress, level = self._cal.feed(env)
        self.calib_progress.emit(kind, progress)
        if level is not None:
            cal = self._settings.calib
            cal = replace(cal, rest=level) if kind == "rest" else replace(cal, mvc=level)
            self.apply(replace(self._settings, calib=cal))
            self.calib_done.emit(kind)
