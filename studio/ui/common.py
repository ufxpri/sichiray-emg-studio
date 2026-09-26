"""Base class for studio windows: hub wiring, geometry memory, refresh throttling."""
from __future__ import annotations

import time

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel, QMainWindow

from ..core.hub import Hub
from ..device.protocol import LinkState
from .theme import MUTED


class StudioWindow(QMainWindow):
    KEY = "window"
    TITLE = "EMG Studio"
    REFRESH_HZ = 30.0

    def __init__(self, hub: Hub) -> None:
        super().__init__()
        self.hub = hub
        self.setWindowTitle(f"{self.TITLE} — EMG Studio")
        self._last = 0.0
        self._gen = -1
        self.status = QLabel()
        self.status.setStyleSheet(f"color: {MUTED}; padding: 2px 6px;")
        self.statusBar().addWidget(self.status, 1)
        hub.ticked.connect(self._on_tick)
        g = QSettings("sichiray", "emg_studio").value(f"geom/{self.KEY}")
        if g is not None:
            self.restoreGeometry(g)

    def closeEvent(self, ev) -> None:
        QSettings("sichiray", "emg_studio").setValue(f"geom/{self.KEY}", self.saveGeometry())
        super().closeEvent(ev)

    def _on_tick(self) -> None:
        if not self.isVisible():
            return
        now = time.monotonic()
        if now - self._last < 1 / self.REFRESH_HZ - 0.002:
            return
        self._last = now
        if self.hub.gen != self._gen:
            self._gen = self.hub.gen
            self.on_mode_change()
        self.refresh()

    def on_mode_change(self) -> None:
        """Mode (HEX/ASCII) or connection changed: sample rate and scale are new."""

    def refresh(self) -> None:
        raise NotImplementedError

    def mode_text(self) -> str:
        m = self.hub.mode
        if not m:
            return "데이터 대기 중"
        s = self.hub.spec
        return f"{m.upper()}  {s.bits}-bit  {s.fs:g} Hz"


STATE_LABEL = {LinkState.DISCONNECTED: "끊김", LinkState.CONNECTING: "연결 중",
               LinkState.CONNECTED: "연결됨", LinkState.ERROR: "오류"}
