"""Window 4: raw input. Values exactly as received, physical channel order, no filtering."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .common import StudioWindow
from .device import CH, IMU_NAMES
from .theme import CH_COLORS, MUTED

SPANS = [1, 2, 5, 10, 30]


class RawWindow(StudioWindow):
    KEY = "raw"
    TITLE = "EMG 원시 데이터"

    def __init__(self, hub) -> None:
        super().__init__(hub)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("표시 시간"))
        self.span = QComboBox()
        self.span.addItems([f"{s} 초" for s in SPANS])
        self.span.setCurrentIndex(1)
        bar.addWidget(self.span)
        bar.addSpacing(16)
        bar.addWidget(QLabel("세로축"))
        self.yscale = QComboBox()
        self.yscale.addItems(["전체 범위 (0 ~ 최댓값)", "자동 확대 (보기만)"])
        self.yscale.currentIndexChanged.connect(self._apply_y)
        bar.addWidget(self.yscale)
        bar.addStretch(1)
        self.info = QLabel()
        self.info.setStyleSheet(f"color: {MUTED};")
        bar.addWidget(self.info)

        self.gl = pg.GraphicsLayoutWidget()
        self.plots, self.curves, self.values = [], [], []
        for c in range(CH):
            p = self.gl.addPlot(row=c, col=0)
            p.setLabel("left", f"CH{c + 1}")
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.getAxis("left").setWidth(48)
            if c < CH - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            if self.plots:
                p.setXLink(self.plots[0])
            self.curves.append(p.plot(pen=pg.mkPen(CH_COLORS[c], width=1)))
            lab = self.gl.addLabel("", row=c, col=1)
            lab.setFixedWidth(64)
            self.values.append(lab)
            self.plots.append(p)
        self.plots[-1].setLabel("bottom", "시간", units="s")

        self.imu_plot = self.gl.addPlot(row=CH, col=0)
        self.imu_plot.setLabel("left", "IMU")
        self.imu_plot.addLegend(offset=(4, 4), colCount=9, labelTextSize="8pt")
        self.imu_plot.setYRange(0, 255, padding=0.02)
        self.imu_plot.setMouseEnabled(x=False, y=False)
        self.imu_plot.hideButtons()
        self.imu_plot.getAxis("left").setWidth(48)
        imu_colors = ["#ff8787", "#69db7c", "#74c0fc", "#ffa8a8", "#b2f2bb", "#a5d8ff", "#ffd43b", "#e599f7", "#ced4da"]
        self.imu_curves = [self.imu_plot.plot(pen=pg.mkPen(imu_colors[i], width=1), name=IMU_NAMES[i]) for i in range(9)]
        self.gl.ci.layout.setRowStretchFactor(CH, 2)
        for c in range(CH):
            self.gl.ci.layout.setRowStretchFactor(c, 2)

        root = QWidget()
        v = QVBoxLayout(root)
        v.addLayout(bar)
        v.addWidget(self.gl, 1)
        self.setCentralWidget(root)
        self.resize(1100, 900)
        self._apply_y()

    def on_mode_change(self) -> None:
        self._apply_y()
        hex_mode = self.hub.mode == "hex"
        self.imu_plot.setVisible(hex_mode)

    def _apply_y(self) -> None:
        full = self.hub.spec["full"]
        for p in self.plots:
            if self.yscale.currentIndex() == 0:
                p.disableAutoRange(axis="y")
                p.setYRange(0, full, padding=0.02)
            else:
                p.enableAutoRange(axis="y")

    def refresh(self) -> None:
        link, fs = self.hub.link, self.hub.fs
        span = SPANS[self.span.currentIndex()]
        data = link.emg.last(int(span * fs))
        n = len(data)
        self.info.setText(f"{self.mode_text()}   ·   {n}샘플 표시")
        if not n:
            for cv in self.curves:
                cv.setData([], [])
            return
        x = (np.arange(n) - n) / fs
        for c in range(CH):
            self.curves[c].setData(x, data[:, c])
            self.values[c].setText(f"<span style='color:{CH_COLORS[c]}'>{int(data[-1, c])}</span>")
        self.plots[0].setXRange(-span, 0, padding=0)
        if self.hub.mode == "hex":
            imu = link.imu.last(int(span * 50))
            if len(imu):
                xi = (np.arange(len(imu)) - len(imu)) / 50.0
                for i in range(9):
                    self.imu_curves[i].setData(xi, imu[:, i])
                self.imu_plot.setXRange(-span, 0, padding=0)
        self.status.setText("가공 없음: 수신한 정수 그대로, 물리 채널 순서 (회전·필터는 필터 창 설정이 적용된 다른 창에서만)")
