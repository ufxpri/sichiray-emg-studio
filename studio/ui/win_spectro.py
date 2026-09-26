"""Window 3: rolling spectrogram per channel, current spectrum, and a noise diagnosis table."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel, QSplitter,
                               QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget, QCheckBox)

from ..core import analysis
from ..device.protocol import CH
from .common import StudioWindow
from .theme import CH_COLORS, GRADE_BG, GRADE_FG, MUTED

HISTORY_S = 10.0
METRIC_WINDOW_S = 4.0

GUIDE = """
<h3>스펙트로그램 읽는 법</h3>
<p>가로축은 시간(오른쪽 끝이 지금), 세로축은 주파수, 밝기는 그 순간 그 주파수의 세기(dB)입니다.
오른쪽 곡선은 최근 2초 스펙트럼입니다.</p>
<ul>
<li><b>근육이 수축하면</b> 20~250 Hz가 넓게, 흐릿한 구름처럼 한꺼번에 밝아집니다. 근전도는 무작위 신호라 특정 주파수에 모이지 않습니다.</li>
<li><b>가로로 곧게 이어지는 선</b>은 한 주파수에 계속 머무는 결정적 간섭입니다. 60/120/180 Hz는 전원, 9.77 Hz는 이 장비의 충전기(§18)입니다. 근육 신호는 이런 모양으로 나오지 않습니다.</li>
<li><b>바닥(0~20 Hz)이 번쩍이는 것</b>은 전극이 움직이거나 밴드가 흔들린 것입니다(움직임 아티팩트).</li>
<li><b>위에서 아래까지 세로로 번쩍이는 띠</b>는 순간적인 충격입니다. 접촉 끊김, 정전기, 밴드를 건드린 경우입니다.</li>
<li><b>모든 채널에 똑같이 나타나는 무늬</b>는 근육이 아니라 공통 잡음입니다. 근육 신호는 채널마다 다릅니다.</li>
</ul>
<p>점선은 EMG 대역의 시작점(20 Hz)입니다. HEX 모드는 0~250 Hz를 볼 수 있지만, ASCII 모드는 67 Hz 샘플링이라
0~33 Hz만 보이고 그 위의 근전도는 아래로 접혀 들어옵니다(에일리어싱, §13).</p>
<p><b>소스</b>를 '필터 적용 후'로 바꾸면 필터 창에서 설정한 전역 필터가 무엇을 걷어냈는지 비교할 수 있습니다.</p>
<h3>진단표</h3>
<p>최근 4초로 계산합니다. 초록 = 양호, 노랑 = 주의, 빨강 = 나쁨, 회색 = 참고값(좋고 나쁨이 상황에 따라 다름).
행 이름에 마우스를 올리면 그 지표가 무엇을 뜻하는지 설명이 나옵니다. 가장 먼저 볼 것은 <b>협대역 간섭</b>(쓸 수 있는 측정인가),
다음은 <b>저주파</b>(밴드가 움직이는가)와 <b>채널 간 상관</b>(공통 잡음인가)입니다.</p>
<p>휴식할 때 표를 한 번 보고, 주먹을 쥐었을 때 다시 보세요. 좋은 채널은 수축 중에 RMS, 양자화 대비, 휴식 대비가 크게 오르고,
EMG 대역 비율이 올라가며, 첨도는 3에 가까워집니다.</p>
"""


class SpectroWindow(StudioWindow):
    KEY = "spectro"
    TITLE = "스펙트로그램 · 노이즈 분석"
    REFRESH_HZ = 20.0

    def __init__(self, hub) -> None:
        super().__init__(hub)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("소스"))
        self.source = QComboBox()
        self.source.addItems(["원본 입력 (중앙값만 뺌)", "필터 적용 후 (전역 필터)"])
        self.source.currentIndexChanged.connect(lambda *_: self._reset())
        bar.addWidget(self.source)
        bar.addSpacing(12)
        self.auto = QCheckBox("밝기 자동")
        self.auto.setChecked(True)
        bar.addWidget(self.auto)
        bar.addWidget(QLabel("범위 dB"))
        self.lo, self.hi = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, v in ((self.lo, -40.0), (self.hi, 30.0)):
            sp.setRange(-120, 80)
            sp.setValue(v)
            sp.setSingleStep(5)
            bar.addWidget(sp)
        bar.addStretch(1)
        self.info = QLabel()
        self.info.setStyleSheet(f"color: {MUTED};")
        bar.addWidget(self.info)

        # ---- spectrogram rows
        self.gl = pg.GraphicsLayoutWidget()
        cmap = pg.colormap.get("inferno")
        self.imgs, self.psd_curves, self.spec_plots, self.psd_plots = [], [], [], []
        for c in range(CH):
            p = self.gl.addPlot(row=c, col=0)
            p.setLabel("left", f"CH{c + 1}  Hz", color=CH_COLORS[c])
            p.getAxis("left").setWidth(52)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            img = pg.ImageItem(axisOrder="row-major")
            img.setColorMap(cmap)
            p.addItem(img)
            p.addItem(pg.InfiniteLine(pos=20, angle=0, pen=pg.mkPen("#ffffff", width=1, style=Qt.DotLine)))
            if c < CH - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            if self.spec_plots:
                p.setXLink(self.spec_plots[0])
            q = self.gl.addPlot(row=c, col=1)
            q.setMouseEnabled(x=False, y=False)
            q.hideButtons()
            q.setYLink(p)
            q.getAxis("left").setStyle(showValues=False)
            q.showGrid(x=True, alpha=0.15)
            if c < CH - 1:
                q.getAxis("bottom").setStyle(showValues=False)
            self.psd_curves.append(q.plot(pen=pg.mkPen(CH_COLORS[c], width=1.2)))
            self.imgs.append(img)
            self.spec_plots.append(p)
            self.psd_plots.append(q)
        self.spec_plots[-1].setLabel("bottom", "시간", units="s")
        self.psd_plots[-1].setLabel("bottom", "dB")
        self.gl.ci.layout.setColumnStretchFactor(0, 5)
        self.gl.ci.layout.setColumnStretchFactor(1, 1)

        # ---- diagnosis table + guide
        self.table = QTableWidget(len(analysis.METRICS) + 1, CH)
        self.table.setHorizontalHeaderLabels([f"CH{c + 1}" for c in range(CH)])
        labels = ["종합"] + [f"{m.label}" + (f" ({m.unit})" if m.unit else "") for m in analysis.METRICS]
        self.table.setVerticalHeaderLabels(labels)
        self.table.verticalHeaderItem(0).setToolTip("아래 지표 중 가장 나쁜 판정입니다.")
        for i, m in enumerate(analysis.METRICS):
            self.table.verticalHeaderItem(i + 1).setToolTip(f"<p style='max-width:360px'>{m.help}</p>")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        for r in range(self.table.rowCount()):
            for c in range(CH):
                it = QTableWidgetItem("-")
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, it)

        guide = QTextBrowser()
        guide.setHtml(GUIDE)
        guide.setOpenExternalLinks(False)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.table)
        right.addWidget(guide)
        right.setSizes([470, 330])

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addLayout(bar)
        lv.addWidget(self.gl, 1)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([900, 560])
        self.setCentralWidget(split)
        self.resize(1500, 950)
        self._metric_t = 0.0
        self._reset()

    # ------------------------------------------------------------ geometry
    def on_mode_change(self) -> None:
        self._reset()

    def _reset(self) -> None:
        fs = self.hub.fs
        self.nfft = int(2 ** np.ceil(np.log2(max(16, fs * 0.5))))   # ~0.5 s: 2 Hz bins at 500 Hz
        self.hop = max(1, round(fs / 20))
        self.ncol = int(HISTORY_S * fs / self.hop)
        self.freqs = np.fft.rfftfreq(self.nfft, 1 / fs)
        self.win = np.hanning(self.nfft)
        self.spec = np.full((CH, len(self.freqs), self.ncol), -80.0, np.float32)
        self.lin = np.zeros((CH, len(self.freqs), self.ncol), np.float32)
        self.buf = np.zeros((0, CH))
        ring = self.ring()
        self.pos = ring.total
        width = self.ncol * self.hop / fs
        for c in range(CH):
            self.imgs[c].setImage(self.spec[c], autoLevels=False, levels=(self.lo.value(), self.hi.value()))
            self.imgs[c].setRect(QRectF(-width, 0, width, fs / 2))
            self.spec_plots[c].setXRange(-width, 0, padding=0)
            self.spec_plots[c].setYRange(0, fs / 2, padding=0)
        self._metric_t = 0.0

    def ring(self):
        st = self.hub.streams
        return st.pre if self.source.currentIndex() == 0 else st.filt

    # ------------------------------------------------------------ update
    def refresh(self) -> None:
        new, self.pos = self.ring().since(self.pos)
        if len(new):
            self.buf = np.vstack([self.buf, new])[-(self.nfft + self.hop * self.ncol):]
        cols = []
        while len(self.buf) >= self.nfft:
            seg = self.buf[:self.nfft]
            seg = (seg - seg.mean(axis=0)) * self.win[:, None]
            P = (np.abs(np.fft.rfft(seg, axis=0)) ** 2) / (self.win ** 2).sum()
            cols.append(P.T)                      # (CH, F)
            self.buf = self.buf[self.hop:]
        if cols:
            k = min(len(cols), self.ncol)
            block = np.stack(cols[-k:], axis=2)   # (CH, F, k)
            self.lin = np.roll(self.lin, -k, axis=2)
            self.lin[:, :, -k:] = block
            self.spec = np.roll(self.spec, -k, axis=2)
            self.spec[:, :, -k:] = 10 * np.log10(block + 1e-6)
            if self.auto.isChecked():
                live = self.spec[:, :, -min(self.ncol, 60):]
                lo, hi = np.percentile(live, 5), np.percentile(live, 99.7)
                self.lo.setValue(float(np.floor(lo)))
                self.hi.setValue(float(np.ceil(max(hi, lo + 10))))
            levels = (self.lo.value(), self.hi.value())
            recent = 10 * np.log10(self.lin[:, :, -40:].mean(axis=2) + 1e-6)   # ~2 s
            for c in range(CH):
                self.imgs[c].setImage(self.spec[c], autoLevels=False, levels=levels)
                self.psd_curves[c].setData(recent[c], self.freqs)
                self.psd_plots[c].setXRange(levels[0], levels[1] + 10, padding=0)
        self.info.setText(f"{self.mode_text()}   ·   FFT {self.nfft}점 ({self.nfft / self.hub.fs:.2f} s, "
                          f"{self.hub.fs / self.nfft:.1f} Hz 간격)")
        now = time.monotonic()
        if now - self._metric_t > 0.5:
            self._metric_t = now
            self._update_table()

    def _update_table(self) -> None:
        fs = self.hub.fs
        n = int(METRIC_WINDOW_S * fs)
        sig = self.ring().last(n)
        raw = self.hub.raw_logical_last(n)   # logical order, like the analysed source
        if len(sig) < max(64, fs) or len(raw) != len(sig):
            self.status.setText("진단표: 데이터가 4초 쌓이면 계산합니다.")
            return
        spec = self.hub.spec
        ctx = dict(center=spec.center, full=spec.full, mode=self.hub.mode)
        vals, grades, lines = analysis.compute(raw, sig, fs, ctx, self.hub.rest_rms())
        order = {"info": 0, "ok": 1, "warn": 2, "bad": 3}
        for c in range(CH):
            worst = max((grades[m.key][c] for m in analysis.METRICS), key=order.get)
            self._cell(0, c, {"ok": "양호", "warn": "주의", "bad": "나쁨", "info": "-"}[worst], worst, bold=True)
            for i, m in enumerate(analysis.METRICS):
                v = vals[m.key][c]
                txt = "-" if not np.isfinite(v) else m.fmt.format(v)
                if m.key == "line" and np.isfinite(lines[c][0]):
                    txt = f"{lines[c][0]:.1f}Hz {v:.0f}%"
                elif m.key == "line":
                    txt = "없음"
                self._cell(i + 1, c, txt, grades[m.key][c])
        rot = self.hub.filter.rotate
        self.status.setText(f"진단표: 최근 {METRIC_WINDOW_S:g}초, 소스 = {self.source.currentText()}"
                            + (f", 채널 회전 +{rot} 적용(논리 채널 순서)" if rot else ""))

    def _cell(self, r: int, c: int, text: str, grade: str, bold: bool = False) -> None:
        it = self.table.item(r, c)
        it.setText(text)
        it.setBackground(QBrush(QColor(GRADE_BG[grade])))
        it.setForeground(QBrush(QColor(GRADE_FG[grade])))
        if bold:
            f = it.font()
            f.setBold(True)
            it.setFont(f)
