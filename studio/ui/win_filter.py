"""Window 6: the global filter chain and calibration. Changes apply to every window at once."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget, QSplitter)
from scipy.signal import welch

from ..core.settings import FilterSpec
from ..device.protocol import CH
from .common import StudioWindow
from .theme import CH_COLORS, MUTED

HELP = {
    "hp": "이 주파수 아래를 걷어냅니다. 20 Hz 아래는 전극 움직임과 기준선 이동이라 근전도가 아닙니다(§13). "
          "차수가 높을수록 경계가 가팔라지지만 시간 지연과 울림이 늘어납니다.",
    "lp": "이 주파수 위를 걷어냅니다. HEX(500 Hz)의 표면 근전도는 250 Hz까지 쓸모가 있어 보통 끕니다. "
          "고주파 잡음이 보일 때만 켜세요.",
    "notch": "전원 주파수(한국 60 Hz)와 그 배수를 좁게 파냅니다. Q가 클수록 폭이 좁아 근전도를 덜 깎습니다.",
    "extra": "쉼표로 구분한 주파수를 추가로 파냅니다. 스펙트로그램에서 가로 선으로 보이는 간섭에 씁니다. "
             "예: 9.77 (충전기, §18). 원인을 없애는 것이 먼저이고, 노치는 차선책입니다.",
    "rotate": "밴드는 착용할 때마다 돌아갑니다. 신호 패턴은 밴드와 함께 채널 단위로 강체 회전합니다(§15). "
              "논리 채널 k = 물리 채널 (k + 회전) mod 8. 원시 창은 항상 물리 순서입니다.",
    "mute": "켠 채널은 필터 출력을 0으로 만듭니다. 전극이 떨어졌거나 간섭이 심한 채널을 뺄 때 씁니다.",
    "env": "포락선 = |필터 출력|을 이 주파수로 저역통과한 값입니다. 힘의 크기를 따라가는 느린 곡선으로, "
           "3D 손과 ML의 입력이 됩니다. 낮을수록 부드럽지만 늦게 반응합니다.",
    "calib": "휴식: 팔 힘을 완전히 빼고 3초. 최대 수축: 주먹을 있는 힘껏 쥐고 3초. 두 값을 재면 포락선을 "
             "0(휴식)~1(최대)로 정규화합니다. 세션마다 진폭이 2배 이상 달라지므로(§23) 착용할 때마다 다시 재세요.",
}


class FilterWindow(StudioWindow):
    KEY = "filter"
    TITLE = "신호 처리 · 필터 · 보정"
    REFRESH_HZ = 15.0

    def __init__(self, hub) -> None:
        super().__init__(hub)
        self._loading = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._commit)

        form = QVBoxLayout()
        # high-pass
        self.hp_on, self.hp_hz, self.hp_ord = QCheckBox("사용"), self._dspin(0.5, 240, 1), self._ispin(1, 8)
        form.addWidget(self._group("고역통과 (Butterworth)", "hp",
                                   [("", self.hp_on), ("차단 주파수 (Hz)", self.hp_hz), ("차수", self.hp_ord)]))
        self.lp_on, self.lp_hz, self.lp_ord = QCheckBox("사용"), self._dspin(1, 249, 1), self._ispin(1, 8)
        form.addWidget(self._group("저역통과 (Butterworth)", "lp",
                                   [("", self.lp_on), ("차단 주파수 (Hz)", self.lp_hz), ("차수", self.lp_ord)]))
        self.n_on = QCheckBox("사용")
        self.n_base = QComboBox()
        self.n_base.addItems(["60", "50"])
        self.n_harm, self.n_q = self._ispin(1, 4), self._dspin(1, 100, 1)
        form.addWidget(self._group("전원 노치", "notch", [("", self.n_on), ("기본 주파수 (Hz)", self.n_base),
                                                         ("배수 개수", self.n_harm), ("Q", self.n_q)]))
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("예: 9.77, 45")
        form.addWidget(self._group("추가 노치", "extra", [("주파수 목록", self.extra)]))
        self.rotate = self._ispin(0, 7)
        mute_box = QWidget()
        mg = QGridLayout(mute_box)
        mg.setContentsMargins(0, 0, 0, 0)
        self.mutes = []
        for c in range(CH):
            cb = QCheckBox(f"{c + 1}")
            cb.setStyleSheet(f"color: {CH_COLORS[c]};")
            mg.addWidget(cb, c // 4, c % 4)
            self.mutes.append(cb)
        form.addWidget(self._group("채널", "rotate", [("회전 (+칸)", self.rotate), ("끄기", mute_box)]))
        self.env_hz = self._dspin(0.5, 30, 0.5)
        form.addWidget(self._group("포락선", "env", [("저역통과 (Hz)", self.env_hz)]))

        # calibration
        cal = QGroupBox("보정 (정규화)")
        cal.setToolTip(HELP["calib"])
        cv = QVBoxLayout(cal)
        row = QHBoxLayout()
        self.b_rest = QPushButton("휴식 측정 (3초)")
        self.b_mvc = QPushButton("최대 수축 측정 (3초)")
        self.b_clear = QPushButton("초기화")
        self.b_rest.clicked.connect(lambda: self._calib("rest"))
        self.b_mvc.clicked.connect(lambda: self._calib("mvc"))
        self.b_clear.clicked.connect(hub.clear_calibration)
        for b in (self.b_rest, self.b_mvc, self.b_clear):
            row.addWidget(b)
        cv.addLayout(row)
        self.cal_bar = QProgressBar()
        self.cal_bar.setRange(0, 100)
        self.cal_bar.setValue(0)
        self.cal_bar.setFormat("")
        cv.addWidget(self.cal_bar)
        self.cal_text = QLabel()
        self.cal_text.setWordWrap(True)
        self.cal_text.setStyleSheet(f"color: {MUTED};")
        cv.addWidget(self.cal_text)
        hint = QLabel(HELP["calib"])
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {MUTED}; font-size: 8pt;")
        cv.addWidget(hint)
        form.addWidget(cal)

        reset = QPushButton("필터 기본값으로 (보정값은 유지)")
        reset.clicked.connect(self._defaults)
        form.addWidget(reset)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color: #ffe8a3;")
        form.addWidget(self.notes)
        form.addStretch(1)

        panel = QWidget()
        panel.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)

        # ---- plots
        top = QHBoxLayout()
        top.addWidget(QLabel("비교 채널"))
        self.ch = QComboBox()
        self.ch.addItems([f"CH{c + 1}" for c in range(CH)])
        top.addWidget(self.ch)
        top.addStretch(1)
        self.info = QLabel()
        self.info.setStyleSheet(f"color: {MUTED};")
        top.addWidget(self.info)

        self.gl = pg.GraphicsLayoutWidget()
        self.p_resp = self.gl.addPlot(row=0, col=0, title="필터 체인 주파수 응답")
        self.p_resp.setLabel("left", "dB")
        self.p_resp.setLabel("bottom", "Hz")
        self.p_resp.showGrid(x=True, y=True, alpha=0.2)
        self.p_resp.setYRange(-60, 5)
        self.c_resp = self.p_resp.plot(pen=pg.mkPen("#ffd43b", width=2))
        self.p_psd = self.gl.addPlot(row=0, col=1, title="스펙트럼: 필터 전 / 후 (최근 4초)")
        self.p_psd.setLabel("bottom", "Hz")
        self.p_psd.setLabel("left", "dB")
        self.p_psd.showGrid(x=True, y=True, alpha=0.2)
        self.p_psd.addLegend(offset=(-4, 4))
        self.c_psd_pre = self.p_psd.plot(pen=pg.mkPen("#868e96", width=1.2), name="전")
        self.c_psd_post = self.p_psd.plot(pen=pg.mkPen("#3bc9db", width=1.5), name="후")
        self.p_time = self.gl.addPlot(row=1, col=0, colspan=2, title="파형: 필터 전 / 후 (최근 2초, 중앙값을 뺀 카운트)")
        self.p_time.setLabel("bottom", "시간", units="s")
        self.p_time.showGrid(x=True, y=True, alpha=0.2)
        self.p_time.addLegend(offset=(-4, 4))
        self.c_pre = self.p_time.plot(pen=pg.mkPen("#868e96", width=1), name="전")
        self.c_post = self.p_time.plot(pen=pg.mkPen("#3bc9db", width=1), name="후")
        self.c_env = self.p_time.plot(pen=pg.mkPen("#ffd43b", width=2), name="포락선")
        self.p_bar = self.gl.addPlot(row=2, col=0, colspan=2, title="채널별 포락선 (카운트) · 정규화 값은 막대 위 숫자")
        self.p_bar.setXRange(-0.6, CH - 0.4)
        self.p_bar.getAxis("bottom").setTicks([[(c, f"CH{c + 1}") for c in range(CH)]])
        self.bars = pg.BarGraphItem(x=np.arange(CH), height=np.zeros(CH), width=0.6,
                                    brushes=[pg.mkBrush(c) for c in CH_COLORS])
        self.p_bar.addItem(self.bars)
        self.rest_marks = pg.ScatterPlotItem(symbol="t1", size=12, brush=pg.mkBrush("#ffffff"))
        self.mvc_marks = pg.ScatterPlotItem(symbol="t", size=12, brush=pg.mkBrush("#ff6b6b"))
        self.p_bar.addItem(self.rest_marks)
        self.p_bar.addItem(self.mvc_marks)
        self.bar_txt = [pg.TextItem("", anchor=(0.5, 1), color="#d7dae0") for _ in range(CH)]
        for t in self.bar_txt:
            self.p_bar.addItem(t)
        for p in (self.p_resp, self.p_psd, self.p_time, self.p_bar):
            p.hideButtons()
        self.gl.ci.layout.setRowStretchFactor(0, 3)
        self.gl.ci.layout.setRowStretchFactor(1, 3)
        self.gl.ci.layout.setRowStretchFactor(2, 2)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addLayout(top)
        rv.addWidget(self.gl, 1)
        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([360, 1000])
        self.setCentralWidget(split)
        self.resize(1400, 900)

        for w in (self.hp_on, self.lp_on, self.n_on, *self.mutes):
            w.toggled.connect(self._changed)
        for w in (self.hp_hz, self.hp_ord, self.lp_hz, self.lp_ord, self.n_harm, self.n_q, self.rotate, self.env_hz):
            w.valueChanged.connect(self._changed)
        self.n_base.currentIndexChanged.connect(self._changed)
        self.extra.textChanged.connect(self._changed)
        hub.settings_changed.connect(self._load)
        hub.calib_progress.connect(self._on_progress)
        hub.calib_done.connect(lambda k: self.cal_bar.setFormat("완료"))
        self._load()

    # ------------------------------------------------------------ widgets
    def _dspin(self, lo: float, hi: float, step: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(2)
        s.setKeyboardTracking(False)
        return s

    def _ispin(self, lo: int, hi: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setKeyboardTracking(False)
        return s

    def _group(self, title: str, help_key: str, rows) -> QGroupBox:
        g = QGroupBox(title)
        g.setToolTip(f"<p style='max-width:340px'>{HELP[help_key]}</p>")
        f = QFormLayout(g)
        for label, w in rows:
            f.addRow(label, w)
        return g

    # ------------------------------------------------------------ settings <-> widgets
    def _load(self) -> None:
        s, f = self.hub.settings, self.hub.filter
        self._loading = True
        self.hp_on.setChecked(f.hp_on)
        self.hp_hz.setValue(f.hp_hz)
        self.hp_ord.setValue(f.hp_order)
        self.lp_on.setChecked(f.lp_on)
        self.lp_hz.setValue(f.lp_hz)
        self.lp_ord.setValue(f.lp_order)
        self.n_on.setChecked(f.notch_on)
        self.n_base.setCurrentText(f"{f.notch_base:g}")
        self.n_harm.setValue(f.notch_harm)
        self.n_q.setValue(f.notch_q)
        if self._parse_extra() != f.extra_notches:
            self.extra.setText(", ".join(f"{v:g}" for v in f.extra_notches))
        self.rotate.setValue(f.rotate)
        for c, cb in enumerate(self.mutes):
            cb.setChecked(bool(f.mute[c]))
        self.env_hz.setValue(s.env_hz)
        self._loading = False
        self.notes.setText("\n".join(self.hub.filter_notes()))
        self._draw_response()
        self._cal_summary()

    def _parse_extra(self) -> tuple[float, ...]:
        out = []
        for tok in self.extra.text().replace(";", ",").split(","):
            try:
                v = float(tok)
            except ValueError:
                continue
            if v > 0:
                out.append(v)
        return tuple(out)

    def _changed(self, *_) -> None:
        if not self._loading:
            self._debounce.start()

    def _commit(self) -> None:
        spec = FilterSpec(
            hp_on=self.hp_on.isChecked(), hp_hz=self.hp_hz.value(), hp_order=self.hp_ord.value(),
            lp_on=self.lp_on.isChecked(), lp_hz=self.lp_hz.value(), lp_order=self.lp_ord.value(),
            notch_on=self.n_on.isChecked(), notch_base=float(self.n_base.currentText()),
            notch_harm=self.n_harm.value(), notch_q=self.n_q.value(), extra_notches=self._parse_extra(),
            rotate=self.rotate.value(), mute=tuple(cb.isChecked() for cb in self.mutes))
        self.hub.apply(replace(self.hub.settings, filter=spec, env_hz=self.env_hz.value()))

    def _defaults(self) -> None:
        self.hub.apply(replace(self.hub.settings, filter=FilterSpec(), env_hz=5.0))

    # ------------------------------------------------------------ calibration
    def _calib(self, kind: str) -> None:
        if self.hub.mode is None:
            self.cal_bar.setFormat("데이터가 없습니다")
            return
        self.cal_bar.setFormat("휴식: 힘 빼세요… %p%" if kind == "rest" else "최대 수축: 꽉 쥐세요… %p%")
        self.cal_bar.setValue(0)
        self.hub.start_calibration(kind)

    def _on_progress(self, kind: str, frac: float) -> None:
        self.cal_bar.setValue(int(frac * 100))

    def _cal_summary(self) -> None:
        s = self.hub.settings.calib
        f = lambda xs: " ".join(f"{v:.1f}" for v in xs) if xs else "없음"
        self.cal_text.setText(f"휴식 포락선: {f(s.rest)}\n최대 포락선: {f(s.mvc)}")

    # ------------------------------------------------------------ plots
    def _draw_response(self) -> None:
        fs = self.hub.fs
        self.c_resp.setData(*self.hub.filter_response())
        self.p_resp.setXRange(0, fs / 2, padding=0)

    def on_mode_change(self) -> None:
        self._load()

    def refresh(self) -> None:
        st, fs, c = self.hub.streams, self.hub.fs, self.ch.currentIndex()
        self.info.setText(self.mode_text())
        n2, n4 = int(2 * fs), int(4 * fs)
        pre, post, env = st.pre.last(n2), st.filt.last(n2), st.env.last(n2)
        if len(pre) and len(pre) == len(post) == len(env):
            x = (np.arange(len(pre)) - len(pre)) / fs
            p0 = pre[:, c] - pre[:, c].mean()
            self.c_pre.setData(x, p0)
            self.c_post.setData(x, post[:, c])
            self.c_env.setData(x, env[:, c])
        pre4, post4 = st.pre.last(n4), st.filt.last(n4)
        if len(pre4) >= 64 and len(pre4) == len(post4):
            nper = min(len(pre4), 512 if fs > 200 else 64)
            f, a = welch(pre4[:, c] - pre4[:, c].mean(), fs=fs, nperseg=nper)
            _, b = welch(post4[:, c], fs=fs, nperseg=nper)
            self.c_psd_pre.setData(f, 10 * np.log10(a + 1e-9))
            self.c_psd_post.setData(f, 10 * np.log10(b + 1e-9))
            self.p_psd.setXRange(0, fs / 2, padding=0)
        e = st.env.last(max(1, int(0.2 * fs)))
        if len(e):
            cur = e.mean(axis=0)
            self.bars.setOpts(height=cur)
            s = self.hub.settings.calib
            if s.rest:
                self.rest_marks.setData(np.arange(CH), s.rest)
            else:
                self.rest_marks.clear()
            if s.mvc:
                self.mvc_marks.setData(np.arange(CH), s.mvc)
            else:
                self.mvc_marks.clear()
            nrm = st.norm.last(1)
            top = max(float(cur.max()), max(s.mvc) if s.mvc else 0.0, 1.0)
            self.p_bar.setYRange(0, top * 1.25, padding=0)
            for k, t in enumerate(self.bar_txt):
                t.setPos(k, cur[k] + top * 0.02)
                t.setText(f"{nrm[0, k]:.2f}" if (s.rest and s.mvc and len(nrm)) else f"{cur[k]:.1f}")
        self.status.setText("설정은 즉시 모든 창에 적용되고 data/studio_settings.json에 저장됩니다.  "
                            "▽ 흰 삼각형 = 휴식 기준, △ 빨간 삼각형 = 최대 수축 기준")
