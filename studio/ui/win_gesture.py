"""Live gesture window: the trained CNN (studio.gesture) on the incoming EMG.

1. pick a model (trained with `python -m studio.gesture.train`, saved under data/models/gesture)
2. calibrate the band placement: about 40 s following the pictures; the signal is used without
   labels, for the channel levels and the network's BatchNorm statistics (AdaBN)
3. the current gesture shows as a hand picture, with the probabilities of every class
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout,
                               QWidget)

from .. import config
from ..gesture.net import FS, GestureNet, KalmanDot, LiveDecoder, Stabilizer
from .common import StudioWindow
from .hand_pic import HandPicture
from .theme import MUTED, PANEL, TEXT

MODEL_DIR = os.path.join(config.DATA_DIR, "models", "gesture")
GESTURE_COLORS = ["#868e96", "#4dabf7", "#ff922b", "#20c997", "#fcc419", "#f783ac", "#51cf66", "#9775fa"]
CALIB_REST_S, CALIB_HOLD_S, CALIB_GAP_S = 4.0, 3.0, 2.0
UNSURE = 0.5            # below this top probability the decision is shown dimmed
SMOOTHING = [("0.4초 (학습과 같음)", 5), ("0.6초", 7), ("0.8초", 9), ("1.0초", 11)]
SWITCHING = [("3회 연속·60% 이상 (권장)", 0.6, 3), ("바로 (채터링 있음)", 0.0, 1), ("5회 연속·70% 이상", 0.7, 5)]
TRAIL = 12              # past map positions drawn as a tail
# map dot Kalman filter: (label, accel, noise); None = raw kNN position
DOT_FILTER = [("중 (권장)", 1.0, 0.3), ("약", 3.0, 0.15), ("강", 0.5, 0.3), ("끔", None, None)]


def colour(label: int) -> str:
    return GESTURE_COLORS[label % len(GESTURE_COLORS)]


class ProbBars(QWidget):
    """One bar per class, the decided one in its gesture colour."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[tuple[int, str]] = []
        self.p = np.zeros(0)
        self.setMinimumWidth(260)

    def set(self, names: list[tuple[int, str]], p: np.ndarray) -> None:
        self.names, self.p = names, p
        self.setMinimumHeight(28 * max(1, len(names)) + 8)
        self.update()

    def paintEvent(self, ev) -> None:
        q = QPainter(self)
        q.setRenderHint(QPainter.Antialiasing)
        if not self.names:
            q.end()
            return
        top = int(np.argmax(self.p)) if len(self.p) else -1
        f = QFont(self.font())
        f.setPointSize(10)
        q.setFont(f)
        name_w, pct_w, h = 96, 48, 28
        bar_w = max(40, self.width() - name_w - pct_w - 12)
        for i, (lab, name) in enumerate(self.names):
            y = 4 + i * h
            v = float(self.p[i]) if i < len(self.p) else 0.0
            q.setPen(QColor(TEXT if i == top else MUTED))
            q.drawText(QRectF(0, y, name_w, h - 6), Qt.AlignVCenter | Qt.AlignLeft, name)
            q.setPen(Qt.NoPen)
            q.setBrush(QColor(PANEL))
            q.drawRoundedRect(QRectF(name_w, y + 4, bar_w, h - 14), 4, 4)
            q.setBrush(QColor(colour(lab) if i == top else "#495057"))
            q.drawRoundedRect(QRectF(name_w, y + 4, max(2.0, bar_w * v), h - 14), 4, 4)
            q.setPen(QColor(TEXT if i == top else MUTED))
            q.drawText(QRectF(name_w + bar_w + 6, y, pct_w, h - 6), Qt.AlignVCenter | Qt.AlignRight,
                       f"{100 * v:.0f}%")
        q.end()


class SignalMapView(pg.PlotWidget):
    """The model's t-SNE map of training windows, with the live window as a moving dot."""

    def __init__(self) -> None:
        super().__init__()
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.setAspectLocked(True)
        for ax in ("left", "bottom"):
            self.getAxis(ax).setStyle(showValues=False)
            self.getAxis(ax).setTicks([[]])
        self.ref = pg.ScatterPlotItem(size=4, pen=None)
        self.addItem(self.ref)
        self.tail = pg.PlotDataItem(pen=pg.mkPen(TEXT, width=1.5))
        self.addItem(self.tail)
        self.raw = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(255, 255, 255, 90))
        self.addItem(self.raw)
        self.dot = pg.ScatterPlotItem(size=18, pen=pg.mkPen("#ffffff", width=2))
        self.addItem(self.dot)
        self.texts: list[pg.TextItem] = []
        self.empty = pg.TextItem("", color=MUTED, anchor=(0.5, 0.5))
        self.addItem(self.empty)
        self.trail: list[np.ndarray] = []
        self.kalman: KalmanDot | None = KalmanDot(accel=DOT_FILTER[0][1], noise=DOT_FILTER[0][2])

    def set_filter(self, accel: float | None, noise: float | None) -> None:
        self.kalman = None if accel is None else KalmanDot(accel=accel, noise=noise)
        self.trail = []

    def set_map(self, net: GestureNet | None) -> None:
        for t in self.texts:
            self.removeItem(t)
        self.texts, self.trail = [], []
        if self.kalman is not None:
            self.kalman.reset()
        self.tail.setData([], [])
        self.dot.setData([], [])
        m = net.map if net else None
        if m is None:
            self.ref.setData([], [])
            self.empty.setText("이 모델에는 지도가 없습니다 (다시 학습하면 생깁니다)" if net else "")
            return
        self.empty.setText("")
        rgba = {int(c): QColor(colour(int(c))) for c in np.unique(m.label)}
        brushes = [pg.mkBrush(rgba[int(c)].red(), rgba[int(c)].green(), rgba[int(c)].blue(), 70) for c in m.label]
        self.ref.setData(m.xy[:, 0], m.xy[:, 1], brush=brushes)
        names = net.meta["names"]
        for c in np.unique(m.label):
            xy = np.median(m.xy[m.label == c], 0)
            t = pg.TextItem(names.get(str(int(c)), str(int(c))), color=TEXT, anchor=(0.5, 0.5))
            f = QFont(self.font())
            f.setBold(True)
            t.setFont(f)
            t.setPos(float(xy[0]), float(xy[1]))
            self.addItem(t)
            self.texts.append(t)
        self.autoRange()

    def push(self, xy: np.ndarray, label: int) -> None:
        """xy: raw kNN position of the newest window. The dot and its tail show the Kalman
        estimate; the faint small dot the raw position."""
        cur = xy if self.kalman is None else self.kalman.update(xy)
        self.trail = (self.trail + [cur])[-TRAIL:]
        tr = np.array(self.trail)
        self.tail.setData(tr[:, 0], tr[:, 1])
        if self.kalman is None:
            self.raw.setData([], [])
        else:
            self.raw.setData([xy[0]], [xy[1]])
        self.dot.setData([cur[0]], [cur[1]], brush=pg.mkBrush(colour(label)))


class GestureWindow(StudioWindow):
    KEY = "gesture"
    TITLE = "실시간 손동작 판정"

    def __init__(self, hub) -> None:
        super().__init__(hub)
        self.net: GestureNet | None = None
        self.dec: LiveDecoder | None = None
        self.pos = 0
        self.last = None              # latest Decision
        self.calib: list[tuple[str, int, float]] | None = None   # (text, label, end time)
        self.calib_t0 = 0.0
        self.calib_note = ""
        self.cost_ms = 0.0
        self.n_dec = 0
        self.t_rate = time.monotonic()
        self.rate = 0.0
        self.stab = Stabilizer(*SWITCHING[0][1:])
        self.shown: int | None = None

        # model row
        top = QHBoxLayout()
        top.addWidget(QLabel("모델"))
        self.models = QComboBox()
        self.models.setMinimumWidth(240)
        self.models.currentIndexChanged.connect(self._load)
        top.addWidget(self.models, 1)
        again = QPushButton("⟳")
        again.setFixedWidth(32)
        again.setToolTip("모델 목록 새로고침")
        again.clicked.connect(self._list)
        top.addWidget(again)
        self.model_info = QLabel()
        self.model_info.setStyleSheet(f"color: {MUTED};")
        self.model_info.setWordWrap(True)

        # calibration row
        cal = QHBoxLayout()
        self.cal_btn = QPushButton("착용 보정 시작 (약 40초)")
        self.cal_btn.setMinimumHeight(34)
        self.cal_btn.clicked.connect(self._calibrate)
        cal.addWidget(self.cal_btn)
        self.cal_bar = QProgressBar()
        self.cal_bar.setRange(0, 1000)
        self.cal_bar.setTextVisible(False)
        self.cal_bar.setFixedHeight(10)
        cal.addWidget(self.cal_bar, 1)
        self.cal_state = QLabel()
        cal.addWidget(self.cal_state)

        # steadiness row
        steady = QHBoxLayout()
        steady.addWidget(QLabel("평활"))
        self.smooth = QComboBox()
        for text, n in SMOOTHING:
            self.smooth.addItem(text, n)
        self.smooth.currentIndexChanged.connect(self._steady)
        steady.addWidget(self.smooth)
        steady.addSpacing(16)
        steady.addWidget(QLabel("화면 전환"))
        self.switch = QComboBox()
        for text, pmin, n in SWITCHING:
            self.switch.addItem(text, (pmin, n))
        self.switch.currentIndexChanged.connect(self._steady)
        steady.addWidget(self.switch)
        steady.addSpacing(16)
        steady.addWidget(QLabel("지도 점 칼만 필터"))
        self.dot_filter = QComboBox()
        for text, acc, noise in DOT_FILTER:
            self.dot_filter.addItem(text, (acc, noise))
        self.dot_filter.setToolTip("지도 위 점의 떨림을 줄입니다. 강할수록 부드럽지만 늦게 따라갑니다. "
                                   "흐린 작은 점은 필터 전 위치입니다.")
        self.dot_filter.currentIndexChanged.connect(lambda: self.map.set_filter(*self.dot_filter.currentData()))
        steady.addWidget(self.dot_filter)
        steady.addStretch(1)
        hint = QLabel("판정이 떨리면 화면 전환 조건을 올리세요 (반응은 그만큼 늦어짐)")
        hint.setStyleSheet(f"color: {MUTED};")
        steady.addWidget(hint)

        # main view
        self.pic = HandPicture()
        self.pic.setMinimumSize(320, 300)
        self.big = QLabel("–")
        f = QFont(self.font())
        f.setPointSize(26)
        f.setBold(True)
        self.big.setFont(f)
        self.big.setAlignment(Qt.AlignCenter)
        self.sub = QLabel()
        self.sub.setAlignment(Qt.AlignCenter)
        self.sub.setStyleSheet(f"color: {MUTED};")
        self.truth = QLabel()
        self.truth.setAlignment(Qt.AlignCenter)
        f2 = QFont(self.font())
        f2.setPointSize(13)
        self.truth.setFont(f2)
        left = QVBoxLayout()
        left.addWidget(self.pic, 1)
        left.addWidget(self.big)
        left.addWidget(self.sub)
        left.addWidget(self.truth)
        self.bars = ProbBars()
        right = QVBoxLayout()
        self.bars_head = head = QLabel("동작별 확률 (최근 0.4초 평균)")
        head.setStyleSheet(f"color: {MUTED};")
        right.addWidget(head)
        right.addWidget(self.bars)
        right.addStretch(1)
        note = QLabel("밴드를 새로 찰 때마다 착용 보정을 다시 하세요. 보정 전에는 학습 때 기준을 써서 판정이 부정확합니다. "
                      "판정은 손을 바꾼 뒤 약 0.2초 늦게 따라옵니다.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {MUTED};")
        right.addWidget(note)
        self.map = SignalMapView()
        self.map.setMinimumSize(320, 300)
        centre = QVBoxLayout()
        mh = QLabel("신호 지도 (학습 창의 t-SNE, 흰 테두리 점 = 지금, 흐린 점 = 필터 전)")
        mh.setStyleSheet(f"color: {MUTED};")
        centre.addWidget(mh)
        centre.addWidget(self.map, 1)
        mid = QHBoxLayout()
        mid.addLayout(left, 3)
        mid.addLayout(centre, 4)
        mid.addLayout(right, 3)

        root = QWidget()
        v = QVBoxLayout(root)
        v.addLayout(top)
        v.addWidget(self.model_info)
        v.addLayout(cal)
        v.addLayout(steady)
        v.addLayout(mid, 1)
        self.setCentralWidget(root)
        self.resize(1240, 640)
        self._list()

    # ------------------------------------------------------------ model
    def _list(self) -> None:
        cur = self.models.currentData()
        self.models.blockSignals(True)
        self.models.clear()
        for p in sorted(glob.glob(os.path.join(MODEL_DIR, "*.npz"))):
            self.models.addItem(os.path.splitext(os.path.basename(p))[0], p)
        if not self.models.count():
            self.models.addItem("(학습된 모델 없음)", None)
        i = self.models.findData(cur)
        self.models.setCurrentIndex(max(0, i))
        self.models.blockSignals(False)
        self._load()

    def _load(self) -> None:
        path = self.models.currentData()
        self.net = self.dec = self.last = None
        self.calib = None
        if not path:
            self.model_info.setText("모델을 먼저 학습하세요:  py -3.13 -m studio.gesture.train --who <이름|all>")
            self._show_state()
            return
        try:
            self.net = GestureNet.load(path)
        except Exception as e:                                       # noqa: BLE001
            self.model_info.setText(f"모델을 읽지 못했습니다: {e}")
            self._show_state()
            return
        m = self.net.meta
        ev = m.get("eval_leave_one_wear_out", {}).get("mean")
        self.model_info.setText(
            f"{', '.join(m.get('people', []))} · 정상 시도 {m.get('trials', '?')}개 · 착용 {len(m.get('wears', []))}개로 학습"
            + (f" · 새로 찼을 때 예상 정확도 {100 * ev:.0f}% (착용 하나 빼고 평가)" if ev else "")
            + f" · {m.get('created_local', '')}")
        self.dec = LiveDecoder(self.net, smooth=self.smooth.currentData())
        self.stab.reset()
        self.shown = None
        self.map.set_map(self.net)
        self.pos = self.hub.streams.t.total
        self._show_state()

    def _steady(self) -> None:
        pmin, n = self.switch.currentData()
        self.stab = Stabilizer(pmin, n)
        self.shown = None
        if self.dec is not None:
            self.dec.smooth = self.smooth.currentData()
        self.bars_head.setText(f"동작별 확률 (최근 {(self.smooth.currentData() - 1) / 10:.1f}초 평균)")

    def _names(self) -> list[tuple[int, str]]:
        n = self.net.meta["names"]
        return [(c, n.get(str(c), n.get(c, str(c)))) for c in self.net.classes]

    def _key(self, label: int) -> str:
        lab = self.net.meta["labels"]
        return lab.get(str(label), lab.get(label, "rest"))

    # ------------------------------------------------------------ calibration
    def _calibrate(self) -> None:
        if self.dec is None:
            return
        if self.calib is not None:                                   # cancel
            self.calib = None
            self.dec.collecting = False
            self.calib_note = "보정을 취소했습니다"
            self._show_state()
            return
        names = dict(self._names())
        t = time.monotonic()
        seq, end = [], t + CALIB_REST_S
        seq.append(("힘을 빼고 쉬세요", 0, end))
        for c in self.net.classes:
            if c == 0:
                continue
            end += CALIB_HOLD_S
            seq.append((f"{names[c]}", c, end))
            end += CALIB_GAP_S
            seq.append(("쉬세요", 0, end))
        self.calib, self.calib_t0 = seq, t
        self.dec.start_collecting()
        self.cal_btn.setText("보정 취소")

    def _calib_step(self) -> None:
        now = time.monotonic()
        seq = self.calib
        total = seq[-1][2] - self.calib_t0
        self.cal_bar.setValue(int(1000 * min(1.0, (now - self.calib_t0) / total)))
        cur = next((s for s in seq if now < s[2]), None)
        if cur is None:
            place = self.dec.finish_collecting()
            self.calib = None
            self.calib_note = (f"보정됨 {time.strftime('%H:%M')} · 창 {place.windows}개" if place else
                               "보정 실패: 신호가 부족합니다 (연결과 밴드를 확인하세요)")
            self._show_state()
            return
        text, label, end = cur
        self.pic.set_gesture(self._key(label), colour(label))
        self.big.setText(text)
        self.big.setStyleSheet(f"color: {TEXT};")
        self.sub.setText(f"보정 중 · {end - now:.1f}초 · 정답 표시 없이 신호 세기와 기준점만 잡습니다")

    def _show_state(self) -> None:
        self.cal_btn.setText("착용 보정 시작 (약 40초)")
        self.cal_btn.setEnabled(self.dec is not None)
        self.cal_bar.setValue(0)
        if self.dec is None:
            self.cal_state.setText("")
            self.big.setText("–")
            self.sub.setText("")
            self.bars.set([], np.zeros(0))
            return
        p = self.dec.placement
        self.cal_state.setText(self.calib_note or ("보정 전: 학습 때 기준 사용 (부정확)" if not p.windows else ""))
        self.cal_state.setStyleSheet(f"color: {'#ffd43b' if not p.windows else MUTED};")

    def _show_truth(self) -> None:
        """When playing a recording: what the screen asked for at that moment."""
        rp = self.hub.link.replay
        pr = rp.prompt() if rp is not None and self.calib is None else None
        if pr is None:
            self.truth.setText("")
            return
        kind, name, since = pr
        what = {"ready": "준비", "hold": "유지", "rest": "휴식"}.get(kind, kind)
        shown = self.big.text().rstrip("?")
        target = name if kind in ("hold", "ready") else "휴식"
        ok = kind == "hold" and since > 0.5 and shown == target or kind == "rest" and since > 0.5 and shown == "휴식"
        bad = kind in ("hold", "rest") and since > 0.8 and shown != target
        col = "#69db7c" if ok else "#ff8787" if bad else MUTED
        self.truth.setText(f"녹화 당시 안내: {name} ({what} {since:.1f}초)")
        self.truth.setStyleSheet(f"color: {col};")

    # ------------------------------------------------------------ live
    def on_mode_change(self) -> None:
        self.pos = 0
        if self.dec:
            self.dec.reset()

    def refresh(self) -> None:
        s = self.hub.streams
        tt, end = s.t.since(self.pos)
        cen, _ = s.cen.since(self.pos)
        self.pos = end
        k = min(len(tt), len(cen))
        ok_rate = abs(self.hub.fs - FS) < 1
        if self.dec is not None and k and ok_rate:
            t0 = time.perf_counter()
            out = self.dec.feed(tt[-k:, 0], cen[-k:])
            if out:
                self.cost_ms = 1000 * (time.perf_counter() - t0) / len(out)
                self.n_dec += len(out)
                self.last = out[-1]
                for d in out:
                    self.shown = self.stab.update(d)
                if self.net.map is not None and self.calib is None:
                    for xy in self.net.map.place(np.stack([d.emb for d in out])):
                        self.map.push(xy, out[-1].label)
        now = time.monotonic()
        if now - self.t_rate > 1.0:
            self.rate, self.n_dec, self.t_rate = self.n_dec / (now - self.t_rate), 0, now
        if self.calib is not None:
            self._calib_step()
        elif self.dec is not None and self.last is not None:
            d = self.last
            names = dict(self._names())
            lab = d.label if self.shown is None else self.shown
            pr = float(d.proba[self.net.classes.index(lab)])
            sure = pr >= UNSURE
            self.pic.set_gesture(self._key(lab), colour(lab), dim=not sure)
            self.big.setText(names[lab] if sure else f"{names[lab]}?")
            self.big.setStyleSheet(f"color: {colour(lab) if sure else MUTED};")
            self.sub.setText(f"확률 {100 * pr:.0f}%" + ("  ·  불확실" if not sure else "")
                             + (f"  ·  지금 창은 {names[d.label]} 쪽" if d.label != lab else ""))
            self.bars.set(self._names(), d.proba)
        self._show_truth()
        stale = self.last is not None and (s.t.total == 0 or time.perf_counter() - self.last.t > 1.0)
        parts = [self.mode_text()]
        if not ok_rate and self.hub.mode:
            parts.append("판정은 HEX 500 Hz 모드에서만 됩니다 (밴드 버튼 1번)")
        if self.dec is not None:
            parts.append(f"판정 {self.rate:.0f}회/초 · 계산 {self.cost_ms:.1f} ms")
            if self.dec.gaps:
                parts.append(f"신호 끊김 {self.dec.gaps}회")
        if stale:
            parts.append("새 신호 없음")
        self.status.setText("   ·   ".join(parts))
