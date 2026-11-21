"""Gesture recording: participant name and notes, then a guided random session.

Pages: setup -> run (prompt on screen) -> between sets -> saving -> done.
The runner is stepped by a 10 ms timer so phase changes reach the screen promptly;
everything else refreshes with the hub.
"""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QDoubleSpinBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from ..record.protocol import NAMES, PRESETS, Plan, Runner
from .common import StudioWindow
from .hand_pic import HandPicture
from ..device.protocol import CH
from .theme import CH_COLORS, MUTED, TEXT

EMG_SPAN_S = 4.0   # seconds of raw EMG shown in the recorder's strip

PHASE = {
    "ready": ("준비", "#ffd43b"),
    "hold": ("유지", "#69db7c"),
    "rest": ("휴식", MUTED),
}


def big(pt: int, bold: bool = True) -> QFont:
    f = QFont()
    f.setPointSize(pt)
    f.setBold(bold)
    return f


class Preview(QLabel):
    """Latest camera frame, scaled down, so the participant can keep the hand in view."""

    def __init__(self) -> None:
        super().__init__("카메라 꺼짐")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(320, 180)
        self.setStyleSheet(f"background: #000; color: {MUTED};")
        self._seq = -1

    def update_from(self, camera) -> None:
        fr, seq = camera.latest()
        if fr is None or seq == self._seq or not fr.jpeg:
            if fr is None:
                self.setText("카메라 꺼짐")
            return
        self._seq = seq
        img = QImage.fromData(fr.jpeg).mirrored(True, False)
        self.setPixmap(QPixmap.fromImage(img).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class EmgStrip(pg.GraphicsLayoutWidget):
    """Compact 8-channel raw EMG strip so the participant can see the signal is coming in."""

    def __init__(self, height: int = 220) -> None:
        super().__init__()
        self.setFixedHeight(height)
        self.plots, self.curves = [], []
        for c in range(CH):
            p = self.addPlot(row=c, col=0)
            p.setLabel("left", f"{c + 1}")
            p.getAxis("left").setWidth(22)
            p.getAxis("left").setStyle(showValues=False)
            p.getAxis("bottom").setStyle(showValues=False)
            p.hideButtons()
            p.setMouseEnabled(x=False, y=False)
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            p.enableAutoRange(axis="y")
            if self.plots:
                p.setXLink(self.plots[0])
            self.curves.append(p.plot(pen=pg.mkPen(CH_COLORS[c], width=1)))
            self.plots.append(p)
        self.ci.layout.setSpacing(0)
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.note = QLabel("EMG 신호 없음", self)
        self.note.setStyleSheet(f"color: {MUTED}; background: transparent;")
        self.note.move(32, 4)

    def update_from(self, hub) -> None:
        fs = hub.fs or 1.0
        data = hub.raw_last(int(EMG_SPAN_S * fs))
        n = len(data)
        self.note.setVisible(n == 0)
        if not n:
            for cv in self.curves:
                cv.setData([], [])
            return
        x = (np.arange(n) - n) / fs
        for c in range(CH):
            self.curves[c].setData(x, data[:, c])
        self.plots[0].setXRange(-EMG_SPAN_S, 0, padding=0)


class RecordWindow(StudioWindow):
    KEY = "record"
    TITLE = "손동작 녹화"
    REFRESH_HZ = 15.0

    def __init__(self, hub) -> None:
        super().__init__(hub)
        self.rec = hub.recorder
        self.plan = Plan()
        self.names = dict(NAMES)     # on-screen names, preset may override (e.g. fist shown as "0")
        self.hints: dict[int, str] = {}
        self.keys: dict[int, str] = {}
        self.runner: Runner | None = None
        self.pages = QStackedWidget()
        self.pages.addWidget(self._setup_page())
        self.pages.addWidget(self._run_page())
        self.pages.addWidget(self._between_page())
        self.pages.addWidget(self._done_page())
        self.setCentralWidget(self.pages)
        self.resize(960, 820)
        self._clock = QTimer(self)
        self._clock.setTimerType(Qt.PreciseTimer)
        self._clock.timeout.connect(self._step)
        self._saving = False

    # ------------------------------------------------------------ pages
    def _setup_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        title = QLabel("손동작 녹화")
        title.setFont(big(18))
        v.addWidget(title)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet(f"color: {MUTED};")
        v.addWidget(self.info)

        form = QFormLayout()
        proto = QHBoxLayout()
        self.preset = QComboBox()
        for key, (label, _set, ids, _names) in PRESETS.items():
            self.preset.addItem(f"{label} ({len(ids)}개)", key)
        self.preset.currentIndexChanged.connect(self._plan_changed)
        proto.addWidget(self.preset, 1)
        self.reps_box = QSpinBox()
        self.reps_box.setRange(1, 20)
        self.reps_box.setPrefix("동작마다 ")
        self.reps_box.setSuffix("회")
        self.reps_box.valueChanged.connect(self._plan_changed)
        proto.addWidget(self.reps_box)
        self.sets_box = QSpinBox()
        self.sets_box.setRange(1, 10)
        self.sets_box.setSuffix("세트")
        self.sets_box.valueChanged.connect(self._plan_changed)
        proto.addWidget(self.sets_box)
        form.addRow("동작 묶음", proto)
        self.gallery = QHBoxLayout()
        self.gallery.setSpacing(6)
        gal = QWidget()
        gal.setLayout(self.gallery)
        form.addRow("손 모양", gal)
        timing = QHBoxLayout()
        self.time_box = {}
        for key, label in (("ready_s", "준비"), ("hold_s", "유지"), ("rest_s", "휴식")):
            b = QDoubleSpinBox()
            b.setRange(0.5, 30.0)
            b.setSingleStep(0.5)
            b.setDecimals(1)
            b.setPrefix(f"{label} ")
            b.setSuffix("초")
            b.valueChanged.connect(self._plan_changed)
            timing.addWidget(b)
            self.time_box[key] = b
        reset = QPushButton("기본값")
        reset.setToolTip("준비 1초 · 유지 3초 · 휴식 2초, 전체 9동작 5회 2세트")
        reset.clicked.connect(lambda: self._set_plan(Plan()))
        timing.addWidget(reset)
        timing.addStretch(1)
        form.addRow("동작 시간", timing)
        self.name = QLineEdit()
        self.name.setPlaceholderText("예: 홍길동")
        self.name.textChanged.connect(self._check)
        form.addRow("이름", self.name)
        self.note = QPlainTextEdit()
        self.note.setPlaceholderText("착용 상태, 피부, 배터리, 특이사항 등")
        self.note.setFixedHeight(70)
        form.addRow("실험 노트", self.note)
        v.addLayout(form)

        row = QHBoxLayout()
        checks = QVBoxLayout()
        self.emg_state = QLabel()
        self.cam_state = QLabel()
        self.no_video = QCheckBox("카메라 없이 녹화 (테스트용)")
        self.no_video.toggled.connect(self._check)
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("카메라"))
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 9)
        cam_row.addWidget(self.cam_index)
        self.cam_btn = QPushButton("카메라 시작 (영상만)")
        self.cam_btn.setToolTip("여기서 바로 카메라를 켭니다. 손 인식 모델은 싣지 않습니다 "
                                "(필요하면 '카메라 손 인식' 창의 [손 인식 모델 로드]).")
        self.cam_btn.clicked.connect(self._toggle_camera)
        cam_row.addWidget(self.cam_btn)
        cam_row.addStretch(1)
        for x in (self.emg_state, self.cam_state):
            checks.addWidget(x)
        checks.addLayout(cam_row)
        checks.addWidget(self.no_video)
        checks.addStretch(1)
        row.addLayout(checks, 1)
        self.preview0 = Preview()
        row.addWidget(self.preview0)
        v.addLayout(row)

        self.emg0 = EmgStrip(170)
        v.addWidget(self.emg0)
        v.addStretch(1)
        self.where = QLabel(f"저장 위치: {self.rec.root}")
        self.where.setStyleSheet(f"color: {MUTED};")
        v.addWidget(self.where)
        self.start_btn = QPushButton("세션 시작")
        self.start_btn.setMinimumHeight(44)
        self.start_btn.clicked.connect(self._start)
        v.addWidget(self.start_btn)
        self._set_plan(self.plan)
        return w

    def _set_plan(self, p: Plan) -> None:
        """Put a plan into the setup controls (does not change self.plan until started)."""
        key = next((k for k, (_, gs, ids, _n) in PRESETS.items()
                    if gs == p.gesture_set and tuple(ids) == tuple(p.gestures)), "all")
        self.preset.setCurrentIndex(max(0, self.preset.findData(key)))
        self.reps_box.setValue(p.reps)
        self.sets_box.setValue(p.sets)
        for key, b in self.time_box.items():
            b.setValue(getattr(p, key))
        self._plan_changed()

    def _plan_from_controls(self, seed: int = 0) -> Plan:
        _label, gset, ids, names = PRESETS[self.preset.currentData()]
        p = Plan(sets=self.sets_box.value(), reps=self.reps_box.value(), seed=seed, gestures=tuple(ids),
                 gesture_set=gset, **{k: b.value() for k, b in self.time_box.items()})
        self.names = {**p.gset.names, **names}
        self.hints = p.gset.hints
        self.keys = p.gset.labels
        return p

    def _plan_changed(self) -> None:
        p = self._plan_from_controls()
        n = p.sets * p.trials_per_set
        secs = n * p.trial_s
        dur = f"약 {secs / 60:.0f}분" if secs >= 90 else f"약 {secs:.0f}초"
        gest = ", ".join(self.names[i] for i in p.gestures)
        self._fill_gallery(p.gestures)
        self.info.setText(f"동작 {len(p.gestures)}개({gest})가 랜덤 순서로 나옵니다. "
                          f"한 번에 준비 {p.ready_s:g}초 → 유지 {p.hold_s:g}초 → 휴식 {p.rest_s:g}초, "
                          f"동작마다 {p.reps}회씩 {p.sets}세트 (총 {n}회, {dur}"
                          + (" + 세트 사이 휴식" if p.sets > 1 else "") + ").\n"
                          "EMG 원시 데이터와 카메라 영상 전체 프레임이 원본 그대로 저장됩니다. 손 관절은 카메라 창에서 "
                          "손 인식을 켰으면 실시간으로 함께 저장되고, 껐으면 녹화 뒤 영상에서 뽑습니다 "
                          "(worker/hand_offline.py, GPU 있는 PC 권장).")

    def _fill_gallery(self, ids) -> None:
        """One small hand picture per gesture of the selected preset, for briefing the participant."""
        if getattr(self, "_gallery_ids", None) == (tuple(ids), dict(self.names)):
            return
        self._gallery_ids = (tuple(ids), dict(self.names))
        while self.gallery.count():
            it = self.gallery.takeAt(0)
            old = it.widget()
            if old is not None:
                old.setParent(None)      # off screen now, not at the next event loop pass
                old.deleteLater()
        for i in ids:
            cell = QWidget()
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(0)
            pic = HandPicture(caption=False)
            pic.setFixedSize(84, 84)
            pic.set_gesture(self.keys[i], TEXT)
            pic.setToolTip(self.hints[i])
            cv.addWidget(pic)
            lbl = QLabel(self.names[i])
            lbl.setAlignment(Qt.AlignCenter)
            cv.addWidget(lbl)
            self.gallery.addWidget(cell)
        self.gallery.addStretch(1)

    def _run_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.counter = QLabel()
        self.counter.setFont(big(12, False))
        top.addWidget(self.counter, 1)
        abort = QPushButton("중단")
        abort.clicked.connect(self._abort)
        top.addWidget(abort)
        v.addLayout(top)

        self.phase_lbl = QLabel()
        self.phase_lbl.setAlignment(Qt.AlignCenter)
        self.phase_lbl.setFont(big(20))
        self.gesture_lbl = QLabel()
        self.gesture_lbl.setAlignment(Qt.AlignCenter)
        self.gesture_lbl.setFont(big(48))
        self.hand_pic = HandPicture()
        self.hand_pic.setMinimumHeight(260)
        self.hint_lbl = QLabel()
        self.hint_lbl.setAlignment(Qt.AlignCenter)
        self.hint_lbl.setFont(big(14, False))
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(14)
        self.next_lbl = QLabel()
        self.next_lbl.setAlignment(Qt.AlignCenter)
        self.next_lbl.setStyleSheet(f"color: {MUTED};")
        v.addStretch(1)
        for x in (self.phase_lbl, self.gesture_lbl, self.hand_pic, self.hint_lbl):
            v.addWidget(x, 3 if x is self.hand_pic else 0)
        v.addSpacing(16)
        v.addWidget(self.bar)
        v.addWidget(self.next_lbl)
        v.addStretch(1)
        bottom = QHBoxLayout()
        self.emg1 = EmgStrip(180)
        bottom.addWidget(self.emg1, 1)
        self.preview1 = Preview()
        bottom.addWidget(self.preview1)
        v.addLayout(bottom)
        return w

    def _toggle_camera(self) -> None:
        cam = self.hub.camera
        if cam.running:
            cam.stop()
        else:
            cam.start(self.cam_index.value(), model=False)

    def _between_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addStretch(1)
        self.between_lbl = QLabel()
        self.between_lbl.setAlignment(Qt.AlignCenter)
        self.between_lbl.setFont(big(20))
        v.addWidget(self.between_lbl)
        sub = QLabel("팔을 쉬게 하세요. 밴드는 벗지 마세요. 녹화는 계속됩니다.")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {MUTED};")
        v.addWidget(sub)
        v.addSpacing(20)
        self.next_set_btn = QPushButton()
        self.next_set_btn.setMinimumHeight(44)
        self.next_set_btn.clicked.connect(self._next_set)
        v.addWidget(self.next_set_btn)
        stop = QPushButton("여기서 끝내기")
        stop.clicked.connect(self._abort)
        v.addWidget(stop)
        v.addStretch(1)
        return w

    def _done_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.done_title = QLabel()
        self.done_title.setFont(big(18))
        v.addWidget(self.done_title)
        self.done_text = QLabel()
        self.done_text.setWordWrap(True)
        self.done_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.done_text)
        v.addStretch(1)
        row = QHBoxLayout()
        self.open_btn = QPushButton("폴더 열기")
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.rec.folder)))
        row.addWidget(self.open_btn)
        again = QPushButton("새 세션")
        again.clicked.connect(self._again)
        row.addWidget(again)
        v.addLayout(row)
        return w

    # ------------------------------------------------------------ flow
    def _video_ok(self) -> bool:
        fr, _ = self.hub.camera.latest()
        return self.hub.camera.running and fr is not None

    def _check(self) -> None:
        emg_ok = self.hub.mode == "hex"
        cam_ok = self._video_ok()
        self.emg_state.setText("✓ EMG 연결됨 (HEX 500 Hz)" if emg_ok else
                               "✗ EMG: 런처에서 암밴드를 연결하세요 (HEX 모드 필요)")
        self.emg_state.setStyleSheet(f"color: {'#69db7c' if emg_ok else '#ff8787'};")
        if cam_ok:
            self.cam_state.setText("✓ 카메라 실행 중 · " + ("손 인식 켜짐" if self.hub.camera.has_model
                                                            else "영상만 (관절은 녹화 뒤 추출)"))
        else:
            self.cam_state.setText("✗ 카메라 꺼짐: 아래 [카메라 시작]을 누르세요 (손 인식 없이 영상만 녹화해도 됩니다)")
        self.cam_state.setStyleSheet(f"color: {'#69db7c' if cam_ok else '#ff8787'};")
        self.start_btn.setEnabled(bool(self.name.text().strip()) and emg_ok and (cam_ok or self.no_video.isChecked()))

    def _start(self) -> None:
        self.plan = self._plan_from_controls(seed=int(time.time()))
        self.runner = Runner(self.plan)
        self.rec.start(self.name.text(), self.note.toPlainText(), self.plan, want_video=self._video_ok(),
                       shown_names=self.names)
        self.pages.setCurrentIndex(1)
        self.runner.start_set(time.perf_counter())
        self._clock.start(10)
        self._show()

    def _next_set(self) -> None:
        self.pages.setCurrentIndex(1)
        self.runner.start_set(time.perf_counter())
        self._clock.start(10)
        self._show()

    def _step(self) -> None:
        r = self.runner
        if r is None:
            return
        now = time.perf_counter()
        if r.update(now) or r.phase in ("ready", "hold", "rest"):
            self._show(now)

    def _show(self, now: float | None = None) -> None:
        r = self.runner
        now = time.perf_counter() if now is None else now
        if r.phase == "between":
            self._clock.stop()
            self.between_lbl.setText(f"세트 {r.set_idx + 1} / {self.plan.sets} 완료")
            self.next_set_btn.setText(f"세트 {r.set_idx + 2} 시작")
            self.pages.setCurrentIndex(2)
            return
        if r.phase == "done":
            self._end("complete")
            return
        name, color = PHASE[r.phase]
        g = r.gesture
        self.counter.setText(f"세트 {r.set_idx + 1}/{self.plan.sets}  ·  {r.trial + 1}/{self.plan.trials_per_set}")
        self.phase_lbl.setText(name)
        self.phase_lbl.setStyleSheet(f"color: {color};")
        if r.phase == "rest":
            nxt = r.next_gesture
            if nxt is not None:
                # the next gesture is shown big during rest so the participant can get ready
                self.gesture_lbl.setText(f"다음  {self.names[nxt]}")
                self.gesture_lbl.setStyleSheet(f"color: {TEXT};")
                self.hand_pic.set_gesture(self.keys[nxt], TEXT, dim=True)
                self.hint_lbl.setText(f"손에 힘을 빼세요  ·  다음 동작: {self.hints[nxt]}")
                self.next_lbl.setText("")
            else:
                self.gesture_lbl.setText("휴식")
                self.gesture_lbl.setStyleSheet(f"color: {MUTED};")
                self.hand_pic.set_gesture("rest", MUTED, dim=True)
                self.hint_lbl.setText("손에 힘을 빼세요")
                self.next_lbl.setText("세트 마지막")
        else:
            self.gesture_lbl.setText(self.names[g])
            self.gesture_lbl.setStyleSheet(f"color: {color if r.phase == 'hold' else TEXT};")
            self.hand_pic.set_gesture(self.keys[g], color)
            self.hint_lbl.setText(self.hints[g])
            self.next_lbl.setText("")
        self.bar.setValue(int(r.phase_progress(now) * 1000))
        self.bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")

    def _abort(self) -> None:
        if self.runner is None or self._saving:
            return
        self.runner.abort(time.perf_counter())
        self._end("aborted")

    def _end(self, status: str) -> None:
        self._clock.stop()
        self._saving = True
        self._status = status
        self.rec.stop()
        self.done_title.setText("저장 중…")
        self.done_text.setText("카메라 워커가 영상 파일을 마무리하고 있습니다.")
        self.open_btn.setEnabled(False)
        self.pages.setCurrentIndex(3)

    def _finish(self) -> None:
        s = self.rec.finish(self.runner, self._status)
        self._saving = False
        self.open_btn.setEnabled(True)
        self.done_title.setText("녹화 완료" if s["status"] == "complete" else "녹화 중단 (지금까지 저장됨)")
        e = s["emg"]
        lines = [f"폴더: {s['folder']}",
                 f"완료한 동작: {s['trials_done']} / {self.plan.sets * self.plan.trials_per_set}",
                 f"EMG: 샘플 {e['samples']:,}개, 장치 시계 기준 {e.get('fs_device_clock', '-')} Hz, "
                 f"누락 추정 프레임 {e.get('frames_missing_est', '-')}, "
                 f"배터리 {e.get('battery_start', '-')} → {e.get('battery_end', '-')}%",
                 (f"손 관절: {s['hand']['frames']}프레임 (오른손 {s['hand']['with_right_hand']})"
                  if s["hand"].get("source") == "live" else
                  "손 관절: 녹화 안 함 → 영상에서 추출: worker/hand_offline.py <폴더>")]
        v = s.get("video")
        if v:
            lines.append(f"영상: {v.get('frames', 0)}프레임, 놓친 프레임 {v.get('dropped', 0)}"
                         + (f", 오류 {v['error']}" if v.get("error") else ""))
        else:
            lines.append("영상: 없음")
        self.done_text.setText("\n".join(lines))

    def _again(self) -> None:
        self.runner = None
        self.note.clear()
        self.pages.setCurrentIndex(0)

    # ------------------------------------------------------------ hub ticks
    def refresh(self) -> None:
        page = self.pages.currentIndex()
        if page == 0:
            self._check()
            self.preview0.update_from(self.hub.camera)
            self.emg0.update_from(self.hub)
            running = self.hub.camera.running
            self.cam_btn.setText("카메라 정지" if running else "카메라 시작 (영상만)")
            self.cam_index.setEnabled(not running)
        elif page == 1:
            self.preview1.update_from(self.hub.camera)
            self.emg1.update_from(self.hub)
        elif page == 3 and self._saving and not self.rec.video_pending():
            self._finish()
        self.status.setText(self.mode_text())

    def closeEvent(self, ev) -> None:
        if self.runner is not None and self.rec.active:
            self.runner.abort(time.perf_counter())
            self._end("aborted")
            deadline = time.monotonic() + 5
            while self.rec.video_pending() and time.monotonic() < deadline:
                time.sleep(0.05)
            self._finish()
        super().closeEvent(ev)
