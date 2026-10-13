"""Gesture recording: participant name and notes, then a guided random session.

Pages: setup -> run (prompt on screen) -> between sets -> saving -> done.
The runner is stepped by a 10 ms timer so phase changes reach the screen promptly;
everything else refreshes with the hub.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from ..record.protocol import GESTURES, HINTS, NAMES, Plan, Runner
from .common import StudioWindow
from .theme import MUTED, TEXT

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


class RecordWindow(StudioWindow):
    KEY = "record"
    TITLE = "손동작 녹화"
    REFRESH_HZ = 15.0

    def __init__(self, hub) -> None:
        super().__init__(hub)
        self.rec = hub.recorder
        self.plan = Plan()
        self.runner: Runner | None = None
        self.pages = QStackedWidget()
        self.pages.addWidget(self._setup_page())
        self.pages.addWidget(self._run_page())
        self.pages.addWidget(self._between_page())
        self.pages.addWidget(self._done_page())
        self.setCentralWidget(self.pages)
        self.resize(900, 640)
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
        p = self.plan
        n = p.sets * p.trials_per_set
        minutes = n * p.trial_s / 60
        gest = ", ".join(g[2] for g in GESTURES)
        info = QLabel(f"동작 {len(GESTURES)}개({gest})가 랜덤 순서로 나옵니다. "
                      f"한 번에 준비 {p.ready_s:g}초 → 유지 {p.hold_s:g}초 → 휴식 {p.rest_s:g}초, "
                      f"동작마다 {p.reps}회씩 {p.sets}세트 (총 {n}회, 약 {minutes:.0f}분 + 세트 사이 휴식).\n"
                      "EMG 원시 데이터, 카메라 영상 전체 프레임, 손 관절이 원본 그대로 저장됩니다.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {MUTED};")
        v.addWidget(info)

        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("예: 홍길동")
        self.name.textChanged.connect(self._check)
        form.addRow("이름", self.name)
        self.note = QPlainTextEdit()
        self.note.setPlaceholderText("착용 상태, 피부, 배터리, 특이사항 등")
        self.note.setFixedHeight(110)
        form.addRow("실험 노트", self.note)
        v.addLayout(form)

        row = QHBoxLayout()
        checks = QVBoxLayout()
        self.emg_state = QLabel()
        self.cam_state = QLabel()
        self.no_video = QCheckBox("카메라 없이 녹화 (테스트용)")
        self.no_video.toggled.connect(self._check)
        for x in (self.emg_state, self.cam_state, self.no_video):
            checks.addWidget(x)
        checks.addStretch(1)
        row.addLayout(checks, 1)
        self.preview0 = Preview()
        row.addWidget(self.preview0)
        v.addLayout(row)

        v.addStretch(1)
        self.where = QLabel(f"저장 위치: {self.rec.root}")
        self.where.setStyleSheet(f"color: {MUTED};")
        v.addWidget(self.where)
        self.start_btn = QPushButton("세션 시작")
        self.start_btn.setMinimumHeight(44)
        self.start_btn.clicked.connect(self._start)
        v.addWidget(self.start_btn)
        return w

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
        self.gesture_lbl.setFont(big(72))
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
        for x in (self.phase_lbl, self.gesture_lbl, self.hint_lbl):
            v.addWidget(x)
        v.addSpacing(16)
        v.addWidget(self.bar)
        v.addWidget(self.next_lbl)
        v.addStretch(1)
        self.preview1 = Preview()
        v.addWidget(self.preview1, 0, Qt.AlignRight)
        return w

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
        self.cam_state.setText("✓ 카메라 실행 중" if cam_ok else
                               "✗ 카메라: '카메라 손 인식' 창에서 [시작]을 누르세요")
        self.cam_state.setStyleSheet(f"color: {'#69db7c' if cam_ok else '#ff8787'};")
        self.start_btn.setEnabled(bool(self.name.text().strip()) and emg_ok and (cam_ok or self.no_video.isChecked()))

    def _start(self) -> None:
        self.plan = Plan(seed=int(time.time()))
        self.runner = Runner(self.plan)
        self.rec.start(self.name.text(), self.note.toPlainText(), self.plan, want_video=self._video_ok())
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
            self.gesture_lbl.setText("휴식")
            self.gesture_lbl.setStyleSheet(f"color: {MUTED};")
            self.hint_lbl.setText("손에 힘을 빼세요")
            nxt = r.next_gesture
            self.next_lbl.setText(f"다음: {NAMES[nxt]}" if nxt is not None else "세트 마지막")
        else:
            self.gesture_lbl.setText(NAMES[g])
            self.gesture_lbl.setStyleSheet(f"color: {color if r.phase == 'hold' else TEXT};")
            self.hint_lbl.setText(HINTS[g])
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
                 f"손 관절: {s['hand']['frames']}프레임 (오른손 {s['hand']['with_right_hand']})"]
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
        elif page == 1:
            self.preview1.update_from(self.hub.camera)
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
