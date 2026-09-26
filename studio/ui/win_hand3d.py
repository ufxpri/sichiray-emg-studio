"""Window 2: EMG-reconstructed hand (left) next to the camera-detected hand (right),
with on-the-spot training. Both views share the camera's wrist orientation, so only
the finger articulation differs. Talks to the pose learner only through its API.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..hand.mano_np import get_mano
from ..hand.skeleton import FINGERS, flexion
from .common import StudioWindow
from .gl_hand import SKIN_CAM, SKIN_EMG, HandView
from .theme import FINGER_COLORS, MUTED

HOLD_S = 0.5   # keep showing the last camera hand this long after it disappears


class Hand3DWindow(StudioWindow):
    KEY = "hand3d"
    TITLE = "3D 손 · EMG 복원"
    REFRESH_HZ = 30.0

    def __init__(self, hub) -> None:
        super().__init__(hub)
        self.cam = hub.camera
        self.pose = hub.pose
        self.v_emg = HandView("EMG 복원", SKIN_EMG)
        self.v_cam = HandView("카메라 (정답)", SKIN_CAM)
        self._cam_states = (self.v_emg.cam_state(), self.v_cam.cam_state())

        # ---- side panel
        side = QWidget()
        sv = QVBoxLayout(side)
        opts = QHBoxLayout()
        self.show_mesh = QCheckBox("메시")
        self.show_mesh.setChecked(True)
        self.show_bones = QCheckBox("뼈대")
        self.show_bones.setChecked(True)
        opts.addWidget(self.show_mesh)
        opts.addWidget(self.show_bones)
        reset = QPushButton("시점 초기화")
        reset.clicked.connect(lambda: (self.v_emg.reset(), self.v_cam.reset()))
        opts.addWidget(reset)
        sv.addLayout(opts)

        box = QGroupBox("현장 학습")
        bv = QVBoxLayout(box)
        self.b_collect = QPushButton("● 수집 시작")
        self.b_collect.setCheckable(True)
        self.b_collect.toggled.connect(self._collect)
        bv.addWidget(self.b_collect)
        self.count = QLabel()
        bv.addWidget(self.count)
        row = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItems(["릿지 (즉시)", "MLP (수 초)"])
        row.addWidget(self.kind)
        self.b_train = QPushButton("학습")
        self.b_train.clicked.connect(self._train)
        row.addWidget(self.b_train)
        bv.addLayout(row)
        self.predict = QCheckBox("EMG 예측 (왼쪽 손 움직이기)")
        self.predict.toggled.connect(self.pose.set_predicting)
        bv.addWidget(self.predict)
        clear = QPushButton("수집 데이터 지우기")
        clear.clicked.connect(self._clear)
        bv.addWidget(clear)
        self.report = QLabel()
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bv.addWidget(self.report)
        self.msg = QLabel()
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet("color: #ffe8a3;")
        bv.addWidget(self.msg)
        guide = QLabel("수집 방법: 오른팔에 밴드, 오른손을 카메라에 보이게 두고 1~2분 동안 손가락을 천천히 다양하게 "
                       "움직이세요. 주먹, 펴기, 손가락 하나씩, 집기, 힘 빼기까지 고루 넣습니다. 수집 중에는 밴드를 건드리지 "
                       "마세요. 착용할 때마다 신호가 달라지므로(§23) 밴드를 다시 차면 새로 수집하는 것이 좋습니다.")
        guide.setWordWrap(True)
        guide.setStyleSheet(f"color: {MUTED}; font-size: 8pt;")
        bv.addWidget(guide)
        sv.addWidget(box)

        fb = QGroupBox("손가락 굽힘 (°)  위 = EMG, 아래 = 카메라")
        g = QGridLayout(fb)
        self.bars_emg, self.bars_cam, self.nums = [], [], []
        for i, (name, _) in enumerate(FINGERS):
            lab = QLabel(name)
            lab.setStyleSheet(f"color: {FINGER_COLORS[i]};")
            g.addWidget(lab, 2 * i, 0, 2, 1)
            for j, store in enumerate((self.bars_emg, self.bars_cam)):
                bar = QProgressBar()
                bar.setRange(0, 270)
                bar.setTextVisible(False)
                bar.setFixedHeight(9)
                color = "#74c0fc" if j == 0 else FINGER_COLORS[i]
                bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
                g.addWidget(bar, 2 * i + j, 1)
                store.append(bar)
            num = QLabel("-")
            num.setMinimumWidth(70)
            num.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            g.addWidget(num, 2 * i, 2, 2, 1)
            self.nums.append(num)
        sv.addWidget(fb)
        help_ = QLabel("두 뷰는 시점이 연동됩니다. 왼쪽 드래그 = 회전, 휠 = 확대, 가운데 드래그 = 이동")
        help_.setWordWrap(True)
        help_.setStyleSheet(f"color: {MUTED}; font-size: 8pt;")
        sv.addWidget(help_)
        sv.addStretch(1)

        split = QSplitter()
        split.addWidget(self.v_emg)
        split.addWidget(self.v_cam)
        split.addWidget(side)
        split.setSizes([560, 560, 330])
        self.setCentralWidget(split)
        self.resize(1480, 780)
        self._seq = -1
        self._last_seen = 0.0
        self._refusal, self._refusal_t = "", 0.0

    # ------------------------------------------------------------ controls
    def _collect(self, on: bool) -> None:
        if on:
            why = self.pose.start_collecting()
            if why:
                self._refuse(why)
                self.b_collect.setChecked(False)
                return
        else:
            self.pose.stop_collecting()
        self.b_collect.setText("■ 수집 중지" if on else "● 수집 시작")

    def _refuse(self, why: str | None) -> None:
        """Show why a command was refused, for a few seconds."""
        self._refusal, self._refusal_t = why or "", time.monotonic()

    def _train(self) -> None:
        self._refuse(self.pose.train(["ridge", "mlp"][self.kind.currentIndex()]))

    def _clear(self) -> None:
        if QMessageBox.question(self, "데이터 지우기", f"수집한 {self.pose.count:,}개 샘플을 지울까요?") \
                == QMessageBox.Yes:
            self._refuse(self.pose.clear())

    def _sync_views(self) -> None:
        a, b = self.v_emg.cam_state(), self.v_cam.cam_state()
        pa, pb = self._cam_states
        if a != pa:
            self.v_cam.set_cam_state(a)
            b = a
        elif b != pb:
            self.v_emg.set_cam_state(b)
            a = b
        self._cam_states = (a, b)

    # ------------------------------------------------------------ update
    def refresh(self) -> None:
        self._sync_views()
        mano = get_mano()
        p = self.pose
        self.count.setText(f"샘플 {p.count:,}개 · 약 {p.seconds():.0f}초 분량"
                           + ("  (수집 중)" if p.collecting else ""))
        self.b_train.setEnabled(not p.training)
        self.predict.setEnabled(p.model is not None)
        if self.predict.isChecked() != p.predicting:   # the learner is the source of truth
            self.predict.blockSignals(True)
            self.predict.setChecked(p.predicting)
            self.predict.blockSignals(False)
        self.report.setText(p.report or "아직 학습한 모델이 없습니다.")
        self.msg.setText(self._refusal or p.message)
        if self._refusal and time.monotonic() - self._refusal_t > 6:
            self._refusal = ""
        mesh_on, bones_on = self.show_mesh.isChecked(), self.show_bones.isChecked()

        # camera hand (right)
        flex_cam = None
        frame, seq = self.cam.latest()
        hand = frame.right_hand() if frame and self.cam.running else None
        if hand is not None:
            self._last_seen = time.monotonic()
            if seq != self._seq:
                self._seq = seq
                self.v_cam.show_hand(hand.verts, hand.kp3d, self.cam.faces, mesh_on, bones_on)
            flex_cam = flexion(hand.kp3d)
            self.v_cam.title.setText("카메라 (정답) · 오른손")
        elif time.monotonic() - self._last_seen > HOLD_S:
            self.v_cam.clear()
            self.v_cam.title.setText("카메라 (정답) · 손 없음" if self.cam.running else "카메라 (정답) · 카메라 꺼짐")

        # EMG hand
        flex_emg = None
        pred = p.predicted_hand()
        if pred is not None and mano is not None:
            verts, joints = mano.forward(*pred)
            self.v_emg.show_hand(verts, joints, mano.faces, mesh_on, bones_on)
            flex_emg = flexion(joints)
            self.v_emg.title.setText("EMG 복원 · 예측 중")
        else:
            self.v_emg.clear()
            self.v_emg.title.setText("EMG 복원 · " + ("모델 없음 — 수집 후 학습하세요" if p.model is None
                                                     else "'EMG 예측'을 켜세요"))
        for i in range(5):
            self.bars_emg[i].setValue(int(min(flex_emg[i], 270)) if flex_emg is not None else 0)
            self.bars_cam[i].setValue(int(min(flex_cam[i], 270)) if flex_cam is not None else 0)
            e = f"{flex_emg[i]:.0f}" if flex_emg is not None else "-"
            c = f"{flex_cam[i]:.0f}" if flex_cam is not None else "-"
            self.nums[i].setText(f"{e} / {c}")
        self.status.setText("왼쪽 손목 방향은 카메라 값을 빌려 씁니다 (EMG로는 팔의 자세를 알 수 없음). "
                            "손가락 모양만 EMG에서 나옵니다.")
