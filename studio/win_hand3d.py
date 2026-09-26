"""Window 2: EMG-reconstructed hand (left) next to the camera-detected hand (right),
with on-the-spot training.

Coordinates: WiLoR gives camera space (x right, y down, z away, metres). Shown in
centimetres with the wrist at the origin, remapped to GL as X = x, Y = z, Z = -y,
so the default view looks from where the camera stands. Both views share the
camera's wrist orientation, so only the finger articulation differs.
"""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import GL_BLEND, GL_DEPTH_TEST, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget)

from .camera import BONES, FINGER_COLORS, FINGERS
from .common import StudioWindow
from .mano_np import Mano
from .pose import flexion_from_joints
from .theme import BG, MUTED

HOLD_S = 0.5
SKIN_CAM = (0.93, 0.78, 0.68, 1.0)
SKIN_EMG = (0.62, 0.78, 0.95, 1.0)
ON_TOP = {GL_DEPTH_TEST: False, GL_BLEND: True, "glBlendFunc": (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)}


def to_gl(p: np.ndarray, root: np.ndarray) -> np.ndarray:
    q = (p - root) * 100.0
    return np.stack([q[..., 0], q[..., 2], -q[..., 1]], axis=-1)


def rgba(hex_color: str) -> tuple:
    c = pg.mkColor(hex_color)
    return (c.redF(), c.greenF(), c.blueF(), 1.0)


class HandView(QWidget):
    """One titled GL view with a mesh, skeleton and grid."""

    def __init__(self, title: str, skin: tuple) -> None:
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: bold; padding: 4px;")
        self.title.setAlignment(Qt.AlignCenter)
        v.addWidget(self.title)
        self.gl = gl.GLViewWidget()
        self.gl.setBackgroundColor(BG)
        v.addWidget(self.gl, 1)
        grid = gl.GLGridItem()
        grid.setSize(40, 40)
        grid.setSpacing(2, 2)
        grid.translate(0, 0, -12)
        grid.setColor((1, 1, 1, 0.08))
        self.gl.addItem(grid)
        self.mesh = gl.GLMeshItem(smooth=True, shader="shaded", color=skin, glOptions="opaque")
        self.mesh.setVisible(False)   # an empty GLMeshItem raises on every draw
        self.gl.addItem(self.mesh)
        self.lines = gl.GLLinePlotItem(mode="lines", width=4, antialias=True)
        self.lines.setGLOptions(ON_TOP)
        self.gl.addItem(self.lines)
        self.dots = gl.GLScatterPlotItem(size=9, color=(1, 1, 1, 1))
        self.dots.setGLOptions(ON_TOP)
        self.gl.addItem(self.dots)
        self.colors = np.array([rgba(FINGER_COLORS[f]) for a, b, f in BONES for _ in (0, 1)])
        self.reset()

    def reset(self) -> None:
        self.gl.setCameraPosition(distance=45, elevation=8, azimuth=-90)
        self.gl.opts["center"] = pg.Vector(0, 0, 0)

    def cam_state(self) -> tuple:
        o = self.gl.opts
        c = o["center"]
        return (round(o["distance"], 3), round(o["elevation"], 3), round(o["azimuth"], 3),
                round(c.x(), 3), round(c.y(), 3), round(c.z(), 3))

    def set_cam_state(self, s: tuple) -> None:
        self.gl.opts.update(distance=s[0], elevation=s[1], azimuth=s[2], center=pg.Vector(s[3], s[4], s[5]))
        self.gl.update()

    def show_hand(self, verts, joints, faces, mesh_on: bool, bones_on: bool) -> None:
        root = joints[0]
        kp = to_gl(joints, root)
        if mesh_on and verts is not None and faces is not None:
            self.mesh.setMeshData(vertexes=to_gl(verts, root), faces=faces, smooth=True)
            self.mesh.setVisible(True)
        else:
            self.mesh.setVisible(False)
        if bones_on:
            self.lines.setData(pos=np.array([kp[i] for a, b, _ in BONES for i in (a, b)]), color=self.colors)
            self.dots.setData(pos=kp)
        self.lines.setVisible(bones_on)
        self.dots.setVisible(bones_on)

    def clear(self) -> None:
        for it in (self.mesh, self.lines, self.dots):
            it.setVisible(False)


class Hand3DWindow(StudioWindow):
    KEY = "hand3d"
    TITLE = "3D 손 · EMG 복원"
    REFRESH_HZ = 30.0

    def __init__(self, hub) -> None:
        super().__init__(hub)
        self.cam = hub.camera
        self.pose = hub.pose
        self.mano: Mano | None = None
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
        self.b_train.clicked.connect(lambda: self.pose.train_async(["ridge", "mlp"][self.kind.currentIndex()]))
        row.addWidget(self.b_train)
        bv.addLayout(row)
        self.predict = QCheckBox("EMG 예측 (왼쪽 손 움직이기)")
        self.predict.toggled.connect(lambda on: setattr(self.pose, "predicting", on))
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

    # ------------------------------------------------------------ controls
    def _collect(self, on: bool) -> None:
        if on and not self.cam.running:
            self.pose.message = "카메라 손 인식 창에서 먼저 '시작'을 누르세요."
            self.b_collect.setChecked(False)
            return
        if on and self.hub.mode is None:
            self.pose.message = "밴드가 연결되지 않았습니다."
            self.b_collect.setChecked(False)
            return
        self.pose.set_collecting(on)
        self.b_collect.setText("■ 수집 중지" if on else "● 수집 시작")

    def _clear(self) -> None:
        if QMessageBox.question(self, "데이터 지우기", f"수집한 {self.pose.count:,}개 샘플을 지울까요?") \
                == QMessageBox.Yes:
            self.pose.clear()

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
        if self.mano is None and Mano.available():
            self.mano = Mano()
        if self.pose.mano is None and self.mano is not None:
            self.pose.mano = self.mano
        p = self.pose
        self.count.setText(f"샘플 {p.count:,}개 · 약 {p.seconds():.0f}초 분량"
                           + ("  (수집 중)" if p.collecting else ""))
        self.b_train.setEnabled(not p.training)
        self.predict.setEnabled(p.model is not None)
        self.report.setText(p.report or "아직 학습한 모델이 없습니다.")
        self.msg.setText(p.message)
        mesh_on, bones_on = self.show_mesh.isChecked(), self.show_bones.isChecked()

        # camera hand (right)
        flex_cam = None
        frame, seq = self.cam.latest()
        hand = None
        if frame and self.cam.running:
            hands = [h for h in frame["hands"] if h["is_right"] == 1]
            hand = hands[0] if hands else None
        if hand is not None:
            self._last_seen = time.monotonic()
            if seq != self._seq:
                self._seq = seq
                self.v_cam.show_hand(hand["verts"], hand["kp3d"], self.cam.faces, mesh_on, bones_on)
            flex_cam = flexion_from_joints(hand["kp3d"])
            self.v_cam.title.setText("카메라 (정답) · 오른손")
        elif time.monotonic() - self._last_seen > HOLD_S:
            self.v_cam.clear()
            self.v_cam.title.setText("카메라 (정답) · 손 없음" if self.cam.running else "카메라 (정답) · 카메라 꺼짐")

        # EMG hand
        flex_emg = None
        if p.predicting and p.pred is not None and self.mano is not None:
            verts, joints = self.mano.forward(p.last_go, p.pred, p.last_betas)
            self.v_emg.show_hand(verts, joints, self.mano.faces, mesh_on, bones_on)
            flex_emg = flexion_from_joints(joints)
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
