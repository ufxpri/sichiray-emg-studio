"""Window 1: webcam with the detected right hand's skeleton drawn over it."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from ..hand.camera_link import HandFrame
from ..hand.skeleton import BONES
from .common import StudioWindow
from .theme import BG, FINGER_COLORS, MUTED

RESOLUTIONS = [(1280, 720), (640, 480), (1920, 1080)]


class VideoView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.img: QImage | None = None
        self.frame: HandFrame | None = None
        self.mirror = False
        self.show_bones = True
        self.show_box = True
        self.message = ""
        self.setMinimumSize(480, 270)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BG))
        if self.img is None:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, self.message or "카메라 대기 중")
            return
        iw, ih = self.img.width(), self.img.height()
        s = min(self.width() / iw, self.height() / ih)
        ox, oy = (self.width() - iw * s) / 2, (self.height() - ih * s) / 2
        hands = self.frame.hands if self.frame else ()
        p.save()
        p.translate(ox, oy)
        p.scale(s, s)
        if self.mirror:
            p.translate(iw, 0)
            p.scale(-1, 1)
        p.drawImage(0, 0, self.img)
        p.setRenderHint(QPainter.Antialiasing)
        for h in hands:
            if self.show_box:
                x0, y0, x1, y1 = h.bbox
                p.setPen(QPen(QColor("#ffffff"), 3 / s, Qt.DashLine))
                p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            if self.show_bones:
                for a, b, f in BONES:
                    p.setPen(QPen(QColor(FINGER_COLORS[f]), 4 / s))
                    p.drawLine(QPointF(*h.kp2d[a]), QPointF(*h.kp2d[b]))
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#ffffff"))
                for x, y in h.kp2d:
                    p.drawEllipse(QPointF(x, y), 4 / s, 4 / s)
        p.restore()
        p.setFont(QFont("Malgun Gothic", 10, QFont.Bold))   # labels unmirrored, so they stay readable
        for h in hands:
            x0 = iw - h.bbox[2] if self.mirror else h.bbox[0]
            p.setPen(QColor("#ffffff"))
            p.drawText(QPointF(ox + x0 * s, oy + h.bbox[1] * s - 6), "오른손" if h.is_right else "왼손")


class CameraWindow(StudioWindow):
    KEY = "camera"
    TITLE = "카메라 손 인식"
    REFRESH_HZ = 60.0

    def __init__(self, hub) -> None:
        super().__init__(hub)
        self.cam = hub.camera
        bar = QHBoxLayout()
        bar.addWidget(QLabel("카메라"))
        self.index = QSpinBox()
        self.index.setRange(0, 9)
        bar.addWidget(self.index)
        self.res = QComboBox()
        self.res.addItems([f"{w}×{h}" for w, h in RESOLUTIONS])
        bar.addWidget(self.res)
        self.btn = QPushButton("시작")
        self.btn.clicked.connect(self._toggle)
        bar.addWidget(self.btn)
        bar.addSpacing(12)
        right = QLabel("오른손만 인식")
        right.setStyleSheet(f"color: {MUTED};")
        bar.addWidget(right)
        self.mirror = QCheckBox("거울 모드")
        self.bones = QCheckBox("뼈대")
        self.bones.setChecked(True)
        self.box = QCheckBox("상자")
        self.box.setChecked(True)
        for w in (self.mirror, self.bones, self.box):
            bar.addWidget(w)
        bar.addStretch(1)

        self.view = VideoView()
        self.stats = QLabel()
        self.stats.setStyleSheet(f"color: {MUTED};")
        root = QWidget()
        v = QVBoxLayout(root)
        v.addLayout(bar)
        v.addWidget(self.view, 1)
        v.addWidget(self.stats)
        self.setCentralWidget(root)
        self.resize(1100, 720)
        self._seq = -1

    def _toggle(self) -> None:
        if self.cam.running:
            self.cam.stop()
            self.view.img = None
        else:
            w, h = RESOLUTIONS[self.res.currentIndex()]
            self.cam.start(self.index.value(), w, h)

    def refresh(self) -> None:
        cam = self.cam
        running = cam.running
        self.btn.setText("정지" if running else "시작")
        self.index.setEnabled(not running)
        self.res.setEnabled(not running)
        self.view.mirror = self.mirror.isChecked()
        self.view.show_bones = self.bones.isChecked()
        self.view.show_box = self.box.isChecked()
        frame, seq = cam.latest()
        if frame is not None and seq != self._seq and running:
            self._seq = seq
            img = QImage.fromData(frame.jpeg, "JPG")
            if not img.isNull():
                self.view.img = img
            self.view.frame = frame
        status = cam.error or cam.status
        self.view.message = status
        self.view.update()
        if frame and running:
            ms = frame.ms
            self.stats.setText(
                f"카메라 {frame.cam_fps:.0f} fps · 추론 {cam.rx_fps():.0f} fps · 검출 {ms['detect']:.0f} ms "
                f"({'YOLO' if frame.how == 'yolo' else '추적'}) · WiLoR {ms['wilor']:.0f} ms · "
                f"촬영→결과 지연 {frame.lag_ms:.0f} ms · 오른손 {'있음' if frame.right_hand() else '없음'}")
        self.status.setText(status)
