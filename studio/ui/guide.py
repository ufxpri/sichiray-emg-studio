"""Intro page: how to put the band on so every wearer lands in the same place.

Rules match sichiray_emg8/docs/wearing_guide.md. Channel numbers are written on the pods,
1..8 clockwise; pod 5 carries the charging port (on its edge) and the button (on its face).
Right arm, port toward the wrist, palm down with pod 5 on top: thumb side 3, palm side 1,
little-finger side 7. Seeing 7 on the thumb side means the band is on backwards.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import theme

KEY_POD = 5
SKIN = QColor("#5b5650")
BAND = QColor("#23262c")
ACCENT = QColor("#ffa94d")

RULES = [
    ("오른팔", "오른팔에 찹니다."),
    ("방향", "5번 마디(충전 단자·버튼)의 충전 단자가 <b>손목 쪽</b>을 향하게 합니다. "
             "거꾸로 차면 채널 순서가 거울처럼 뒤집혀 소프트웨어로 바로잡을 수 없습니다."),
    ("높이", "팔꿈치 안쪽 접히는 선에서 손가락 서너 개 폭 아래, 전완에서 가장 두꺼운 곳."),
    ("회전", "팔을 모니터 쪽으로 뻗고 손바닥을 바닥으로. <b>5번 마디(버튼)가 팔 윗면 한가운데</b>, "
             "가운뎃손가락과 일직선."),
    ("확인", "엄지 쪽 옆면에 <b>3번</b>, 손바닥 쪽에 <b>1번</b>, 새끼손가락 쪽에 <b>7번</b>이 옵니다. "
             "엄지 쪽이 7번이면 거꾸로 찬 것이니 다시 찹니다."),
    ("조임", "손목을 돌려도 밴드가 따라 돌지 않을 만큼 단단히."),
    ("피부", "건조하면 전극 자리를 물로 살짝 적시고, 충전기는 뽑고 측정합니다."),
]

# Virtual canvas; the drawing is scaled to fit the widget.
VW, VH = 520.0, 600.0
CX = VW / 2
# First-person view: the wearer looks down along their own right arm stretched toward the
# monitor, so the elbow is near (bottom, wide) and the hand far (top, narrow).
WRIST_Y, WRIST_R = 190.0, 40.0
BAND_Y, BAND_R = 400.0, 66.0
ELBOW_Y = 500.0
BOTTOM_R = 84.0
SQUASH = 0.45        # a cross-section seen from above: its upper half is the back of the arm
RING_R = BAND_R + 4


def pod_angle(ch: int) -> float:
    """Angle of a pod around the arm as the wearer sees it: 5 on top (back of the forearm),
    3 on the left (thumb), 1 underneath (palm), 7 on the right (little finger)."""
    return math.radians(90 + (KEY_POD - ch) * 45)


def pod_pos(ch: int) -> QPointF:
    a = pod_angle(ch)
    return QPointF(CX + RING_R * math.cos(a), BAND_Y - RING_R * SQUASH * math.sin(a))


def _text(p: QPainter, x: float, y: float, s: str, color: QColor | str, w: float = 160,
          align=Qt.AlignCenter, bold: bool = False) -> None:
    f = QFont(p.font())
    f.setBold(bold)
    p.setFont(f)
    p.setPen(QColor(color))
    p.drawText(QRectF(x - w / 2, y - 11, w, 22), align, s)


class ArmView(QWidget):
    """The wearer's own right arm stretched toward the monitor, palm down, kept abstract."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(400, 450)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        sc = min(self.width() / VW, self.height() / VH)
        p.translate((self.width() - VW * sc) / 2, (self.height() - VH * sc) / 2)
        p.scale(sc, sc)
        p.setClipRect(QRectF(0, 0, VW, VH))

        # arm: one trapezoid; hand: a rounded block and a thumb toward the left
        p.setPen(Qt.NoPen)
        p.setBrush(SKIN)
        arm = QPainterPath()
        arm.moveTo(CX - WRIST_R, WRIST_Y)
        arm.lineTo(CX + WRIST_R, WRIST_Y)
        arm.lineTo(CX + BOTTOM_R, VH)
        arm.lineTo(CX - BOTTOM_R, VH)
        arm.closeSubpath()
        p.drawPath(arm)
        p.drawRoundedRect(QRectF(CX - 44, WRIST_Y - 150, 88, 160), 30, 30)
        p.setPen(QPen(SKIN, 22, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(CX - 30, WRIST_Y - 30), QPointF(CX - 78, WRIST_Y - 92))

        # band: the visible upper half of the ring, one thick arc
        p.setPen(QPen(BAND, 30, Qt.SolidLine, Qt.FlatCap))
        p.drawArc(QRectF(CX - RING_R, BAND_Y - RING_R * SQUASH, 2 * RING_R, 2 * RING_R * SQUASH),
                  0, 180 * 16)

        # pods on the visible half; 5 highlighted, with the charging port on its wrist-side edge
        for ch in (3, 4, 5, 6, 7):
            c = pod_pos(ch)
            key = ch == KEY_POD
            p.setPen(Qt.NoPen)
            p.setBrush(ACCENT if key else QColor(theme.CH_COLORS[ch - 1]))
            p.drawEllipse(c, 13, 13)
            _text(p, c.x(), c.y(), str(ch), "#15171c", 26, bold=True)
        top = pod_pos(KEY_POD)
        p.setPen(QPen(ACCENT, 2))
        p.drawLine(QPointF(top.x(), top.y() - 18), QPointF(top.x(), top.y() - 60))
        p.drawLine(QPointF(top.x(), top.y() - 60), QPointF(top.x() - 6, top.y() - 52))
        p.drawLine(QPointF(top.x(), top.y() - 60), QPointF(top.x() + 6, top.y() - 52))

        # labels
        _text(p, top.x() + 70, top.y() - 44, "충전 단자 → 손목", ACCENT, 120, Qt.AlignLeft | Qt.AlignVCenter)
        _text(p, top.x() + 70, top.y() - 22, "5 · 버튼 (팔 윗면)", ACCENT, 120, Qt.AlignLeft | Qt.AlignVCenter)
        l, r = pod_pos(3), pod_pos(7)
        _text(p, l.x() - 70, l.y(), "엄지 쪽", theme.TEXT, 100, Qt.AlignRight | Qt.AlignVCenter)
        _text(p, r.x() + 70, r.y(), "새끼손가락 쪽", theme.TEXT, 100, Qt.AlignLeft | Qt.AlignVCenter)
        _text(p, CX, BAND_Y + 46, "1 · 손바닥 쪽 (아래)", theme.MUTED, 160)
        _text(p, CX, 16, "↑ 모니터", theme.MUTED, 120)
        # elbow-to-band distance
        x = CX + BOTTOM_R + 30
        p.setPen(QPen(QColor(theme.MUTED), 1))
        p.drawLine(QPointF(x, BAND_Y + 16), QPointF(x, ELBOW_Y))
        p.drawLine(QPointF(x - 5, BAND_Y + 16), QPointF(x + 5, BAND_Y + 16))
        p.drawLine(QPointF(x - 5, ELBOW_Y), QPointF(x + 5, ELBOW_Y))
        _text(p, x + 62, (BAND_Y + ELBOW_Y) / 2, "손가락 3~4개", theme.MUTED, 110, Qt.AlignLeft | Qt.AlignVCenter)
        _text(p, x + 62, ELBOW_Y, "팔꿈치", theme.MUTED, 110, Qt.AlignLeft | Qt.AlignVCenter)
        p.end()


class WearGuide(QWidget):
    """Shown when the studio starts; `done` fires when the wearer is ready."""

    done = Signal()

    def __init__(self) -> None:
        super().__init__()
        title = QLabel("암밴드 착용 가이드")
        tf = QFont(self.font())
        tf.setPointSize(tf.pointSize() + 6)
        tf.setBold(True)
        title.setFont(tf)

        rules = "".join(
            f"<p style='margin: 0 0 10px 0;'><b style='color:{ACCENT.name()};'>{i}. {head}</b><br>{body}</p>"
            for i, (head, body) in enumerate(RULES, 1))
        text = QLabel(
            f"<p style='color:{theme.MUTED}; margin: 0 0 14px 0;'>누가 차도 같은 자리에 오도록 정한 규칙입니다. "
            "남는 한 칸 안팎의 어긋남은 이후 회전 보정으로 맞춥니다. "
            "그림은 오른팔을 모니터 쪽으로 뻗고 손바닥을 아래로 한 채 내려다본 모습입니다.</p>" + rules)
        text.setWordWrap(True)
        text.setTextFormat(Qt.RichText)
        text.setAlignment(Qt.AlignTop)

        go = QPushButton("착용 완료 — 시작하기")
        go.setMinimumHeight(40)
        go.setStyleSheet("background: #2b8a3e; color: white; font-weight: bold;")
        go.clicked.connect(self.done.emit)

        left = QVBoxLayout()
        left.addWidget(title)
        left.addSpacing(6)
        left.addWidget(text, 1)
        left.addWidget(go)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(360)

        h = QHBoxLayout(self)
        h.addWidget(lw)
        h.addSpacing(10)
        h.addWidget(ArmView(), 1)
