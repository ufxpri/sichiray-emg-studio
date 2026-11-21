"""Hand-shape example pictures for the recorder prompt.

A right hand, palm down, as the participant sees their own hand. Finger and sideways
wrist gestures are drawn from above; wrist flexion / extension from the thumb side,
because seen from above they only foreshorten. Gestures that move the wrist also show
the neutral hand as a dashed outline and an arrow for the direction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .theme import MUTED

SKIN = "#e8b993"
SKIN_EDGE = "#b07a55"


@dataclass(frozen=True)
class Pose:
    t: float = 0.3                                   # thumb fold 0 open .. 1 folded across the palm
    f: tuple[float, float, float, float] = (0.35, 0.38, 0.42, 0.45)   # index, middle, ring, little fold
    dev: float = 0.0                                 # -1 thumb side (radial) .. +1 little side (ulnar)
    flex: float = 0.0                                # +1 towards the palm .. -1 towards the back


RELAXED = (0.25, 0.28, 0.3, 0.35)
# keyed by gesture key (GestureSet.labels, any set); unknown keys fall back to the rest pose
POSES = {
    "rest": Pose(),
    "scissors": Pose(1, (0, 0, 1, 1)),
    "rock": Pose(1, (1, 1, 1, 1)),
    "paper": Pose(0, (0, 0, 0, 0)),
    "one": Pose(1, (0, 1, 1, 1)),
    "two": Pose(1, (0, 0, 1, 1)),
    "three": Pose(1, (0, 0, 0, 1)),
    "four": Pose(1, (0, 0, 0, 0)),
    "five": Pose(0, (0, 0, 0, 0)),
    "fist": Pose(1, (1, 1, 1, 1)),
    "open": Pose(0, (0, 0, 0, 0)),
    "index": Pose(1, (0, 1, 1, 1)),
    "wrist_ext": Pose(0.25, RELAXED, flex=-1),
    "wrist_flex": Pose(0.35, RELAXED, flex=1),
    "radial_dev": Pose(0.3, RELAXED, dev=-1),
    "ulnar_dev": Pose(0.3, RELAXED, dev=1),
}

DEV_DEG = 25.0
FLEX_DEG = 60.0
FB = (-36, -12, 12, 36)          # finger base x, index .. little (thumb on the left)
FL = (72, 80, 74, 60)            # finger length
TOP_VIEW = QRectF(-205, -210, 410, 370)
SIDE_VIEW = QRectF(-165, -205, 380, 410)


class HandPicture(QWidget):
    def __init__(self, parent=None, caption: bool = True) -> None:
        super().__init__(parent)
        self.pose = POSES["rest"]
        self.color = QColor(MUTED)
        self.dim = False
        self.caption = caption
        self.setMinimumSize(80, 80)

    def set_gesture(self, key: str, color: str | None = None, dim: bool = False) -> None:
        self.pose = POSES.get(key, POSES["rest"])
        self.color = QColor(color or MUTED)
        self.dim = dim
        self.update()

    # ------------------------------------------------------------ painting
    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.dim:
            p.setOpacity(0.45)
        side = self.pose.flex != 0
        scene = SIDE_VIEW if side else TOP_VIEW
        cap_h = 18 if self.caption and self.height() > 160 else 0
        area = QRectF(self.rect()).adjusted(4, 4, -4, -4 - cap_h)
        s = min(area.width() / scene.width(), area.height() / scene.height())
        p.save()
        p.translate(area.center())
        p.scale(s, s)
        p.translate(-scene.center())
        (self._side if side else self._top)(p, s)
        p.restore()
        if cap_h:
            p.setPen(QColor(MUTED))
            f = QFont(self.font())
            f.setPointSize(9)
            p.setFont(f)
            p.drawText(QRectF(0, self.height() - cap_h - 2, self.width(), cap_h), Qt.AlignCenter,
                       "옆(엄지 쪽)에서 본 오른손 · 손바닥 아래" if side else "위에서 본 오른손 · 손바닥 아래")
        p.end()

    def _pens(self, s: float, ghost: bool):
        if ghost:
            pen = QPen(QColor(MUTED), 1.6 / s * 1.4, Qt.DashLine)
            return pen, Qt.NoBrush
        return QPen(self.color, max(2.5, 3.0 / s)), QColor(SKIN)

    def _top(self, p: QPainter, s: float) -> None:
        pose = self.pose
        # forearm (does not move)
        p.setPen(QPen(QColor(SKIN_EDGE), 2))
        p.setBrush(QColor(SKIN))
        p.drawRoundedRect(QRectF(-40, -4, 80, 168), 30, 30)
        if pose.dev:
            self._top_hand(p, s, Pose(pose.t, pose.f), ghost=True)
        self._top_hand(p, s, pose, ghost=False)
        if pose.dev:
            self._arc_arrow(p, s, QPointF(0, 0), 188, -90, -90 + pose.dev * DEV_DEG)

    def _top_hand(self, p: QPainter, s: float, pose: Pose, ghost: bool) -> None:
        pen, brush = self._pens(s, ghost)
        p.save()
        p.rotate(pose.dev * DEV_DEG)
        p.translate(0, -28)
        p.setPen(pen)
        p.setBrush(brush)
        for bx, L, f in zip(FB, FL, pose.f):
            lv = L * (1 - 0.68 * f)
            p.drawRoundedRect(QRectF(bx - 10, -58 - lv, 20, lv + 10), 10, 10)
        p.drawRoundedRect(QRectF(-52, -64, 104, 92), 24, 24)
        if not ghost:       # knuckle line of folded fingers, so a fist reads as a fist
            p.save()
            p.setPen(QPen(QColor(SKIN_EDGE), 1.5))
            for bx, L, f in zip(FB, FL, pose.f):
                if f > 0.6:
                    p.drawLine(QPointF(bx - 7, -50), QPointF(bx + 7, -50))
            p.restore()
        a = -55 + 90 * pose.t
        ln = 52 - 14 * pose.t
        p.translate(-44, -6)
        p.rotate(a)
        p.drawRoundedRect(QRectF(-10, -ln, 20, ln + 8), 10, 10)
        p.restore()

    def _side(self, p: QPainter, s: float) -> None:
        pose = self.pose
        p.setPen(QPen(QColor(SKIN_EDGE), 2))
        p.setBrush(QColor(SKIN))
        p.drawRoundedRect(QRectF(-170, -28, 178, 56), 26, 26)
        self._side_hand(p, s, 0.0, ghost=True)
        self._side_hand(p, s, pose.flex * FLEX_DEG, ghost=False)
        self._arc_arrow(p, s, QPointF(0, 0), 190, 0, pose.flex * FLEX_DEG)
        if s < 0.4:          # thumbnail: the labels would only be clutter
            return
        f = QFont(self.font())
        f.setPointSizeF(15)
        p.setFont(f)
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(-165, -64, 150, 30), Qt.AlignLeft | Qt.AlignVCenter, "손등")
        p.drawText(QRectF(-165, 34, 150, 30), Qt.AlignLeft | Qt.AlignVCenter, "손바닥")

    def _side_hand(self, p: QPainter, s: float, deg: float, ghost: bool) -> None:
        pen, brush = self._pens(s, ghost)
        p.save()
        p.rotate(deg)
        p.setPen(pen)
        p.setBrush(brush)
        # relaxed fingers: a slightly curled band in front of the palm
        fingers = QPainterPath()
        fingers.moveTo(78, -18)
        fingers.cubicTo(120, -20, 158, -12, 172, 6)
        fingers.cubicTo(176, 14, 168, 20, 160, 16)
        fingers.cubicTo(140, 6, 110, 10, 78, 14)
        fingers.closeSubpath()
        p.drawPath(fingers)
        p.drawRoundedRect(QRectF(-6, -24, 96, 44), 20, 20)
        p.drawRoundedRect(QRectF(26, 6, 52, 22), 11, 11)       # thumb on the near side
        p.restore()

    def _arc_arrow(self, p: QPainter, s: float, c: QPointF, r: float, a0: float, a1: float) -> None:
        """Arc from angle a0 to a1 (degrees, Qt screen angles: 0 = +x, positive = clockwise)."""
        pen = QPen(self.color, max(3.0, 4.0 / s), Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        pts = [QPointF(c.x() + r * math.cos(math.radians(a0 + (a1 - a0) * i / 24)),
                       c.y() + r * math.sin(math.radians(a0 + (a1 - a0) * i / 24))) for i in range(25)]
        path.moveTo(pts[0])
        for q in pts[1:]:
            path.lineTo(q)
        p.drawPath(path)
        # arrow head along the tangent at the end
        sgn = 1 if a1 > a0 else -1
        t = math.radians(a1)
        tx, ty = -math.sin(t) * sgn, math.cos(t) * sgn
        nx, ny = -ty, tx
        tip = pts[-1]
        h = 16
        head = QPainterPath()
        head.moveTo(tip + QPointF(tx * h, ty * h))
        head.lineTo(tip + QPointF(nx * h * 0.7, ny * h * 0.7))
        head.lineTo(tip - QPointF(nx * h * 0.7, ny * h * 0.7))
        head.closeSubpath()
        p.setPen(Qt.NoPen)
        p.setBrush(self.color)
        p.drawPath(head)
