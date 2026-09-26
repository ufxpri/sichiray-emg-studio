"""A titled 3D view of one hand: MANO mesh, 21-joint skeleton drawn on top, grid.

Coordinates: WiLoR gives camera space (x right, y down, z away, metres). Shown in
centimetres with the wrist at the origin, remapped to GL as X = x, Y = z, Z = -y,
so the default view looks from where the camera stands.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import GL_BLEND, GL_DEPTH_TEST, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..hand.skeleton import BONES
from .theme import BG, FINGER_COLORS

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
