"""Shared look: dark Qt palette, pyqtgraph defaults, channel and grade colours."""
from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtGui import QColor, QFont, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

BG = "#15171c"
PANEL = "#1d2027"
TEXT = "#d7dae0"
MUTED = "#8a909c"
GRID = "#2c313a"

# eight channels, evenly spaced hues, similar lightness so none dominates
CH_COLORS = ["#ff6b6b", "#ffa94d", "#ffd43b", "#8ce99a", "#3bc9db", "#74c0fc", "#b197fc", "#f783ac"]
GRADE_BG = {"ok": "#1f4d2e", "warn": "#6b5315", "bad": "#6e1f24", "info": "#23262d"}
GRADE_FG = {"ok": "#b2f2bb", "warn": "#ffe8a3", "bad": "#ffc9c9", "info": TEXT}
GRADE_LABEL = {"ok": "양호", "warn": "주의", "bad": "나쁨", "info": "-"}


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    for role, c in [
        (QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, PANEL),
        (QPalette.AlternateBase, BG), (QPalette.Text, TEXT), (QPalette.Button, PANEL),
        (QPalette.ButtonText, TEXT), (QPalette.ToolTipBase, PANEL), (QPalette.ToolTipText, TEXT),
        (QPalette.Highlight, "#3b5bdb"), (QPalette.HighlightedText, "#ffffff"),
        (QPalette.PlaceholderText, MUTED),
    ]:
        p.setColor(role, QColor(c))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(MUTED))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(MUTED))
    app.setPalette(p)
    app.setFont(QFont("Malgun Gothic", 9))
    app.setStyleSheet(
        "QToolTip { color: %s; background: %s; border: 1px solid %s; padding: 4px; }"
        "QGroupBox { border: 1px solid %s; border-radius: 4px; margin-top: 10px; padding-top: 6px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: %s; }"
        % (TEXT, PANEL, GRID, GRID, MUTED))
    pg.setConfigOptions(background=BG, foreground=MUTED, antialias=False)


def mono() -> QFont:
    f = QFont("Consolas", 9)
    f.setStyleHint(QFont.Monospace)
    return f
