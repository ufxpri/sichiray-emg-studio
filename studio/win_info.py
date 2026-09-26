"""Window 5: link and sensor information, plus a byte console in HEX or ASCII."""
from __future__ import annotations

import html

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QRadioButton, QSplitter, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from .common import StudioWindow
from .device import FRAME_LEN, HDR, IMU_NAMES, MODES, TAIL
from .theme import GRADE_FG, MUTED, TEXT, mono

# byte ranges of a HEX frame and their colours in the console
FIELDS = [(0, 3, MUTED, "헤더 AA AA 5F"), (3, 7, "#66d9e8", "타임스탬프(ms)"), (7, 10, "#ffa94d", "가속도"),
          (10, 13, "#ffd43b", "자이로"), (13, 16, "#da77f2", "각도"), (16, 96, TEXT, "EMG 10샘플×8ch"),
          (96, 97, "#69db7c", "배터리"), (97, 98, MUTED, "꼬리 55")]
BAD = "#ff6b6b"

ROWS = [
    ("source", "연결"), ("mode", "출력 모드"), ("uptime", "연결 시간"), ("idle", "마지막 수신"),
    ("bytes", "수신 총량"), ("bps", "전송률"), ("sps", "샘플률"), ("frames", "정상 프레임"),
    ("resync", "재동기로 버린 바이트"), ("lost", "누락 프레임"), ("irregular", "불규칙 타임스탬프"),
    ("tsgap", "타임스탬프 간격"), ("ts", "장치 시각"), ("lines", "ASCII 줄 (정상/불량)"),
    ("battery", "배터리"), ("imu", "IMU 원시값"), ("modes", "모드 전환 횟수"),
]
TIPS = {
    "resync": "프레임 경계(AA AA 5F … 55)가 맞지 않아 버린 바이트 수입니다. 무선 구간의 깨짐이나 모드 전환 직후에 생깁니다.",
    "lost": "장치 타임스탬프는 프레임마다 20 ms씩 늘어납니다. 간격이 40 ms면 프레임 1개가 오는 도중에 사라진 것입니다.",
    "irregular": "20의 배수가 아닌 간격입니다. 장치가 재시작됐거나 파싱이 어긋난 경우입니다.",
    "bps": "이론값: HEX 98 B × 50 = 4,900 B/s, ASCII 약 2,675 B/s(§12). 크게 낮으면 무선 연결이 약하거나 끊기는 중입니다.",
    "sps": "초당 EMG 샘플 수입니다. HEX는 500, ASCII는 약 67이어야 합니다.",
    "idle": "마지막으로 바이트를 받은 뒤 흐른 시간입니다. 0.5초를 넘으면 스트림이 멈춘 것입니다.",
}


def _fmt_bytes(n: int) -> str:
    return f"{n:,} B" if n < 10240 else f"{n / 1024:,.1f} KB" if n < 10 * 1024 ** 2 else f"{n / 1024 ** 2:,.1f} MB"


class InfoWindow(StudioWindow):
    KEY = "info"
    TITLE = "센서 정보"
    REFRESH_HZ = 10.0

    def __init__(self, hub) -> None:
        super().__init__(hub)
        # ---- counters
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self.vals: dict[str, QLabel] = {}
        half = (len(ROWS) + 1) // 2
        for i, (key, name) in enumerate(ROWS):
            r, c = i % half, (i // half) * 2
            k = QLabel(name)
            k.setStyleSheet(f"color: {MUTED};")
            val = QLabel("-")
            val.setFont(mono())
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if key in TIPS:
                k.setToolTip(TIPS[key])
                val.setToolTip(TIPS[key])
                k.setText(name + " ⓘ")
            grid.addWidget(k, r, c)
            grid.addWidget(val, r, c + 1)
            self.vals[key] = val
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        top = QWidget()
        top.setLayout(grid)

        # ---- history plots
        self.gl = pg.GraphicsLayoutWidget()
        self.p_bps = self.gl.addPlot(row=0, col=0)
        self.p_bps.setLabel("left", "B/s")
        self.p_bps.showGrid(x=True, y=True, alpha=0.15)
        self.c_bps = self.p_bps.plot(pen=pg.mkPen("#74c0fc", width=1.5))
        self.l_bps = pg.InfiniteLine(angle=0, pen=pg.mkPen("#74c0fc", style=Qt.DashLine, width=1))
        self.p_bps.addItem(self.l_bps)
        self.p_batt = self.gl.addPlot(row=1, col=0)
        self.p_batt.setLabel("left", "배터리 %")
        self.p_batt.setLabel("bottom", "연결 후 시간", units="s")
        self.p_batt.setYRange(0, 100)
        self.p_batt.showGrid(x=True, y=True, alpha=0.15)
        self.c_batt = self.p_batt.plot(pen=pg.mkPen("#69db7c", width=1.5), stepMode=None)
        for p in (self.p_bps, self.p_batt):
            p.hideButtons()
        self.gl.setMinimumWidth(380)

        upper = QSplitter()
        upper.addWidget(top)
        upper.addWidget(self.gl)
        upper.setSizes([520, 480])

        # ---- byte console
        cbar = QHBoxLayout()
        cbar.addWidget(QLabel("바이트 콘솔"))
        self.r_hex, self.r_asc = QRadioButton("HEX"), QRadioButton("ASCII")
        self.r_hex.setChecked(True)
        grp = QButtonGroup(self)
        for r in (self.r_hex, self.r_asc):
            grp.addButton(r)
            cbar.addWidget(r)
        grp.buttonToggled.connect(lambda *_: self._reset_console())
        self.frame_break = QCheckBox("프레임마다 줄바꿈")
        self.frame_break.setChecked(True)
        self.frame_break.setToolTip("HEX 표시에서 AA AA 5F 헤더마다 새 줄을 시작하고 필드별로 색을 입힙니다.")
        cbar.addWidget(self.frame_break)
        self.freeze = QCheckBox("화면 멈춤")
        self.freeze.setToolTip("콘솔 표시만 멈춥니다. 수신과 통계는 계속됩니다.")
        cbar.addWidget(self.freeze)
        clear = QPushButton("지우기")
        clear.clicked.connect(lambda: self.console.clear())
        cbar.addWidget(clear)
        cbar.addStretch(1)
        legend = "  ".join(f"<span style='color:{c}'>■ {name}</span>" for _, _, c, name in FIELDS)
        legend += f"  <span style='color:{BAD}'>■ 프레임 밖 바이트</span>"
        self.legend = QLabel(legend)
        cbar.addWidget(self.legend)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(mono())
        self.console.setMaximumBlockCount(600)
        self.console.setLineWrapMode(QPlainTextEdit.NoWrap)

        lower = QWidget()
        lv = QVBoxLayout(lower)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addLayout(cbar)
        lv.addWidget(self.console, 1)

        split = QSplitter(Qt.Vertical)
        split.addWidget(upper)
        split.addWidget(lower)
        split.setSizes([330, 470])
        self.setCentralWidget(split)
        self.resize(1250, 820)

        self.cursor = hub.link.bytes.total
        self.pend = bytearray()

    # ------------------------------------------------------------ counters
    def _set(self, key: str, text: str, grade: str | None = None) -> None:
        lab = self.vals[key]
        lab.setText(text)
        lab.setStyleSheet(f"color: {GRADE_FG[grade]};" if grade and grade != "ok" else "")

    def refresh(self) -> None:
        s = self.hub.link.snapshot()
        m = s["mode"]
        spec = MODES.get(m or "hex")
        st = s["status"] + (f" — {s['error']}" if s["error"] else "")
        self._set("source", f"{s['source'] or '-'}  ({st})", "bad" if s["error"] else None)
        self._set("mode", f"{m.upper()}  {spec['bits']}-bit  명목 {spec['fs']:g} Hz" if m else "감지 대기")
        self._set("uptime", f"{s['uptime']:.0f} s")
        idle = s["idle"]
        self._set("idle", "-" if idle is None else f"{idle * 1000:.0f} ms 전",
                  None if idle is None else "bad" if idle > 0.5 else None)
        self._set("bytes", _fmt_bytes(s["bytes_total"]))
        ratio = s["bps"] / spec["bps"] if m else 0
        self._set("bps", f"{s['bps']:,.0f} B/s  (이론 {spec['bps']:,} · {ratio:.0%})",
                  "warn" if m and ratio < 0.9 else None)
        self._set("sps", f"{s['sps']:.0f} /s  (명목 {spec['fs']:g})",
                  "warn" if m and s["sps"] < spec["fs"] * 0.9 else None)
        self._set("frames", f"{s['frames_ok']:,}" if m == "hex" else "- (HEX 전용)")
        self._set("resync", f"{s['resync_bytes']:,}" if m == "hex" else "-", "warn" if s["resync_bytes"] and m == "hex" else None)
        self._set("lost", f"{s['lost_frames']:,}" if m == "hex" else "-", "warn" if s["lost_frames"] and m == "hex" else None)
        self._set("irregular", f"{s['irregular_ts']:,}" if m == "hex" else "-",
                  "warn" if s["irregular_ts"] and m == "hex" else None)
        if s["ts_mean"] is not None and m == "hex":
            self._set("tsgap", f"평균 {s['ts_mean']:.2f} / 최소 {s['ts_min']} / 최대 {s['ts_max']} ms (정상 20)",
                      "warn" if s["ts_max"] != 20 or s["ts_min"] != 20 else None)
        else:
            self._set("tsgap", "-")
        self._set("ts", f"{s['last_ts']:,} ms" if s["last_ts"] is not None else "-")
        self._set("lines", f"{s['lines_ok']:,} / {s['lines_bad']:,}" if m == "ascii" else "- (ASCII 전용)",
                  "warn" if s["lines_bad"] and m == "ascii" else None)
        b = s["battery"]
        self._set("battery", "-" if b is None else f"{b} %", None if b is None else "bad" if b < 10 else "warn" if b < 20 else None)
        if s["imu"] and m == "hex":
            self._set("imu", " ".join(f"{n}={v}" for n, v in zip(IMU_NAMES, s["imu"])))
        else:
            self._set("imu", "- (HEX 모드에서만 전송)")
        self._set("modes", str(s["mode_changes"]))

        h = np.array(self.hub.rate_hist) if self.hub.rate_hist else np.zeros((0, 3))
        if len(h):
            self.c_bps.setData(h[:, 0], h[:, 1])
        self.l_bps.setPos(spec["bps"])
        bh = np.array(s["batt_hist"]) if s["batt_hist"] else np.zeros((0, 2))
        if len(bh):
            self.c_batt.setData(bh[:, 0], bh[:, 1])
        self.status.setText("ⓘ 표시가 있는 항목은 마우스를 올리면 설명이 나옵니다.")
        self._pump_console()

    # ------------------------------------------------------------ console
    def _reset_console(self) -> None:
        self.console.clear()
        self.pend.clear()
        self.legend.setVisible(self.r_hex.isChecked())

    def _pump_console(self) -> None:
        data, self.cursor = self.hub.link.bytes.since(self.cursor)
        if self.freeze.isChecked():
            self.pend.clear()
            return
        self.pend += data
        lines = self._hex_lines() if self.r_hex.isChecked() else self._ascii_lines()
        if lines:
            at_end = self.console.verticalScrollBar().value() >= self.console.verticalScrollBar().maximum() - 4
            self.console.appendHtml("<br>".join(lines[-300:]))
            if at_end:
                self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _hex_lines(self) -> list[str]:
        p, out = self.pend, []
        if not self.frame_break.isChecked():
            n = len(p) // 32 * 32
            out = [" ".join(f"{b:02X}" for b in p[i:i + 32]) for i in range(0, n, 32)]
            del p[:n]
            return out
        while True:
            i = p.find(HDR)
            if i < 0:
                if len(p) > 4096:  # nothing frame-like: flush as plain rows
                    out.append(self._span(p[:-2], BAD))
                    del p[:-2]
                break
            if i > 0:
                out.append(self._span(p[:i], BAD))
                del p[:i]
            if len(p) < FRAME_LEN:
                break
            if p[FRAME_LEN - 1] == TAIL:
                out.append(self._frame_html(bytes(p[:FRAME_LEN])))
                del p[:FRAME_LEN]
                continue
            j = p.find(HDR, 3)   # broken frame: show it up to the next header
            end = j if j > 0 else len(p)
            out.append(self._span(p[:end], BAD))
            del p[:end]
        return out

    @staticmethod
    def _span(b: bytes, color: str) -> str:
        return f"<span style='color:{color}'>{' '.join(f'{x:02X}' for x in b)}</span>"

    @staticmethod
    def _frame_html(b: bytes) -> str:
        parts = []
        for lo, hi, color, _ in FIELDS:
            if lo == 16:  # EMG: group each 8-channel sample
                groups = [" ".join(f"{x:02X}" for x in b[k:k + 8]) for k in range(16, 96, 8)]
                parts.append(f"<span style='color:{color}'>{' │ '.join(groups)}</span>")
            else:
                parts.append(f"<span style='color:{color}'>{' '.join(f'{x:02X}' for x in b[lo:hi])}</span>")
        return "&nbsp;&nbsp;".join(parts)

    def _ascii_lines(self) -> list[str]:
        p, out = self.pend, []
        while True:
            i = p.find(b"\n")
            if i < 0:
                if len(p) > 512:
                    out.append(self._ascii_html(bytes(p)))
                    p.clear()
                break
            out.append(self._ascii_html(bytes(p[:i])))
            del p[:i + 1]
        return out

    @staticmethod
    def _ascii_html(b: bytes) -> str:
        s = "".join(chr(x) if 32 <= x < 127 else "·" for x in b.rstrip(b"\r"))
        return html.escape(s).replace(" ", "&nbsp;")
