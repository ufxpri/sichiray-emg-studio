"""EMG Studio launcher: connect to the armband and open the analysis windows.

  python emg_studio.py                      # pick port in the window
  python emg_studio.py --port COM4 --open all
  python emg_studio.py --demo hex --open all
  python emg_studio.py --demo hex --open all --snapshot shots --after 12   # save PNGs and quit

Requires: pip install PySide6 pyqtgraph pyserial numpy scipy
"""
from __future__ import annotations

import argparse
import os
import sys

from studio import theme  # sets the Qt binding for pyqtgraph before it is imported
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QPushButton, QVBoxLayout, QWidget)

from studio.device import find_ch340, list_ports
from studio.hub import Hub
from studio.win_camera import CameraWindow
from studio.win_filter import FilterWindow
from studio.win_hand3d import Hand3DWindow
from studio.win_info import InfoWindow
from studio.win_raw import RawWindow
from studio.win_spectro import SpectroWindow

WINDOWS = {
    "camera": (CameraWindow, "카메라 손 인식", "웹캠 + GPU 손 관절 오버레이 (WiLoR, 별도 프로세스)"),
    "hand3d": (Hand3DWindow, "3D 손", "카메라가 인식한 손의 3D 메시·뼈대, 손가락 굽힘"),
    "raw": (RawWindow, "EMG 원시 데이터", "수신한 값 그대로 8채널 + IMU"),
    "info": (InfoWindow, "센서 정보", "전송률, 프레임, 배터리, HEX/ASCII 바이트 콘솔"),
    "spectro": (SpectroWindow, "스펙트로그램 · 노이즈", "채널별 스펙트로그램과 노이즈 진단표"),
    "filter": (FilterWindow, "신호 처리 · 필터", "전역 밴드패스·노치·회전·보정"),
}
DEMO = {"데모 (HEX 500 Hz)": "hex", "데모 (ASCII 67 Hz)": "ascii"}


class Launcher(QMainWindow):
    def __init__(self, hub: Hub) -> None:
        super().__init__()
        self.hub = hub
        self.wins: dict[str, QMainWindow] = {}
        self.setWindowTitle("EMG Studio")

        conn = QGroupBox("연결")
        ch = QHBoxLayout(conn)
        self.port = QComboBox()
        self.port.setMinimumWidth(260)
        ch.addWidget(self.port, 1)
        refresh = QPushButton("⟳")
        refresh.setToolTip("포트 목록 새로고침")
        refresh.setFixedWidth(32)
        refresh.clicked.connect(self._ports)
        ch.addWidget(refresh)
        self.btn = QPushButton("연결")
        self.btn.clicked.connect(self._toggle)
        ch.addWidget(self.btn)
        self.demo_btn = QPushButton("모드 버튼")
        self.demo_btn.setToolTip("데모 장치의 HEX ↔ ASCII 모드를 바꿉니다 (실제 밴드는 본체 버튼: 2번 클릭 = ASCII, 1번 = HEX).")
        self.demo_btn.clicked.connect(lambda: self.hub.link.demo and self.hub.link.demo.toggle())
        ch.addWidget(self.demo_btn)

        self.state = QLabel("끊김")
        self.state.setStyleSheet(f"color: {theme.MUTED}; padding: 4px;")

        wg = QGroupBox("창")
        grid = QGridLayout(wg)
        for i, (key, (_, name, desc)) in enumerate(WINDOWS.items()):
            b = QPushButton(name)
            b.setMinimumHeight(40)
            b.clicked.connect(lambda _=False, k=key: self.open(k))
            grid.addWidget(b, i, 0)
            d = QLabel(desc)
            d.setStyleSheet(f"color: {theme.MUTED};")
            grid.addWidget(d, i, 1)
        allb = QPushButton("모두 열기")
        allb.clicked.connect(lambda: [self.open(k) for k in WINDOWS])
        grid.addWidget(allb, len(WINDOWS), 0, 1, 2)

        root = QWidget()
        v = QVBoxLayout(root)
        v.addWidget(conn)
        v.addWidget(self.state)
        v.addWidget(wg)
        v.addStretch(1)
        self.setCentralWidget(root)
        self.resize(620, 470)
        self._ports()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._status)
        self.timer.start(250)

    def _ports(self) -> None:
        self.port.clear()
        ch340 = set(find_ch340())
        for dev, desc in list_ports():
            self.port.addItem(f"{dev}  ★ 암밴드 동글 (CH340)" if dev in ch340 else f"{dev}  {desc}", dev)
        for label, mode in DEMO.items():
            self.port.addItem(label, f"demo:{mode}")
        idx = next((i for i in range(self.port.count()) if self.port.itemData(i) in ch340), 0)
        self.port.setCurrentIndex(idx)

    def connect_to(self, target: str) -> None:
        if target.startswith("demo:"):
            self.hub.link.open_demo(target[5:])
        else:
            self.hub.link.open_serial(target)

    def _toggle(self) -> None:
        if self.hub.link.status in ("연결됨", "연결 중"):
            self.hub.link.close()
        else:
            self.connect_to(self.port.currentData())

    def _status(self) -> None:
        s = self.hub.link.snapshot()
        on = s["status"] in ("연결됨", "연결 중")
        self.btn.setText("해제" if on else "연결")
        self.port.setEnabled(not on)
        self.demo_btn.setVisible(self.hub.link.demo is not None)
        parts = [f"{s['source'] or '-'}: {s['status']}"]
        if s["error"]:
            parts.append(s["error"])
        if s["mode"]:
            parts.append(f"{s['mode'].upper()} · {s['sps']:.0f} 샘플/s · {s['bps']:,.0f} B/s")
        elif on:
            parts.append("데이터 대기 중 — 밴드 전원을 켜세요")
        if s["battery"] is not None:
            parts.append(f"배터리 {s['battery']}%")
        self.state.setText("   ·   ".join(parts))

    def open(self, key: str) -> QMainWindow:
        w = self.wins.get(key)
        if w is None:
            w = self.wins[key] = WINDOWS[key][0](self.hub)
        w.show()
        w.raise_()
        w.activateWindow()
        return w

    def closeEvent(self, ev) -> None:
        for w in self.wins.values():
            w.close()
        self.hub.link.close()
        self.hub.camera.stop()
        super().closeEvent(ev)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="COM port to open at start")
    ap.add_argument("--demo", choices=["hex", "ascii"], help="start the synthetic device")
    ap.add_argument("--open", default="", help="windows to open: camera,hand3d,raw,info,spectro,filter or all")
    ap.add_argument("--snapshot", help="save each window as PNG into this folder, then quit")
    ap.add_argument("--after", type=float, default=10.0, help="seconds before --snapshot")
    ap.add_argument("--camera", type=int, help="start the hand worker on this camera index")
    ap.add_argument("--camera-video", default="", help="test: feed the hand worker from this video file")
    ap.add_argument("--toggle-at", type=float, help="demo: press the mode button after this many seconds")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    theme.apply(app)
    hub = Hub()
    win = Launcher(hub)
    win.show()
    if args.demo:
        win.connect_to(f"demo:{args.demo}")
    elif args.port:
        win.connect_to(args.port)
    if args.camera is not None:
        hub.camera.start(args.camera, video=args.camera_video)
    keys = list(WINDOWS) if args.open == "all" else [k for k in args.open.split(",") if k]
    for k in keys:
        win.open(k)
    if args.toggle_at is not None:
        QTimer.singleShot(int(args.toggle_at * 1000), lambda: hub.link.demo and hub.link.demo.toggle())
    if args.snapshot:
        os.makedirs(args.snapshot, exist_ok=True)

        def shoot() -> None:
            win.grab().save(os.path.join(args.snapshot, "launcher.png"))
            for k, w in win.wins.items():
                w.grab().save(os.path.join(args.snapshot, f"{k}.png"))
            app.quit()
        QTimer.singleShot(int(args.after * 1000), shoot)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
