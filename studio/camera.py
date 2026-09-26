"""Studio side of the hand worker: starts studio/hand_worker.py in the hand venv
and receives frames, 2D/3D joints and MANO vertices over localhost TCP.

The worker lives in its own interpreter (PyTorch, numpy<2) so inference never
holds the studio's GIL. Frame timestamps are perf_counter_ns() taken in the
worker right after the camera read; on Windows that clock is system-wide, so
it can be lined up with EMG arrival times in the recording stage.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np

HOME = os.path.join(os.path.expanduser("~"), ".emg_hand")
PYTHON = os.environ.get("EMG_HAND_PY", os.path.join(HOME, "venv", "Scripts", "python.exe"))
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_worker.py")
LOG = os.path.join(HOME, "worker.log")
PORT = int(os.environ.get("EMG_HAND_PORT", "8790"))

# OpenPose/MANO 21-joint order: wrist, then 4 joints per finger thumb..pinky
FINGERS = [("엄지", [0, 1, 2, 3, 4]), ("검지", [0, 5, 6, 7, 8]), ("중지", [0, 9, 10, 11, 12]),
           ("약지", [0, 13, 14, 15, 16]), ("소지", [0, 17, 18, 19, 20])]
FINGER_COLORS = ["#ff6b6b", "#ffd43b", "#69db7c", "#4dabf7", "#da77f2"]
BONES = [(c[i], c[i + 1], f) for f, (_, c) in enumerate(FINGERS) for i in range(4)]


class CameraLink:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self.lock = threading.Lock()
        self.status = "정지"
        self.error = ""
        self.faces: np.ndarray | None = None
        self.device = ""
        self.frame: dict | None = None    # newest header, + "jpeg", + per-hand "verts"
        self.seq = 0                      # increments per received frame
        self._rx_times: list[float] = []
        self._stop = threading.Event()

    @staticmethod
    def available() -> tuple[bool, str]:
        if not os.path.exists(PYTHON):
            return False, (f"손 인식 환경이 없습니다: {PYTHON}\n"
                           "README의 '카메라 손 인식 설치' 절을 따라 설치하세요.")
        return True, ""

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, camera: int = 0, width: int = 1280, height: int = 720, fps: int = 30, video: str = "") -> None:
        self.stop()
        ok, msg = self.available()
        if not ok:
            self.status, self.error = "오류", msg
            return
        os.makedirs(HOME, exist_ok=True)
        log = open(LOG, "w", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            [PYTHON, WORKER, "--camera", str(camera), "--width", str(width), "--height", str(height),
             "--fps", str(fps), "--port", str(PORT)] + (["--video", video] if video else []),
            stdout=log, stderr=log, creationflags=flags)
        self.status, self.error = "워커 시작 중", ""
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.status = "정지"

    # ------------------------------------------------------------ receiving
    def _connect(self) -> socket.socket | None:
        t0 = time.time()
        while not self._stop.is_set() and time.time() - t0 < 60:
            if self.proc and self.proc.poll() is not None:
                return None
            try:
                s = socket.create_connection(("127.0.0.1", PORT), timeout=1)
                s.settimeout(None)
                return s
            except OSError:
                time.sleep(0.3)
        return None

    def _recv(self, s: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("worker closed")
            buf += chunk
        return bytes(buf)

    def _run(self) -> None:
        s = self._connect()
        if s is None:
            if not self._stop.is_set():
                self.status, self.error = "오류", "워커에 연결하지 못했습니다. " + self._log_tail()
            return
        self.sock = s
        self.status = "연결됨"
        try:
            while not self._stop.is_set():
                hl, jl, bl = struct.unpack("<III", self._recv(s, 12))
                head = json.loads(self._recv(s, hl))
                jpeg = self._recv(s, jl) if jl else b""
                blob = self._recv(s, bl) if bl else b""
                self._handle(head, jpeg, blob)
        except (OSError, ConnectionError, ValueError):
            if not self._stop.is_set():
                self.status, self.error = "오류", "워커가 종료됐습니다. " + self._log_tail()

    def _handle(self, head: dict, jpeg: bytes, blob: bytes) -> None:
        kind = head.get("type")
        if kind == "status":
            self.status = head["text"]
        elif kind == "error":
            self.status, self.error = "오류", head["text"]
        elif kind == "hello":
            self.faces = np.array(head["faces"], np.int32)
            self.device = head.get("device", "")
            self.status = f"실행 중 · {self.device} · 모델 로딩 {head.get('load_s')}s"
        elif kind == "frame":
            verts = np.frombuffer(blob, np.float32).reshape(-1, 778, 3) if blob else np.zeros((0, 778, 3))
            for i, h in enumerate(head["hands"]):
                h["verts"] = verts[i]
                for k in ("kp2d", "kp3d", "cam_t", "hand_pose"):
                    h[k] = np.array(h[k], np.float32)
            head["jpeg"] = jpeg
            now = time.monotonic()
            with self.lock:
                self.frame = head
                self.seq += 1
                self._rx_times = [t for t in self._rx_times if now - t < 1.0] + [now]

    def _log_tail(self) -> str:
        try:
            with open(LOG, encoding="utf-8", errors="replace") as fh:
                lines = [ln.strip() for ln in fh if ln.strip() and "WARNING: You are using a MANO" not in ln]
            return " / ".join(lines[-3:])
        except OSError:
            return ""

    # ------------------------------------------------------------ reading
    def latest(self) -> tuple[dict | None, int]:
        with self.lock:
            return self.frame, self.seq

    def rx_fps(self) -> float:
        with self.lock:
            return float(len(self._rx_times))

    def pick(self, frame: dict | None) -> dict | None:
        """The right hand (the worker only reports right hands for now)."""
        if not frame:
            return None
        hands = [h for h in frame["hands"] if h["is_right"] == 1]
        return hands[0] if hands else None


def flexion(kp3d: np.ndarray) -> list[float]:
    """Total bend of each finger in degrees: the sum of the angles between
    consecutive bones from the wrist to the tip. 0 = straight, ~250 = fist."""
    out = []
    for _, chain in FINGERS:
        p = kp3d[chain]
        v = np.diff(p, axis=0)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        cos = np.clip((v[:-1] * v[1:]).sum(axis=1), -1, 1)
        out.append(float(np.degrees(np.arccos(cos)).sum()))
    return out
