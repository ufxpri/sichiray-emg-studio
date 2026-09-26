"""Studio side of the hand worker: starts worker/hand_worker.py in the hand venv and
receives frames, 2D/3D joints and MANO vertices over localhost TCP.

The worker lives in its own interpreter (PyTorch, numpy<2) so inference never holds
the studio's GIL. Frame timestamps are perf_counter_ns() taken in the worker right
after the camera read; on Windows that clock is system-wide, the same clock the EMG
samples are put on.

Wire format (owned by the worker, versioned by config.HAND_PROTO): u32 header len,
u32 jpeg len, u32 blob len, header JSON, JPEG, float32 vertices (778x3 per hand).
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
from dataclasses import dataclass

import numpy as np

from .. import config

N_VERTS = 778


@dataclass(frozen=True)
class HandObs:
    is_right: bool
    det_conf: float           # YOLO confidence, -1 while tracking
    bbox: tuple[float, float, float, float]
    kp2d: np.ndarray          # (21, 2) image pixels
    kp3d: np.ndarray          # (21, 3) metres, camera axes, not translated
    verts: np.ndarray         # (778, 3)
    global_orient: np.ndarray  # (3,) axis-angle, wrist relative to camera
    hand_pose: np.ndarray     # (15, 3) axis-angle, finger joints relative to parents
    betas: np.ndarray         # (10,) MANO shape

    @classmethod
    def decode(cls, h: dict, verts: np.ndarray) -> "HandObs":
        a = lambda k, shape: np.asarray(h[k], np.float32).reshape(shape)
        return cls(bool(h["is_right"]), float(h["det_conf"]), tuple(h["bbox"]), a("kp2d", (21, 2)),
                   a("kp3d", (21, 3)), verts, a("global_orient", (3,)), a("hand_pose", (15, 3)), a("betas", (10,)))


@dataclass(frozen=True)
class HandFrame:
    seq: int
    t_capture: float          # perf_counter seconds at camera read
    width: int
    height: int
    hands: tuple[HandObs, ...]
    how: str                  # "yolo" (detected) or "track" (previous keypoint box)
    cam_fps: float
    ms: dict                  # detect / wilor / total, milliseconds
    lag_ms: float             # capture -> result sent
    jpeg: bytes

    def right_hand(self) -> HandObs | None:
        return next((h for h in self.hands if h.is_right), None)


class CameraLink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._status = "정지"
        self._error = ""
        self._frame: HandFrame | None = None
        self._seq = 0
        self._rx: list[float] = []
        self.faces: np.ndarray | None = None
        self.device = ""

    # ------------------------------------------------------------ lifecycle
    @staticmethod
    def available() -> tuple[bool, str]:
        if not os.path.exists(config.HAND_PY):
            return False, (f"손 인식 환경이 없습니다: {config.HAND_PY}\n"
                           "README의 '카메라 손 인식 설치' 절을 따라 설치하세요.")
        return True, ""

    @property
    def running(self) -> bool:
        p = self._proc
        return p is not None and p.poll() is None

    def start(self, camera: int = 0, width: int = 1280, height: int = 720, fps: int = 30, video: str = "") -> None:
        self.stop()
        ok, msg = self.available()
        if not ok:
            self._set(None, "오류", msg)
            return
        os.makedirs(config.HAND_HOME, exist_ok=True)
        log = open(config.HAND_LOG, "w", encoding="utf-8", errors="replace")
        args = [config.HAND_PY, config.WORKER, "--camera", str(camera), "--width", str(width),
                "--height", str(height), "--fps", str(fps), "--port", str(config.HAND_PORT),
                "--models", config.HAND_MODELS, "--mano-out", config.MANO_NPZ]
        if video:
            args += ["--video", video]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(args, stdout=log, stderr=log, creationflags=flags)
        stop = threading.Event()
        with self._lock:
            self._proc, self._stop = proc, stop
            self._status, self._error, self._frame = "워커 시작 중", "", None
        self._thread = threading.Thread(target=self._run, args=(proc, stop), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            stop, proc, sock, thread = self._stop, self._proc, self._sock, self._thread
            self._stop = self._proc = self._sock = self._thread = None
            self._status = "정지"
        if stop:
            stop.set()
        if sock:
            try:
                sock.close()
            except OSError:
                pass
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if thread:
            thread.join(timeout=3)

    # ------------------------------------------------------------ receiving (own thread)
    def _set(self, stop: threading.Event | None, status: str, error: str | None = None) -> None:
        with self._lock:
            if stop is not None and stop is not self._stop:   # a superseded thread
                return
            self._status = status
            if error is not None:
                self._error = error

    def _connect(self, proc: subprocess.Popen, stop: threading.Event) -> socket.socket | None:
        t0 = time.time()
        while not stop.is_set() and time.time() - t0 < 60:
            if proc.poll() is not None:
                return None
            try:
                s = socket.create_connection(("127.0.0.1", config.HAND_PORT), timeout=1)
                s.settimeout(None)
                return s
            except OSError:
                time.sleep(0.3)
        return None

    @staticmethod
    def _recv(s: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("worker closed")
            buf += chunk
        return bytes(buf)

    def _run(self, proc: subprocess.Popen, stop: threading.Event) -> None:
        s = self._connect(proc, stop)
        if s is None:
            if not stop.is_set():
                self._set(stop, "오류", "워커에 연결하지 못했습니다. " + self._log_tail())
            return
        with self._lock:
            if stop is not self._stop:
                s.close()
                return
            self._sock = s
        self._set(stop, "연결됨")
        try:
            while not stop.is_set():
                hl, jl, bl = struct.unpack("<III", self._recv(s, 12))
                head = json.loads(self._recv(s, hl))
                jpeg = self._recv(s, jl) if jl else b""
                blob = self._recv(s, bl) if bl else b""
                self._handle(stop, head, jpeg, blob)
        except (OSError, ConnectionError, ValueError):
            if not stop.is_set():
                self._set(stop, "오류", "워커가 종료됐습니다. " + self._log_tail())

    def _handle(self, stop: threading.Event, head: dict, jpeg: bytes, blob: bytes) -> None:
        kind = head.get("type")
        if kind == "status":
            self._set(stop, head["text"])
        elif kind == "error":
            self._set(stop, "오류", head["text"])
        elif kind == "hello":
            if head.get("proto") != config.HAND_PROTO:
                self._set(stop, "오류", f"워커 프로토콜 {head.get('proto')} ≠ {config.HAND_PROTO}. 워커와 Studio 버전을 맞추세요.")
                stop.set()
                return
            self.faces = np.array(head["faces"], np.int32)
            self.device = head.get("device", "")
            self._set(stop, f"실행 중 · {self.device} · 모델 로딩 {head.get('load_s')}s")
        elif kind == "frame":
            verts = np.frombuffer(blob, np.float32).reshape(-1, N_VERTS, 3) if blob else np.zeros((0, N_VERTS, 3))
            hands = tuple(HandObs.decode(h, verts[i]) for i, h in enumerate(head["hands"]))
            now = time.monotonic()
            with self._lock:
                if stop is not self._stop:
                    return
                self._seq += 1
                self._frame = HandFrame(self._seq, head["t_ns"] / 1e9, head["w"], head["h"], hands, head["how"],
                                        head["cam_fps"], head["ms"], head["lag_ms"], jpeg)
                self._rx = [t for t in self._rx if now - t < 1.0] + [now]

    @staticmethod
    def _log_tail() -> str:
        try:
            with open(config.HAND_LOG, encoding="utf-8", errors="replace") as fh:
                lines = [ln.strip() for ln in fh if ln.strip() and "WARNING: You are using a MANO" not in ln]
            return " / ".join(lines[-3:])
        except OSError:
            return ""

    # ------------------------------------------------------------ reading (any thread)
    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def latest(self) -> tuple[HandFrame | None, int]:
        with self._lock:
            return self._frame, self._seq

    def rx_fps(self) -> float:
        with self._lock:
            return float(len(self._rx))
