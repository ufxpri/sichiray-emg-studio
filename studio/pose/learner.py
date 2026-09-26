"""EMG -> hand pose, learned on the spot from the camera: the facade the UI talks to.

Collect  each camera frame with a right hand -> (unfiltered EMG window, finger pose)
Train    snapshot on the main thread -> Trainer on a worker thread -> result back
         on the main thread through a queued signal (no shared mutable state)
Predict  live filtered EMG -> features -> model -> smoothed pose, only while the
         filter settings match the ones the model was trained with

The target is WiLoR's finger articulation (15 joints x axis-angle, wrist-relative).
Wrist orientation towards the camera depends on how the arm is held, which forearm
EMG cannot see; the EMG hand borrows the camera's for display.
"""
from __future__ import annotations

import threading
from typing import Protocol

import numpy as np
from PySide6.QtCore import QObject, Signal

from .. import config
from ..core.settings import FilterSpec
from ..hand.camera_link import HandFrame
from ..hand.mano_np import get_mano
from .collector import FrameCollector
from .dataset import PoseDataset
from .features import LONG_S, features
from .models import PoseModel
from .trainer import MIN_SAMPLES, Trainer, TrainResult

SMOOTH = 0.3   # EMA weight of each new prediction at 30 Hz


class EmgSource(Protocol):
    @property
    def fs(self) -> float: ...

    @property
    def mode(self) -> str | None: ...

    @property
    def filter(self) -> FilterSpec: ...

    def recent_raw(self, n: int) -> tuple[np.ndarray, np.ndarray]: ...

    def recent_filtered(self, n: int) -> np.ndarray: ...


class FrameSource(Protocol):
    @property
    def running(self) -> bool: ...

    def latest(self) -> tuple[HandFrame | None, int]: ...


class PoseLearner(QObject):
    _finished = Signal(object)   # TrainResult or Exception, emitted from the training thread

    def __init__(self, emg: EmgSource, frames: FrameSource,
                 data_path: str = config.POSE_DATA_PATH, model_path: str = config.POSE_MODEL_PATH) -> None:
        super().__init__()
        self._emg, self._frames = emg, frames
        self._data_path, self._model_path = data_path, model_path
        self.dataset, self.message = PoseDataset.load(data_path)
        self.model = PoseModel.load(model_path)
        self._collector = FrameCollector()
        self._collecting = self._predicting = self._training = False
        self._cam_seq = -1
        self._pred: np.ndarray | None = None
        self._wrist = (np.array([np.pi / 2, 0, 0]), np.zeros(10))   # last camera orientation, shape
        self._finished.connect(self._on_finished)

    # ------------------------------------------------------------ state
    @property
    def collecting(self) -> bool:
        return self._collecting

    @property
    def predicting(self) -> bool:
        return self._predicting

    @property
    def training(self) -> bool:
        return self._training

    @property
    def report(self) -> str:
        return self.model.report if self.model else ""

    @property
    def count(self) -> int:
        return self.dataset.count

    def seconds(self) -> float:
        return self.dataset.seconds()

    def filter_matches(self) -> bool:
        return self.model is None or self.model.filter_sig == self._emg.filter.signature()

    # ------------------------------------------------------------ commands (main thread)
    def start_collecting(self) -> str | None:
        """-> why it cannot start, or None."""
        if self._training:
            return "학습이 끝난 뒤에 수집하세요."
        if not self._frames.running:
            return "카메라 손 인식 창에서 먼저 '시작'을 누르세요."
        if self._emg.mode is None:
            return "밴드가 연결되지 않았습니다."
        self.dataset.new_session()
        self._collector.reset()
        self._cam_seq = self._frames.latest()[1]
        self._collecting = True
        self.message = ""
        return None

    def stop_collecting(self) -> None:
        if self._collecting:
            self._collecting = False
            self.dataset.save(self._data_path)

    def clear(self) -> str | None:
        if self._training or self._collecting:
            return "수집이나 학습 중에는 지울 수 없습니다."
        self.dataset.clear()
        self.dataset.save(self._data_path)
        self.message = "데이터를 지웠습니다."
        return None

    def set_predicting(self, on: bool) -> None:
        self._predicting = on and self.model is not None
        self._pred = None

    def train(self, kind: str = "ridge") -> str | None:
        if self._training:
            return "이미 학습 중입니다."
        if self._collecting:
            return "수집을 멈춘 뒤에 학습하세요."
        if self.count < MIN_SAMPLES:
            return f"샘플이 부족합니다 ({self.count}개). 최소 {MIN_SAMPLES}개, 권장 1,000개 이상(약 1분)."
        snap, spec = self.dataset.snapshot(), self._emg.filter   # copies taken on this thread
        self._training = True
        self.message = "학습 중…"

        def work() -> None:
            try:
                self._finished.emit(Trainer(get_mano()).train(snap, spec, kind))
            except Exception as e:   # reported in the window, not a crash
                self._finished.emit(e)

        threading.Thread(target=work, daemon=True).start()
        return None

    def train_sync(self, kind: str = "ridge") -> TrainResult:
        """Same as train() without a thread (scripts, tests)."""
        result = Trainer(get_mano()).train(self.dataset.snapshot(), self._emg.filter, kind)
        self._on_finished(result)
        return result

    def _on_finished(self, result) -> None:
        self._training = False
        if isinstance(result, Exception):
            self.message = f"학습 실패: {result}"
            return
        self.model = result.model
        self.model.save(self._model_path)
        self._pred = None
        self.message = "학습 완료. 'EMG 예측'을 켜면 왼쪽 손이 움직입니다."

    # ------------------------------------------------------------ clock (main thread, from the hub)
    def tick(self) -> None:
        frame, seq = self._frames.latest()
        if frame is not None and seq != self._cam_seq:
            self._cam_seq = seq
            hand = frame.right_hand()
            if hand is not None:
                self._wrist = (hand.global_orient.astype(float), hand.betas.astype(float))
                if self._collecting:
                    self._collector.add(frame.t_capture, hand.hand_pose)
        if self._collecting:
            msg = self._collector.drain(self._emg, self.dataset)
            if msg:
                self.message = msg
        if self._predicting and self.model is not None:
            self._predict()

    def _predict(self) -> None:
        m, fs = self.model, self._emg.fs
        if not self.filter_matches():
            self._pred = None
            self.message = (f"필터 설정이 학습 때와 다릅니다 (학습: {m.filter_desc}). "
                            "설정을 되돌리거나 같은 데이터로 다시 학습하세요.")
            return
        if abs(m.fs - fs) > 1:
            self._pred = None
            self.message = f"모델은 {m.fs:g} Hz로 학습됐습니다. 출력 모드를 맞추세요."
            return
        n = int(LONG_S * fs)
        x = self._emg.recent_filtered(n)
        if len(x) < n:
            return
        if self.message.startswith(("필터 설정이", "모델은")):
            self.message = ""
        y = m.predict(features(x, fs)).reshape(15, 3)
        self._pred = y if self._pred is None else (1 - SMOOTH) * self._pred + SMOOTH * y

    # ------------------------------------------------------------ output
    def predicted_hand(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(wrist orientation borrowed from the camera, predicted finger pose, hand shape)."""
        if not self._predicting or self._pred is None:
            return None
        go, betas = self._wrist
        return go, self._pred.copy(), betas
