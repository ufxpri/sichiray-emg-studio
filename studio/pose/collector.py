"""Pairs each camera frame with the EMG window that ends at its capture time.

Camera results arrive ~100 ms after capture and EMG within tens of ms, so a
frame waits in `pending` until EMG up to its time is in the buffer.
"""
from __future__ import annotations

import numpy as np

from .dataset import PoseDataset
from .features import RAW_S

BUFFER_S = 4.0    # how much recent EMG to search
WAIT_S = 1.0      # give up on a frame whose EMG has not arrived by then
GAP_S = 0.05      # EMG must exist within this of the frame time


class FrameCollector:
    def __init__(self) -> None:
        self.pending: list[tuple[float, np.ndarray]] = []

    def reset(self) -> None:
        self.pending.clear()

    def add(self, t: float, pose: np.ndarray) -> None:
        self.pending.append((t, pose))

    def drain(self, emg, dataset: PoseDataset) -> str | None:
        """Move every frame whose EMG is available into `dataset`. -> a message, or None."""
        if emg.mode is None or not self.pending:
            return None
        fs = emg.fs
        if not dataset.accepts(fs):
            self.pending.clear()
            return f"기존 데이터는 {dataset.fs:g} Hz입니다. 출력 모드를 맞추거나 데이터를 지우세요."
        L = int(RAW_S * fs)
        tt, x = emg.recent_raw(int(BUFFER_S * fs))
        if len(tt) < L:
            return None
        keep = []
        for t, pose in self.pending:
            if t > tt[-1]:                          # EMG for this moment not here yet
                if t - tt[-1] < WAIT_S:
                    keep.append((t, pose))
                continue
            i = int(np.searchsorted(tt, t, side="right"))
            if i < L or tt[i - 1] < t - GAP_S:      # too old for the buffer, or a gap in EMG
                continue
            dataset.append(x[i - L:i], pose, t, fs)
        self.pending = keep
        return None
