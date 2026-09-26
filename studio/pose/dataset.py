"""Collected (EMG window, hand pose) pairs.

Stored unfiltered - raw minus centre, physical channel order, int16, with a 1 s
pre-roll - so the same recording can be retrained under any filter setting.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSnapshot:
    """An immutable copy, in time order, safe to hand to a training thread."""
    raw: np.ndarray    # (N, L, 8) int16
    Y: np.ndarray      # (N, 45) hand pose, axis-angle
    T: np.ndarray      # (N,) perf_counter seconds of the camera frame
    S: np.ndarray      # (N,) session number
    fs: float
    seconds: float


class PoseDataset:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._raw: list[np.ndarray] = []
        self._Y: list[np.ndarray] = []
        self._T: list[float] = []
        self._S: list[int] = []
        self.fs: float | None = None
        self.session = 0

    # ------------------------------------------------------------ building
    def new_session(self) -> None:
        with self._lock:
            if self._raw:
                self.session += 1

    def accepts(self, fs: float) -> bool:
        return self.fs is None or abs(self.fs - fs) <= 1

    def append(self, raw: np.ndarray, pose: np.ndarray, t: float, fs: float) -> None:
        with self._lock:
            self._raw.append(np.round(raw).astype(np.int16))
            self._Y.append(np.asarray(pose, np.float32).reshape(-1))
            self._T.append(float(t))
            self._S.append(self.session)
            self.fs = fs

    def clear(self) -> None:
        with self._lock:
            self._raw, self._Y, self._T, self._S = [], [], [], []
            self.fs, self.session = None, 0

    # ------------------------------------------------------------ reading
    @property
    def count(self) -> int:
        with self._lock:
            return len(self._raw)

    def seconds(self) -> float:
        with self._lock:
            return self._seconds(np.array(self._T), np.array(self._S))

    @staticmethod
    def _seconds(T: np.ndarray, S: np.ndarray) -> float:
        return float(sum(T[S == s].max() - T[S == s].min() for s in np.unique(S))) if len(T) else 0.0

    def snapshot(self) -> DatasetSnapshot | None:
        with self._lock:
            if not self._raw:
                return None
            T, S = np.array(self._T), np.array(self._S)
            o = np.argsort(T, kind="stable")
            return DatasetSnapshot(np.array(self._raw)[o], np.array(self._Y, np.float64)[o], T[o], S[o],
                                   float(self.fs), self._seconds(T, S))

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        with self._lock:
            if not self._raw:
                if os.path.exists(path):
                    os.remove(path)
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, raw=np.array(self._raw, np.int16), Y=np.array(self._Y, np.float32),
                                T=np.array(self._T), S=np.array(self._S), fs=self.fs)

    @classmethod
    def load(cls, path: str) -> tuple["PoseDataset", str]:
        """-> (dataset, message for the user, empty when all is well)."""
        ds = cls()
        try:
            d = np.load(path)
        except OSError:
            return ds, ""
        if "raw" not in d.files:
            return ds, "이전 형식의 수집 데이터(필터 후 저장)는 불러오지 않았습니다. 다시 수집하세요."
        ds._raw, ds._Y = list(d["raw"]), list(d["Y"].reshape(len(d["Y"]), -1))
        ds._T, ds._S = [float(t) for t in d["T"]], [int(s) for s in d["S"]]
        ds.fs = float(d["fs"])
        ds.session = max(ds._S) + 1 if ds._S else 0
        return ds, ""
