"""Rest / maximum-contraction calibration as a small state machine."""
from __future__ import annotations

import numpy as np

SETTLE_S = 0.3   # skip the start while the envelope settles


class Calibrator:
    def __init__(self) -> None:
        self.kind: str | None = None
        self._need = 0
        self._fs = 1.0
        self._env: list[np.ndarray] = []

    @property
    def active(self) -> bool:
        return self.kind is not None

    def start(self, kind: str, seconds: float, fs: float) -> None:
        """kind: 'rest' or 'mvc'."""
        self.kind, self._need, self._fs, self._env = kind, int(seconds * fs), fs, []

    def feed(self, env: np.ndarray) -> tuple[float, tuple[float, ...] | None]:
        """Add envelope rows. -> (progress 0..1, per-channel level once done)."""
        if not self.active:
            return 0.0, None
        self._env.append(env)
        got = sum(len(a) for a in self._env)
        if got < self._need:
            return got / self._need, None
        e = np.vstack(self._env)
        e = e[min(len(e) // 4, int(SETTLE_S * self._fs)):]
        level = e.mean(axis=0) if self.kind == "rest" else np.percentile(e, 95, axis=0)
        self.kind = None
        return 1.0, tuple(float(v) for v in level)
