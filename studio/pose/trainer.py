"""DatasetSnapshot + filter -> trained PoseModel + an honest report. Pure: no Qt, no
shared state, so it can run on any thread."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.filters import filter_offline
from ..core.settings import FilterSpec
from ..hand.mano_np import Mano
from ..hand.skeleton import FINGERS
from .dataset import DatasetSnapshot
from .evaluate import (chance_test, detrend, finger_error, fit_on_head, gain_pct, n_validation, pose_error,
                       select_ridge)
from .features import LONG_S, features
from .models import REGRESSORS, PoseModel, RidgeRegressor

MIN_SAMPLES = 100
P_SIGNIFICANT = 0.05


@dataclass(frozen=True)
class TrainResult:
    model: PoseModel
    label: str
    n: int
    n_val: int
    err: float           # validation joint error, degrees
    base: float          # same for 'always the average hand'
    fast_gain: float     # ridge gain on detrended data, %
    p: float             # chance of reaching fast_gain with mismatched labels
    chance95: float
    n_shift: int
    seconds: float
    finger: tuple[np.ndarray, np.ndarray] | None   # per-finger bend error: model, average hand

    @property
    def gain(self) -> float:
        return gain_pct(self.err, self.base)

    @property
    def significant(self) -> bool:
        return self.fast_gain > 0 and self.p < P_SIGNIFICANT

    def verdict(self) -> str:
        v = (f"우연보다 높음 (p<{max(self.p, 1 / self.n_shift):.2f})" if self.significant else
             f"우연과 구분되지 않음 (p={self.p:.2f}) — EMG가 손 모양을 설명한다고 볼 수 없음")
        return v + (" · 데이터가 1분 미만이라 판정이 불안정" if self.seconds < 60 else "")

    def format_ko(self) -> str:
        fingers = ("  ".join(f"{name} {a:.0f}/{b:.0f}" for (name, _), a, b in zip(FINGERS, *self.finger))
                   if self.finger is not None else "-")
        return (f"{self.label} · 샘플 {self.n:,} (검증 = 마지막 {self.n_val}개)\n"
                f"검증 관절 오차 {self.err:.1f}°  /  평균 손으로 찍기 {self.base:.1f}°  →  {self.gain:+.0f}%\n"
                f"빠른 변화만(5초 이동평균 제거): 릿지 {self.fast_gain:+.0f}% / 우연 95%선 {self.chance95:+.0f}% "
                f"(라벨 시간 이동 {self.n_shift}회)\n"
                f"→ {self.verdict()}\n"
                f"손가락 굽힘 오차° (모델/평균 손): {fingers}\n"
                f"필터: {self.model.filter_desc}")


class Trainer:
    def __init__(self, mano: Mano | None = None) -> None:
        self.mano = mano

    @staticmethod
    def feature_matrix(snap: DatasetSnapshot, spec: FilterSpec) -> np.ndarray:
        """Re-filter every stored window with `spec`, keep the context, take features."""
        keep = int(LONG_S * snap.fs)
        return np.array([features(filter_offline(w, spec, snap.fs)[-keep:], snap.fs) for w in snap.raw])

    def train(self, snap: DatasetSnapshot, spec: FilterSpec, kind: str) -> TrainResult:
        X, Y = self.feature_matrix(snap, spec), snap.Y
        n_val = n_validation(len(X))
        if kind == "ridge":
            err, alpha, P, _ = select_ridge(X, Y, n_val)
            make = lambda: RidgeRegressor(alpha)
        else:
            make = REGRESSORS[kind]
            err, _, P = fit_on_head(make(), X, Y, n_val)
        ymu = Y[:-n_val].mean(axis=0)
        base = pose_error(np.broadcast_to(ymu, Y[-n_val:].shape), Y[-n_val:])
        fast_gain, p, c95, n_shift = chance_test(detrend(X, snap.T, snap.S), detrend(Y, snap.T, snap.S), n_val)
        reg = make()
        model = PoseModel.train(reg, X, Y, snap.fs, spec.signature(), spec.describe())
        result = TrainResult(model, reg.label, len(X), n_val, err, base, fast_gain, p, c95, n_shift,
                             snap.seconds, finger_error(P, Y[-n_val:], ymu, self.mano))
        model.report = result.format_ko()
        return result

