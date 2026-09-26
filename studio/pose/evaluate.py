"""How good is a pose model, honestly.

Validation is the last 20% in time, never shuffled: windows from the same moment
are nearly identical, and a random split once gave this project an 86% that was
really 0.3% (NOTES §8). Every result is shown next to the laziest model -
'always the average hand' - and next to chance.

Chance: the same fit with the labels slid along time, which keeps each signal's
own rhythm but breaks which EMG belongs to which hand. That test alone is fooled
by slow drift shared by accident (electrode, fatigue, posture): unrelated random
walks came out 'real' in 100% of trials. After removing a 5 s moving average from
both sides it was 0%, with full power on a real fast relation (NOTES §26).
"""
from __future__ import annotations

import numpy as np

from ..hand.mano_np import Mano, rodrigues
from ..hand.skeleton import flexion
from .models import RidgeRegressor, Standardizer

VAL_FRAC = 0.2
DETREND_S = 5.0
ALPHAS = (0.1, 1, 10, 100, 1000)
N_SHIFT = 40


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-joint rotation difference in degrees. a, b: (..., 15, 3) axis-angle."""
    tr = np.einsum("...ij,...ij->...", rodrigues(a), rodrigues(b))   # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def pose_error(P: np.ndarray, Y: np.ndarray) -> float:
    return float(geodesic_deg(P.reshape(-1, 15, 3), Y.reshape(-1, 15, 3)).mean())


def n_validation(n: int) -> int:
    return max(20, int(n * VAL_FRAC))


def fit_on_head(reg, X: np.ndarray, Y: np.ndarray, n_val: int) -> tuple[float, float, np.ndarray]:
    """Fit on all but the last n_val rows, predict those. -> (error, baseline error, predictions)."""
    Xtr, Ytr, Xva, Yva = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
    sc = Standardizer.fit(Xtr, Ytr)
    reg.fit(sc.x(Xtr), sc.y(Ytr))
    P = reg.predict(sc.x(Xva)) + sc.ymu
    return pose_error(P, Yva), pose_error(np.broadcast_to(sc.ymu, Yva.shape), Yva), P


def gain_pct(err: float, base: float) -> float:
    return (1 - err / base) * 100 if base > 0 else 0.0


def select_ridge(X: np.ndarray, Y: np.ndarray, n_val: int) -> tuple[float, float, np.ndarray, float]:
    """Best ridge alpha on the validation tail. -> (error, alpha, predictions, gain %)."""
    best = None
    for alpha in ALPHAS:
        err, base, P = fit_on_head(RidgeRegressor(alpha), X, Y, n_val)
        if best is None or err < best[0]:
            best = (err, alpha, P, gain_pct(err, base))
    return best


def detrend(A: np.ndarray, T: np.ndarray, S: np.ndarray, win_s: float = DETREND_S) -> np.ndarray:
    """Subtract a centred moving average of +-win_s/2 seconds, per session. Rows in time order."""
    out = np.empty_like(A, dtype=float)
    for s in np.unique(S):
        idx = np.flatnonzero(S == s)
        t, a = T[idx], A[idx].astype(float)
        cs = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
        lo = np.searchsorted(t, t - win_s / 2)
        hi = np.searchsorted(t, t + win_s / 2, side="right")
        out[idx] = a - (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    return out


def chance_test(X: np.ndarray, Y: np.ndarray, n_val: int, n_shift: int = N_SHIFT) -> tuple[float, float, float, int]:
    """Ridge gain on (X, Y) against the same with Y slid along time. Use on detrended data.
    -> (gain %, p, chance 95th percentile %, shifts used)."""
    gain = select_ridge(X, Y, n_val)[3]
    shifts = np.unique((np.linspace(0.1, 0.9, n_shift) * len(Y)).astype(int))
    chance = np.array([select_ridge(X, np.roll(Y, k, axis=0), n_val)[3] for k in shifts])
    return gain, float((chance >= gain).mean()), float(np.percentile(chance, 95)), len(shifts)


def finger_error(P: np.ndarray, Y: np.ndarray, ymu: np.ndarray, mano: Mano | None):
    """Mean absolute finger-bend error (deg) of the model and of the average hand, per finger.
    Not correlation: over a short stretch two unrelated signals correlate by chance
    (demo EMG vs a video gave r = 0.5-0.8 at 0% gain)."""
    if mano is None:
        return None
    go = np.zeros(3)
    flex = lambda p: flexion(mano.forward(go, np.reshape(p, (15, 3)))[1])
    fy = np.array([flex(y) for y in Y])
    return np.abs(np.array([flex(p) for p in P]) - fy).mean(axis=0), np.abs(flex(ymu) - fy).mean(axis=0)
