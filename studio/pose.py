"""EMG -> hand pose, learned on the spot from the camera.

Collect: every camera frame with a right hand becomes one sample. The target is
WiLoR's finger articulation (15 joints x axis-angle = 45 numbers, relative to
the wrist). The input is the EMG window that ends at the frame's capture time,
both on the perf_counter clock.

What is stored is the *unfiltered* window (raw minus centre, physical channel
order, int16) with a 1 s pre-roll. Training runs the current global filter over
it, and the model remembers that filter. Prediction uses the live filtered
stream, so it only runs while the filter settings match the model's - otherwise
a changed rotation or cut-off would silently feed the model a different signal.
The same recording can be retrained under different filters to compare them. Wrist orientation relative to the
camera is not a target: it depends on how the arm is held, which the forearm
EMG cannot see. The EMG hand borrows the camera's wrist orientation for display.

Train: features from each window, then ridge regression (closed form, instant)
or a small MLP. Validation is the last 20% in time, never shuffled: windows
from the same moment are nearly identical, and a random split once gave this
project an 86% that was really 0.3% (NOTES §8). The report always shows the
error of the laziest possible model - predicting the average hand - next to the
real one, so 'better than nothing' is visible at a glance.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from .camera import FINGERS
from .mano_np import Mano, rodrigues
from .pipeline import describe, filter_offline, signature

WIN_S = 0.2     # main feature window
LONG_S = 0.5    # longer context (sustained contraction)
PREROLL_S = 1.0  # stored before the feature context so offline filters settle (60 Hz notch Q30: ~0.8 s)
RAW_S = PREROLL_S + LONG_S
VAL_FRAC = 0.2
FEAT_VERSION = 2   # 2: model carries its filter signature


def features(x: np.ndarray, fs: float) -> np.ndarray:
    """x: (n >= LONG_S*fs, 8) filtered EMG ending at the target time -> (64,)."""
    w = max(4, int(WIN_S * fs))
    s = x[-w:]
    q = max(1, w // 4)
    eps = 0.1
    rms = np.sqrt((s ** 2).mean(axis=0))
    sub = [np.sqrt((s[i * q:(i + 1) * q] ** 2).mean(axis=0)) for i in range(4)]
    wl = np.abs(np.diff(s, axis=0)).mean(axis=0)
    sign = np.signbit(s[:-1]) != np.signbit(s[1:])
    zc = (sign & (np.abs(np.diff(s, axis=0)) > 1.0)).mean(axis=0)
    longr = np.sqrt((x ** 2).mean(axis=0))
    return np.concatenate([np.log(rms + eps), *[np.log(v + eps) for v in sub], np.log(wl + eps), zc,
                           np.log(longr + eps)])


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-joint rotation difference in degrees. a, b: (..., 15, 3) axis-angle."""
    Ra, Rb = rodrigues(a), rodrigues(b)
    tr = np.einsum("...ij,...ij->...", Ra, Rb)   # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


DETREND_S = 5.0


def detrend(A: np.ndarray, T: np.ndarray, S: np.ndarray, win_s: float = DETREND_S) -> np.ndarray:
    """Subtract a centred moving average of +-win_s/2 seconds, per recording session.
    Rows must be in time order. Removes the slow drifts (electrode, fatigue, posture)
    that two unrelated signals share by accident."""
    out = np.empty_like(A, dtype=float)
    for s in np.unique(S):
        idx = np.flatnonzero(S == s)
        t, a = T[idx], A[idx].astype(float)
        cs = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
        lo = np.searchsorted(t, t - win_s / 2)
        hi = np.searchsorted(t, t + win_s / 2, side="right")
        out[idx] = a - (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    return out


def chance_test(X: np.ndarray, Y: np.ndarray, n_val: int, fit, n_shift: int = 40):
    """Gain of `fit` on (X, Y) against the same fit with Y slid along time.
    Sliding keeps each signal's own rhythm but breaks which EMG belongs to which hand.
    Only trustworthy on detrended data: with slow drift in both signals, the plain
    version called unrelated random walks 'real' in 100% of trials (0% after detrending).
    -> (gain %, p, chance 95th percentile %, shifts used)."""
    gain = fit(X, Y, n_val)[3]
    shifts = np.unique((np.linspace(0.1, 0.9, n_shift) * len(Y)).astype(int))
    chance = np.array([fit(X, np.roll(Y, k, axis=0), n_val)[3] for k in shifts])
    return gain, float((chance >= gain).mean()), float(np.percentile(chance, 95)), len(shifts)


def flexion_from_joints(j: np.ndarray) -> np.ndarray:
    out = []
    for _, chain in FINGERS:
        v = np.diff(j[chain], axis=0)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        out.append(np.degrees(np.arccos(np.clip((v[:-1] * v[1:]).sum(axis=1), -1, 1))).sum())
    return np.array(out)


class PoseLearner:
    def __init__(self, hub) -> None:
        from .hub import DATA_DIR
        self.hub = hub
        self.path_data = os.path.join(DATA_DIR, "pose_dataset.npz")
        self.path_model = os.path.join(DATA_DIR, "pose_model.npz")
        self.mano = Mano() if Mano.available() else None
        self.collecting = False
        self.predicting = False
        self.training = False
        self.message = ""
        self.report = ""
        self.cam_seq = -1
        self.pending: list[tuple] = []
        self.session = 0
        self.fs: float | None = None
        self.win: list[np.ndarray] = []
        self.Y: list[np.ndarray] = []
        self.T: list[float] = []
        self.S: list[int] = []
        self.model: dict | None = None
        self.pred: np.ndarray | None = None   # smoothed (15, 3)
        self.last_go = np.array([np.pi / 2, 0, 0])
        self.last_betas = np.zeros(10)
        self.last_cam_t = 0.0
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            d = np.load(self.path_data)
            if "raw" not in d.files:   # older format stored filtered windows only
                self.message = "이전 형식의 수집 데이터(필터 후 저장)는 불러오지 않았습니다. 다시 수집하세요."
                raise KeyError("old format")
            self.win = list(d["raw"])
            self.Y = list(d["Y"])
            self.T = list(d["T"])
            self.S = list(d["S"])
            self.fs = float(d["fs"])
            self.session = int(max(self.S)) + 1 if self.S else 0
        except (OSError, KeyError, ValueError):
            pass
        try:
            m = np.load(self.path_model, allow_pickle=True)
            if int(m["version"]) == FEAT_VERSION:
                self.model = {k: m[k] for k in m.files}
                self.report = str(m["report"])
        except (OSError, KeyError, ValueError):
            pass

    def save_data(self) -> None:
        if not self.win:
            if os.path.exists(self.path_data):
                os.remove(self.path_data)
            return
        os.makedirs(os.path.dirname(self.path_data), exist_ok=True)
        np.savez_compressed(self.path_data, raw=np.array(self.win, np.int16), Y=np.array(self.Y, np.float32),
                            T=np.array(self.T), S=np.array(self.S), fs=self.fs)

    def clear(self) -> None:
        self.win, self.Y, self.T, self.S = [], [], [], []
        self.fs = None
        self.session = 0
        self.save_data()
        self.message = "데이터를 지웠습니다."

    @property
    def count(self) -> int:
        return len(self.win)

    def seconds(self) -> float:
        if not self.T:
            return 0.0
        T, S = np.array(self.T), np.array(self.S)
        return float(sum(T[S == s].max() - T[S == s].min() for s in np.unique(S)))

    # ------------------------------------------------------------ collection
    def set_collecting(self, on: bool) -> None:
        if on and not self.collecting:
            self.session += 1 if self.win else 0
            self.pending.clear()
            self.cam_seq = self.hub.camera.latest()[1]
        if not on and self.collecting:
            self.save_data()
        self.collecting = on

    def _right_hand(self, frame: dict | None) -> dict | None:
        if not frame:
            return None
        hands = [h for h in frame["hands"] if h["is_right"] == 1]
        return hands[0] if hands else None

    def tick(self) -> None:
        cam = self.hub.camera
        frame, seq = cam.latest()
        if seq != self.cam_seq and frame is not None:
            self.cam_seq = seq
            hand = self._right_hand(frame)
            if hand is not None:
                self.last_go = np.array(hand["global_orient"], float)
                self.last_betas = np.array(hand["betas"], float)
                self.last_cam_t = time.monotonic()
                if self.collecting:
                    self.pending.append((frame["t_ns"] / 1e9, np.asarray(hand["hand_pose"], np.float32)))
        if self.collecting:
            self._drain()
        if self.predicting and self.model is not None:
            self._predict()

    def _drain(self) -> None:
        """Turn pending camera frames into samples once EMG up to their time has arrived."""
        fs = self.hub.fs
        if self.hub.mode is None or not self.pending:
            return
        if self.fs is not None and abs(self.fs - fs) > 1:
            self.message = f"기존 데이터는 {self.fs:g} Hz입니다. 모드를 맞추거나 데이터를 지우세요."
            self.pending.clear()
            return
        L = int(RAW_S * fs)
        tt = self.hub.t.last(int(4 * fs))[:, 0]
        x = self.hub.cen.last(len(tt))
        if len(tt) < L or len(x) != len(tt):
            return
        keep = []
        for t, pose in self.pending:
            if t > tt[-1]:                       # EMG for this moment not here yet
                if t - tt[-1] < 1.0:
                    keep.append((t, pose))
                continue
            i = int(np.searchsorted(tt, t, side="right"))
            if i < L or tt[i - 1] < t - 0.05:     # too old for the buffer, or a gap in EMG
                continue
            self.win.append(np.round(x[i - L:i]).astype(np.int16))
            self.Y.append(pose)
            self.T.append(t)
            self.S.append(self.session)
            self.fs = fs
        self.pending = keep
        self.message = ""

    # ------------------------------------------------------------ training
    def train_async(self, kind: str = "ridge") -> None:
        if self.training:
            return
        if self.count < 100:
            self.message = f"샘플이 부족합니다 ({self.count}개). 최소 100개, 권장 1,000개 이상(약 1분)."
            return
        self.training = True
        self.message = "학습 중…"
        threading.Thread(target=self._train, args=(kind,), daemon=True).start()

    def _train(self, kind: str) -> None:
        import copy
        try:
            fs = self.fs
            settings = copy.deepcopy(self.hub.settings)   # the filter this model will expect
            order = np.argsort(np.array(self.T))
            W = np.array(self.win)[order]
            Y = np.array(self.Y, np.float64)[order].reshape(len(order), -1)
            keep = int(LONG_S * fs)
            X = np.array([features(filter_offline(w, settings, fs)[-keep:], fs) for w in W])
            n_val = max(20, int(len(X) * VAL_FRAC))
            Xtr, Ytr, Xva, Yva = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
            mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
            ymu = Ytr.mean(axis=0)
            if kind == "ridge":
                err, alpha, Pva, _ = self._ridge_select(X, Y, n_val)
                detail = f"릿지 α={alpha:g}"
            else:
                from sklearn.neural_network import MLPRegressor
                net = MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-3, max_iter=400,
                                   early_stopping=True, random_state=0)
                net.fit((Xtr - mu) / sd, Ytr - ymu)
                Pva = net.predict((Xva - mu) / sd) + ymu
                err = geodesic_deg(Pva.reshape(-1, 15, 3), Yva.reshape(-1, 15, 3)).mean()
                detail = "MLP 128×128"
            base = geodesic_deg(np.broadcast_to(ymu, Yva.shape).reshape(-1, 15, 3), Yva.reshape(-1, 15, 3)).mean()
            finger = self._finger_err(Pva, Yva, Ytr.mean(axis=0))
            # final model on everything
            mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
            ymu = Y.mean(axis=0)
            model = dict(version=FEAT_VERSION, kind=kind, fs=fs, mu=mu, sd=sd, ymu=ymu,
                         sig=signature(settings), filt=describe(settings))
            if kind == "ridge":
                model["W"] = self._ridge((X - mu) / sd, Y - ymu, alpha)
            else:
                net.fit((X - mu) / sd, Y - ymu)
                model["net"] = np.array(net, dtype=object)
            gain = (1 - err / base) * 100 if base > 0 else 0
            T = np.array(self.T)[order]
            S = np.array(self.S)[order]
            fast_gain, p, c95, n_shift = chance_test(detrend(X, T, S), detrend(Y, T, S), n_val, self._ridge_select)
            if fast_gain <= 0 or p >= 0.05:
                verdict = f"우연과 구분되지 않음 (p={p:.2f}) — EMG가 손 모양을 설명한다고 볼 수 없음"
            else:
                verdict = f"우연보다 높음 (p<{max(p, 1 / n_shift):.2f})"
            if self.seconds() < 60:
                verdict += " · 데이터가 1분 미만이라 판정이 불안정"
            rs = ("  ".join(f"{name} {a:.0f}/{b:.0f}" for (name, _), a, b in zip(FINGERS, *finger))
                  if finger is not None else "-")
            self.report = (f"{detail} · 샘플 {len(X):,} (검증 = 마지막 {n_val}개)\n"
                           f"검증 관절 오차 {err:.1f}°  /  평균 손으로 찍기 {base:.1f}°  →  {gain:+.0f}%\n"
                           f"빠른 변화만(5초 이동평균 제거): 릿지 {fast_gain:+.0f}% / 우연 95%선 {c95:+.0f}% "
                           f"(라벨 시간 이동 {n_shift}회)\n"
                           f"→ {verdict}\n"
                           f"손가락 굽힘 오차° (모델/평균 손): {rs}\n"
                           f"필터: {describe(settings)}")
            model["report"] = self.report
            self.model = model
            np.savez(self.path_model, **model)
            self.message = "학습 완료. 'EMG 예측'을 켜면 왼쪽 손이 움직입니다."
            self.pred = None
        except Exception as e:   # surfaced in the window, not a crash
            self.message = f"학습 실패: {e}"
        finally:
            self.training = False

    def _ridge_select(self, X: np.ndarray, Y: np.ndarray, n_val: int):
        """Fit on all but the last n_val rows, pick alpha on those. -> (err, alpha, P_val, gain %)."""
        Xtr, Ytr, Xva, Yva = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
        mu, sd, ymu = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6, Ytr.mean(axis=0)
        best = None
        for alpha in (0.1, 1, 10, 100, 1000):
            P = ((Xva - mu) / sd) @ self._ridge((Xtr - mu) / sd, Ytr - ymu, alpha) + ymu
            err = geodesic_deg(P.reshape(-1, 15, 3), Yva.reshape(-1, 15, 3)).mean()
            if best is None or err < best[0]:
                best = (err, alpha, P)
        base = geodesic_deg(np.broadcast_to(ymu, Yva.shape).reshape(-1, 15, 3), Yva.reshape(-1, 15, 3)).mean()
        return best[0], best[1], best[2], ((1 - best[0] / base) * 100 if base > 0 else 0.0)

    @staticmethod
    def _ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
        return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)

    def _finger_err(self, P: np.ndarray, Y: np.ndarray, ymu: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Mean absolute finger-bend error (deg) of the model and of 'always the average hand'.
        Not correlation: over a short validation stretch of a slowly moving hand, two unrelated
        signals easily correlate by chance (demo EMG vs a video gave r = 0.5-0.8 at 0% gain)."""
        if self.mano is None:
            return None
        go = np.zeros(3)
        flex = lambda p: flexion_from_joints(self.mano.forward(go, p.reshape(15, 3))[1])
        fp = np.array([flex(p) for p in P])
        fy = np.array([flex(y) for y in Y])
        fm = flex(ymu)
        return np.abs(fp - fy).mean(axis=0), np.abs(fm - fy).mean(axis=0)

    # ------------------------------------------------------------ prediction
    def filter_matches(self) -> bool:
        return self.model is None or str(self.model["sig"]) == signature(self.hub.settings)

    def _predict(self) -> None:
        m, fs = self.model, self.hub.fs
        if not self.filter_matches():
            self.pred = None
            self.message = ("필터 설정이 학습 때와 다릅니다 "
                            f"(학습: {m['filt']}). "
                            "설정을 되돌리거나 같은 데이터로 다시 학습하세요.")
            return
        if self.message.startswith("필터 설정이"):
            self.message = ""
        if abs(float(m["fs"]) - fs) > 1:
            self.message = f"모델은 {float(m['fs']):g} Hz로 학습됐습니다. 출력 모드를 맞추세요."
            return
        x = self.hub.filt.last(int(LONG_S * fs))
        if len(x) < int(LONG_S * fs):
            return
        f = (features(x, fs) - m["mu"]) / m["sd"]
        if m["kind"] == "ridge":
            y = f @ m["W"] + m["ymu"]
        else:
            y = m["net"].item().predict(f[None])[0] + m["ymu"]
        y = y.reshape(15, 3)
        self.pred = y if self.pred is None else 0.7 * self.pred + 0.3 * y
