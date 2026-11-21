"""The gesture classifier shared by training (studio.gesture.train, PyTorch) and the live
window (numpy only, so the studio runs without PyTorch).

Pipeline, as in the research report's final model (exp 0015), made causal for live use:
  raw - 127 (physical channel order)
  -> causal band-pass 20-240 Hz + notches 60/120/180/240 Hz, restarted after every gap
  -> 200 ms windows, one decision every 100 ms
  -> divide by the channel RMS of the current band placement (from calibration)
  -> 1D CNN (3 conv + BatchNorm + ReLU, average pool, linear) with AdaBN: BatchNorm statistics
     re-estimated from the current placement's windows
  -> mean log-probability over the last 5 windows (0.4 s), past only
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

FS = 500.0
WIN = 100                   # 200 ms
STEP = 50                   # 100 ms between decisions
SMOOTH = 5                  # windows averaged (0.4 s)
HOLD_S = (0.5, 3.0)         # training windows after the hold prompt
REST_S = (0.5, 2.0)         # ... and after the rest prompt
CENTER = 127.0
MAX_STEP_S = 1.5 / FS       # a larger step between samples is a gap
BN_EPS = 1e-5
MODEL_VERSION = 1


def filter_sos(fs: float = FS) -> np.ndarray:
    sos = butter(4, [20, 240], btype="band", fs=fs, output="sos")
    notch = [tf2sos(*iirnotch(f0, 30, fs)) for f0 in (60, 120, 180, 240) if f0 < fs / 2]
    return np.vstack([sos, *notch])


def causal_filter(x: np.ndarray, t: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Filter each gap-free run forward only, starting from the steady state of its first sample."""
    out = np.zeros_like(x, dtype=np.float64)
    breaks = np.flatnonzero(np.diff(t) > MAX_STEP_S) + 1
    zi0 = sosfilt_zi(sos)
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(t)]):
        if b > a:
            out[a:b], _ = sosfilt(sos, x[a:b], axis=0, zi=zi0[:, :, None] * x[a][None, None, :])
    return out


def windows(x: np.ndarray, t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Start indices of the gap-free windows lying inside [t0, t1)."""
    i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
    s = np.arange(i0, i1 - WIN + 1, STEP)
    return s[t[s + WIN - 1] - t[s] < (WIN + 1) / FS] if len(s) else s


# ------------------------------------------------------------------ network (numpy)
def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int) -> np.ndarray:
    """x (N, C, L), w (O, C, K), 'same' padding K//2 -> (N, O, L')."""
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (k // 2, k // 2)))
    v = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)[:, :, ::stride]     # (N, C, L', K)
    return np.einsum("nclk,ock->nol", v, w, optimize=True) + b[None, :, None]


@dataclass
class BNStats:
    mean: list[np.ndarray]
    var: list[np.ndarray]


@dataclass
class GestureNet:
    conv: list[tuple[np.ndarray, np.ndarray, int]]     # (weight, bias, stride) per layer
    bn: list[tuple[np.ndarray, np.ndarray]]            # (gamma, beta) per layer
    fc: tuple[np.ndarray, np.ndarray]                  # (weight (K, 64), bias)
    train_stats: BNStats                               # BatchNorm statistics of the training data
    train_rms: np.ndarray                              # mean channel RMS of the training placements
    meta: dict = field(default_factory=dict)           # labels, names, people, fs, ...

    @property
    def classes(self) -> list[int]:
        return [int(c) for c in self.meta["classes"]]

    def _layers(self, x: np.ndarray, stats: BNStats | None, collect: list | None = None):
        for i, ((w, b, s), (g, be)) in enumerate(zip(self.conv, self.bn)):
            x = _conv1d(x, w, b, s)
            if collect is not None:                    # AdaBN: this layer's statistics from x itself
                m, v = x.mean(axis=(0, 2)), x.var(axis=(0, 2))
                collect.append((m, v))
            else:
                m, v = stats.mean[i], stats.var[i]
            x = np.maximum(0.0, (x - m[None, :, None]) / np.sqrt(v[None, :, None] + BN_EPS) * g[None, :, None]
                           + be[None, :, None])
        return x.mean(axis=2)

    def embed(self, x: np.ndarray, stats: BNStats) -> np.ndarray:
        return self._layers(x.astype(np.float64), stats)

    def log_proba(self, x: np.ndarray, stats: BNStats) -> np.ndarray:
        z = self.embed(x, stats) @ self.fc[0].T + self.fc[1]
        z -= z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    def adapt(self, x: np.ndarray) -> BNStats:
        """AdaBN: BatchNorm statistics of these (unlabelled, RMS-normalised) windows, layer by layer."""
        got: list = []
        self._layers(x.astype(np.float64), None, got)
        return BNStats([m for m, _ in got], [v for _, v in got])

    # -------------------------------------------------------------- storage (plain arrays, no pickle)
    def save(self, path: str) -> None:
        arr = {"meta": np.array(json.dumps(self.meta, ensure_ascii=False)), "version": np.array(MODEL_VERSION),
               "train_rms": self.train_rms, "fc_w": self.fc[0], "fc_b": self.fc[1]}
        for i, ((w, b, s), (g, be)) in enumerate(zip(self.conv, self.bn)):
            arr |= {f"c{i}_w": w, f"c{i}_b": b, f"c{i}_s": np.array(s), f"bn{i}_g": g, f"bn{i}_b": be,
                    f"bn{i}_m": self.train_stats.mean[i], f"bn{i}_v": self.train_stats.var[i]}
        np.savez(path, **arr)

    @classmethod
    def load(cls, path: str) -> "GestureNet":
        d = np.load(path)
        if int(d["version"]) != MODEL_VERSION:
            raise ValueError(f"모델 형식 {int(d['version'])} 은(는) 이 버전에서 읽을 수 없습니다")
        n = sum(1 for k in d.files if k.endswith("_s") and k.startswith("c"))
        return cls(conv=[(d[f"c{i}_w"], d[f"c{i}_b"], int(d[f"c{i}_s"])) for i in range(n)],
                   bn=[(d[f"bn{i}_g"], d[f"bn{i}_b"]) for i in range(n)],
                   fc=(d["fc_w"], d["fc_b"]),
                   train_stats=BNStats([d[f"bn{i}_m"] for i in range(n)], [d[f"bn{i}_v"] for i in range(n)]),
                   train_rms=d["train_rms"], meta=json.loads(str(d["meta"])))


# ------------------------------------------------------------------ calibration of a band placement
@dataclass
class Placement:
    """What a band placement needs: channel RMS for the input, BatchNorm statistics (AdaBN)."""
    rms: np.ndarray
    stats: BNStats
    windows: int = 0            # how many windows it was estimated from (0 = training defaults)

    @classmethod
    def default(cls, net: GestureNet) -> "Placement":
        return cls(net.train_rms, net.train_stats, 0)

    @classmethod
    def estimate(cls, net: GestureNet, wins: np.ndarray) -> "Placement":
        """wins (N, 8, WIN) filtered, not yet normalised."""
        rms = np.sqrt((wins.astype(np.float64) ** 2).mean(axis=(0, 2))) + 1e-6
        return cls(rms, net.adapt(wins / rms[None, :, None]), len(wins))


# ------------------------------------------------------------------ live decoding
@dataclass
class Decision:
    t: float                    # time of the window's last sample
    proba: np.ndarray           # smoothed class probabilities (classes order)
    raw: np.ndarray             # this window alone
    label: int                  # argmax of proba, as a gesture label


class LiveDecoder:
    """Feed raw samples as they arrive; get a decision every 100 ms of gap-free signal.

    Keeps the causal filter state across calls. A gap in the sample times restarts the filter
    and the smoothing, as in training. While `collecting`, filtered windows are also kept for
    estimating a new Placement."""

    def __init__(self, net: GestureNet, fs: float = FS) -> None:
        self.net = net
        self.sos = filter_sos(fs)
        self.placement = Placement.default(net)
        self.collecting = False
        self.collected: list[np.ndarray] = []
        self.gaps = 0
        self.reset()

    def reset(self) -> None:
        self._zi = None
        self._buf = np.zeros((0, 8))
        self._last_t = None
        self._since = 0
        self._hist: list[np.ndarray] = []

    def feed(self, t: np.ndarray, x: np.ndarray) -> list[Decision]:
        """t (n,) sample times, x (n, 8) values as received minus the centre (127), physical order."""
        out: list[Decision] = []
        if not len(t):
            return out
        x = np.asarray(x, np.float64)
        # split at gaps (also against the previous call's last sample)
        prev = np.r_[self._last_t if self._last_t is not None else t[0], t[:-1]]
        cut = np.flatnonzero(t - prev > MAX_STEP_S)
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]):
            if a in cut:
                self.gaps += 1
                self.reset()
            out += self._run(t[a:b], x[a:b])
        self._last_t = float(t[-1])
        return out

    def _run(self, t: np.ndarray, x: np.ndarray) -> list[Decision]:
        if not len(t):
            return []
        if self._zi is None:
            self._zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self._zi = sosfilt(self.sos, x, axis=0, zi=self._zi)
        buf = np.vstack([self._buf, y])
        base = len(buf) - len(y)
        # a decision every STEP samples, on the window ending at that sample
        ends = [base + j + 1 for j in range(len(y)) if (self._since + j + 1) % STEP == 0]
        self._since = (self._since + len(y)) % STEP
        self._buf = buf[-WIN:]
        ends = [e for e in ends if e >= WIN]
        if not ends:
            return []
        W = np.stack([buf[e - WIN:e].T for e in ends])                # (k, 8, WIN)
        if self.collecting:
            self.collected.extend(W.copy())
        p = self.placement
        out = []
        for e, lp in zip(ends, self.net.log_proba(W / p.rms[None, :, None], p.stats)):
            self._hist = (self._hist + [lp])[-SMOOTH:]
            sm = np.mean(self._hist, axis=0)
            pr = np.exp(sm - sm.max())
            pr /= pr.sum()
            out.append(Decision(float(t[e - base - 1]), pr, np.exp(lp), self.net.classes[int(pr.argmax())]))
        return out

    # -------------------------------------------------------------- calibration
    def start_collecting(self) -> None:
        self.collected = []
        self.collecting = True

    def finish_collecting(self, min_windows: int = 100) -> Placement | None:
        self.collecting = False
        wins, self.collected = self.collected, []
        if len(wins) < min_windows:
            return None
        self.placement = Placement.estimate(self.net, np.stack(wins))
        self._hist = []
        return self.placement
