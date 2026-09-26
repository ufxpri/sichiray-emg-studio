"""EMG window -> feature vector. Used identically for training and live prediction."""
from __future__ import annotations

import numpy as np

WIN_S = 0.2       # main window
LONG_S = 0.5      # longer context (sustained contraction)
PREROLL_S = 1.0   # stored before the context so offline filters settle (60 Hz notch Q30: ~0.8 s)
RAW_S = PREROLL_S + LONG_S
N_FEATURES = 64
EPS = 0.1         # counts; below the quantisation floor (0.29 RMS)


def features(x: np.ndarray, fs: float) -> np.ndarray:
    """x: (>= LONG_S*fs, 8) filtered EMG ending at the target time -> (64,).

    per channel: log RMS (200 ms), log RMS of its four 50 ms quarters, log waveform
    length, zero-crossing rate (with a 1-count dead band), log RMS (500 ms)."""
    w = max(4, int(WIN_S * fs))
    s = x[-w:]
    q = max(1, w // 4)
    rms = np.sqrt((s ** 2).mean(axis=0))
    sub = [np.sqrt((s[i * q:(i + 1) * q] ** 2).mean(axis=0)) for i in range(4)]
    d = np.diff(s, axis=0)
    wl = np.abs(d).mean(axis=0)
    zc = ((np.signbit(s[:-1]) != np.signbit(s[1:])) & (np.abs(d) > 1.0)).mean(axis=0)
    longr = np.sqrt((x ** 2).mean(axis=0))
    return np.concatenate([np.log(rms + EPS), *[np.log(v + EPS) for v in sub], np.log(wl + EPS), zc,
                           np.log(longr + EPS)])
