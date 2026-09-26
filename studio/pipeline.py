"""The one global signal chain. Every window that shows 'filtered' data reads it.

raw counts -> subtract mode centre -> channel rotation -> high-pass -> low-pass
-> mains notches -> extra notches -> mute  = filtered
filtered -> |x| -> low-pass                                     = envelope
envelope -> (env - rest) / (mvc - rest)                          = normalised 0..1

Filters run causally (sosfilt with carried state) because this is live data.
A stage that cannot exist at the current sample rate is skipped and reported,
e.g. a 20 Hz high-pass is still possible in ASCII mode (Nyquist 33 Hz) but a
60 Hz notch is not.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

CH = 8


@dataclass
class Settings:
    hp_on: bool = True
    hp_hz: float = 20.0          # below this: electrode movement, not EMG (NOTES §13)
    hp_order: int = 4
    lp_on: bool = False
    lp_hz: float = 200.0
    lp_order: int = 4
    notch_on: bool = True
    notch_base: float = 60.0     # Korea mains
    notch_harm: int = 4          # base, 2x, 3x, 4x (up to Nyquist)
    notch_q: float = 30.0
    extra_notches: list[float] = field(default_factory=list)
    rotate: int = 0              # logical ch k = physical ch (k + rotate) % 8 (NOTES §15)
    mute: list[bool] = field(default_factory=lambda: [False] * CH)
    env_hz: float = 5.0
    rest: list[float] | None = None   # envelope at rest, per logical channel
    mvc: list[float] | None = None    # envelope at maximum contraction

    @classmethod
    def load(cls, path: str) -> "Settings":
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, ensure_ascii=False, indent=2)


def design(s: Settings, fs: float) -> tuple[np.ndarray, list[str]]:
    """Second-order sections for the whole chain, plus notes on skipped stages."""
    nyq = fs / 2
    parts, notes = [], []
    if s.hp_on:
        if s.hp_hz < nyq * 0.95:
            parts.append(butter(s.hp_order, s.hp_hz, "highpass", fs=fs, output="sos"))
        else:
            notes.append(f"고역통과 {s.hp_hz:g} Hz: 나이퀴스트 {nyq:.1f} Hz 이상이라 생략")
    if s.lp_on:
        if s.lp_hz < nyq * 0.95:
            parts.append(butter(s.lp_order, s.lp_hz, "lowpass", fs=fs, output="sos"))
        else:
            notes.append(f"저역통과 {s.lp_hz:g} Hz: 나이퀴스트 {nyq:.1f} Hz 이상이라 생략")
    freqs = []
    if s.notch_on:
        freqs += [s.notch_base * k for k in range(1, s.notch_harm + 1)]
    freqs += list(s.extra_notches)
    for f0 in freqs:
        if 0 < f0 < nyq * 0.98:
            b, a = iirnotch(f0, s.notch_q, fs=fs)
            parts.append(tf2sos(b, a))
        else:
            notes.append(f"노치 {f0:g} Hz: 나이퀴스트 {nyq:.1f} Hz 이상이라 생략")
    sos = np.vstack(parts) if parts else np.zeros((0, 6))
    return sos, notes


FILTER_KEYS = ("hp_on", "hp_hz", "hp_order", "lp_on", "lp_hz", "lp_order", "notch_on", "notch_base",
               "notch_harm", "notch_q", "extra_notches", "rotate", "mute")


def signature(s: Settings) -> str:
    """Everything that changes the filtered signal (not the envelope or calibration)."""
    d = asdict(s)
    return json.dumps({k: d[k] for k in FILTER_KEYS}, sort_keys=True)


def describe(s: Settings) -> str:
    parts = []
    if s.hp_on:
        parts.append(f"HPF {s.hp_hz:g}Hz")
    if s.lp_on:
        parts.append(f"LPF {s.lp_hz:g}Hz")
    if s.notch_on:
        parts.append(f"노치 {s.notch_base:g}Hz×{s.notch_harm}")
    if s.extra_notches:
        parts.append("추가 노치 " + ",".join(f"{f:g}" for f in s.extra_notches))
    if s.rotate:
        parts.append(f"회전 +{s.rotate}")
    if any(s.mute):
        parts.append("끔 " + ",".join(str(i + 1) for i, m in enumerate(s.mute) if m))
    return " · ".join(parts) or "필터 없음"


def filter_offline(cen: np.ndarray, s: Settings, fs: float) -> np.ndarray:
    """Same chain as Pipeline.process on a stored window: cen is raw minus the mode
    centre in physical channel order. Starts in steady state for the first sample,
    so the window needs a pre-roll before the part that is used."""
    x = np.roll(np.asarray(cen, np.float64), -s.rotate, axis=1)
    sos, _ = design(s, fs)
    if len(sos):
        zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
        x, _ = sosfilt(sos, x, axis=0, zi=zi)
    x[:, np.array(s.mute, bool)] = 0.0
    return x


class Pipeline:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.fs = 500.0
        self.center = 127.0
        self.configure(settings)

    def configure(self, settings: Settings, fs: float | None = None, center: float | None = None) -> None:
        self.s = settings
        if fs:
            self.fs = fs
        if center is not None:
            self.center = center
        self.sos, self.notes = design(settings, self.fs)
        self.env_sos = butter(2, min(settings.env_hz, self.fs / 2 * 0.9), "lowpass", fs=self.fs, output="sos")
        self.zi = None
        self.env_zi = None

    def centred(self, raw: np.ndarray) -> np.ndarray:
        return np.roll(raw - self.center, -self.s.rotate, axis=1)

    def process(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.centred(raw)
        if len(self.sos):
            if self.zi is None:  # start in steady state for the first sample, no step transient
                self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
            y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        else:
            y = x
        y[:, np.array(self.s.mute, bool)] = 0.0
        r = np.abs(y)
        if self.env_zi is None:
            self.env_zi = sosfilt_zi(self.env_sos)[:, :, None] * r[0][None, None, :]
        env, self.env_zi = sosfilt(self.env_sos, r, axis=0, zi=self.env_zi)
        if self.s.rest and self.s.mvc:
            rest, mvc = np.array(self.s.rest), np.array(self.s.mvc)
            norm = np.clip((env - rest) / np.maximum(mvc - rest, 1e-6), 0, 1.5)
        else:
            norm = np.zeros_like(env)
        return y, env, norm
