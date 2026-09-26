"""Filter design and the stateless (offline) form of the chain.

physical raw - centre -> rotation -> high-pass -> low-pass -> mains notches
-> extra notches -> mute  = filtered, in logical channel order.

A stage that cannot exist at the sample rate is skipped and reported: a 20 Hz
high-pass still works in ASCII mode (Nyquist 33 Hz), a 60 Hz notch does not.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfreqz, tf2sos

from .settings import FilterSpec


def design(spec: FilterSpec, fs: float) -> tuple[np.ndarray, list[str]]:
    """Second-order sections for the whole chain, plus notes on skipped stages."""
    nyq = fs / 2
    parts, notes = [], []
    if spec.hp_on:
        if spec.hp_hz < nyq * 0.95:
            parts.append(butter(spec.hp_order, spec.hp_hz, "highpass", fs=fs, output="sos"))
        else:
            notes.append(f"고역통과 {spec.hp_hz:g} Hz: 나이퀴스트 {nyq:.1f} Hz 이상이라 생략")
    if spec.lp_on:
        if spec.lp_hz < nyq * 0.95:
            parts.append(butter(spec.lp_order, spec.lp_hz, "lowpass", fs=fs, output="sos"))
        else:
            notes.append(f"저역통과 {spec.lp_hz:g} Hz: 나이퀴스트 {nyq:.1f} Hz 이상이라 생략")
    freqs = [spec.notch_base * k for k in range(1, spec.notch_harm + 1)] if spec.notch_on else []
    for f0 in freqs + list(spec.extra_notches):
        if 0 < f0 < nyq * 0.98:
            b, a = iirnotch(f0, spec.notch_q, fs=fs)
            parts.append(tf2sos(b, a))
        else:
            notes.append(f"노치 {f0:g} Hz: 나이퀴스트 {nyq:.1f} Hz 이상이라 생략")
    return (np.vstack(parts) if parts else np.zeros((0, 6))), notes


def response(sos: np.ndarray, fs: float, n: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Frequency (Hz) and gain (dB) of the chain."""
    if not len(sos):
        return np.array([0.0, fs / 2]), np.zeros(2)
    w, h = sosfreqz(sos, worN=n, fs=fs)
    return w, 20 * np.log10(np.maximum(np.abs(h), 1e-6))


def steady_state(sos: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Filter state as if `first` had been the input forever: no start-up step."""
    return sosfilt_zi(sos)[:, :, None] * first[None, None, :]


def filter_offline(cen: np.ndarray, spec: FilterSpec, fs: float) -> np.ndarray:
    """The live chain applied to a stored window (`cen` = raw minus centre, physical
    order). Needs a pre-roll before the part that is used, for the filters to settle."""
    x = spec.to_logical(np.asarray(cen, np.float64))
    sos, _ = design(spec, fs)
    if len(sos):
        x, _ = sosfilt(sos, x, axis=0, zi=steady_state(sos, x[0]))
    x[:, np.array(spec.mute, bool)] = 0.0
    return x
