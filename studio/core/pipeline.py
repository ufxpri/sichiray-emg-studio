"""The live (causal, stateful) signal chain every 'filtered' view reads.

raw -> centre -> rotation -> filters -> mute                  = filtered
filtered -> |x| -> low-pass                                    = envelope
envelope -> (env - rest) / (mvc - rest), when calibrated       = normalised 0..1
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from ..device.protocol import ModeSpec, spec_for
from .filters import design, response, steady_state
from .settings import Settings


class Pipeline:
    def __init__(self, settings: Settings, mode: ModeSpec | None = None) -> None:
        self.configure(settings, mode or spec_for(None))

    def configure(self, settings: Settings, mode: ModeSpec) -> None:
        """New settings or sample rate: rebuild filters, restart their state."""
        self.settings, self.mode = settings, mode
        self.sos, self.notes = design(settings.filter, mode.fs)
        self.env_sos = butter(2, min(settings.env_hz, mode.fs / 2 * 0.9), "lowpass", fs=mode.fs, output="sos")
        self._zi = self._env_zi = None

    def response(self) -> tuple[np.ndarray, np.ndarray]:
        return response(self.sos, self.mode.fs)

    def centred(self, raw: np.ndarray) -> np.ndarray:
        """Raw minus the mode centre, physical order."""
        return raw - self.mode.center

    def process(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """-> (logical centred input, filtered, envelope, normalised)."""
        x = self.settings.filter.to_logical(self.centred(raw))
        if len(self.sos):
            if self._zi is None:
                self._zi = steady_state(self.sos, x[0])
            y, self._zi = sosfilt(self.sos, x, axis=0, zi=self._zi)
        else:
            y = x.copy()
        y[:, np.array(self.settings.filter.mute, bool)] = 0.0
        r = np.abs(y)
        if self._env_zi is None:
            self._env_zi = steady_state(self.env_sos, r[0])
        env, self._env_zi = sosfilt(self.env_sos, r, axis=0, zi=self._env_zi)
        cal = self.settings.calib
        if cal.complete:
            rest, mvc = np.array(cal.rest), np.array(cal.mvc)
            norm = np.clip((env - rest) / np.maximum(mvc - rest, 1e-6), 0, 1.5)
        else:
            norm = np.zeros_like(env)
        return x, y, env, norm
