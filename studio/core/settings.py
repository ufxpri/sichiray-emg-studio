"""User settings, split by what they affect.

FilterSpec    everything that changes the filtered signal. Frozen: a model stores
              its signature and refuses to predict when the live one differs, so
              the signature is taken over *all* fields, never a hand-kept list.
Calibration   rest / maximum-contraction envelope levels, per logical channel.
Settings      both, plus the envelope smoothing. Changed only by building a new
              value (dataclasses.replace) and handing it to Hub.apply().

The JSON file stays flat (the format written since the first version).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from ..device.protocol import CH


@dataclass(frozen=True)
class FilterSpec:
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
    extra_notches: tuple[float, ...] = ()
    rotate: int = 0              # logical ch k = physical ch (k + rotate) % 8 (NOTES §15)
    mute: tuple[bool, ...] = (False,) * CH

    def signature(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def describe(self) -> str:
        parts = []
        if self.hp_on:
            parts.append(f"HPF {self.hp_hz:g}Hz")
        if self.lp_on:
            parts.append(f"LPF {self.lp_hz:g}Hz")
        if self.notch_on:
            parts.append(f"노치 {self.notch_base:g}Hz×{self.notch_harm}")
        if self.extra_notches:
            parts.append("추가 노치 " + ",".join(f"{f:g}" for f in self.extra_notches))
        if self.rotate:
            parts.append(f"회전 +{self.rotate}")
        if any(self.mute):
            parts.append("끔 " + ",".join(str(i + 1) for i, m in enumerate(self.mute) if m))
        return " · ".join(parts) or "필터 없음"

    def to_logical(self, x: np.ndarray) -> np.ndarray:
        """Physical channel order -> logical (rotation applied). The one place this rule lives."""
        return np.roll(x, -self.rotate, axis=-1)


@dataclass(frozen=True)
class Calibration:
    rest: tuple[float, ...] | None = None   # envelope at rest
    mvc: tuple[float, ...] | None = None    # envelope at maximum contraction

    @property
    def complete(self) -> bool:
        return self.rest is not None and self.mvc is not None

    def rest_rms(self) -> np.ndarray | None:
        """Resting RMS of the filtered signal from the rest envelope.
        For a zero-mean Gaussian, mean(|x|) = RMS * sqrt(2/pi)."""
        return None if self.rest is None else np.array(self.rest) / np.sqrt(2 / np.pi)


@dataclass(frozen=True)
class Settings:
    filter: FilterSpec = field(default_factory=FilterSpec)
    env_hz: float = 5.0
    calib: Calibration = field(default_factory=Calibration)

    def with_filter(self, **changes) -> "Settings":
        return replace(self, filter=replace(self.filter, **changes))

    # ------------------------------------------------------------ JSON (flat)
    @classmethod
    def load(cls, path: str) -> "Settings":
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return cls()
        try:
            names = {f.name for f in fields(FilterSpec)}
            fkw = {k: v for k, v in d.items() if k in names}
            for k in ("extra_notches", "mute"):
                if k in fkw:
                    fkw[k] = tuple(fkw[k])
            tup = lambda v: None if v is None else tuple(float(x) for x in v)
            return cls(FilterSpec(**fkw), float(d.get("env_hz", 5.0)),
                       Calibration(tup(d.get("rest")), tup(d.get("mvc"))))
        except (TypeError, ValueError):
            return cls()

    def save(self, path: str) -> None:
        d = asdict(self.filter)
        d.update(env_hz=self.env_hz, rest=self.calib.rest, mvc=self.calib.mvc)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
