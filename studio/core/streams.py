"""The processed streams the hub keeps, aligned row for row."""
from __future__ import annotations

import numpy as np

from ..device.protocol import CH
from .ring import Ring

BUFFER_S = 60


class StreamStore:
    """t     perf_counter seconds of each row (same clock as camera frames)
    cen   raw minus mode centre, physical order   (what the pose learner stores)
    pre   cen in logical order                     (input to the filters)
    filt  filtered, logical order
    env   envelope
    norm  envelope normalised by the rest/MVC calibration"""

    def __init__(self, cap: int = BUFFER_S * 500) -> None:
        self.t = Ring(cap, 1)
        self.cen, self.pre, self.filt, self.env, self.norm = (Ring(cap, CH) for _ in range(5))

    def clear(self) -> None:
        for r in (self.t, self.cen, self.pre, self.filt, self.env, self.norm):
            r.clear()

    def write(self, t, cen, pre, filt, env, norm) -> None:
        for r, v in ((self.t, t), (self.cen, cen), (self.pre, pre), (self.filt, filt), (self.env, env),
                     (self.norm, norm)):
            r.write(v)

    def recent_raw(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Times (n,) and unfiltered centred samples (n, 8), physical order."""
        t = self.t.last(n)[:, 0]
        cen = self.cen.last(len(t))
        k = min(len(t), len(cen))
        return t[len(t) - k:], cen[len(cen) - k:]
