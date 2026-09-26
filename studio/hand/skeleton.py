"""The 21-joint hand skeleton (OpenPose order, as WiLoR reports it) and what is
computed from it. Display colours live in ui.theme, not here."""
from __future__ import annotations

import numpy as np

# wrist, then 4 joints per finger, thumb .. little finger
FINGERS = [("엄지", [0, 1, 2, 3, 4]), ("검지", [0, 5, 6, 7, 8]), ("중지", [0, 9, 10, 11, 12]),
           ("약지", [0, 13, 14, 15, 16]), ("소지", [0, 17, 18, 19, 20])]
BONES = [(c[i], c[i + 1], f) for f, (_, c) in enumerate(FINGERS) for i in range(4)]


def flexion(joints: np.ndarray) -> np.ndarray:
    """Total bend of each finger in degrees: the sum of the angles between consecutive
    bones from the wrist to the tip. A relaxed hand is tens of degrees, a fist > 200."""
    out = []
    for _, chain in FINGERS:
        v = np.diff(joints[chain], axis=0)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        out.append(np.degrees(np.arccos(np.clip((v[:-1] * v[1:]).sum(axis=1), -1, 1))).sum())
    return np.array(out)
