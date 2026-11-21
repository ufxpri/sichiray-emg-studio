"""A recorded session played back as if the armband were on: the bytes of emg_bytes.bin are
handed to the Link at the pace they arrived, so they go through the same parsers, filters and
windows as a live band (dropouts included). Also knows which prompt was on screen at each
moment of the recording, so a window can show the truth next to its own output."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .sources import SourceError


class ReplaySource:
    def __init__(self, folder: str, speed: float = 1.0) -> None:
        self.folder = folder
        self.name = "재생: " + os.path.basename(folder.rstrip("/\\"))
        self.speed = speed
        self._blob = b""
        self._t0 = 0.0          # perf_counter when playback started
        self._rec0 = 0.0        # recording clock of the first chunk
        self._i = 0

    def open(self) -> None:
        try:
            e = np.load(os.path.join(self.folder, "emg.npz"))
            with open(os.path.join(self.folder, "emg_bytes.bin"), "rb") as fh:
                self._blob = fh.read()
            p = np.load(os.path.join(self.folder, "prompts.npz"))
            meta = json.load(open(os.path.join(self.folder, "meta.json"), encoding="utf-8"))
        except (OSError, KeyError, ValueError) as ex:
            raise SourceError(f"재생할 수 없는 세션 폴더입니다: {ex}") from ex
        self.arrival, self.offset, self.length = e["chunk_arrival"], e["chunk_offset"], e["chunk_len"]
        if not len(self.arrival):
            raise SourceError("이 세션에는 받은 바이트 기록이 없습니다")
        self.p_kind, self.p_t, self.p_label = p["kind"], p["t"], p["label"]
        names = meta.get("label_names_shown") or meta.get("label_names_ko") or {}
        self.names = {int(k): v for k, v in names.items()}
        self.participant = meta.get("participant", "")
        self._rec0 = float(self.arrival[0])
        self._t0 = time.perf_counter()
        self._i = 0

    @property
    def duration(self) -> float:
        return float(self.arrival[-1] - self._rec0) / self.speed

    def now_recording(self) -> float:
        """The recording-clock moment being played now."""
        return self._rec0 + (time.perf_counter() - self._t0) * self.speed

    def read(self) -> bytes:
        if self._i >= len(self.arrival):
            time.sleep(0.05)
            return b""
        due = self.now_recording()
        j = int(np.searchsorted(self.arrival, due, side="right"))
        if j <= self._i:
            time.sleep(min(0.02, max(0.0, (self.arrival[self._i] - due) / self.speed)))
            return b""
        a, b = int(self.offset[self._i]), int(self.offset[j - 1] + self.length[j - 1])
        self._i = j
        return self._blob[a:b]

    def close(self) -> None:
        self._blob = b""

    # ------------------------------------------------------------ what the screen asked for
    def prompt(self) -> tuple[str, str, float] | None:
        """(kind, gesture name, seconds into the phase) shown at the moment being played."""
        if not len(self.p_t):
            return None
        now = self.now_recording()
        i = int(np.searchsorted(self.p_t, now, side="right")) - 1
        if i < 0:
            return None
        kind = str(self.p_kind[i])
        lab = int(self.p_label[i])
        if kind == "rest" or kind in ("set_start", "set_end", "break"):
            name = self.names.get(0, "휴식")
        else:
            name = self.names.get(lab, str(lab))
        return kind, name, float(now - self.p_t[i])

    @property
    def finished(self) -> bool:
        return self._i >= len(self.arrival)
