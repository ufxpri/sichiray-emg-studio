"""Link: one reader thread over a ByteSource, turning bytes into timestamped samples.

Both parsers see every byte, so a button press mid-stream is followed without a
reconnect. Whichever parser keeps producing valid output decides the mode; a
mode change bumps `gen` so readers know the old samples belong to another scale.
Nothing here filters or rescales: the rings hold values exactly as received.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..core.ring import ByteLog, Ring
from .clock import ClockAligner
from .demo import DemoSource
from .parsers import AsciiParser, HexParser
from .protocol import CH, LinkState
from .sources import ByteSource, SerialSource, SourceError
from .stats import LinkSnapshot, LinkStats

BUFFER_S = 60


class Link:
    def __init__(self) -> None:
        self.emg = Ring(BUFFER_S * 500, CH)        # values as received, physical channel order
        self.tsamp = Ring(BUFFER_S * 500, 1)       # perf_counter seconds of each sample
        self.imu = Ring(BUFFER_S * 50, 9)
        self.bytes = ByteLog()
        self.lock = threading.Lock()               # guards everything below and the ring pairs
        self._stats = LinkStats()
        self._clock = ClockAligner()
        self.mode: str | None = None
        self.gen = 0
        self.state = LinkState.DISCONNECTED
        self.error = ""
        self.source_name = ""
        self._source: ByteSource | None = None
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._hex, self._ascii = HexParser(), AsciiParser()
        self._vote = ("", 0)

    # ------------------------------------------------------------ lifecycle
    def open(self, source: ByteSource) -> None:
        self.close()
        stop = threading.Event()
        with self.lock:
            self._hex, self._ascii = HexParser(), AsciiParser()
            self._stats.reset()
            self._set_mode(None)
            self._source, self._stop = source, stop
            self.source_name, self.error, self.state = source.name, "", LinkState.CONNECTING
        self._thread = threading.Thread(target=self._run, args=(source, stop), daemon=True)
        self._thread.start()

    def open_serial(self, port: str) -> None:
        self.open(SerialSource(port))

    def open_demo(self, mode: str = "hex") -> None:
        self.open(DemoSource(mode))

    def close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = self._stop = self._source = None
        self.state = LinkState.DISCONNECTED

    @property
    def is_demo(self) -> bool:
        return isinstance(self._source, DemoSource)

    def press_mode_button(self) -> bool:
        """Only the demo device can be switched from software (NOTES §12: no serial command exists)."""
        src = self._source
        if hasattr(src, "press_mode_button"):
            src.press_mode_button()
            return True
        return False

    def _run(self, source: ByteSource, stop: threading.Event) -> None:
        """Owns `source` for its whole life; `stop` is this thread's own event, so a
        restart can never leave an old thread reading on behalf of the new one."""
        try:
            source.open()
            self._set_state(stop, LinkState.CONNECTED)
            while not stop.is_set():
                data = source.read()
                if data:
                    self.ingest(data)
        except SourceError as e:
            self._set_state(stop, LinkState.ERROR, str(e))
        finally:
            source.close()

    def _set_state(self, stop: threading.Event, state: LinkState, error: str = "") -> None:
        with self.lock:
            if self._stop is stop:                 # ignore a superseded thread
                self.state, self.error = state, error

    # ------------------------------------------------------------ parsing
    def _set_mode(self, mode: str | None) -> None:
        self.mode = mode
        self.gen += 1
        for r in (self.emg, self.tsamp, self.imu):
            r.clear()
        self._clock.reset()
        if mode:
            self._stats.record_mode_change()
        else:
            self._stats.reset_mode()

    def _elect(self, mode: str, units: int, need: int) -> None:
        if not units or mode == self.mode:
            if units:
                self._vote = ("", 0)
            return
        cand, n = self._vote
        n = n + units if cand == mode else units
        self._vote = (mode, n)
        if n >= need or self.mode is None:
            self._set_mode(mode)
            self._vote = ("", 0)

    def ingest(self, data: bytes) -> None:
        now, pc = time.monotonic(), time.perf_counter()
        self.bytes.add(data)
        frames, dropped = self._hex.feed(data)
        rows, bad = self._ascii.feed(data)
        with self.lock:
            st = self._stats
            st.record_rx(len(data), now)
            self._elect("hex", len(frames), 2)
            self._elect("ascii", len(rows), 3)
            if self.mode == "hex":
                st.record_resync(dropped)
                for f in frames:
                    st.record_frame(f.ts, f.imu, f.battery, now)
                    self.emg.write(f.emg)
                    self.tsamp.write(self._clock.stamp(f.ts, pc))
                    self.imu.write(f.imu)
            elif self.mode == "ascii":
                st.record_lines(len(rows), bad, now)
                if rows:                           # no device clock in ASCII: arrival time
                    self.emg.write(rows)
                    self.tsamp.write(np.full(len(rows), pc))

    # ------------------------------------------------------------ reading
    def read_since(self, gen: int, pos: int) -> tuple[int, np.ndarray, np.ndarray, int]:
        """New samples and their times since (gen, pos), read atomically. A different
        gen means the mode changed: the reader starts over from the new gen's first sample."""
        with self.lock:
            if gen != self.gen:
                pos = 0
            raw, new_pos = self.emg.since(pos)
            t, _ = self.tsamp.since(pos)
            n = min(len(raw), len(t))
            return self.gen, raw[len(raw) - n:], t[len(t) - n:, 0], new_pos

    def snapshot(self) -> LinkSnapshot:
        with self.lock:
            return self._stats.snapshot(source=self.source_name, state=self.state, error=self.error,
                                        mode=self.mode, gen=self.gen)
