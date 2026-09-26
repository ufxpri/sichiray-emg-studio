"""Device link: owns the serial port and turns bytes into samples and statistics.

The armband has two output modes, switched only by its button (NOTES §12):
  HEX   98-byte frames, 50 frames/s, 10 samples x 8 ch of 8-bit EMG + IMU + battery
  ASCII one text line per sample, 8 space-separated 12-bit integers, ~67 Hz

Both parsers see every byte, so a button press mid-stream is followed without a
reconnect. Whichever parser keeps producing valid output decides the mode; a
mode change bumps `gen` so readers know the old samples belong to another scale.

Nothing here filters or rescales. Rings hold the values exactly as received.

Every EMG sample also gets a time on the perf_counter clock (`tsamp`, seconds),
the clock the camera worker stamps frames with. HEX frames carry a device
timestamp (+20 ms per frame), so sample times come from that, shifted onto the
host clock by the smallest arrival-minus-device offset seen (the least-delayed
frame). Serial bursts therefore do not smear the timing. ASCII has no device
clock and uses arrival time.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

CH = 8
BAUD = 115200
FRAME_LEN = 98
HDR = b"\xAA\xAA\x5F"
TAIL = 0x55
SAMPLES_PER_FRAME = 10
FRAME_MS = 20

MODES = {
    # bps: what the link should carry. hex = 98 B x 50 frames; ascii measured in §12.
    "hex": dict(fs=500.0, full=255, center=127, bits=8, bps=FRAME_LEN * 50),
    "ascii": dict(fs=67.1, full=4095, center=2048, bits=12, bps=2675),
}
IMU_NAMES = ["AccX", "AccY", "AccZ", "GyX", "GyY", "GyZ", "Pitch", "Roll", "Yaw"]


def find_ch340() -> list[str]:
    from serial.tools import list_ports
    return [p.device for p in list_ports.comports() if (p.vid, p.pid) == (0x1A86, 0x7523)]


def list_ports() -> list[tuple[str, str]]:
    from serial.tools import list_ports as lp
    return [(p.device, p.description) for p in lp.comports()]


class Ring:
    """Fixed-size sample store. `total` counts every row ever written, so a reader
    can ask for exactly the rows it has not seen yet."""

    def __init__(self, cap: int, cols: int) -> None:
        self.buf = np.zeros((cap, cols), np.float64)
        self.cap, self.cols = cap, cols
        self.total = 0
        self.lock = threading.Lock()

    def clear(self) -> None:
        with self.lock:
            self.total = 0

    def write(self, rows) -> None:
        rows = np.asarray(rows, np.float64).reshape(-1, self.cols)
        k = len(rows)
        if not k:
            return
        with self.lock:
            if k > self.cap:
                rows = rows[-self.cap:]
                self.total += k - self.cap
                k = self.cap
            i = self.total % self.cap
            first = min(k, self.cap - i)
            self.buf[i:i + first] = rows[:first]
            self.buf[:k - first] = rows[first:]
            self.total += k

    def _take(self, start: int, end: int) -> np.ndarray:
        return self.buf[np.arange(start, end) % self.cap].copy()

    def last(self, n: int) -> np.ndarray:
        with self.lock:
            n = min(n, self.total, self.cap)
            return self._take(self.total - n, self.total)

    def since(self, pos: int) -> tuple[np.ndarray, int]:
        with self.lock:
            start = max(pos, self.total - self.cap)
            if start > self.total:  # ring was cleared under the reader
                start = 0
            return self._take(start, self.total), self.total


class ByteLog:
    """Recent raw chunks for the byte console, with a running sequence number."""

    def __init__(self, max_bytes: int = 256 * 1024) -> None:
        self.chunks: deque[bytes] = deque()
        self.size = 0
        self.max = max_bytes
        self.total = 0
        self.lock = threading.Lock()

    def add(self, data: bytes) -> None:
        with self.lock:
            self.chunks.append(data)
            self.size += len(data)
            self.total += len(data)
            while self.size > self.max:
                self.size -= len(self.chunks.popleft())

    def since(self, pos: int) -> tuple[bytes, int]:
        with self.lock:
            want = self.total - pos
            if want <= 0:
                return b"", self.total
            blob = b"".join(self.chunks)
            return blob[-min(want, len(blob)):], self.total


class HexParser:
    """Bytes in, (frames, discarded byte count) out. Resyncs on a bad tail."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def feed(self, data: bytes) -> tuple[list[tuple], int]:
        self.buf += data
        out, dropped = [], 0
        while True:
            i = self.buf.find(HDR)
            if i < 0:
                keep = 2 if self.buf.endswith(b"\xAA\xAA") else 1 if self.buf.endswith(b"\xAA") else 0
                dropped += len(self.buf) - keep
                del self.buf[:len(self.buf) - keep]
                break
            if i:
                dropped += i
                del self.buf[:i]
            if len(self.buf) < FRAME_LEN:
                break
            if self.buf[FRAME_LEN - 1] == TAIL:
                b = bytes(self.buf[:FRAME_LEN])
                del self.buf[:FRAME_LEN]
                emg = np.frombuffer(b, np.uint8, 80, 16).reshape(SAMPLES_PER_FRAME, CH)
                out.append((int.from_bytes(b[3:7], "big"), list(b[7:16]), emg, b[96]))
            else:
                dropped += 1
                del self.buf[:1]
        return out, dropped


class AsciiParser:
    """Bytes in, (valid rows, bad line count) out. A valid line is exactly 8 integers
    in 0..4095; anything else is counted, never guessed at."""

    def __init__(self) -> None:
        self.buf = b""

    def feed(self, data: bytes) -> tuple[list[list[int]], int]:
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        if len(self.buf) > 256:  # binary stream with no newline in sight
            self.buf = self.buf[-64:]
        rows, bad = [], 0
        for ln in lines:
            tok = ln.strip(b"\r \t").split()
            if len(tok) == CH and all(t.isdigit() for t in tok):
                vals = [int(t) for t in tok]
                if max(vals) <= 4095:
                    rows.append(vals)
                    continue
            if ln.strip():
                bad += 1
        return rows, bad


class Stats:
    """Everything the info window shows. Mutated only under Link.lock."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t0 = time.monotonic()
        self.bytes_total = 0
        self.rx_win: deque[tuple[float, int]] = deque()
        self.smp_win: deque[tuple[float, int]] = deque()
        self.last_rx = 0.0
        self.bps_hist: deque[tuple[float, float]] = deque(maxlen=3600)
        self.batt_hist: deque[tuple[float, int]] = deque(maxlen=20000)
        self.battery: int | None = None
        self.imu: list[int] | None = None
        self.mode_changes = 0
        self.reset_mode()

    def reset_mode(self) -> None:
        self.frames_ok = 0
        self.resync_bytes = 0
        self.lost_frames = 0
        self.irregular_ts = 0
        self.last_ts: int | None = None
        self.ts_deltas: deque[int] = deque(maxlen=250)
        self.lines_ok = 0
        self.lines_bad = 0

    @staticmethod
    def _rate(win: deque, now: float, span: float = 1.0) -> float:
        while win and now - win[0][0] > span:
            win.popleft()
        return sum(n for _, n in win) / span


class Link:
    def __init__(self) -> None:
        self.emg = Ring(60 * 500, CH)
        self.tsamp = Ring(60 * 500, 1)
        self._off: float | None = None
        self.imu = Ring(60 * 50, 9)
        self.bytes = ByteLog()
        self.stats = Stats()
        self.lock = threading.Lock()
        self.mode: str | None = None
        self.gen = 0
        self.source = ""       # "COM4", "데모"
        self.status = "끊김"
        self.error = ""
        self.demo: DemoSource | None = None
        self._hex = HexParser()
        self._ascii = AsciiParser()
        self._vote = ("", 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def open_serial(self, port: str) -> None:
        self._start(port, lambda: self._serial_loop(port))

    def open_demo(self, mode: str = "hex") -> None:
        self._start("데모", self._demo_loop, DemoSource(mode))

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None
        self.demo = None
        self.status = "끊김"

    def _start(self, name: str, target, demo: "DemoSource | None" = None) -> None:
        self.close()
        self.demo = demo
        self._stop = threading.Event()
        self._hex, self._ascii = HexParser(), AsciiParser()
        with self.lock:
            self.stats.reset()
            self._set_mode(None)
        self.source, self.error, self.status = name, "", "연결 중"
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def _serial_loop(self, port: str) -> None:
        import serial
        try:
            ser = serial.Serial(port, BAUD, timeout=0.05)
        except (OSError, serial.SerialException) as e:
            busy = "PermissionError" in str(e) or "Access is denied" in str(e) or "거부" in str(e)
            self.error = ("다른 프로그램이 포트를 사용 중입니다 (벤더 뷰어, sscom, 다른 EMG Studio 등). 그 프로그램을 닫고 다시 연결하세요."
                          if busy else str(e))
            self.status = "오류"
            return
        self.status = "연결됨"
        with ser:
            while not self._stop.is_set():
                try:
                    # read what is waiting; a large blocking read returns in ~1 s bursts on Windows
                    data = ser.read(max(1, ser.in_waiting))
                except (OSError, serial.SerialException) as e:
                    self.error, self.status = str(e), "오류"
                    return
                if data:
                    self.ingest(data)
        self.status = "끊김"

    def _demo_loop(self) -> None:
        self.status = "연결됨"
        next_t = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue
            self.ingest(self.demo.step())
            next_t += self.demo.period

    # ------------------------------------------------------------ parsing
    def _set_mode(self, mode: str | None) -> None:
        self.mode = mode
        self.gen += 1
        self.emg.clear()
        self.tsamp.clear()
        self.imu.clear()
        self._off = None
        self.stats.reset_mode()
        if mode:
            self.stats.mode_changes += 1

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
        now = time.monotonic()
        pc = time.perf_counter()
        self.bytes.add(data)
        frames, dropped = self._hex.feed(data)
        rows, bad = self._ascii.feed(data)
        with self.lock:
            st = self.stats
            st.bytes_total += len(data)
            st.rx_win.append((now, len(data)))
            st.last_rx = now
            self._elect("hex", len(frames), 2)
            self._elect("ascii", len(rows), 3)
            if self.mode == "hex":
                st.resync_bytes += dropped
                for ts, imu, emg, batt in frames:
                    self._on_frame(st, now, pc, ts, imu, emg, batt)
            elif self.mode == "ascii":
                st.lines_bad += bad
                if rows:
                    st.lines_ok += len(rows)
                    self.emg.write(rows)
                    self.tsamp.write(np.full(len(rows), pc))
                    st.smp_win.append((now, len(rows)))

    def _on_frame(self, st: Stats, now: float, pc: float, ts: int, imu, emg, batt: int) -> None:
        st.frames_ok += 1
        if st.last_ts is not None:
            d = ts - st.last_ts
            st.ts_deltas.append(d)
            if d != FRAME_MS:
                if d > FRAME_MS and d % FRAME_MS == 0:
                    st.lost_frames += d // FRAME_MS - 1
                else:
                    st.irregular_ts += 1
        st.last_ts = ts
        off = pc - ts / 1000.0
        if self._off is None or off < self._off or off - self._off > 1.0:   # >1 s: device clock restarted
            self._off = off
        else:
            self._off += 2e-6   # let the estimate follow slow crystal drift
        last = ts / 1000.0 + self._off   # assume the stamp marks the frame's last sample (±20 ms unknown)
        self.emg.write(emg)
        self.tsamp.write(last - (SAMPLES_PER_FRAME - 1 - np.arange(SAMPLES_PER_FRAME)) * 0.002)
        self.imu.write(imu)
        st.smp_win.append((now, SAMPLES_PER_FRAME))
        st.imu = imu
        if batt != st.battery or not st.batt_hist or now - st.batt_hist[-1][0] > 5:
            st.batt_hist.append((now - st.t0, batt))
        st.battery = batt

    # ------------------------------------------------------------ reading
    def snapshot(self) -> dict:
        """Consistent copy of the counters, for display."""
        with self.lock:
            st = self.stats
            now = time.monotonic()
            bps = Stats._rate(st.rx_win, now)
            sps = Stats._rate(st.smp_win, now)
            d = list(st.ts_deltas)
            return dict(
                source=self.source, status=self.status, error=self.error, mode=self.mode,
                gen=self.gen, uptime=now - st.t0, idle=(now - st.last_rx) if st.last_rx else None,
                bytes_total=st.bytes_total, bps=bps, sps=sps,
                frames_ok=st.frames_ok, resync_bytes=st.resync_bytes, lost_frames=st.lost_frames,
                irregular_ts=st.irregular_ts, last_ts=st.last_ts,
                ts_mean=float(np.mean(d)) if d else None,
                ts_min=min(d) if d else None, ts_max=max(d) if d else None,
                lines_ok=st.lines_ok, lines_bad=st.lines_bad, battery=st.battery,
                imu=st.imu, mode_changes=st.mode_changes,
                batt_hist=list(st.batt_hist),
            )


class DemoSource:
    """Synthetic armband that emits real HEX frames or ASCII lines, so the demo
    exercises the same parsers as the device.

    Channels carry deliberate faults so the noise table has something to find:
      all   a small 60 Hz mains hum (common to every channel)
      ch2   slow electrode movement (0.5-3 Hz wander)
      ch6   poor contact: almost no muscle signal, higher noise floor
      ch8   a charger-like 9.77 Hz sawtooth line (NOTES §18)
    Muscle activity cycles every 8 s: rest, fist (all), rest, 'index' (ch3-4), rest.
    Occasionally a frame is dropped or garbage bytes are inserted.
    """

    GAIN = np.array([1.0, 0.8, 1.2, 1.1, 1.6, 0.15, 0.7, 0.9])

    def __init__(self, mode: str = "hex") -> None:
        from scipy.signal import butter, sosfilt_zi
        self.mode = mode
        self.rng = np.random.default_rng(1)
        self.n = 0          # 500 Hz sample counter
        self.ts = 0
        self.frame_no = 0
        self.ascii_acc = 0.0
        self.batt = 87.0
        self.sos = butter(4, [30, 150], "bandpass", fs=500, output="sos")
        self.zi = np.repeat(sosfilt_zi(self.sos)[:, :, None] * 0, CH, axis=2)
        self.wander = np.zeros(CH)

    @property
    def period(self) -> float:
        return FRAME_MS / 1000 if self.mode == "hex" else 1 / MODES["ascii"]["fs"]

    def toggle(self) -> None:
        """What the armband button does: double = ASCII, single = HEX."""
        self.mode = "ascii" if self.mode == "hex" else "hex"

    def _synth(self, k: int) -> np.ndarray:
        """k samples at 500 Hz, in 8-bit counts around 0."""
        from scipy.signal import sosfilt
        t = (self.n + np.arange(k)) / 500.0
        self.n += k
        cyc = t % 8.0
        fist = ((cyc > 2) & (cyc < 4)).astype(float)
        index = ((cyc > 5) & (cyc < 7)).astype(float)
        env = fist[:, None] * self.GAIN[None, :] * 14
        env[:, 2:4] += index[:, None] * 12
        mus, self.zi = sosfilt(self.sos, self.rng.standard_normal((k, CH)), axis=0, zi=self.zi)
        x = mus * env * 2.2 + 0.35 * self.rng.standard_normal((k, CH))
        x += 0.8 * np.sin(2 * np.pi * 60 * t)[:, None]
        x[:, 5] += 1.2 * self.rng.standard_normal(k)
        self.wander = 0.995 * self.wander + 0.25 * self.rng.standard_normal(CH)
        x[:, 1] += self.wander[1] * 3 + 5 * np.sin(2 * np.pi * 1.3 * t) * (np.sin(2 * np.pi * 0.07 * t) > 0.3)
        x[:, 7] += 7 * ((t * 9.77) % 1.0 - 0.5)
        return x

    def step(self) -> bytes:
        if self.mode == "hex":
            return self._frame()
        x = self._synth(max(1, round(500 / MODES["ascii"]["fs"])))[-1]
        v = np.clip(np.round(2048 + x * 16), 0, 4095).astype(int)
        return (" ".join(str(i) for i in v) + "\r\n").encode()

    def _frame(self) -> bytes:
        x = self._synth(SAMPLES_PER_FRAME)
        self.frame_no += 1
        self.ts += FRAME_MS
        if self.frame_no % 600 == 0:   # a lost frame: timestamp jumps, bytes never arrive
            self.ts += FRAME_MS
        emg = np.clip(np.round(127 + x), 0, 255).astype(np.uint8)
        t = self.n / 500.0
        imu = [127 + int(20 * np.sin(t * 0.9 + i)) for i in range(6)]
        imu += [127 + int(40 * np.sin(t * 0.7)), 127 + int(30 * np.sin(t * 0.4)), 127]
        self.batt = max(0.0, self.batt - FRAME_MS / 1000 / 60)
        body = bytes([0xAA, 0xAA, 0x5F]) + self.ts.to_bytes(4, "big") + bytes(imu) \
            + emg.tobytes() + bytes([int(self.batt), TAIL])
        if self.frame_no % 450 == 0:   # line noise: a few stray bytes before the frame
            body = bytes([0x13, 0xAA, 0x00]) + body
        return body
