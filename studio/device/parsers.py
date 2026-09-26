"""Bytes -> HEX frames / ASCII rows. Nothing is guessed: what does not parse is counted."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import CH, FIELD, FRAME_LEN, HDR, SAMPLES_PER_FRAME, TAIL


@dataclass(frozen=True)
class HexFrame:
    ts: int                 # device clock, ms (+20 per frame)
    imu: tuple[int, ...]    # AccX..Yaw, raw bytes
    emg: np.ndarray         # (10, 8) uint8
    battery: int            # 0..100 %


def is_frame(buf: bytes | bytearray) -> bool:
    """`buf` starts with a complete, well-formed frame."""
    return len(buf) >= FRAME_LEN and buf[:len(HDR)] == HDR and buf[FRAME_LEN - 1] == TAIL


def decode(b: bytes) -> HexFrame:
    emg = np.frombuffer(b[FIELD["emg"]], np.uint8).reshape(SAMPLES_PER_FRAME, CH)
    imu = tuple(b[FIELD["acc"].start:FIELD["angle"].stop])
    return HexFrame(int.from_bytes(b[FIELD["timestamp"]], "big"), imu, emg, b[FIELD["battery"].start])


class HexParser:
    """Feed bytes, get (frames, discarded byte count). Resyncs one byte at a time on a bad tail."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def feed(self, data: bytes) -> tuple[list[HexFrame], int]:
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
            if is_frame(self.buf):
                out.append(decode(bytes(self.buf[:FRAME_LEN])))
                del self.buf[:FRAME_LEN]
            else:
                dropped += 1
                del self.buf[:1]
        return out, dropped

    @staticmethod
    def segment(buf: bytearray) -> list[tuple[str, bytes]]:
        """Split `buf` (consumed in place) into ("frame", bytes) and ("junk", bytes) for
        display. An incomplete tail stays in `buf` for the next call."""
        out = []
        while True:
            i = buf.find(HDR)
            if i < 0:
                if len(buf) > 4096:            # nothing frame-like: flush as junk
                    out.append(("junk", bytes(buf[:-2])))
                    del buf[:-2]
                break
            if i:
                out.append(("junk", bytes(buf[:i])))
                del buf[:i]
            if len(buf) < FRAME_LEN:
                break
            if is_frame(buf):
                out.append(("frame", bytes(buf[:FRAME_LEN])))
                del buf[:FRAME_LEN]
                continue
            j = buf.find(HDR, len(HDR))        # broken frame: junk up to the next header
            end = j if j > 0 else len(buf)
            out.append(("junk", bytes(buf[:end])))
            del buf[:end]
        return out


class AsciiParser:
    """Feed bytes, get (valid rows, bad line count). A valid line is exactly 8 integers in 0..4095."""

    def __init__(self) -> None:
        self.buf = b""

    def feed(self, data: bytes) -> tuple[list[list[int]], int]:
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        if len(self.buf) > 256:                # binary stream with no newline in sight
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
