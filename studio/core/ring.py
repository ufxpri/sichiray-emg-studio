"""Thread-safe fixed-size buffers with a running write counter, so each reader can
ask for exactly what it has not seen yet."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np


class Ring:
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
    """Recent raw chunks for the byte console."""

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
