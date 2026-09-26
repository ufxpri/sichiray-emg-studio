"""Behaviour tests for the parts that must not change during refactoring:
parsing, buffers, timestamps, the filter chain, noise metrics, and the learner's
evaluation maths. No Qt window, no serial port, no GPU needed."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from studio.device import (CH, FRAME_LEN, AsciiParser, DemoSource, HexParser, Link, Ring)
from studio.pipeline import Pipeline, Settings, filter_offline, signature
from studio import analysis
from studio.pose import PoseLearner, features, geodesic_deg


# ---------------------------------------------------------------- buffers
def test_ring_wraps_and_reads_since():
    r = Ring(10, 2)
    r.write(np.arange(24).reshape(12, 2))          # more than capacity
    assert r.total == 12
    last = r.last(3)
    assert last[:, 0].tolist() == [18, 20, 22]
    rows, pos = r.since(9)
    assert pos == 12 and rows[:, 0].tolist() == [18, 20, 22]
    r.clear()
    rows, pos = r.since(12)                        # reader ahead of a cleared ring
    assert pos == 0 and len(rows) == 0


# ---------------------------------------------------------------- parsers
def _demo_bytes(mode: str, n: int) -> bytes:
    d = DemoSource(mode)
    return b"".join(d.step() for _ in range(n))


def test_hex_parser_frames_and_resync():
    data = _demo_bytes("hex", 50)
    frames, dropped = HexParser().feed(data)
    # the demo inserts 3 stray bytes before frame 450 only; 50 frames are clean
    assert len(frames) == 50 and dropped == 0
    ts, imu, emg, batt = frames[0]
    assert emg.shape == (10, CH) and len(imu) == 9 and 0 <= batt <= 100
    garbage = b"\x01\x02\xAA\x03" + data[:FRAME_LEN * 3]
    frames, dropped = HexParser().feed(garbage)
    assert len(frames) == 3 and dropped == 4


def test_hex_parser_across_chunk_boundaries():
    data = _demo_bytes("hex", 20)
    p, got = HexParser(), 0
    for i in range(0, len(data), 7):               # feed in awkward 7-byte chunks
        got += len(p.feed(data[i:i + 7])[0])
    assert got == 20


def test_ascii_parser_accepts_only_eight_ints():
    rows, bad = AsciiParser().feed(b"1 2 3 4 5 6 7 8\r\n1 2 3\r\n9 9 9 9 9 9 9 5000\r\npartial 1 2")
    assert rows == [[1, 2, 3, 4, 5, 6, 7, 8]] and bad == 2


# ---------------------------------------------------------------- link
def test_link_elects_mode_and_stamps_samples():
    link = Link()
    link.ingest(_demo_bytes("hex", 20))
    assert link.mode == "hex"
    assert link.emg.total == 200 and link.tsamp.total == 200
    t = link.tsamp.last(200)[:, 0]
    assert np.allclose(np.diff(t)[:9], 0.002)      # 500 Hz inside a frame
    gen = link.gen
    link.ingest(_demo_bytes("ascii", 10))          # button press: ASCII now
    assert link.mode == "ascii" and link.gen == gen + 1
    assert link.emg.total == link.tsamp.total > 0


# ---------------------------------------------------------------- filters
def test_offline_filter_matches_live_pipeline():
    s = Settings(rotate=3, extra_notches=[9.77])
    s.mute[5] = True
    fs, center = 500.0, 127.0
    raw = 127 + np.random.default_rng(0).normal(0, 5, (1500, CH)).round()
    live = Pipeline(s)
    live.configure(s, fs, center)
    y = np.vstack([live.process(raw[i:i + 17])[0] for i in range(0, len(raw), 17)])
    off = filter_offline(raw - center, s, fs)
    assert np.abs(off[-250:] - y[-250:]).max() < 0.1
    assert np.all(y[:, 5] == 0)                    # muted logical channel


def test_signature_tracks_filter_but_not_calibration():
    a = Settings()
    b = copy.deepcopy(a)
    b.rest = [1.0] * CH                            # calibration is not part of the filter
    assert signature(a) == signature(b)
    b.rotate = 2
    assert signature(a) != signature(b)
    json.loads(signature(a))


def test_design_skips_stages_above_nyquist():
    from studio.pipeline import design
    sos, notes = design(Settings(), 67.1)          # ASCII rate: 60 Hz notch impossible
    assert len(notes) >= 1 and len(sos) > 0


# ---------------------------------------------------------------- noise metrics
def test_noise_metrics_flag_charger_line():
    fs, n = 500.0, 2000
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    raw = 127 + rng.normal(0, 3, (n, CH)).round()
    raw[:, 7] += np.round(12 * ((t * 9.77) % 1.0 - 0.5))     # charger-like sawtooth on ch8
    sig = raw - 127
    vals, grades, lines = analysis.compute(raw, sig, fs, dict(center=127, full=255, mode="hex"), None)
    assert set(grades) == {m.key for m in analysis.METRICS}
    assert grades["line"][7] == "bad" and abs(lines[7][0] - 9.77) < 1.0
    assert grades["line"][0] in ("ok", "info")


# ---------------------------------------------------------------- learner maths
def test_features_and_geodesic():
    x = np.random.default_rng(2).normal(0, 3, (250, CH))
    f = features(x, 500.0)
    assert f.shape == (64,) and np.isfinite(f).all()
    a = np.random.default_rng(3).normal(0, 0.3, (4, 15, 3))
    assert np.allclose(geodesic_deg(a, a), 0, atol=1e-3)


def _learner() -> PoseLearner:
    return PoseLearner.__new__(PoseLearner)       # maths only, no hub


def test_chance_test_separates_real_relation_from_spurious_drift():
    from studio.pose import chance_test, detrend
    rng = np.random.default_rng(4)
    n = 600
    T = np.arange(n) / 17.0                        # ~17 camera frames per second
    S = np.zeros(n, int)
    X = rng.normal(size=(n, 64)).cumsum(axis=0) * 0.05 + rng.normal(size=(n, 64))
    z = rng.normal(size=n)
    X[:, 0] += 2 * z
    Y = np.zeros((n, 45))
    Y[:, 9:18] = 0.3 * np.tanh(z)[:, None]         # pose driven by a fast EMG feature...
    Y += rng.normal(0, 0.2, (n, 45)).cumsum(axis=0) * 0.05   # ...plus slow drift
    pl = _learner()
    gain, p, _, _ = chance_test(detrend(X, T, S), detrend(Y, T, S), 120, pl._ridge_select)
    assert gain > 20 and p < 0.05
    # Two unrelated slow drifts. Without detrending this was called real (p = 0.00) because
    # both keep drifting into the validation stretch; ridge alone reported +28%.
    Ynull = rng.normal(0, 0.2, (n, 45)).cumsum(axis=0) * 0.1
    assert pl._ridge_select(X, Ynull, 120)[3] > 10              # the spurious 'gain' is there...
    gain, p, _, _ = chance_test(detrend(X, T, S), detrend(Ynull, T, S), 120, pl._ridge_select)
    assert p >= 0.05                                            # ...and the test is not fooled
