"""Behaviour tests: parsing, buffers, timestamps, the filter chain, noise metrics,
the learner's evaluation maths and persistence. No window, serial port or GPU."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from studio.core import analysis
from studio.core.filters import design, filter_offline
from studio.core.pipeline import Pipeline
from studio.core.ring import Ring
from studio.core.settings import Calibration, FilterSpec, Settings
from studio.device.demo import DemoSource
from studio.device.link import Link
from studio.device.parsers import AsciiParser, HexParser
from studio.device.protocol import CH, FRAME_LEN, MODES
from studio.pose.evaluate import chance_test, detrend, geodesic_deg, select_ridge
from studio.pose.features import features
from studio.pose.models import MlpRegressor, PoseModel, RidgeRegressor


# ---------------------------------------------------------------- buffers
def test_ring_wraps_and_reads_since():
    r = Ring(10, 2)
    r.write(np.arange(24).reshape(12, 2))          # more than capacity
    assert r.total == 12
    assert r.last(3)[:, 0].tolist() == [18, 20, 22]
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
    assert len(frames) == 50 and dropped == 0
    f = frames[0]
    assert f.emg.shape == (10, CH) and len(f.imu) == 9 and 0 <= f.battery <= 100 and f.ts == 20
    frames, dropped = HexParser().feed(b"\x01\x02\xAA\x03" + data[:FRAME_LEN * 3])
    assert len(frames) == 3 and dropped == 4


def test_hex_parser_across_chunk_boundaries():
    data = _demo_bytes("hex", 20)
    p, got = HexParser(), 0
    for i in range(0, len(data), 7):               # feed in awkward 7-byte chunks
        got += len(p.feed(data[i:i + 7])[0])
    assert got == 20


def test_console_segmentation_matches_parser():
    data = _demo_bytes("hex", 5)
    buf = bytearray(b"\x13\x37" + data + data[:40])
    seg = HexParser.segment(buf)
    assert [k for k, _ in seg] == ["junk"] + ["frame"] * 5
    assert len(buf) == 40                          # incomplete frame kept for next time


def test_ascii_parser_accepts_only_eight_ints():
    rows, bad = AsciiParser().feed(b"1 2 3 4 5 6 7 8\r\n1 2 3\r\n9 9 9 9 9 9 9 5000\r\npartial 1 2")
    assert rows == [[1, 2, 3, 4, 5, 6, 7, 8]] and bad == 2


# ---------------------------------------------------------------- link
def test_link_elects_mode_and_stamps_samples():
    link = Link()
    link.ingest(_demo_bytes("hex", 20))
    assert link.mode == "hex"
    gen, raw, t, pos = link.read_since(-1, 0)
    assert gen == link.gen and raw.shape == (200, CH) and len(t) == 200 and pos == 200
    assert np.allclose(np.diff(t)[:9], 0.002)      # 500 Hz inside a frame
    link.ingest(_demo_bytes("ascii", 10))          # button press: ASCII now
    assert link.mode == "ascii" and link.gen == gen + 1
    gen2, raw, t, _ = link.read_since(gen, pos)    # stale position from the old mode
    assert gen2 == gen + 1 and len(raw) == len(t) and 0 < len(raw) <= 10


def test_snapshot_counts_lost_frames():
    link = Link()
    link.ingest(_demo_bytes("hex", 601))           # frame 600's timestamp skips one frame
    s = link.snapshot()
    assert s.frames_ok == 601 and s.lost_frames == 1 and s.resync_bytes == 3


# ---------------------------------------------------------------- settings and filters
def test_settings_flat_json_roundtrip_and_old_file(tmp_path):
    s = Settings(FilterSpec(rotate=2, extra_notches=(9.77,)), 4.0, Calibration((1.0,) * CH, (5.0,) * CH))
    p = tmp_path / "s.json"
    s.save(str(p))
    assert Settings.load(str(p)) == s
    old = {"hp_on": True, "hp_hz": 30.0, "mute": [False] * CH, "rest": None, "mvc": None, "env_hz": 5.0}
    p.write_text(json.dumps(old), encoding="utf-8")
    assert Settings.load(str(p)).filter.hp_hz == 30.0


def test_signature_tracks_filter_but_not_calibration():
    a = Settings()
    assert a.filter.signature() == replace(a, calib=Calibration((1.0,) * CH)).filter.signature()
    assert a.filter.signature() != a.with_filter(rotate=2).filter.signature()


def test_offline_filter_matches_live_pipeline():
    spec = FilterSpec(rotate=3, extra_notches=(9.77,), mute=tuple(i == 5 for i in range(CH)))
    raw = 127 + np.random.default_rng(0).normal(0, 5, (1500, CH)).round()
    live = Pipeline(Settings(spec), MODES["hex"])
    y = np.vstack([live.process(raw[i:i + 17])[1] for i in range(0, len(raw), 17)])
    off = filter_offline(raw - 127, spec, 500.0)
    assert np.abs(off[-250:] - y[-250:]).max() < 0.1
    assert np.all(y[:, 5] == 0)                    # muted logical channel


def test_design_skips_stages_above_nyquist():
    sos, notes = design(FilterSpec(), 67.1)        # ASCII rate: 60 Hz notch impossible
    assert len(notes) >= 1 and len(sos) > 0


# ---------------------------------------------------------------- noise metrics
def test_noise_metrics_flag_charger_line():
    fs, n = 500.0, 2000
    t = np.arange(n) / fs
    raw = 127 + np.random.default_rng(1).normal(0, 3, (n, CH)).round()
    raw[:, 7] += np.round(12 * ((t * 9.77) % 1.0 - 0.5))     # charger-like sawtooth on ch8
    vals, grades, lines = analysis.compute(raw, raw - 127, fs, dict(center=127, full=255, mode="hex"), None)
    assert set(grades) == {m.key for m in analysis.METRICS}
    assert grades["line"][7] == "bad" and abs(lines[7][0] - 9.77) < 1.0
    assert grades["line"][0] in ("ok", "info")


# ---------------------------------------------------------------- learner maths
def test_features_and_geodesic():
    f = features(np.random.default_rng(2).normal(0, 3, (250, CH)), 500.0)
    assert f.shape == (64,) and np.isfinite(f).all()
    a = np.random.default_rng(3).normal(0, 0.3, (4, 15, 3))
    assert np.allclose(geodesic_deg(a, a), 0, atol=1e-3)


def test_chance_test_separates_real_relation_from_spurious_drift():
    rng = np.random.default_rng(4)
    n = 600
    T, S = np.arange(n) / 17.0, np.zeros(n, int)   # ~17 camera frames per second
    X = rng.normal(size=(n, 64)).cumsum(axis=0) * 0.05 + rng.normal(size=(n, 64))
    z = rng.normal(size=n)
    X[:, 0] += 2 * z
    Y = np.zeros((n, 45))
    Y[:, 9:18] = 0.3 * np.tanh(z)[:, None]         # pose driven by a fast EMG feature...
    Y += rng.normal(0, 0.2, (n, 45)).cumsum(axis=0) * 0.05   # ...plus slow drift
    gain, p, _, _ = chance_test(detrend(X, T, S), detrend(Y, T, S), 120)
    assert gain > 20 and p < 0.05
    # Two unrelated slow drifts: ridge alone reports a sizeable 'gain' (spurious regression)...
    Ynull = rng.normal(0, 0.2, (n, 45)).cumsum(axis=0) * 0.1
    assert select_ridge(X, Ynull, 120)[3] > 10
    gain, p, _, _ = chance_test(detrend(X, T, S), detrend(Ynull, T, S), 120)
    assert p >= 0.05                               # ...and the detrended test is not fooled


def test_pose_model_roundtrip_without_pickle(tmp_path):
    rng = np.random.default_rng(5)
    X, Y = rng.normal(size=(200, 64)), rng.normal(size=(200, 45))
    for reg in (RidgeRegressor(10.0), MlpRegressor()):
        m = PoseModel.train(reg, X, Y, 500.0, FilterSpec().signature(), "desc")
        path = str(tmp_path / f"{reg.kind}.npz")
        m.save(path)
        back = PoseModel.load(path)
        assert back is not None and back.filter_sig == m.filter_sig
        assert np.allclose(back.predict(X[:5]), m.predict(X[:5]), atol=1e-6)
    from sklearn.neural_network import MLPRegressor
    net = MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-3, max_iter=50, random_state=0).fit(X, Y)
    # the numpy forward pass must equal scikit-learn's
    r = MlpRegressor()
    r.Ws, r.bs = list(net.coefs_), list(net.intercepts_)
    assert np.allclose(r.predict(X[:5]), net.predict(X[:5]), atol=1e-6)
