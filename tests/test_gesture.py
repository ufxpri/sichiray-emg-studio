"""studio.gesture.net: the numpy CNN, its storage and the live decoder (no PyTorch needed)."""
import numpy as np

from studio.gesture import net as gn


def _random_net(seed=0, classes=(0, 1, 2, 3)):
    r = np.random.default_rng(seed)
    shapes = [(32, 8, 7, 1), (64, 32, 5, 2), (64, 64, 5, 2)]
    conv = [(r.normal(0, 0.2, (o, c, k)), r.normal(0, 0.1, o), s) for o, c, k, s in shapes]
    bn = [(1 + r.normal(0, 0.1, o), r.normal(0, 0.1, o)) for o, *_ in shapes]
    stats = gn.BNStats([np.zeros(o) for o, *_ in shapes], [np.ones(o) for o, *_ in shapes])
    meta = {"classes": list(classes), "labels": {str(c): k for c, k in zip(classes, ["rest", "fist", "open", "index"])},
            "names": {str(c): f"g{c}" for c in classes}}
    return gn.GestureNet(conv, bn, (r.normal(0, 0.3, (len(classes), 64)), np.zeros(len(classes))), stats,
                         np.ones(8), meta)


def test_conv_matches_direct_sum():
    r = np.random.default_rng(1)
    x, w, b = r.normal(size=(2, 3, 20)), r.normal(size=(4, 3, 5)), r.normal(size=4)
    for stride in (1, 2):
        got = gn._conv1d(x, w, b, stride)
        xp = np.pad(x, ((0, 0), (0, 0), (2, 2)))
        L = (20 - 1) // stride + 1
        ref = np.array([[[(xp[n, :, i * stride:i * stride + 5] * w[o]).sum() + b[o] for i in range(L)]
                         for o in range(4)] for n in range(2)])
        assert got.shape == ref.shape and np.allclose(got, ref)


def test_save_load_roundtrip(tmp_path):
    g = _random_net()
    x = np.random.default_rng(2).normal(size=(5, 8, gn.WIN))
    g.save(tmp_path / "m.npz")
    h = gn.GestureNet.load(str(tmp_path / "m.npz"))
    assert h.classes == [0, 1, 2, 3] and h.meta["names"]["1"] == "g1"
    assert np.allclose(g.log_proba(x, g.train_stats), h.log_proba(x, h.train_stats))


def test_adabn_normalises_each_layer():
    g = _random_net()
    x = np.random.default_rng(3).normal(5, 3, size=(40, 8, gn.WIN))
    st = g.adapt(x)
    # with its own statistics the first layer's pre-activation is centred
    pre = gn._conv1d(x, *g.conv[0][:2], g.conv[0][2])
    assert np.allclose(pre.mean(axis=(0, 2)), st.mean[0]) and np.allclose(pre.var(axis=(0, 2)), st.var[0])


def test_live_decoder_decides_every_100ms_and_restarts_after_gaps():
    g = _random_net()
    dec = gn.LiveDecoder(g)
    r = np.random.default_rng(4)
    t = np.arange(0, 3.0, 1 / gn.FS)
    x = r.normal(0, 10, (len(t), 8))
    t[len(t) // 2:] += 0.5                                     # a 0.5 s dropout half way
    out = []
    for a in range(0, len(t), 17):                             # hub-sized chunks
        out += dec.feed(t[a:a + 17], x[a:a + 17])
    assert dec.gaps == 1
    dt = np.diff([d.t for d in out])
    assert np.allclose(dt[dt < 0.5], 0.1)
    assert all(abs(d.proba.sum() - 1) < 1e-9 and d.label in g.classes for d in out)
    # no decision before a full gap-free window after the gap
    t_gap = t[len(t) // 2]
    assert not any(t_gap <= d.t < t_gap + (gn.WIN - 1) / gn.FS for d in out)


def test_calibration_sets_channel_levels():
    g = _random_net()
    dec = gn.LiveDecoder(g)
    t = np.arange(0, 12.0, 1 / gn.FS)
    x = np.random.default_rng(5).normal(0, 1, (len(t), 8)) * np.arange(1, 9) * 20
    dec.start_collecting()
    for a in range(0, len(t), 17):
        dec.feed(t[a:a + 17], x[a:a + 17])
    place = dec.finish_collecting(min_windows=50)
    assert place is not None and place.windows >= 100
    assert np.all(np.diff(place.rms) > 0)                      # louder channels, larger level
    assert dec.finish_collecting() is None                     # nothing collected since


def test_replay_scores_a_recorded_session(tmp_path):
    from studio.gesture.replay import replay
    from tests.test_record import _synthetic_session
    sess = tmp_path / "s"
    sess.mkdir()
    _synthetic_session(str(sess), gap_trial=99, wrong_trial=99)
    g = _random_net(classes=(0, 1, 2, 3))
    g.save(tmp_path / "m.npz")
    r = replay(str(sess), str(tmp_path / "m.npz"), calib_s=12)
    assert r["calib_windows"] >= 100 and r["scored"] > 50
    assert 0 <= r["balanced_acc"] <= 1 and set(r["recall"]) <= {0, 1, 2, 3}


def _d(label, p, classes=(0, 1, 2)):
    pr = np.full(len(classes), (1 - p) / (len(classes) - 1))
    pr[list(classes).index(label)] = p
    return gn.Decision(0.0, pr, pr, label)


def test_stabilizer_ignores_flicker_and_switches_on_a_steady_change():
    st = gn.Stabilizer(min_prob=0.6, hold=3)
    for lab in (1, 1, 1):
        st.update(_d(lab, 0.9))
    assert st.shown == 1
    for lab in (2, 1, 2, 1, 2, 2, 1):                  # flicker: never 3 in a row
        st.update(_d(lab, 0.9))
    assert st.shown == 1
    for _ in range(2):
        st.update(_d(2, 0.5))                          # 3 in a row but not confident enough
    st.update(_d(2, 0.5))
    assert st.shown == 1
    for _ in range(3):
        st.update(_d(2, 0.8))
    assert st.shown == 2


def test_signal_map_places_reference_points_on_themselves():
    r = np.random.default_rng(6)
    centres = r.normal(0, 5, (3, 64))
    lab = np.repeat([0, 1, 2], 40)
    emb = centres[lab] + r.normal(0, 0.3, (120, 64))
    xy = np.array([[-1, 0], [1, 0], [0, 1]], float)[lab] + r.normal(0, 0.05, (120, 2))
    mu, sd = emb.mean(0), emb.std(0)
    m = gn.SignalMap(((emb - mu) / sd).astype(np.float32), xy, lab, mu, sd)
    new = centres + r.normal(0, 0.3, (3, 64))          # unseen windows of each class
    got = m.place(new)
    assert np.allclose(got, [[-1, 0], [1, 0], [0, 1]], atol=0.1)


def test_replay_source_plays_recorded_bytes_at_their_pace(tmp_path):
    import json
    import time
    from studio.device.replay import ReplaySource
    blob = bytes(range(200))
    arr = np.array([10.0, 10.05, 10.10, 10.15])
    np.savez(tmp_path / "emg.npz", chunk_arrival=arr, chunk_offset=np.array([0, 50, 100, 150]),
             chunk_len=np.array([50, 50, 50, 50]))
    (tmp_path / "emg_bytes.bin").write_bytes(blob)
    np.savez(tmp_path / "prompts.npz", kind=np.array(["ready", "hold", "rest"]), t=np.array([10.0, 10.05, 10.12]),
             label=np.array([2, 2, 2]))
    json.dump({"label_names_shown": {"0": "휴식", "2": "손 펴기"}}, open(tmp_path / "meta.json", "w", encoding="utf-8"))
    src = ReplaySource(str(tmp_path), speed=1.0)
    src.open()
    got, t0 = b"", time.perf_counter()
    while not src.finished and time.perf_counter() - t0 < 2:
        got += src.read()
    assert got == blob and time.perf_counter() - t0 >= 0.14       # not faster than recorded
    assert src.prompt()[:2] == ("rest", "휴식")


def test_kalman_dot_smooths_jitter_and_follows_a_jump():
    r = np.random.default_rng(7)
    k = gn.KalmanDot()
    z = np.r_[np.zeros(40), np.ones(40)]
    Z = np.c_[z, np.zeros_like(z)] + r.normal(0, 0.12, (80, 2))
    out = np.array([k.update(v) for v in Z])
    assert np.std(np.diff(out[15:40, 0])) < 0.5 * np.std(np.diff(Z[15:40, 0]))   # steadier frame to frame
    assert out[50:, 0].mean() > 0.85                             # caught up within 1 s of the jump
    k.reset()
    assert np.allclose(k.update(np.array([0.3, -0.2])), [0.3, -0.2])
