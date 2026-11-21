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
