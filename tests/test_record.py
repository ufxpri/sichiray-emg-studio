import json
import os
import time

import numpy as np

from studio.device.link import Link
from studio.hand.camera_link import CameraLink
from studio.record.protocol import GESTURES, Plan, Runner
from studio.record.session import SessionRecorder


def test_order_balanced_random_no_repeats():
    p = Plan(seed=3)
    sets = p.order()
    assert len(sets) == p.sets
    for seq in sets:
        assert len(seq) == p.trials_per_set
        assert all(seq.count(g[0]) == p.reps for g in GESTURES)
        assert all(a != b for a, b in zip(seq, seq[1:]))
    assert sets[0] != sets[1]
    assert Plan(seed=3).order() == sets


def test_runner_logs_every_phase_on_plan_even_with_late_ticks():
    p = Plan(sets=2, reps=1, seed=1)
    r = Runner(p)
    r.start_set(100.0)
    assert (r.trial, r.phase) == (0, "ready")
    # jump straight into trial 2's hold: trial 0 hold/rest, trial 1 ready/hold/rest, trial 2 ready/hold logged
    r.update(100.0 + 2 * p.trial_s + p.ready_s + 0.5)
    assert (r.trial, r.phase) == (2, "hold")
    kinds = [(e.trial, e.kind) for e in r.events if e.trial >= 0]
    assert kinds == [(0, "ready"), (0, "hold"), (0, "rest"), (1, "ready"), (1, "hold"), (1, "rest"),
                     (2, "ready"), (2, "hold")]
    hold2 = [e for e in r.events if (e.trial, e.kind) == (2, "hold")][0]
    assert hold2.t_plan == 100.0 + 2 * p.trial_s + p.ready_s
    assert hold2.label == r.order[0][2]
    r.update(100.0 + p.trials_per_set * p.trial_s)
    assert r.phase == "between"
    r.start_set(500.0)
    r.update(500.0 + p.trials_per_set * p.trial_s + 1)
    assert r.phase == "done"
    assert sum(e.kind == "hold" for e in r.events) == 2 * p.trials_per_set


def test_session_writes_raw_emg_and_prompts(tmp_path):
    link = Link()
    rec = SessionRecorder(link, CameraLink(), root=str(tmp_path))
    link.open_demo("hex")
    deadline = time.monotonic() + 5
    while link.mode != "hex" and time.monotonic() < deadline:
        time.sleep(0.05)
    plan = Plan(reps=1, seed=2)
    r = Runner(plan)
    folder = rec.start("테스트 사람", "노트", plan, want_video=False)
    t0 = time.perf_counter()
    r.start_set(t0)
    time.sleep(1.2)
    r.update(time.perf_counter())
    r.abort(time.perf_counter())
    rec.stop()
    link.close()
    s = rec.finish(r, "aborted")

    assert os.path.basename(folder).endswith("_테스트_사람")
    emg = np.load(os.path.join(folder, "emg.npz"))
    assert emg["emg"].shape[1] == 8 and len(emg["emg"]) == len(emg["t"]) == 10 * len(emg["frame_ts_ms"])
    assert len(emg["emg"]) > 300
    raw = open(os.path.join(folder, "emg_bytes.bin"), "rb").read()
    hand = np.load(os.path.join(folder, "hand.npz"))
    assert str(hand["source"]) == "live" and s["hand"]["source"] == "live"
    assert len(raw) == emg["chunk_len"].sum() == s["emg"]["bytes"]
    assert 400 < s["emg"]["fs_device_clock"] < 600
    pr = np.load(os.path.join(folder, "prompts.npz"))
    assert list(pr["kind"][:2]) == ["set_start", "ready"] and pr["kind"][-1] == "abort"
    meta = json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))
    assert meta["participant"] == "테스트 사람" and meta["status"] == "aborted"
    assert meta["t0_perf"] <= t0 and meta["labels"]["9"] == "wrist_ext"
    assert os.path.exists(os.path.join(folder, "hand.npz"))


def test_session_without_hand_model_leaves_joints_to_offline(tmp_path):
    link = Link()
    cam = CameraLink()
    cam.has_model = False          # camera-only worker: frames never carry hands
    rec = SessionRecorder(link, cam, root=str(tmp_path))
    link.open_demo("hex")
    deadline = time.monotonic() + 5
    while link.mode != "hex" and time.monotonic() < deadline:
        time.sleep(0.05)
    plan = Plan(reps=1, seed=2)
    r = Runner(plan)
    folder = rec.start("x", "", plan, want_video=False)
    r.start_set(time.perf_counter())
    time.sleep(0.3)
    r.abort(time.perf_counter())
    rec.stop()
    link.close()
    s = rec.finish(r, "aborted")
    assert not os.path.exists(os.path.join(folder, "hand.npz"))   # hand_offline.py writes it later
    assert s["hand"] == {"frames": 0, "with_right_hand": 0, "source": "none"}
    meta = json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))
    assert meta["hand"]["source"] == "none"


def test_plan_gesture_subset():
    from studio.record.protocol import PRESETS
    _label, gset, ids, _names = PRESETS["fist_numbers"]
    p = Plan(sets=1, reps=1, seed=5, gestures=ids, gesture_set=gset)
    assert p.trials_per_set == 6
    (seq,) = p.order()
    assert sorted(seq) == sorted(ids)
    assert "gestures" in __import__("dataclasses").asdict(p)


def test_gesture_sets_have_own_numbering_and_pictures():
    from studio.record.protocol import GESTURE_SETS, PRESETS
    from studio.ui.hand_pic import POSES
    for gs in GESTURE_SETS.values():
        assert 0 not in gs.ids and len(set(gs.ids)) == len(gs.ids)
        assert all(k in POSES for k in gs.labels.values())
    for _label, gset, ids, names in PRESETS.values():
        assert set(ids) <= set(GESTURE_SETS[gset].ids) and set(names) <= set(ids)
    puppet = GESTURE_SETS["puppet"]
    assert puppet.ids == (1, 2, 3, 4, 5, 6, 7)        # its own numbering, not after the basic set
    assert puppet.labels[6] == "radial_dev" and GESTURE_SETS["basic"].labels[6] == "three"


def test_session_meta_records_the_gesture_set(tmp_path):
    from studio.record.protocol import GESTURE_SETS
    link = Link()
    rec = SessionRecorder(link, CameraLink(), root=str(tmp_path))
    plan = Plan(reps=1, seed=4, gesture_set="puppet", gestures=GESTURE_SETS["puppet"].ids)
    r = Runner(plan)
    assert sorted(r.order[0]) == list(range(1, 8))
    folder = rec.start("p", "", plan, want_video=False, shown_names={2: "보"})
    r.start_set(time.perf_counter())
    r.abort(time.perf_counter())
    rec.stop()
    rec.finish(r, "aborted")
    meta = json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))
    assert meta["gesture_set"] == "puppet" and meta["plan"]["gesture_set"] == "puppet"
    assert meta["labels"]["4"] == "wrist_flex" and meta["label_names_ko"]["2"] == "손 펴기"
    assert meta["label_names_shown"]["2"] == "보" and len(meta["gestures"]) == 7


def test_plan_fixed_list_for_makeup():
    p = Plan(sets=1, reps=1, seed=3, gesture_set="puppet", gestures=(1, 4), fixed=(4, 1, 4))
    assert p.trials_per_set == 3
    (seq,) = p.order()
    assert sorted(seq) == [1, 4, 4] and seq == [4, 1, 4]       # the only order without a repeat


def _synthetic_session(folder, gap_trial, wrong_trial):
    """3 gestures x 4 reps, each gesture a different loud channel; one dropout, one wrong gesture."""
    fs, rng = 500.0, np.random.default_rng(0)
    plan = Plan(sets=1, reps=4, seed=1, gesture_set="puppet", gestures=(1, 2, 3))
    (order,) = plan.order()
    kinds, ts, trials, labels = [], [], [], []
    for k, L in enumerate(order):
        t0 = k * plan.trial_s
        for kind, off in (("ready", 0), ("hold", plan.ready_s), ("rest", plan.ready_s + plan.hold_s)):
            kinds.append(kind); ts.append(t0 + off); trials.append(k); labels.append(L)
    kinds.append("set_end"); ts.append(len(order) * plan.trial_s); trials.append(-1); labels.append(0)
    t = np.arange(0, ts[-1], 1 / fs)
    amp = np.full((len(t), 8), 2.0)
    for k, L in enumerate(order):
        h0 = k * plan.trial_s + plan.ready_s
        m = (t >= h0) & (t < h0 + plan.hold_s)
        done = order[k + 1] if k == wrong_trial else L      # the wrong trial does the next gesture
        amp[m, (done - 1) * 2] = 25.0
    emg = np.clip(127 + rng.normal(0, 1, amp.shape) * amp, 0, 255).astype(np.uint8)
    h0 = gap_trial * plan.trial_s + plan.ready_s
    keep = ~((t >= h0 + 0.5) & (t < h0 + 2.5))
    np.savez(os.path.join(folder, "emg.npz"), emg=emg[keep], t=t[keep])
    np.savez(os.path.join(folder, "prompts.npz"), kind=np.array(kinds), t=np.array(ts), t_plan=np.array(ts),
             set=np.zeros(len(ts), np.int16), trial=np.array(trials, np.int16), label=np.array(labels, np.int16))
    meta = {"gesture_set": "puppet", "plan": __import__("dataclasses").asdict(plan), "participant": "x",
            "label_names_shown": {"0": "휴식", "1": "주먹", "2": "손 펴기", "3": "검지 펴기"}}
    json.dump(meta, open(os.path.join(folder, "meta.json"), "w", encoding="utf-8"))
    return order


def test_quality_flags_dropout_and_wrong_gesture(tmp_path):
    from studio.record.quality import check_session
    order = _synthetic_session(str(tmp_path), gap_trial=2, wrong_trial=5)
    assert order[5] != order[6]           # plans never repeat a gesture back to back
    sc = check_session(str(tmp_path))
    bad = {c.trial: c for c in sc.bad}
    assert set(bad) == {2, 5}
    assert "누락" in bad[2].reasons[0] and bad[2].coverage < 0.5
    assert bad[5].predicted == order[6] and bad[5].own < 0.3
    assert sc.cv_accuracy > 0.95


def test_dataset_keeps_clean_trials_and_valid_makeups(tmp_path):
    from studio.record.dataset import export
    from studio.record.protocol import GESTURE_SETS
    base = tmp_path / "s" / "20260101-100000_a"
    base.mkdir(parents=True)
    order = _synthetic_session(str(base), gap_trial=2, wrong_trial=5)
    gs = GESTURE_SETS["puppet"]
    meta = json.load(open(base / "meta.json", encoding="utf-8"))
    meta.update(prompts={"trials_done": len(order)}, labels={str(k): v for k, v in gs.labels.items()},
                label_names_ko={str(k): v for k, v in gs.names.items()})
    json.dump(meta, open(base / "meta.json", "w", encoding="utf-8"))
    # a makeup recorded with the band in place: same synthetic generator, re-doing trial 2 (the dropout)
    mk = tmp_path / "s" / "20260101-101000_a"
    mk.mkdir()
    _synthetic_session(str(mk), gap_trial=99, wrong_trial=99)
    m2 = dict(meta, makeup_of={"session": base.name, "replaces": [{"set": 0, "trial": 2, "label": order[2]}]})
    json.dump(m2, open(mk / "meta.json", "w", encoding="utf-8"))
    info = export(str(tmp_path / "s"), str(tmp_path / "out"))
    d = np.load(tmp_path / "out" / "emg.npz")
    # base: 12 trials - dropout (re-recorded) - wrong gesture = 10; makeup: all 12 match the original
    assert info["trials"] == 22 and len(d["label"]) == 22
    assert info["session_notes"][mk.name].count("유효") == 1
    i = 0
    s = slice(d["offset"][i], d["offset"][i] + d["length"][i])
    assert d["t"][s][0] < 0 < d["t_rest"][i] < d["t"][s][-1]


def test_split_writes_one_file_per_trial_by_category(tmp_path):
    from studio.record.dataset import split
    from studio.record.protocol import GESTURE_SETS
    base = tmp_path / "s" / "20260101-100000_a"
    base.mkdir(parents=True)
    order = _synthetic_session(str(base), gap_trial=2, wrong_trial=5)
    gs = GESTURE_SETS["puppet"]
    meta = json.load(open(base / "meta.json", encoding="utf-8"))
    meta.update(prompts={"trials_done": len(order)}, labels={str(k): v for k, v in gs.labels.items()},
                label_names_ko={str(k): v for k, v in gs.names.items()})
    json.dump(meta, open(base / "meta.json", "w", encoding="utf-8"))
    info = split(str(tmp_path / "s"), str(tmp_path / "out"))
    assert info["files"] == {"정상": 10, "데이터누락": 1, "신호이상": 1}
    out = tmp_path / "out" / "x"
    assert len(list((out / "데이터누락").iterdir())) == 1 and len(list((out / "정상").iterdir())) == 10
    (f,) = (out / "신호이상").iterdir()
    lines = f.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "t,phase," + ",".join(f"ch{c}" for c in range(1, 9))
    assert {ln.split(",")[1] for ln in lines[1:]} == {"ready", "hold", "rest"}
