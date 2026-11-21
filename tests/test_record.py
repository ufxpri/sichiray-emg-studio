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
