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
    assert len(raw) == emg["chunk_len"].sum() == s["emg"]["bytes"]
    assert 400 < s["emg"]["fs_device_clock"] < 600
    pr = np.load(os.path.join(folder, "prompts.npz"))
    assert list(pr["kind"][:2]) == ["set_start", "ready"] and pr["kind"][-1] == "abort"
    meta = json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))
    assert meta["participant"] == "테스트 사람" and meta["status"] == "aborted"
    assert meta["t0_perf"] <= t0 and meta["labels"]["9"] == "wrist_ext"
    assert os.path.exists(os.path.join(folder, "hand.npz"))
