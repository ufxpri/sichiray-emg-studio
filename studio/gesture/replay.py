"""Play a recorded session through the live decoder, as if the band were on: a model test
without the band.

    python -m studio.gesture.replay data/sessions/<session> --model data/models/gesture/<name>.npz

Calibration uses the first --calib seconds of the session (no labels, like the window's
calibration). Every decision is then scored against the prompt on screen at that moment, in the
steady parts the model is trained on (hold 0.5-3.0 s, rest 0.5-2.0 s after the prompt). For a fair
number, the model must not have been trained on this session (train with --exclude).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import net as gn


def replay(folder: str, model: str, calib_s: float = 40.0, chunk: int = 17, smooth: int = gn.SMOOTH,
           min_prob: float = 0.0, hold: int = 1):
    g = gn.GestureNet.load(model)
    e = np.load(os.path.join(folder, "emg.npz"))
    p = np.load(os.path.join(folder, "prompts.npz"))
    t, x = e["t"], e["emg"].astype(np.float64) - gn.CENTER
    dec = gn.LiveDecoder(g, smooth=smooth)
    stab = gn.Stabilizer(min_prob, hold)
    t_cal = t[0] + calib_s
    dec.start_collecting()
    out, calibrated = [], False
    for a in range(0, len(t), chunk):
        if not calibrated and t[a] >= t_cal:
            dec.finish_collecting(min_windows=50)
            calibrated = True
        ds = dec.feed(t[a:a + chunk], x[a:a + chunk])
        if calibrated:
            for d in ds:
                shown = stab.update(d)
                out.append(gn.Decision(d.t, d.proba, d.raw, shown if shown is not None else d.label))
    # truth: the prompt showing at each decision, steady part only
    kind, pt, lab = p["kind"], p["t"], p["label"]
    truth, pred = [], []
    for d in out:
        i = int(np.searchsorted(pt, d.t, side="right")) - 1
        if i < 0 or kind[i] not in ("hold", "rest"):
            continue
        since = d.t - pt[i]
        lo, hi = gn.HOLD_S if kind[i] == "hold" else gn.REST_S
        if lo + gn.WIN / gn.FS <= since <= hi:
            truth.append(int(lab[i]) if kind[i] == "hold" else 0)
            pred.append(d.label)
    truth, pred = np.array(truth, int), np.array(pred, int)
    if not len(truth):
        raise SystemExit("채점할 판정이 없습니다 (보정 시간이 세션보다 길거나, 신호가 없습니다)")
    rec = {int(c): float(np.mean(pred[truth == c] == c)) for c in np.unique(truth)}
    shown_seq = np.array([d.label for d in out])
    minutes = (out[-1].t - out[0].t) / 60 if len(out) > 1 else 1.0
    prompts = int(np.sum(np.isin(kind, ("hold", "rest")) & (pt >= out[0].t) & (pt <= out[-1].t))) if out else 0
    return {"decisions": len(out), "scored": len(truth), "balanced_acc": float(np.mean(list(rec.values()))),
            "recall": rec, "gaps": dec.gaps,
            "switches_per_min": float(np.sum(shown_seq[1:] != shown_seq[:-1]) / minutes),
            "prompt_changes_per_min": prompts / minutes, "calib_windows": dec.placement.windows,
            "names": g.meta["names"], "truth": truth, "pred": pred, "classes": g.classes}


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="replay a recorded session through the live gesture decoder")
    ap.add_argument("session")
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", type=float, default=40.0, help="seconds at the start used for calibration")
    ap.add_argument("--smooth", type=int, default=gn.SMOOTH, help="windows averaged (x 0.1 s)")
    ap.add_argument("--min-prob", type=float, default=0.0, help="stabilizer: probability needed to switch")
    ap.add_argument("--hold", type=int, default=1, help="stabilizer: decisions in a row needed to switch")
    a = ap.parse_args(argv)
    r = replay(a.session, a.model, a.calib, smooth=a.smooth, min_prob=a.min_prob, hold=a.hold)
    n = r["names"]
    name = lambda c: n.get(str(c), str(c))
    print(f"{os.path.basename(a.session.rstrip('/\\'))}: 판정 {r['decisions']}회, 채점 {r['scored']}회 "
          f"(보정 창 {r['calib_windows']}개, 끊김 {r['gaps']}회)")
    print(f"화면 전환 {r['switches_per_min']:.0f}회/분 (안내가 바뀐 횟수 {r['prompt_changes_per_min']:.0f}회/분)")
    print(f"균형 정확도 {r['balanced_acc']:.3f}  " + ", ".join(f"{name(c)} {v:.2f}" for c, v in r["recall"].items()))
    cl = r["classes"]
    print("\n행 = 정답, 열 = 판정 (%)")
    print(" " * 12 + "".join(f"{name(c)[:5]:>7s}" for c in cl))
    for c in cl:
        m = r["truth"] == c
        if m.any():
            print(f"{name(c)[:10]:12s}" + "".join(f"{100 * np.mean(r['pred'][m] == k):7.0f}" for k in cl))


if __name__ == "__main__":
    main()
