"""Train the live gesture classifier on the clean trials of the recorded sessions.

Needs PyTorch (here: Python 3.13); the studio itself only loads the saved arrays.

    python -m studio.gesture.train --who 조승준 [--eval]
    python -m studio.gesture.train --who all --out data/models/gesture/all.npz

Trials: those studio.record.dataset keeps (quality check passed, not re-recorded by a makeup).
Windows: hold 0.5-3.0 s labelled with the gesture, rest 0.5-2.0 s labelled rest; each band
placement (wear) normalised by its own channel RMS and adapted with AdaBN, as the live window does
after its calibration.

--eval  leave one wear out: train on the other wears, calibrate on the held-out wear's windows
        (no labels), score smoothed windows. This is the honest number for putting the band on
        again; the in-wear cross-validation of the trial map is optimistic.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                                  # pragma: no cover
    raise SystemExit("학습에는 PyTorch가 필요합니다 (예: py -3.13 -m studio.gesture.train ...)")
from scipy.interpolate import CubicSpline

from .. import config
from ..record import dataset as ds
from . import net as gn

EPOCHS, BATCH = 30, 128


# ------------------------------------------------------------------ data
def load_windows(root: str, who: str, gesture_set: str = "puppet", exclude: tuple[str, ...] = ()):
    trials, *_rest, meta = ds.collect(root, gesture_set)
    kept = [k for k in trials if k.kept and (who == "all" or k.participant == who)
            and not any(e in k.wear for e in exclude)]
    if not kept:
        raise SystemExit(f"'{who}'의 정상 시도가 없습니다")
    sos = gn.filter_sos()
    X, Y, U, TR, TT = [], [], [], [], []
    by_session = defaultdict(list)
    for n, k in enumerate(kept):
        by_session[k.session].append(n)
    for s, idx in by_session.items():
        e = np.load(os.path.join(root, s, "emg.npz"))
        p = np.load(os.path.join(root, s, "prompts.npz"))
        t = e["t"]
        x = gn.causal_filter(e["emg"].astype(np.float64) - gn.CENTER, t, sos).astype(np.float32)
        kind, pt = p["kind"], p["t"]
        for n in idx:
            k = kept[n]
            j = int(np.flatnonzero((kind == "hold") & (p["set"] == k.set) & (p["trial"] == k.trial))[0])
            for t0, (a, b), lab in ((pt[j], gn.HOLD_S, k.label), (pt[j + 1], gn.REST_S, 0)):
                for s0 in gn.windows(x, t, t0 + a, t0 + b):
                    X.append(x[s0:s0 + gn.WIN].T)
                    Y.append(lab)
                    U.append(k.wear)
                    TR.append(n * 2 + (lab == 0))          # one sequence per hold and per rest phase
                    TT.append(t[s0])
    any_meta = meta[kept[0].session]
    info = {"labels": {int(a): b for a, b in any_meta["labels"].items()},
            "names": {int(a): b for a, b in any_meta["label_names_ko"].items()},
            "people": sorted({k.participant for k in kept}), "wears": sorted({k.wear for k in kept}),
            "trials": len(kept), "gesture_set": gesture_set}
    return (np.stack(X), np.array(Y), np.array(U), np.array(TR), np.array(TT)), info


def normalise(X: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, dict]:
    out, rms = X.copy(), {}
    for u in np.unique(U):
        m = U == u
        rms[u] = np.sqrt((X[m].astype(np.float64) ** 2).mean(axis=(0, 2))) + 1e-6
        out[m] /= rms[u][None, :, None].astype(np.float32)
    return out, rms


# ------------------------------------------------------------------ model (PyTorch twin of net.GestureNet)
class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.c = nn.ModuleList([nn.Conv1d(8, 32, 7, padding=3), nn.Conv1d(32, 64, 5, stride=2, padding=2),
                                nn.Conv1d(64, 64, 5, stride=2, padding=2)])
        self.b = nn.ModuleList([nn.BatchNorm1d(32), nn.BatchNorm1d(64), nn.BatchNorm1d(64)])
        self.drop, self.fc = nn.Dropout(0.3), nn.Linear(64, n_classes)

    def embed(self, x):
        for c, b in zip(self.c, self.b):
            x = F.relu(b(c(x)))
        return x.mean(-1)

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))


def gain_sigma(X: np.ndarray, Y: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Per channel: spread across placements of the class-mean log RMS, overall gain removed."""
    lr = np.log(np.sqrt((X.astype(np.float64) ** 2).mean(-1)) + 1e-6)
    rows = []
    for c in np.unique(Y):
        per = [lr[(U == u) & (Y == c)].mean(0) for u in np.unique(U) if np.sum((U == u) & (Y == c)) > 3]
        if len(per) > 1:
            per = np.array(per)
            rows.append((per - per.mean(1, keepdims=True)).std(0))
    return np.mean(rows, 0) if rows else np.zeros(8)


def augment(xb: np.ndarray, sig: np.ndarray, rng) -> np.ndarray:
    n = len(xb)
    on = rng.random(n) < 0.5
    g = np.exp(rng.normal(0, 1, (n, 8)) * sig)
    knots = np.linspace(0, gn.WIN - 1, 4)
    warp = np.stack([CubicSpline(knots, rng.normal(1, 0.1, 4))(np.arange(gn.WIN)) for _ in range(n)])
    return xb * np.where(on[:, None, None], g[:, :, None] * warp[:, None, :], 1.0).astype(np.float32)


def fit(X, Y, U, classes, seed=0, log=None):
    """X normalised. -> trained Net (eval mode)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    yi = np.searchsorted(classes, Y)
    net = Net(len(classes))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    nb = int(np.ceil(len(X) / BATCH))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * nb)
    cnt = np.bincount(yi, minlength=len(classes)).astype(float)
    wts = torch.tensor(cnt.sum() / (len(classes) * np.maximum(cnt, 1)), dtype=torch.float32)
    sig = gain_sigma(X, Y, U)
    yt = torch.tensor(yi)
    for ep in range(EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(len(X)), nb):
            loss = F.cross_entropy(net(torch.tensor(augment(X[b], sig, rng))), yt[b], weight=wts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        if log and (ep + 1) % 10 == 0:
            log(f"  epoch {ep + 1}/{EPOCHS}  loss {loss.item():.3f}")
    net.eval()
    return net, sig


def export(net: Net, rms: dict, meta: dict) -> gn.GestureNet:
    a = lambda p: p.detach().cpu().numpy().astype(np.float64)
    return gn.GestureNet(
        conv=[(a(c.weight), a(c.bias), c.stride[0]) for c in net.c],
        bn=[(a(b.weight), a(b.bias)) for b in net.b],
        fc=(a(net.fc.weight), a(net.fc.bias)),
        train_stats=gn.BNStats([a(b.running_mean) for b in net.b], [a(b.running_var) for b in net.b]),
        train_rms=np.mean(list(rms.values()), axis=0), meta=meta)


def check_twin(net: Net, g: gn.GestureNet, X: np.ndarray) -> float:
    """Largest difference between the PyTorch and the numpy forward pass (training statistics)."""
    with torch.no_grad():
        ref = F.log_softmax(net(torch.tensor(X)), -1).numpy()
    return float(np.abs(ref - g.log_proba(X, g.train_stats)).max())


def smoothed_accuracy(g: gn.GestureNet, place: gn.Placement, X, Y, TR, TT) -> tuple[float, dict]:
    """Balanced accuracy of the live decision (0.4 s smoothing inside each hold / rest phase)."""
    lp = g.log_proba(X / place.rms[None, :, None], place.stats)
    pred = np.empty(len(X), int)
    for s in np.unique(TR):
        m = np.flatnonzero(TR == s)
        m = m[np.argsort(TT[m])]
        L = lp[m]
        S = np.array([L[max(0, i - gn.SMOOTH + 1):i + 1].mean(0) for i in range(len(L))])
        pred[m] = np.array(g.classes)[S.argmax(1)]
    rec = {int(c): float(np.mean(pred[Y == c] == c)) for c in np.unique(Y)}
    return float(np.mean(list(rec.values()))), rec


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="train the live gesture classifier")
    ap.add_argument("--who", required=True, help="participant name, or 'all'")
    ap.add_argument("--root", default=os.path.join(config.DATA_DIR, "sessions"))
    ap.add_argument("--out", help="model file (default data/models/gesture/<who>.npz)")
    ap.add_argument("--set", default="puppet")
    ap.add_argument("--eval", action="store_true", help="also score leaving one wear out")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="leave out these wears (full-session names or parts, e.g. 111826), to test on them")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    out = a.out or os.path.join(config.DATA_DIR, "models", "gesture", f"{a.who}.npz")
    t0 = time.time()
    (X, Y, U, TR, TT), info = load_windows(a.root, a.who, a.set, tuple(a.exclude))
    classes = np.array(sorted(np.unique(Y)))
    print(f"{a.who}: 시도 {info['trials']}개, 착용 {len(info['wears'])}개, 창 {len(X):,}개")
    Xn, rms = normalise(X, U)
    names = info["names"]

    report = {}
    if a.eval:
        if len(info["wears"]) < 2:
            print("착용이 하나뿐이라 착용 교차 평가를 건너뜁니다")
        else:
            print("착용 하나씩 빼고 평가 (새로 찼을 때를 흉내):")
            scores = []
            for w in info["wears"]:
                te = U == w
                net, _ = fit(Xn[~te], Y[~te], U[~te], classes, seed=1)
                g = export(net, {k: v for k, v in rms.items() if k != w}, {"classes": classes.tolist()})
                place = gn.Placement.estimate(g, X[te])                   # calibration: no labels
                acc, rec = smoothed_accuracy(g, place, X[te], Y[te], TR[te], TT[te])
                acc0, _ = smoothed_accuracy(g, gn.Placement.default(g), X[te], Y[te], TR[te], TT[te])
                scores.append(acc)
                report[w] = {"balanced_acc": round(acc, 3), "without_calibration": round(acc0, 3),
                             "recall": {names[c]: round(v, 2) for c, v in rec.items()}}
                print(f"  {w}: {acc:.3f} (보정 없이 {acc0:.3f})  " +
                      ", ".join(f"{names[c]} {v:.2f}" for c, v in rec.items()))
            report["mean"] = round(float(np.mean(scores)), 3)
            print(f"  평균 {np.mean(scores):.3f}")

    net, sig = fit(Xn, Y, U, classes, seed=0, log=print)
    meta = {"classes": classes.tolist(), "labels": {int(c): info["labels"][int(c)] for c in classes},
            "names": {int(c): names[int(c)] for c in classes}, "people": info["people"],
            "wears": info["wears"], "trials": info["trials"], "windows": int(len(X)),
            "gesture_set": a.set, "fs": gn.FS, "gain_sigma": [round(float(s), 3) for s in sig],
            "eval_leave_one_wear_out": report, "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "studio_commit": ds._commit()}
    g = export(net, rms, meta)
    diff = check_twin(net, g, Xn[:256])
    if diff > 1e-3:
        raise SystemExit(f"numpy 계산이 PyTorch와 다릅니다 (최대 차이 {diff:g})")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    g.save(out)
    print(f"저장: {out}  (numpy/PyTorch 최대 차이 {diff:.1e}, {time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
