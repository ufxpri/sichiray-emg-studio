"""Per-trial quality check of a recorded gesture session: which trials need recording again.

Two kinds of problem:
- data: EMG samples missing during the hold (wireless dropouts show up as gaps in `t`;
  the device clock keeps counting, so the gap length is exact).
- signal: the trial does not look like its gesture. Each trial is classified by a
  shrinkage LDA trained on every *other* trial of the session (leave one trial out), so a
  wrong posture, a different gesture or no effort at all shows up as a trial whose windows
  mostly go to another class. Rest windows (from the rest phases) are a class too, so a
  gesture that was not done is caught as "rest".

    python -m studio.record.quality <session folder> [...]
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

FS = 500.0
WIN_S, STEP_S = 0.2, 0.05
HOLD_SKIP_S = 0.5           # reaction / onset at the start of the hold
REST_SKIP_S = 0.7           # relaxing at the start of the rest
MIN_COVERAGE = 0.9          # fraction of the hold's samples that must be present
MAX_GAP_S = 0.2             # longest tolerated hole inside the hold
MIN_OWN = 0.3               # below this fraction of windows classified as its own gesture: re-record
DOUBT_OWN = 0.5             # below this: borderline, shown but not selected by default
MIN_WINDOWS = 10            # below this many windows the signal check is not possible
NO_REFERENCE = "같은 동작의 다른 정상 시도가 없어 신호 비교 불가"


@dataclass
class TrialCheck:
    set: int
    trial: int
    label: int
    rep: int                    # occurrence of this label in the session, 0-based
    coverage: float             # present / expected samples in the hold
    max_gap_s: float
    own: float = float("nan")   # fraction of windows classified as the trial's gesture
    predicted: int = -1         # most frequent class among the trial's windows
    reasons: list[str] = field(default_factory=list)    # re-record
    doubts: list[str] = field(default_factory=list)     # borderline, worth a look

    @property
    def bad(self) -> bool:
        return bool(self.reasons)


@dataclass
class SessionCheck:
    folder: str
    meta: dict
    trials: list[TrialCheck]
    names: dict[int, str]
    cv_accuracy: float          # window accuracy of the leave-one-trial-out classifier on clean trials
    notes: list[str]
    hold_windows: list = field(default_factory=list, repr=False)   # feature rows per trial
    rest_windows: np.ndarray | None = field(default=None, repr=False)

    @property
    def bad(self) -> list[TrialCheck]:
        return [c for c in self.trials if c.bad]


def _filter(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Band-pass 20-240 Hz + mains notches, run separately on each gap-free run of samples."""
    sos = butter(4, [20, 240], btype="band", fs=FS, output="sos")
    notch = np.vstack([tf2sos(*iirnotch(f0, 30, FS)) for f0 in (60, 120, 180, 240)])
    sos = np.vstack([sos, notch])
    out = np.zeros_like(x)
    breaks = np.flatnonzero(np.diff(t) > 1.5 / FS) + 1
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(t)]):
        if b - a > 60:
            out[a:b] = sosfiltfilt(sos, x[a:b], axis=0)
    return out


def _features(w: np.ndarray) -> np.ndarray:
    rms = np.sqrt((w ** 2).mean(0))
    wl = np.abs(np.diff(w, axis=0)).mean(0)
    return np.concatenate([np.log(rms + 1e-6), np.log(wl + 1e-6)])


def _windows(x: np.ndarray, t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Feature rows of the gap-free windows inside [t0, t1)."""
    win, step = int(WIN_S * FS), int(STEP_S * FS)
    i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
    rows = []
    for s in range(i0, i1 - win + 1, step):
        if t[s + win - 1] - t[s] < (win + 1) / FS:      # no hole inside the window
            rows.append(_features(x[s:s + win]))
    return np.asarray(rows).reshape(-1, 16)


class _LDA:
    """Shrinkage LDA on standardised features (numpy only)."""

    def fit(self, X: np.ndarray, y: np.ndarray, shrink: float = 0.1) -> "_LDA":
        self.mu0, self.sd0 = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu0) / self.sd0
        self.classes = np.unique(y)
        means = np.array([Z[y == c].mean(0) for c in self.classes])
        R = Z - means[np.searchsorted(self.classes, y)]
        S = R.T @ R / max(1, len(Z) - len(self.classes))
        S = (1 - shrink) * S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        P = np.linalg.solve(S, means.T)
        self.W, self.b = P, -0.5 * np.sum(means.T * P, axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu0) / self.sd0
        return self.classes[np.argmax(Z @ self.W + self.b, axis=1)]


def _segments(folder: str):
    """Per trial: TrialCheck with the data verdict, hold feature windows; plus all rest windows."""
    meta = json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))
    names = {int(k): v for k, v in (meta.get("label_names_shown") or meta.get("label_names_ko") or {}).items()}
    e = np.load(os.path.join(folder, "emg.npz"))
    p = np.load(os.path.join(folder, "prompts.npz"))
    t = e["t"]
    x = _filter(e["emg"].astype(np.float64) - 127.0, t)
    kind, pt, sets, trial, lab = p["kind"], p["t"], p["set"], p["trial"], p["label"]
    hold_s = float(meta.get("plan", {}).get("hold_s", 3.0))
    checks: list[TrialCheck] = []
    hold_rows, rest_rows = [], []
    reps: dict[int, int] = {}
    for i in np.flatnonzero(kind == "hold"):
        t0 = pt[i]
        t1 = pt[i + 1] if i + 1 < len(kind) else t0 + hold_s
        L = int(lab[i])
        r = reps.get(L, 0)
        reps[L] = r + 1
        i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
        ts = np.r_[t0, t[i0:i1], t1]
        c = TrialCheck(int(sets[i]), int(trial[i]), L, r, coverage=min(1.0, (i1 - i0) / ((t1 - t0) * FS)),
                       max_gap_s=float(np.diff(ts).max()))
        if c.coverage < MIN_COVERAGE or c.max_gap_s > MAX_GAP_S:
            c.reasons.append(f"EMG 누락 (유지 구간의 {100 * (1 - c.coverage):.0f}%, 최대 {c.max_gap_s:.1f}초 끊김)")
        checks.append(c)
        hold_rows.append(_windows(x, t, t0 + HOLD_SKIP_S, t1))
        if i + 1 < len(kind) and kind[i + 1] == "rest":
            r_end = pt[i + 2] if i + 2 < len(kind) else pt[i + 1] + 2.0
            rest_rows.append(_windows(x, t, pt[i + 1] + REST_SKIP_S, r_end))
    rest_X = np.vstack([r for r in rest_rows if len(r)]) if any(len(r) for r in rest_rows) else np.zeros((0, 16))
    return meta, names, checks, hold_rows, rest_X


def check_session(folder: str) -> SessionCheck:
    return check_sessions([folder])[0]


def check_sessions(folders: list[str]) -> list[SessionCheck]:
    """Check several folders recorded in one band placement together (e.g. an old recording
    split into one folder per set), so each trial is compared with all the others."""
    parts = [_segments(f) for f in folders]
    names = parts[0][1]
    checks = [c for _, _, cs, _, _ in parts for c in cs]
    hold_rows = [h for _, _, _, hs, _ in parts for h in hs]
    rests = [r for *_, r in parts if len(r)]
    rest_X = np.vstack(rests) if rests else np.zeros((0, 16))
    notes = []

    # --- signal: leave-one-trial-out LDA over gesture trials + rest. Two passes: trials that
    # fail the first pass are left out of the training data of the second, so one bad trial
    # does not blur its gesture's class for the others.
    usable = [k for k, c in enumerate(checks) if len(hold_rows[k]) >= MIN_WINDOWS]
    clean = [k for k in usable if not checks[k].bad]
    data_reasons = {k: list(c.reasons) for k, c in enumerate(checks)}
    for _pass in range(2):
        correct = total = 0
        for k, c in enumerate(checks):
            c.reasons = list(data_reasons[k])
            c.doubts = []
        for k in usable:
            c = checks[k]
            train = [j for j in clean if j != k]
            if c.label not in {checks[j].label for j in train}:
                c.reasons.append(NO_REFERENCE)
                continue
            X = np.vstack([hold_rows[j] for j in train] + [rest_X])
            y = np.concatenate([np.full(len(hold_rows[j]), checks[j].label) for j in train]
                               + [np.zeros(len(rest_X), int)])
            pred = _LDA().fit(X, y).predict(hold_rows[k])
            c.own = float(np.mean(pred == c.label))
            c.predicted = int(np.bincount(pred).argmax())
            other = names.get(c.predicted, str(c.predicted))
            if c.own < MIN_OWN:
                c.reasons.append(f"신호가 다른 동작처럼 보임 ({100 * c.own:.0f}%만 본인 동작, 대부분 '{other}')")
            elif c.own < DOUBT_OWN:
                c.doubts.append(f"애매함 ({100 * c.own:.0f}%만 본인 동작, 나머지 대부분 '{other}')")
            elif k in clean:
                correct += int(np.sum(pred == c.label))
                total += len(pred)
        clean = [k for k in clean if not checks[k].bad]
    for k, c in enumerate(checks):
        if k not in usable and not any("누락" in r for r in c.reasons):
            c.reasons.append("유효 신호 구간이 너무 짧음")
    if len(rest_X) == 0:
        notes.append("휴식 구간 신호가 없어 '동작 안 함' 판정은 빠졌습니다")
    if len(folders) > 1:
        notes.append(f"같은 착용의 폴더 {len(folders)}개를 함께 판정했습니다")
    cv = correct / total if total else float("nan")
    out, i = [], 0
    for f, (meta, nm, cs, hs, rx) in zip(folders, parts):
        out.append(SessionCheck(f, meta, checks[i:i + len(cs)], nm, cv, list(notes), hold_rows[i:i + len(cs)],
                                rest_X if len(folders) > 1 else rx))
        i += len(cs)
    return out


@dataclass
class MakeupCheck:
    folder: str
    meta: dict
    trials: list[TrialCheck]
    rest_ok: float          # fraction of the makeup's rest windows the original's model calls rest
    median_own: float       # median over the makeup's trials of the fraction classified as their gesture
    valid: bool             # recorded with the band where it was for the original

    @property
    def why_invalid(self) -> str:
        return ("" if self.valid else
                f"착용 상태가 원본과 다름 (휴식 인식 {100 * self.rest_ok:.0f}%, 본인 동작 중앙값 {100 * self.median_own:.0f}%)")


MAKEUP_REST_OK = 0.5        # makeup valid if the original's model still recognises its rest ...
MAKEUP_MEDIAN_OWN = 0.7     # ... or its trials mostly as their own gesture


def check_makeup(base: SessionCheck, folder: str, exclude: set[tuple[int, int]] = frozenset()) -> MakeupCheck:
    """Judge a makeup session against the session it completes: a model trained on the
    original's clean trials (minus `exclude`, the (set, trial) the makeups replace) and its rest.
    Comparing a makeup with itself would pass a band that was moved in between, because its
    trials then agree with each other; the original is what they have to match."""
    meta, names, checks, hold_rows, rest_X = _segments(folder)
    keep = [k for k, c in enumerate(base.trials) if not c.bad and (c.set, c.trial) not in exclude
            and len(base.hold_windows[k])]
    X = np.vstack([base.hold_windows[k] for k in keep] + [base.rest_windows])
    y = np.concatenate([np.full(len(base.hold_windows[k]), base.trials[k].label) for k in keep]
                       + [np.zeros(len(base.rest_windows), int)])
    mdl = _LDA().fit(X, y)
    rest_ok = float(np.mean(mdl.predict(rest_X) == 0)) if len(rest_X) else 0.0
    owns = []
    for c, W in zip(checks, hold_rows):
        if len(W) >= MIN_WINDOWS:
            pred = mdl.predict(W)
            c.own = float(np.mean(pred == c.label))
            c.predicted = int(np.bincount(pred).argmax())
            owns.append(c.own)
    med = float(np.median(owns)) if owns else 0.0
    valid = rest_ok >= MAKEUP_REST_OK or med >= MAKEUP_MEDIAN_OWN
    for c, W in zip(checks, hold_rows):
        if not valid:
            c.reasons.append("착용 상태가 원본과 달라 무효")
        elif len(W) < MIN_WINDOWS:
            if not c.reasons:
                c.reasons.append("유효 신호 구간이 너무 짧음")
        elif c.own < MIN_OWN:
            other = names.get(c.predicted, str(c.predicted))
            c.reasons.append(f"원본과 비교해 다른 동작처럼 보임 ({100 * c.own:.0f}%만 본인 동작, 대부분 '{other}')")
        elif c.own < DOUBT_OWN:
            c.doubts.append(f"애매함 ({100 * c.own:.0f}%만 본인 동작)")
    return MakeupCheck(folder, meta, checks, rest_ok, med, valid)


def report(sc: SessionCheck) -> str:
    n = sc.names
    lines = [f"{os.path.basename(sc.folder)}: 시도 {len(sc.trials)}개 중 보충 필요 {len(sc.bad)}개 "
             f"(정상 시도끼리 교차 분류 정확도 {sc.cv_accuracy:.2f})"]
    for c in sc.trials:
        if c.bad or c.doubts:
            lines.append(f"  {'보충' if c.bad else '확인'}  세트{c.set + 1} #{c.trial + 1:2d}  "
                         f"{n.get(c.label, c.label):10s} {c.rep + 1}회차  " + " / ".join(c.reasons + c.doubts))
    lines += [f"  참고: {m}" for m in sc.notes]
    return "\n".join(lines)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for f in sys.argv[1:]:
        print(report(check_session(f)))
