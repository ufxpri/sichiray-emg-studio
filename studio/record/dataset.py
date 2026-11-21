"""Build a training dataset from the recorded sessions, keeping only trials that passed the checks.

    python -m studio.record.dataset [--root data/sessions] [--out data/datasets/<name>] [--set puppet]
    python -m studio.record.dataset --split --out data/datasets/<name>   # one CSV per trial, by quality

Sessions are grouped by wear: a full session plus the makeup sessions that complete it
(makeups of makeups included). Which trials count:
- full session: studio.record.quality.check_session (EMG complete, signal matches its
  gesture among the session's other trials); trials a makeup re-recorded are dropped.
- makeup session: quality.check_makeup against its original; if the band was moved in
  between, none of its trials count. Trials a later makeup re-recorded are dropped.
- sessions that never finished one full set (false starts) are left out.

Every kept trial is exported raw (uint8 as received, no filtering) from its ready prompt to
the end of its rest, so onset and relaxation are there too; times are relative to the hold
prompt. See the README.md written next to the data.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import quality as q


DATA, SIGNAL, GOOD, UNJUDGED = "데이터누락", "신호이상", "정상", "판정불가"
CATEGORIES = (GOOD, DATA, SIGNAL, UNJUDGED)


@dataclass
class Trial:
    participant: str
    wear: str               # the full session this trial's band placement belongs to
    session: str
    source: str             # "base" or "makeup"
    set: int
    trial: int
    rep: int
    label: int
    own: float
    coverage: float
    max_gap_s: float
    doubt: bool
    reasons: tuple[str, ...] = ()
    replaced: bool = False  # a later makeup re-recorded this trial

    @property
    def category(self) -> str:
        """데이터누락: EMG missing in the hold (dropout) or too little left to judge.
        신호이상: EMG complete but the signal does not match its gesture (wrong posture, another
        gesture, no effort, or a makeup recorded after the band was moved).
        판정불가: EMG complete, but no other normal trial of the same gesture in the placement to
        compare with (sessions with two repetitions per gesture), so the signal is unverified.
        정상: none of these."""
        if any("누락" in r or "너무 짧음" in r for r in self.reasons):
            return DATA
        if self.reasons and all(r == q.NO_REFERENCE for r in self.reasons):
            return UNJUDGED
        return SIGNAL if self.reasons else GOOD

    @property
    def kept(self) -> bool:
        return self.category == GOOD and not self.replaced



def _meta(folder: str) -> dict:
    return json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))


def wear_key(session: str) -> str:
    """The band placement a full session belongs to. Old recordings were split into one folder
    per set (<name>_set1, <name>_set2) without taking the band off: same placement."""
    return re.sub(r"_set\d+$", "", session)


def collect(root: str, gesture_set: str = "puppet"):
    """-> every recorded trial (Trial, see .category / .kept), per-session notes, recorded
    counts per (participant, label), session metas."""
    names = sorted(os.path.basename(d.rstrip("/\\")) for d in glob.glob(os.path.join(root, "*", ""))
                   if all(os.path.exists(os.path.join(d, f)) for f in ("meta.json", "emg.npz", "prompts.npz")))
    meta = {n: _meta(os.path.join(root, n)) for n in names}
    names = [n for n in names if meta[n].get("gesture_set", "basic") == gesture_set]

    def base_of(n: str) -> str:
        mk = meta[n].get("makeup_of")
        return base_of(mk["session"]) if mk and mk["session"] in meta else n

    replaced = defaultdict(set)      # session -> {(set, trial)} re-recorded by a makeup
    for n in names:
        mk = meta[n].get("makeup_of")
        if mk:
            replaced[mk["session"]] |= {(r["set"], r["trial"]) for r in mk["replaces"]}

    trials: list[Trial] = []
    notes: list[tuple[str, str]] = []
    recorded = defaultdict(int)
    bases = [n for n in names if not meta[n].get("makeup_of")]
    checked: dict[str, q.SessionCheck] = {}

    def check(b: str) -> q.SessionCheck:
        """Full sessions of one placement are checked together."""
        if b not in checked:
            group = [n for n in bases if wear_key(n) == wear_key(b) and n in complete]
            for sc in q.check_sessions([os.path.join(root, n) for n in group]):
                checked[os.path.basename(sc.folder)] = sc
        return checked[b]

    complete = set()
    for b in bases:
        m = meta[b]
        plan = m.get("plan", {})
        planned = plan.get("gestures") or [k for k in m.get("labels", {}) if int(k)]
        if m["prompts"]["trials_done"] >= plan.get("reps", 1) * len(planned):
            complete.add(b)
    for b in bases:
        m = meta[b]
        plan = m.get("plan", {})
        # old sessions have no gesture list in the plan: every label of their set was asked
        planned = plan.get("gestures") or [k for k in m.get("labels", {}) if int(k)]
        if m["prompts"]["trials_done"] < plan.get("reps", 1) * len(planned):
            notes.append((b, f"제외: 1세트를 끝내지 못한 세션 ({m['prompts']['trials_done']}회)"))
            continue
        who = m.get("participant", "")
        sc = check(b)
        groups = [(b, "base", sc.trials)]
        for n in sorted(x for x in names if x != b and base_of(x) == b):
            mc = q.check_makeup(sc, os.path.join(root, n), replaced[b])
            groups.append((n, "makeup", mc.trials))
            notes.append((n, f"{b}의 보충: " + (mc.why_invalid or
                                               f"유효 (휴식 인식 {100 * mc.rest_ok:.0f}%, 본인 동작 중앙값 {100 * mc.median_own:.0f}%)")))
        for n, src, checks in groups:
            for c in checks:
                recorded[(who, c.label)] += 1
                trials.append(Trial(who, wear_key(b), n, src, c.set, c.trial, c.rep, c.label, c.own, c.coverage,
                                    c.max_gap_s, bool(c.doubts), tuple(c.reasons),
                                    (c.set, c.trial) in replaced[n]))
    return trials, notes, recorded, meta


class _Segments:
    """Raw EMG of one trial, ready prompt .. end of rest, times relative to the hold prompt."""

    def __init__(self, root: str, meta: dict) -> None:
        self.root, self.meta, self.cache = root, meta, {}

    def __call__(self, k: Trial):
        if k.session not in self.cache:
            self.cache.clear()
            e = np.load(os.path.join(self.root, k.session, "emg.npz"))
            p = np.load(os.path.join(self.root, k.session, "prompts.npz"))
            self.cache[k.session] = (e["emg"], e["t"], p)
        x, t, p = self.cache[k.session]
        kind, pt = p["kind"], p["t"]
        i = int(np.flatnonzero((kind == "hold") & (p["set"] == k.set) & (p["trial"] == k.trial))[0])
        t_ready, t_hold, t_rest = pt[i - 1], pt[i], pt[i + 1]
        t_end = pt[i + 2] if i + 2 < len(kind) else t_rest + self.meta[k.session]["plan"]["rest_s"]
        a, b = int(np.searchsorted(t, t_ready)), int(np.searchsorted(t, t_end))
        return (x[a:b], (t[a:b] - t_hold).astype(np.float32),
                float(t_ready - t_hold), float(t_rest - t_hold), float(t_end - t_hold))


def export(root: str, out: str, gesture_set: str = "puppet") -> dict:
    trials, notes, recorded, meta = collect(root, gesture_set)
    kept = [k for k in trials if k.kept]
    if not kept:
        raise SystemExit("내보낼 정상 시도가 없습니다")
    os.makedirs(out, exist_ok=True)
    people = list(dict.fromkeys(k.participant for k in kept))
    wears = list(dict.fromkeys(k.wear for k in kept))
    any_meta = meta[kept[0].session]
    labels = {int(k): v for k, v in any_meta["labels"].items()}
    names = {int(k): v for k, v in any_meta["label_names_ko"].items()}

    # ------------------------------------------------------------ raw segments
    emg, tt, rows = [], [], []
    offset = 0
    segment = _Segments(root, meta)
    for idx, k in enumerate(kept):
        x, t, t_ready, t_rest, t_end = segment(k)
        emg.append(x)
        tt.append(t)
        rows.append(dict(index=idx, participant=k.participant, participant_id=people.index(k.participant),
                         wear=k.wear, wear_id=wears.index(k.wear), session=k.session, source=k.source,
                         set=k.set, trial=k.trial, rep=k.rep, label=k.label, key=labels[k.label],
                         name=names[k.label], own=round(k.own, 3), doubt=int(k.doubt),
                         coverage=round(k.coverage, 3), max_gap_s=round(k.max_gap_s, 3),
                         t_ready=round(t_ready, 4), t_rest=round(t_rest, 4),
                         t_end=round(t_end, 4), offset=offset, length=len(x)))
        offset += len(x)
    col = lambda key, dt: np.array([r[key] for r in rows], dt)
    np.savez(os.path.join(out, "emg.npz"),
             emg=np.concatenate(emg), t=np.concatenate(tt),
             offset=col("offset", np.int64), length=col("length", np.int64), label=col("label", np.int16),
             participant_id=col("participant_id", np.int16), wear_id=col("wear_id", np.int16),
             t_ready=col("t_ready", np.float32), t_rest=col("t_rest", np.float32), t_end=col("t_end", np.float32),
             own=col("own", np.float32), doubt=col("doubt", np.int8))
    with open(os.path.join(out, "trials.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------ summary
    good = defaultdict(int)
    for k in kept:
        good[(k.participant, k.label)] += 1
    gl = sorted(labels)[1:]
    lines = ["| 사람 | " + " | ".join(names[l] for l in gl) + " | 합계 |", "|---" * (len(gl) + 2) + "|"]
    for who in people:
        cells = []
        for l in gl:
            g, bad = good[(who, l)], recorded[(who, l)] - good[(who, l)]
            cells.append(f"{g} ({bad})" if bad else str(g))
        tg = sum(good[(who, l)] for l in gl)
        tb = sum(recorded[(who, l)] for l in gl) - tg
        lines.append(f"| {who} | " + " | ".join(cells) + f" | {tg} ({tb}) |")
    info = {
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "gesture_set": gesture_set, "fs": q.FS, "labels": labels, "label_names_ko": names,
        "participants": people, "wears": wears, "trials": len(rows), "samples": offset,
        "sources": sorted({k.session for k in kept}),
        "criteria": {k: getattr(q, k) for k in ("MIN_COVERAGE", "MAX_GAP_S", "MIN_OWN", "DOUBT_OWN", "HOLD_SKIP_S",
                                                 "REST_SKIP_S", "MAKEUP_REST_OK", "MAKEUP_MEDIAN_OWN")},
        "session_notes": dict(notes),
        "studio_commit": _commit(),
    }
    json.dump(info, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README.format(n=len(rows), people=len(people), wears=len(wears), table="\n".join(lines),
                               notes="\n".join(f"- `{s}`: {t}" for s, t in notes) or "- 없음",
                               labels=", ".join(f"{k} {v}" for k, v in labels.items())))
    return info | {"table": "\n".join(lines)}


def split(root: str, out: str, gesture_set: str = "puppet") -> dict:
    """Every recorded trial as its own CSV file, sorted by participant and category:

        out/<participant>/<정상|데이터누락|신호이상>/<label>_<name>_<session time>_s<set>t<trial>.csv

    One file = one trial, ready + hold + rest; columns t (s, 0 = hold prompt), phase
    (ready / hold / rest), ch1..ch8 (raw uint8). out/index.csv lists every file with the reason."""
    trials, notes, recorded, meta = collect(root, gesture_set)
    if not trials:
        raise SystemExit("녹화된 시도가 없습니다")
    os.makedirs(out, exist_ok=True)
    any_meta = meta[trials[0].session]
    labels = {int(k): v for k, v in any_meta["labels"].items()}
    names = {int(k): v for k, v in any_meta["label_names_ko"].items()}
    segment = _Segments(root, meta)
    rows = []
    for k in trials:
        x, t, t_ready, t_rest, t_end = segment(k)
        cat = k.category
        folder = os.path.join(out, k.participant, cat)
        os.makedirs(folder, exist_ok=True)
        stamp = k.session.split("_")[0].split("-")[-1]
        fname = f"{k.label}_{names[k.label].replace(' ', '')}_{stamp}_s{k.set + 1}t{k.trial + 1:02d}.csv"
        phase = np.where(t < 0, "ready", np.where(t < t_rest, "hold", "rest"))
        with open(os.path.join(folder, fname), "w", encoding="utf-8", newline="") as fh:
            fh.write("t,phase," + ",".join(f"ch{c + 1}" for c in range(8)) + "\n")
            fh.writelines(f"{tv:.4f},{ph}," + ",".join(map(str, xv)) + "\n"
                          for tv, ph, xv in zip(t.tolist(), phase.tolist(), x.tolist()))
        rows.append(dict(participant=k.participant, category=cat, file=f"{k.participant}/{cat}/{fname}",
                         label=k.label, key=labels[k.label], name=names[k.label], session=k.session,
                         source=k.source, wear=k.wear, set=k.set, trial=k.trial, rep=k.rep,
                         reason=" / ".join(k.reasons), replaced_by_makeup=int(k.replaced),
                         in_dataset=int(k.kept), own="" if k.own != k.own else round(k.own, 3),
                         doubt=int(k.doubt), coverage=round(k.coverage, 3), max_gap_s=round(k.max_gap_s, 3),
                         t_ready=round(t_ready, 4), t_rest=round(t_rest, 4), t_end=round(t_end, 4),
                         samples=len(x)))
    with open(os.path.join(out, "index.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    people = list(dict.fromkeys(k.participant for k in trials))
    gl = sorted(labels)[1:]
    count = defaultdict(int)
    for k in trials:
        count[(k.participant, k.category, k.label)] += 1
    head = "| 사람 | 분류 | " + " | ".join(names[l] for l in gl) + " | 합계 |"
    lines = [head, "|---" * (len(gl) + 3) + "|"]
    for who in people:
        for cat in CATEGORIES:
            cells = [count[(who, cat, l)] for l in gl]
            lines.append(f"| {who if cat == GOOD else ''} | {cat} | " + " | ".join(map(str, cells))
                         + f" | {sum(cells)} |")
    table = "\n".join(lines)
    kinds = ("다른 동작처럼 보임", "착용 상태가 원본과 다름 (보충 녹화 무효)", "비교할 시도 없음")

    def kind(k: Trial) -> str:
        r = " ".join(k.reasons)
        return kinds[1] if "착용" in r else kinds[2] if "비교 불가" in r else kinds[0]
    sig = defaultdict(int)
    for k in trials:
        if k.category == SIGNAL:
            sig[(k.participant, kind(k))] += 1
    detail = "\n".join(["| 사람 | " + " | ".join(kinds) + " |", "|---" * (len(kinds) + 1) + "|"]
                       + [f"| {who} | " + " | ".join(str(sig[(who, c)]) for c in kinds) + " |" for who in people])
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(SPLIT_README.format(n=len(rows), table=table, detail=detail,
                                     notes="\n".join(f"- `{s}`: {t}" for s, t in notes) or "- 없음"))
    return {"trials": len(rows), "participants": people, "table": table, "session_notes": dict(notes),
            "files": {c: sum(1 for r in rows if r["category"] == c) for c in CATEGORIES}}


SPLIT_README = """# EMG 시도별 품질 분류 (시도 하나 = 파일 하나)

`python -m studio.record.dataset --split` 으로 만든 폴더입니다. 녹화된 시도 {n}개를 하나씩
CSV 파일로 나누고, 사람마다 **정상 / 데이터누락 / 신호이상 / 판정불가** 폴더로 분류했습니다.
녹화를 더 하면 같은 명령으로 다시 만들면 됩니다 (판정은 매번 처음부터 다시 계산).

## 폴더와 파일

    <사람>/<정상|데이터누락|신호이상>/<동작번호>_<동작이름>_<세션시각>_s<세트>t<시도>.csv

예: `정한결/데이터누락/2_손펴기_102730_s1t11.csv` = 정한결, 10:27:30 세션, 1세트 11번째 시도, 손 펴기.

파일 하나는 시도 하나의 **준비(1초) → 동작 유지(3초) → 휴식(2초)** 전체입니다.

| 열 | 내용 |
|---|---|
| `t` | 초. **유지 안내가 뜬 순간이 0**. 준비는 음수, 유지는 0 ~ 약 3, 휴식은 그 뒤 |
| `phase` | `ready` / `hold` / `rest` (화면 안내 기준) |
| `ch1` ~ `ch8` | 밴드 8채널 원본 값 (uint8, 중심 127, 필터 안 함, 500 Hz) |

샘플 간격은 2 ms 입니다. `t` 가 그보다 크게 건너뛰는 곳이 무선 끊김으로 데이터가 빠진 곳입니다.
필터를 걸지 않은 원본 그대로 넣은 것은, 학습할 때 필터와 창 길이를 자유롭게 정하기 위해서입니다.

`index.csv` (엑셀로 열림) 에 모든 파일이 한 줄씩 있습니다: 분류, **분류한 이유(`reason`)**,
원본 세션, 보충 녹화 여부, 판정 점수(`own`, `coverage`, `max_gap_s`), 학습 데이터셋 포함 여부(`in_dataset`).

## 왜 이렇게 나눴나

녹화가 잘못되는 원인은 크게 두 가지이고, 성격이 달라서 따로 나눴습니다.

1. **기계 쪽 문제** — 블루투스가 끊겨서 신호 자체가 비어 있음. 사람은 동작을 제대로 했어도 데이터가 없습니다.
2. **사람 쪽 문제** — 신호는 다 들어왔는데, 자세가 틀렸거나 다른 동작을 했거나 힘을 안 줬음.

둘을 섞어 두면 "이 사람은 이 동작을 못 한다"와 "그때 배터리가 약했다"를 구분할 수 없고, 다시 녹화할 때
대처도 다릅니다 (1번은 배터리·거리 확인, 2번은 동작 설명을 다시). 그래서 판정도 두 단계로 따로 합니다.

### 1단계: 데이터누락 — 신호가 빠졌는가

**동작 유지 구간(3초)** 만 봅니다. 준비·휴식 구간의 짧은 끊김은 학습에 쓰는 부분이 아니라서 따지지 않습니다.

| 기준 | 값 | 이유 |
|---|---|---|
| 들어온 샘플 비율 | **90% 미만이면 누락** | 3초 중 0.3초 이상 빠지면 그 시도의 대표성이 떨어짐 |
| 가장 긴 끊김 | **0.2초 초과면 누락** | 분석 창이 0.2초라, 이보다 긴 구멍은 창 하나가 통째로 비어 동작 중간이 사라짐 |
| 남은 분석 창 | **10개 미만이면 누락** ("유효 신호 구간이 너무 짧음") | 남은 신호로는 2단계 판정 자체를 할 수 없음 |

밴드 내부 시계는 끊겨도 계속 가기 때문에 `t` 의 빈 곳 길이가 정확히 빠진 시간입니다.
끊김은 주로 배터리가 약할 때 몰려서 생겼습니다.

### 2단계: 신호이상 — 그 동작처럼 보이는가

신호가 다 들어온 시도는 "정말 그 동작을 했는가"를 봅니다. 정답 동작을 사람이 일일이 영상으로 확인할 수 없으니,
**같은 사람의 다른 시도들과 비교** 합니다.

1. 유지 구간의 앞 0.5초(반응·힘 주는 중)를 빼고, 0.2초 창(0.05초씩 이동)마다 채널별 세기(log RMS)와
   파형 길이를 계산합니다 (8채널 × 2 = 16개 특징). 필터는 20~240 Hz 대역 + 60 Hz 계열 전원 잡음 제거.
2. **판정할 시도 하나만 빼고** 같은 세션의 나머지 시도 + 휴식 구간으로 분류기(LDA)를 학습합니다.
   휴식도 하나의 동작으로 넣어서, 힘을 안 준 시도는 "휴식"으로 잡히게 했습니다.
3. 빼 둔 시도의 창들을 분류해서, **본인 동작으로 분류된 비율(`own`)** 을 봅니다.

| `own` | 판정 | 이유 |
|---|---|---|
| **30% 미만** | 신호이상 | 대부분의 시간이 다른 동작(또는 휴식)과 더 닮음 → 자세가 틀렸거나 다른 동작을 함 |
| 30 ~ 50% | 정상 (애매, `doubt`=1) | 비슷한 동작끼리(예: 손 펴기 ↔ 손목 젖히기)는 원래 일부 겹침. 버리기엔 아깝고 영상 확인 권장 |
| 50% 이상 | 정상 | |

이 판정은 두 번 돌립니다. 처음 판정에서 이상으로 나온 시도는 두 번째 판정의 학습에서 빼서,
잘못된 시도 하나가 그 동작의 기준을 흐려 다른 정상 시도까지 이상으로 몰리지 않게 했습니다.

### 보충 녹화는 원본과 비교

보충 녹화 세션은 시도 수가 적고, 밴드를 다시 찼으면 보충 녹화끼리는 서로 닮아 보여서 자기들끼리 비교하면
다 통과해 버립니다. 그래서 **원본 세션의 정상 시도로 학습한 분류기** 로 판정합니다.

- 원본 분류기가 보충 녹화의 휴식을 휴식으로 50% 이상 알아보거나, 동작들의 `own` 중앙값이 70% 이상이면 → 같은 착용 상태, 유효
- 둘 다 아니면 → **밴드 위치가 바뀐 것**으로 보고 그 세션의 시도를 모두 신호이상("착용 상태가 원본과 달라 무효")

밴드를 다시 차면 같은 사람·같은 동작도 신호가 크게 달라져서(세션 간 정확도 0.4~0.8) 원본과 섞어 쓸 수 없기 때문입니다.
이 시도들은 신호 자체는 멀쩡할 수 있으니, 필요하면 `reason` 으로 골라 따로 쓰세요.

### 판정불가

신호는 끊김 없이 다 들어왔지만, 같은 착용 안에 **같은 동작의 다른 정상 시도가 없어서** 2단계 비교를 할 수 없었던 시도입니다.
동작당 2회만 녹화한 세션에서 한 시도가 이상하면 남은 하나가 여기로 옵니다. 신호가 이상하다는 뜻이 아니라 확인하지 못했다는 뜻이라
신호이상과 따로 두었고, 확인되지 않았으므로 학습용 데이터셋에는 넣지 않았습니다. 영상으로 확인하면 살릴 수 있습니다.

### 같은 착용의 폴더는 함께 판정

예전 녹화처럼 한 번 찬 상태로 녹화한 것을 세트별 폴더(`..._set1`, `..._set2`)로 나눈 경우, 두 폴더를 한 착용으로 묶어
함께 판정합니다. 비교할 시도가 두 배가 되어 판정이 더 정확해지고, 학습할 때도 같은 착용으로 다룹니다.

### 둘 다 해당하면

누락이 있으면 신호 판정이 빠진 부분 때문에 틀어지므로, **데이터누락이 우선** 입니다. `reason` 에는 두 이유가 다 적혀 있습니다.

### 정상인데 `in_dataset` = 0 인 것

판정은 통과했지만 녹화 중에 애매해서 보충 녹화로 다시 찍은 시도입니다. 같은 동작이 두 번 세어지지 않게
학습용 데이터셋(`puppet-20261121`)에서는 뺐습니다. 신호는 정상이라 정상 폴더에 그대로 두었습니다.

### 빠진 세션

1세트를 끝내지 못하고 멈춘 세션(이름에 `_결측`)은 넣지 않았습니다. 시도가 2~10개뿐이라 같은 동작의 다른 시도가 없어
2단계 판정을 할 수 없고, 신호는 끊김 없이 다 들어와 데이터누락도 아닙니다.

### 한계

- 같은 동작을 같은 방식으로 **여러 번 똑같이 틀리면** 서로 닮아서 정상으로 통과할 수 있습니다 (비교 기준이 본인의 다른 시도라서).
- 30% 기준은 경험값입니다. 애매(`doubt`=1)한 시도는 세션 폴더의 영상으로 확인하는 게 가장 확실합니다.

## 사람별 개수

{table}

신호이상의 이유:

{detail}

## 세션별 메모

{notes}
"""


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        return ""


README = """# EMG gesture dataset (clean trials only)

Built by `python -m studio.record.dataset` from the recorded sessions. {n} trials,
{people} participants, {wears} wears (a wear = one full session + the makeups recorded with the
band in the same place). Only trials that passed the quality checks are here; criteria and
the source sessions are in `meta.json`.

## Trials per participant: good (excluded)

{table}

Excluded = recorded but judged unusable: EMG dropout, signal that looks like another gesture,
a makeup recorded after the band was moved, or a trial a makeup re-recorded.

## Sessions

{notes}

## Files

| File | Contents |
|---|---|
| `emg.npz` | `emg` (N, 8) uint8 raw samples as received (centre 127, unfiltered, 500 Hz), all trials concatenated; `t` (N,) seconds relative to the trial's **hold** prompt; per trial: `offset`, `length` into `emg`, `label`, `participant_id`, `wear_id`, `t_ready` (<0), `t_rest`, `t_end` phase boundaries in the same time base, `own` (fraction of hold windows the check classified as the trial's gesture), `doubt` (1 = borderline, 0.3-0.5) |
| `trials.csv` | the same per-trial table with names: participant, wear (full session), session, source (base / makeup), set, trial, rep, label, key, name, own, doubt, coverage, max_gap_s, ... |
| `meta.json` | labels, participants, wears, criteria, notes per session, studio commit |

Labels: {labels}.

A trial runs from its ready prompt to the end of its rest. Participants start moving at the
ready prompt, so label by time with care: the gesture is roughly `t_ready + 0.2 s .. t_rest`,
the steady part `0.5 .. t_rest`, rest `t_rest + 0.7 .. t_end`. Gaps in `t` larger than 2 ms are
missing samples (the hold itself is complete; ready and rest may have small holes).

```python
import numpy as np
d = np.load("emg.npz")
for i in range(len(d["label"])):
    s = slice(d["offset"][i], d["offset"][i] + d["length"][i])
    x, t = d["emg"][s].astype(np.float32) - 127, d["t"][s]
    hold = (t >= 0.5) & (t < d["t_rest"][i])
```

Normalise per wear (`wear_id`), not across: the band placement changes between wears.
"""


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="export the clean trials of recorded sessions")
    ap.add_argument("--root", default=os.path.join("data", "sessions"))
    ap.add_argument("--out", default=os.path.join("data", "datasets", f"puppet-{time.strftime('%Y%m%d-%H%M')}"))
    ap.add_argument("--set", default="puppet", help="gesture set to export")
    ap.add_argument("--split", action="store_true",
                    help="every recorded trial as its own CSV, by participant and 정상/데이터누락/신호이상")
    a = ap.parse_args()
    if a.split:
        info = split(a.root, a.out, a.set)
        print(f"{a.out}: {info['trials']} trials, " + ", ".join(f"{c} {n}" for c, n in info["files"].items()))
        print()
        print(info["table"])
        raise SystemExit
    info = export(a.root, a.out, a.set)
    print(f"{a.out}: {info['trials']} trials, {info['samples']:,} samples, "
          f"{len(info['participants'])} participants, {len(info['wears'])} wears\n")
    print(info["table"])
    for s, t in info["session_notes"].items():
        print(f"  {s}: {t}")
