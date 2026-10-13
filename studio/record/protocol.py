"""The guided gesture session: which gestures, in what order, for how long.

Each trial is ready (the gesture is announced) -> hold (do it) -> rest (relax). Within a
set every gesture appears REPS times in a random order with no gesture twice in a row,
like the robot-hand game the data is for. The runner is driven by a clock the caller
passes in, so it can be tested without a GUI; every phase change is logged with the
time it was actually shown, next to the time it was planned for.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# label id 0 is rest; ids are stable across sessions, never renumber
GESTURES = [
    # id, key, name shown, how to do it
    (1, "scissors", "가위", "검지·중지를 펴고 나머지는 접기"),
    (2, "rock", "주먹", "다섯 손가락 모두 접기"),
    (3, "paper", "보", "다섯 손가락 모두 펴기"),
    (4, "one", "1", "검지만 펴기"),
    (5, "two", "2", "검지·중지 펴기"),
    (6, "three", "3", "검지·중지·약지 펴기"),
    (7, "four", "4", "엄지만 접고 네 손가락 펴기"),
    (8, "five", "5", "다섯 손가락 모두 펴기"),
    (9, "wrist_ext", "손목 젖히기", "손바닥은 아래, 손등 쪽으로 손목을 끝까지 꺾기"),
]
REST_ID = 0
PHASES = ("ready", "hold", "rest")
LABELS = {0: "rest", **{g[0]: g[1] for g in GESTURES}}
NAMES = {0: "휴식", **{g[0]: g[2] for g in GESTURES}}
HINTS = {g[0]: g[3] for g in GESTURES}


@dataclass(frozen=True)
class Plan:
    sets: int = 2
    reps: int = 5
    ready_s: float = 1.0
    hold_s: float = 3.0
    rest_s: float = 2.0
    seed: int = 0

    @property
    def trial_s(self) -> float:
        return self.ready_s + self.hold_s + self.rest_s

    @property
    def trials_per_set(self) -> int:
        return self.reps * len(GESTURES)

    def order(self) -> list[list[int]]:
        """Gesture ids per set: each gesture `reps` times, shuffled, no immediate repeats."""
        rng = random.Random(self.seed)
        out = []
        for _ in range(self.sets):
            while True:
                seq = [g[0] for g in GESTURES for _ in range(self.reps)]
                rng.shuffle(seq)
                if all(a != b for a, b in zip(seq, seq[1:])):
                    break
            out.append(seq)
        return out


@dataclass
class Event:
    t: float            # when it was shown (caller's clock, seconds)
    t_plan: float       # when it was scheduled
    kind: str           # set_start, ready, hold, rest, set_end, abort
    set: int
    trial: int          # index within the set, -1 for set events
    label: int          # gesture id, 0 for rest / set events


@dataclass
class Runner:
    """Steps through one set at a time. Call `start_set(now)`, then `update(now)` often."""
    plan: Plan
    order: list[list[int]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    set_idx: int = -1
    trial: int = -1
    phase: str = "idle"          # idle, ready, hold, rest, between, done
    _t0: float = 0.0             # planned start of the current set

    def __post_init__(self) -> None:
        if not self.order:
            self.order = self.plan.order()

    @property
    def gesture(self) -> int:
        if self.phase in ("ready", "hold", "rest") and 0 <= self.trial < len(self.order[self.set_idx]):
            return self.order[self.set_idx][self.trial]
        return REST_ID

    @property
    def next_gesture(self) -> int | None:
        seq = self.order[self.set_idx] if self.set_idx >= 0 else []
        return seq[self.trial + 1] if self.trial + 1 < len(seq) else None

    def start_set(self, now: float) -> None:
        self.set_idx += 1
        self.trial = -1
        self._t0 = now
        self.events.append(Event(now, now, "set_start", self.set_idx, -1, REST_ID))
        self.update(now)

    def phase_progress(self, now: float) -> float:
        """0..1 through the current phase."""
        if self.phase not in ("ready", "hold", "rest"):
            return 0.0
        p = self.plan
        start = self._t0 + self.trial * p.trial_s
        off = {"ready": 0.0, "hold": p.ready_s, "rest": p.ready_s + p.hold_s}[self.phase]
        dur = {"ready": p.ready_s, "hold": p.hold_s, "rest": p.rest_s}[self.phase]
        return min(1.0, max(0.0, (now - start - off) / dur))

    def update(self, now: float) -> bool:
        """Advance to whatever phase `now` falls in. -> True if the phase changed.
        Times are planned from the set start, so a late tick never shifts later trials,
        and every boundary passed since the last call is logged, in order."""
        if self.phase in ("idle", "between", "done") and self.trial >= 0:
            return False
        p = self.plan
        n = len(self.order[self.set_idx])
        k = int((now - self._t0) // p.trial_s)
        if k >= n:
            self._advance(n - 1, 2, now)
            self.trial = n
            self._log(now, self._t0 + n * p.trial_s, "set_end", -1)
            self.phase = "done" if self.set_idx + 1 >= p.sets else "between"
            return True
        into = now - self._t0 - k * p.trial_s
        i = 0 if into < p.ready_s else 1 if into < p.ready_s + p.hold_s else 2
        return self._advance(k, i, now)

    def _advance(self, k: int, i: int, now: float) -> bool:
        cur = (self.trial, PHASES.index(self.phase)) if self.trial >= 0 else (0, -1)
        if (k, i) <= cur:
            return False
        p = self.plan
        kk, ii = cur[0], cur[1] + 1
        while (kk, ii) <= (k, i):
            if ii == 3:
                kk, ii = kk + 1, 0
                continue
            plan_t = self._t0 + kk * p.trial_s + (0.0, p.ready_s, p.ready_s + p.hold_s)[ii]
            self.events.append(Event(now, plan_t, PHASES[ii], self.set_idx, kk, self.order[self.set_idx][kk]))
            ii += 1
        self.trial, self.phase = k, PHASES[i]
        return True

    def abort(self, now: float) -> None:
        self._log(now, now, "abort", self.trial)
        self.phase = "done"

    def _log(self, now: float, plan_t: float, kind: str, trial: int) -> None:
        self.events.append(Event(now, plan_t, kind, self.set_idx, trial, REST_ID))
