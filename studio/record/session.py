"""One recording session on disk: raw EMG, camera video, hand joints and the prompt log.

Everything is kept as it arrived; nothing is resampled or filtered. Streams run at
their own rates and are joined later through one clock: perf_counter seconds, the
clock the EMG samples and the camera frames are both stamped with. `meta.json` ties
that clock to wall time (t0_perf <-> t0_unix).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict

import numpy as np

from .. import config
from ..device.protocol import FRAME_MS, IMU_NAMES, SAMPLES_PER_FRAME
from .protocol import GESTURES, LABELS, NAMES, Plan, Runner

FORMAT_VERSION = 1
WEAR_RULES = "v1: right arm, pod 5 charging port toward the wrist, 3-4 fingers below the elbow crease, " \
             "pod 5 on the back of the forearm (thumb side 3, palm 1, little finger 7)"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", config.ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        dirty = subprocess.run(["git", "-C", config.ROOT, "status", "--porcelain"], capture_output=True,
                               text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return ""


def safe_name(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in s.strip())
    return keep.strip("_") or "anon"


class SessionRecorder:
    """Taps the link and the camera while a session runs, then writes the folder."""

    def __init__(self, link, camera, root: str = os.path.join(config.DATA_DIR, "sessions")) -> None:
        self.link, self.camera, self.root = link, camera, root
        self.lock = threading.Lock()
        self.folder = ""
        self.meta: dict = {}
        self._bytes_fh = None
        self.active = False
        self.video_on = False
        self._reset()

    def _reset(self) -> None:
        self.chunks: list[tuple[float, int, int]] = []          # arrival, offset, length
        self.f_ts: list[int] = []
        self.f_arrival: list[float] = []
        self.f_emg: list[np.ndarray] = []
        self.f_imu: list[tuple] = []
        self.f_batt: list[int] = []
        self.f_t: list[np.ndarray] = []
        self.modes: list[tuple[float, str | None]] = []
        self.hand: list = []
        self._offset = 0
        self._mode = "?"

    # ------------------------------------------------------------ taps (other threads)
    def _on_bytes(self, data: bytes, pc: float, mode, frames, stamps) -> None:
        with self.lock:
            if not self.active:
                return
            self._bytes_fh.write(data)
            self.chunks.append((pc, self._offset, len(data)))
            self._offset += len(data)
            if mode != self._mode:
                self.modes.append((pc, mode))
                self._mode = mode
            for f, t in zip(frames, stamps):
                self.f_ts.append(f.ts)
                self.f_arrival.append(pc)
                self.f_emg.append(f.emg)
                self.f_imu.append(f.imu)
                self.f_batt.append(f.battery)
                self.f_t.append(t)

    def _on_hand(self, fr) -> None:
        with self.lock:
            if self.active:
                self.hand.append(fr)

    # ------------------------------------------------------------ lifecycle (GUI thread)
    def start(self, name: str, note: str, plan: Plan, want_video: bool) -> str:
        t0_perf, t0_unix = time.perf_counter(), time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(t0_unix))
        self.folder = os.path.join(self.root, f"{stamp}_{safe_name(name)}")
        os.makedirs(self.folder, exist_ok=True)
        self.meta = {
            "format_version": FORMAT_VERSION,
            "participant": name.strip(),
            "note": note,
            "created_local": time.strftime("%Y-%m-%d %H:%M:%S%z", time.localtime(t0_unix)),
            "t0_perf": t0_perf,
            "t0_unix": t0_unix,
            "clock": "perf_counter seconds (Windows QueryPerformanceCounter, shared by studio and camera worker)",
            "studio_commit": git_commit(),
            "wear_rules": WEAR_RULES,
            "rotation_calibration": None,
            "plan": asdict(plan),
            "labels": {str(k): v for k, v in LABELS.items()},
            "label_names_ko": {str(k): v for k, v in NAMES.items()},
            "gestures": [{"id": g[0], "key": g[1], "name": g[2], "how": g[3]} for g in GESTURES],
            "emg_source": self.link.source_name,
            "emg_mode_at_start": self.link.mode,
            "camera_device": self.camera.device,
        }
        self._reset()
        self._bytes_fh = open(os.path.join(self.folder, "emg_bytes.bin"), "wb")
        with self.lock:
            self.active = True
        self.link.tap = self._on_bytes
        self.camera.on_frame = self._on_hand
        self.video_on = bool(want_video and self.camera.start_video(self.folder))
        self.meta["video_requested"] = want_video
        self.meta["video_started"] = self.video_on
        return self.folder

    def stop(self) -> None:
        """Stop collecting. The worker still has to finish the video: poll `finish`."""
        with self.lock:
            self.active = False
        self.link.tap = None
        self.camera.on_frame = None
        if self.video_on:
            self.camera.stop_video()
        self._stop_t = time.monotonic()
        self.meta["t_end_perf"] = time.perf_counter()

    def video_pending(self) -> bool:
        return self.video_on and self.camera.video_result() is None and time.monotonic() - self._stop_t < 15

    def finish(self, runner: Runner, status: str) -> dict:
        """Write everything; returns a short summary for the screen."""
        self._bytes_fh.close()
        d = self.folder
        summary = {"folder": d, "status": status}

        # EMG: frames as decoded, plus the raw byte log they came from
        F = len(self.f_ts)
        ts = np.array(self.f_ts, np.int64)
        emg = np.array(self.f_emg, np.uint8).reshape(-1, 8) if F else np.zeros((0, 8), np.uint8)
        np.savez(os.path.join(d, "emg.npz"),
                 emg=emg,
                 t=np.concatenate(self.f_t) if F else np.zeros(0),
                 frame_ts_ms=ts,
                 frame_arrival=np.array(self.f_arrival),
                 frame_first_sample=np.arange(F) * SAMPLES_PER_FRAME,
                 imu=np.array(self.f_imu, np.uint8).reshape(-1, 9) if F else np.zeros((0, 9), np.uint8),
                 imu_names=np.array(IMU_NAMES),
                 battery=np.array(self.f_batt, np.int16),
                 chunk_arrival=np.array([c[0] for c in self.chunks]),
                 chunk_offset=np.array([c[1] for c in self.chunks], np.int64),
                 chunk_len=np.array([c[2] for c in self.chunks], np.int64))
        emg_info = {"frames": F, "samples": int(len(emg)), "bytes": self._offset,
                    "mode_changes": [[t, m] for t, m in self.modes]}
        if F > 1:
            span_dev = (ts[-1] - ts[0]) / 1000.0
            gaps = np.diff(ts)
            emg_info.update({
                "fs_device_clock": round((F - 1) * SAMPLES_PER_FRAME / span_dev, 3) if span_dev > 0 else None,
                "fs_host_clock": round((F - 1) * SAMPLES_PER_FRAME / (self.f_arrival[-1] - self.f_arrival[0]), 3),
                "frame_gaps": int(np.sum(gaps != FRAME_MS)),
                "frames_missing_est": int(np.sum(np.maximum(gaps // FRAME_MS - 1, 0))),
                "battery_start": int(self.f_batt[0]), "battery_end": int(self.f_batt[-1]),
                "t_first_sample": float(self.f_t[0][0]),
            })
        self.meta["emg"] = emg_info
        summary["emg"] = emg_info

        # hand joints, right hand only (the band is on the right arm)
        H = len(self.hand)
        nan = lambda *shape: np.full((H, *shape), np.nan, np.float32)
        kp2d, kp3d, orient, pose, betas = nan(21, 2), nan(21, 3), nan(3), nan(15, 3), nan(10)
        conf, bbox = np.full(H, np.nan, np.float32), nan(4)
        for i, fr in enumerate(self.hand):
            h = fr.right_hand()
            if h is not None:
                kp2d[i], kp3d[i], orient[i], pose[i], betas[i] = h.kp2d, h.kp3d, h.global_orient, h.hand_pose, h.betas
                conf[i], bbox[i] = h.det_conf, h.bbox
        np.savez(os.path.join(d, "hand.npz"),
                 t=np.array([fr.t_capture for fr in self.hand]),
                 has_right=np.array([fr.right_hand() is not None for fr in self.hand]),
                 n_hands=np.array([len(fr.hands) for fr in self.hand], np.int8),
                 kp2d=kp2d, kp3d=kp3d, global_orient=orient, hand_pose=pose, betas=betas,
                 det_conf=conf, bbox=bbox, how=np.array([fr.how for fr in self.hand]),
                 image_size=np.array([[fr.width, fr.height] for fr in self.hand], np.int32).reshape(-1, 2))
        self.meta["hand"] = {"frames": H, "with_right_hand": int(sum(fr.right_hand() is not None for fr in self.hand))}
        summary["hand"] = self.meta["hand"]

        # video: written by the worker; frame times are perf_counter_ns there
        if self.video_on:
            res = self.camera.video_result() or {"error": "worker did not answer"}
            self.meta["video"] = res
            summary["video"] = res

        # the prompt log
        ev = runner.events
        np.savez(os.path.join(d, "prompts.npz"),
                 t=np.array([e.t for e in ev]), t_plan=np.array([e.t_plan for e in ev]),
                 kind=np.array([e.kind for e in ev]), set=np.array([e.set for e in ev], np.int16),
                 trial=np.array([e.trial for e in ev], np.int16), label=np.array([e.label for e in ev], np.int16),
                 order=np.array(runner.order, np.int16))
        self.meta["prompts"] = {"events": len(ev), "trials_done": sum(e.kind == "hold" for e in ev)}
        self.meta["status"] = status
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(SESSION_README)
        summary["trials_done"] = self.meta["prompts"]["trials_done"]
        return summary


SESSION_README = """# EMG gesture session

Recorded with EMG Studio (Sichiray EMG PRO 8-channel armband). Every stream is stored as
it arrived, at its own rate. All times are **perf_counter seconds**, one clock shared by
the EMG link and the camera worker; `meta.json` maps it to wall time (`t0_perf` <-> `t0_unix`).

| File | Contents |
|---|---|
| `meta.json` | participant, note, plan (sets, reps, timing, seed), labels, wearing rules, rates, battery, studio commit |
| `emg.npz` | `emg` (N, 8) uint8 raw HEX samples, physical channel order 1..8 (centre 127, no filtering); `t` (N,) sample times; per HEX frame (10 samples each): `frame_ts_ms` device clock (+20 ms/frame, exact rate), `frame_arrival`, `imu` (F, 9) raw bytes (`imu_names`), `battery`; `chunk_*` index into `emg_bytes.bin` |
| `emg_bytes.bin` | every byte received from the armband, unparsed |
| `video.avi` | every camera frame (MJPG). The fps in its header is nominal; use `video_frames.npz` for frame times |
| `video_frames.npz` | `t_ns` perf_counter **nanoseconds** of each video frame, `cam_seq` camera frame counter (gaps = frames the camera produced but were not written) |
| `hand.npz` | WiLoR right-hand estimate per processed frame (fewer than video frames): `t`, `has_right`, `kp2d` (pixels), `kp3d` (m), `global_orient`, `hand_pose` (MANO axis-angle), `betas`, `det_conf`, `bbox`; NaN where no right hand |
| `prompts.npz` | what the screen asked for: `t` shown, `t_plan` scheduled, `kind` (set_start / ready / hold / rest / set_end / abort), `set`, `trial`, `label`; `order` the gesture order per set |

Labels (`label`): 0 rest, 1 scissors, 2 rock, 3 paper, 4..8 the numbers 1..5, 9 wrist extension.
Prompt times say when a gesture was **asked for**; the actual movement starts a reaction time
later. A trial is ready (gesture announced) -> hold (do it) -> rest.

EMG sample times: `t` puts the device clock on perf_counter (offset fixed by the least-delayed
frame, about +-20 ms). For an exact rate use `frame_ts_ms`; for the start, `t[0]`.
"""
