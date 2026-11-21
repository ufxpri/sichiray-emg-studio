"""Extract hand joints from a recorded session's video, after the fact.

    <venv>\\python.exe worker\\hand_offline.py <session folder> [<session folder> ...]

Reads video.avi + video_frames.npz (every camera frame with its perf_counter time) and
runs the same detector + WiLoR pipeline as the live worker, frame by frame, writing
hand.npz in the same layout the studio writes during a live session. Every frame gets
an estimate, not only the ones a slow live worker kept up with, so this is the way to
get dense ground truth from a PC recorded without a GPU: record with the camera only,
then run this on a machine with CUDA.

Runs in the hand venv (PyTorch), like hand_worker.py. Does not import studio.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_worker as hw   # noqa: E402

REQUIRED = ("video.avi", "video_frames.npz")


def process(folder: str, est: "hw.HandEstimator", every: int, out_name: str) -> dict:
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(folder, f))]
    if missing:
        raise FileNotFoundError(f"{folder}: {', '.join(missing)} 없음")
    vf = np.load(os.path.join(folder, "video_frames.npz"))
    t_all = vf["t_ns"].astype(np.float64) / 1e9
    cap = cv2.VideoCapture(os.path.join(folder, "video.avi"))
    if not cap.isOpened():
        raise RuntimeError(f"{folder}: video.avi를 열 수 없습니다")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_video, len(t_all))
    if n_video != len(t_all):
        hw.log(f"warning: video has {n_video} frames, video_frames.npz has {len(t_all)}; using {n}")

    rows: list[tuple] = []     # (t, w, h, how, right-hand dict or None, n_hands)
    est.track, est.since_detect = [], 10 ** 9
    t0 = time.time()
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            hw.log(f"warning: read failed at frame {i}; stopping")
            break
        if i % every:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands, how = est.detect(rgb)
        res = est.estimate(rgb, hands)
        if not res and how == "track":
            est.since_detect = 10 ** 9
        right = next((h for h in res if h["is_right"]), None)
        rows.append((t_all[i], frame.shape[1], frame.shape[0], how, right, len(res)))
        if len(rows) % 50 == 0:
            el = time.time() - t0
            hw.log(f"{os.path.basename(folder)}: {i + 1}/{n} frames, {len(rows) / el:.1f} fps, "
                   f"eta {(n - i - 1) / every / max(len(rows) / el, 1e-6):.0f}s")
    cap.release()

    H = len(rows)
    nan = lambda *shape: np.full((H, *shape), np.nan, np.float32)
    kp2d, kp3d, orient, pose, betas = nan(21, 2), nan(21, 3), nan(3), nan(15, 3), nan(10)
    conf, bbox = np.full(H, np.nan, np.float32), nan(4)
    for k, (_, _, _, _, h, _) in enumerate(rows):
        if h is not None:
            kp2d[k], kp3d[k], betas[k] = h["kp2d"], h["kp3d"], h["betas"]
            orient[k], pose[k] = np.reshape(h["global_orient"], 3), np.reshape(h["hand_pose"], (15, 3))
            conf[k], bbox[k] = h["det_conf"], h["bbox"]
    has_right = np.array([r[4] is not None for r in rows])
    np.savez(os.path.join(folder, out_name),
             t=np.array([r[0] for r in rows]), has_right=has_right,
             n_hands=np.array([r[5] for r in rows], np.int8),
             kp2d=kp2d, kp3d=kp3d, global_orient=orient, hand_pose=pose, betas=betas,
             det_conf=conf, bbox=bbox, how=np.array([r[3] for r in rows]),
             image_size=np.array([[r[1], r[2]] for r in rows], np.int32).reshape(-1, 2),
             source=np.array("offline"), video_frame_index=np.arange(0, n, every)[:H])
    info = {"frames": H, "with_right_hand": int(has_right.sum()), "source": "offline",
            "video_frames": n, "every": every, "seconds": round(time.time() - t0, 1),
            "device": str(est.device), "dtype": str(est.dtype).replace("torch.", "")}
    meta_path = os.path.join(folder, "meta.json")
    if os.path.exists(meta_path) and out_name == "hand.npz":
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["hand"] = info
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+", help="session folders (with video.avi and video_frames.npz)")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--redetect", type=int, default=5, help="run YOLO at least every N processed frames")
    ap.add_argument("--every", type=int, default=1, help="process every Nth video frame")
    ap.add_argument("--models", default=hw.MODELS)
    ap.add_argument("--out", default="hand.npz", help="output file name inside the session folder")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args()
    folders = [os.path.abspath(f) for f in args.folders]
    todo = []
    for f in folders:
        if os.path.exists(os.path.join(f, args.out)) and not args.force:
            hw.log(f"skip {f}: {args.out} already exists (--force to redo)")
        else:
            todo.append(f)
    if not todo:
        return
    os.makedirs(hw.HOME, exist_ok=True)
    os.chdir(hw.HOME)   # ultralytics writes a relative runs/ folder
    t0 = time.time()
    est = hw.HandEstimator(args.dtype, args.conf, args.redetect, args.models)
    hw.log(f"model loaded in {time.time() - t0:.1f}s on {est.device}")
    failed = 0
    for f in todo:
        try:
            info = process(f, est, args.every, args.out)
            print(f"{f}: {info['frames']} frames, right hand in {info['with_right_hand']}, {info['seconds']}s")
        except Exception as e:   # keep going with the other folders
            failed += 1
            hw.log(f"FAILED {f}: {e!r}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
