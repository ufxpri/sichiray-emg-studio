"""Camera + 3D hand pose worker. Runs in its own process and its own venv.

    <venv>\\python.exe studio\\hand_worker.py --camera 0 --port 8790

Why a separate process: WiLoR is a 640M-parameter ViT-H plus a YOLO detector,
and PyTorch pins numpy<2 / old ultralytics that the studio does not want. A
separate interpreter also keeps inference off the studio's GIL, so the serial
reader and the plots never wait for the GPU.

Pipeline per frame
  capture thread   newest frame + perf_counter_ns() at read (same clock as the
                   studio on Windows: QueryPerformanceCounter is system-wide)
  detect           YOLO hand boxes, or the previous frame's keypoint box while
                   tracking (YOLO re-runs every --redetect frames or on loss)
  crop             256x256 patch per hand, anti-alias blur only inside the ROI
  WiLoR            MANO pose/shape, 21 joints, 778 vertices. Backbone and refine
                   net run as captured CUDA graphs: eager PyTorch on Windows is
                   launch-bound (GPU ~2% busy); graphs cut the model 64 -> 23 ms.

Must stay standalone: it runs in another interpreter and cannot import studio.
Output: TCP, one client. Each message = u32 header length, u32 jpeg length,
u32 blob length, header JSON, JPEG bytes, blob. The blob is float32 MANO vertices,
778x3 per hand in header order. The first message is {"type": "hello", "faces": ...}.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
import types
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
PROTO = 1   # wire-format version; the studio checks it (studio/config.py HAND_PROTO)
HOME = os.path.join(os.path.expanduser("~"), ".emg_hand")
MODELS = os.path.join(HOME, "models")          # replaced by --models
PATCH = 256
MAX_HANDS = 2


def log(*a) -> None:
    print("[hand_worker]", *a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------- camera
class Camera:
    """Keeps only the newest frame. Reading blocks at the camera rate, so it
    lives in its own thread and the inference loop never waits for exposure."""

    def __init__(self, index: int, width: int, height: int, fps: int, video: str = "") -> None:
        self.video = video
        self.cap = cv2.VideoCapture(video) if video else cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {index}를 열 수 없습니다")
        self.lock = threading.Lock()
        self.frame = None
        self.t_ns = 0
        self.seq = 0
        self.fps = 0.0
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        last = time.perf_counter()
        while not self.stop:
            ok, frame = self.cap.read()
            t = time.perf_counter_ns()
            if not ok:
                if self.video:   # loop the file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            if self.video:
                time.sleep(1 / 30)
            if frame.ndim == 2:   # grayscale cameras
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            with self.lock:
                self.frame, self.t_ns, self.seq = frame, t, self.seq + 1
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now

    def latest(self, after: int):
        """Wait for a frame newer than `after`."""
        while not self.stop:
            with self.lock:
                if self.seq > after:
                    return self.frame, self.t_ns, self.seq
            time.sleep(0.002)
        return None, 0, after


# --------------------------------------------------------------------- model
def load_model(dtype_name: str, models: str):
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=dtype,
                                             wilor_pretrained_dir=models, verbose=False)
    m = pipe.wilor_model
    # constants live on the GPU so forward() never copies from the host
    m.IMAGE_MEAN = m.IMAGE_MEAN.to("cuda", dtype)
    m.IMAGE_STD = m.IMAGE_STD.to("cuda", dtype)
    make_capturable(m.backbone)
    return pipe, dtype


def make_capturable(vit) -> None:
    """wilor_mini's forward_features indexes with a Python list (`[:, [0]]`), which
    copies an index tensor from the host and breaks CUDA graph capture. Same
    computation with plain slices (copied from wilor_mini otherwise)."""
    import torch

    def forward_features(self, x):
        from wilor_mini.models.vit import rot6d_to_rotmat
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        pose_tokens = self.pose_emb(
            self.init_hand_pose.reshape(1, self.NUM_HAND_JOINTS + 1, self.joint_rep_dim)).repeat(B, 1, 1)
        shape_tokens = self.shape_emb(self.init_betas).unsqueeze(1).repeat(B, 1, 1)
        cam_tokens = self.cam_emb(self.init_cam).unsqueeze(1).repeat(B, 1, 1)
        x = torch.cat([pose_tokens, shape_tokens, cam_tokens, x], 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        n = self.NUM_HAND_JOINTS + 1
        pose_feat, shape_feat, cam_feat = x[:, :n], x[:, n:n + 1], x[:, n + 1:n + 2]
        pred_hand_pose = self.decpose(pose_feat).reshape(B, -1) + self.init_hand_pose
        pred_betas = self.decshape(shape_feat).reshape(B, -1) + self.init_betas
        pred_cam = self.deccam(cam_feat).reshape(B, -1) + self.init_cam
        feats = {"hand_pose": pred_hand_pose, "betas": pred_betas, "cam": pred_cam}
        rot = rot6d_to_rotmat(pred_hand_pose).view(B, n, 3, 3)
        params = {"global_orient": rot[:, :1], "hand_pose": rot[:, 1:], "betas": pred_betas}
        img_feat = x[:, n + 2:].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2)
        return params, pred_cam, feats, img_feat

    vit.forward_features = types.MethodType(forward_features, vit)


class GraphedModel:
    """WiLoR's forward with the heavy parts replayed as CUDA graphs.

    Eager PyTorch on Windows is launch-bound here: ~1500 small kernels kept the
    GPU ~2% busy and the forward took ~30 ms. The ViT backbone and the refinement
    net capture cleanly; smplx's MANO layer does not, so it stays eager (twice,
    ~4 ms each). Same math as wilor_mini's WiLor.forward."""

    def __init__(self, model, dtype) -> None:
        import torch
        self.torch, self.m, self.dtype = torch, model, dtype
        self.graphs: dict[tuple[str, int], tuple] = {}

    def _graph(self, name: str, fn, *inputs):
        torch = self.torch
        key = (name, inputs[0].shape[0])
        if key not in self.graphs:
            static = [t.clone() for t in inputs]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    fn(*static)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                out = fn(*static)
            self.graphs[key] = (g, static, out)
        g, static, out = self.graphs[key]
        for dst, src in zip(static, inputs):
            dst.copy_(src)
        g.replay()
        return out

    def _backbone(self, x):
        m = self.m
        x = x.flip(dims=[-1]) / 255.0
        x = ((x - m.IMAGE_MEAN) / m.IMAGE_STD).permute(0, 3, 1, 2)
        tp, cam, feats, vit_out = m.backbone(x[:, :, :, 32:-32])
        return tp["global_orient"], tp["hand_pose"], tp["betas"], cam, feats["hand_pose"], feats["betas"],             feats["cam"], vit_out

    def _refine(self, vit_out, verts, cam, fp, fb, fc, focal):
        p = self.m.refine_net(vit_out, verts, cam, {"hand_pose": fp, "betas": fb, "cam": fc}, focal)
        return p["global_orient"], p["hand_pose"], p["betas"], p["pred_cam"]

    def __call__(self, x):
        import roma
        torch, m = self.torch, self.m
        B = x.shape[0]
        with torch.no_grad():
            go, hp, be, cam, fp, fb, fc, vit_out = self._graph("backbone", self._backbone, x)
            temp = m.mano(global_orient=go.reshape(B, -1, 3, 3), hand_pose=hp.reshape(B, -1, 3, 3),
                          betas=be.reshape(B, -1), pose2rot=False)
            focal = m.FOCAL_LENGTH * torch.ones(B, 2, device=x.device, dtype=x.dtype)
            go, hp, be, cam = self._graph("refine", self._refine, vit_out, temp.vertices, cam, fp, fb, fc, focal)
            out = m.mano(global_orient=go, hand_pose=hp, betas=be, pose2rot=False)
            res = {"pred_cam": cam, "betas": be,
                   "pred_keypoints_3d": out.joints.reshape(B, -1, 3), "pred_vertices": out.vertices.reshape(B, -1, 3),
                   "global_orient": roma.rotmat_to_rotvec(go), "hand_pose": roma.rotmat_to_rotvec(hp)}
        return {k: v.float().cpu().numpy() for k, v in res.items()}


def export_mano(model, path: str) -> None:
    """Dump the MANO layer's buffers as plain arrays for studio/hand/mano_np.py."""
    mano = model.mano
    t = lambda name: getattr(mano, name).detach().float().cpu().numpy()
    np.savez_compressed(path, v_template=t("v_template"), shapedirs=t("shapedirs"), posedirs=t("posedirs"),
                        J_regressor=t("J_regressor"), parents=mano.parents.cpu().numpy(),
                        lbs_weights=t("lbs_weights"), faces=mano.faces.astype(np.int32),
                        extra_joints_idxs=mano.extra_joints_idxs.cpu().numpy(),
                        joint_map=mano.joint_map.cpu().numpy())


class HandEstimator:
    def __init__(self, dtype_name: str, conf: float, redetect: int, models: str = MODELS,
                 mano_out: str = os.path.join(HOME, "mano_right_np.npz")) -> None:
        import torch
        self.torch = torch
        self.pipe, self.dtype = load_model(dtype_name, models)
        self.model = GraphedModel(self.pipe.wilor_model, self.dtype)
        if not os.path.exists(mano_out):
            export_mano(self.pipe.wilor_model, mano_out)
        self.conf = conf
        self.redetect = redetect
        self.track: list[tuple[np.ndarray, int]] = []   # (box, is_right) from last frame
        self.since_detect = 10 ** 9
        from wilor_mini.utils import utils
        self.utils = utils

    def faces(self) -> list:
        return self.pipe.wilor_model.mano.faces.astype(int).tolist()

    def detect(self, rgb: np.ndarray) -> tuple[list[tuple[np.ndarray, int, float]], str]:
        if self.track and self.since_detect < self.redetect:
            self.since_detect += 1
            return [(b, r, -1.0) for b, r in self.track], "track"
        self.since_detect = 0
        det = self.pipe.hand_detector(rgb, conf=self.conf, verbose=False, half=True)[0]
        boxes = det.boxes.data.cpu().numpy()
        boxes = boxes[boxes[:, 5] == 1]   # right hands only (class 1), for now
        boxes = boxes[np.argsort(-boxes[:, 4])][:MAX_HANDS]
        return [(b[:4].copy(), int(b[5]), float(b[4])) for b in boxes], "yolo"

    def estimate(self, rgb: np.ndarray, hands) -> list[dict]:
        if not hands:
            self.track = []
            return []
        torch, utils = self.torch, self.utils
        img_size = np.array([rgb.shape[1], rgb.shape[0]], float)
        patches, meta = [], []
        for box, right, conf in hands:
            center = (box[2:4] + box[0:2]) / 2.0
            size = 2.5 * float((box[2:4] - box[0:2]).max())
            img = rgb
            ds = size / PATCH / 2.0
            if ds > 1.1:   # anti-alias, but only where the patch is cut from
                sigma = (ds - 1) / 2
                pad = int(size / 2 + 4 * sigma)
                x0, y0 = max(0, int(center[0]) - pad), max(0, int(center[1]) - pad)
                x1, y1 = min(rgb.shape[1], int(center[0]) + pad), min(rgb.shape[0], int(center[1]) + pad)
                img = rgb.copy()
                if x1 > x0 and y1 > y0:
                    img[y0:y1, x0:x1] = cv2.GaussianBlur(rgb[y0:y1, x0:x1], (0, 0), sigma)
            patch, _ = utils.generate_image_patch_cv2(img, center[0], center[1], size, size, PATCH, PATCH,
                                                      right == 0, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
            patches.append(patch)
            meta.append((center, size, right, conf, box))
        x = torch.from_numpy(np.stack(patches)).to(device="cuda", dtype=self.dtype)
        out = self.model(x)
        f = 5000.0 / PATCH * img_size.max()
        res, track = [], []
        for i, (center, size, right, conf, box) in enumerate(meta):
            cam = out["pred_cam"][i:i + 1].copy()
            kp3 = out["pred_keypoints_3d"][i:i + 1].copy()
            verts = out["pred_vertices"][i:i + 1].copy()
            go, hp = out["global_orient"][i].copy(), out["hand_pose"][i].copy()
            cam[:, 1] *= (2 * right - 1)
            if right == 0:   # the model only sees right hands; left ones were mirrored in
                kp3[..., 0] *= -1
                verts[..., 0] *= -1
                go[..., 1:] *= -1
                hp[..., 1:] *= -1
            t_full = utils.cam_crop_to_full(cam, center[None], size, img_size[None], f)
            kp2 = utils.perspective_projection(kp3, translation=t_full, focal_length=np.array([[f, f]]),
                                               camera_center=img_size[None] / 2)[0]
            lo, hi = kp2.min(axis=0), kp2.max(axis=0)
            c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.25 + 8
            track.append((np.array([c[0] - half, c[1] - half, c[0] + half, c[1] + half]), right))
            res.append(dict(
                is_right=int(right), det_conf=round(conf, 3), bbox=[round(float(v), 1) for v in box],
                kp2d=np.round(kp2, 1).tolist(), kp3d=np.round(kp3[0], 5).tolist(),
                verts=verts[0].astype(np.float32), cam_t=np.round(t_full[0], 4).tolist(),
                global_orient=np.round(go.reshape(-1), 4).tolist(), hand_pose=np.round(hp.reshape(-1, 3), 4).tolist(),
                betas=np.round(out["betas"][i], 4).tolist(), focal=f,
            ))
        self.track = track
        return res


# --------------------------------------------------------------------- server
def send(conn: socket.socket, header: dict, jpeg: bytes = b"") -> None:
    blob = b""
    for hand in header.get("hands", []):
        blob += hand.pop("verts").tobytes()
    h = json.dumps(header, separators=(",", ":")).encode()
    conn.sendall(struct.pack("<III", len(h), len(jpeg), len(blob)) + h + jpeg + blob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--redetect", type=int, default=5, help="run YOLO at least every N frames")
    ap.add_argument("--jpeg", type=int, default=80)
    ap.add_argument("--video", default="", help="read this file (looped) instead of a camera")
    ap.add_argument("--models", default=MODELS, help="WiLoR/YOLO weights (downloaded here on first run)")
    ap.add_argument("--mano-out", default=os.path.join(HOME, "mano_right_np.npz"),
                    help="where to export MANO as plain arrays for the studio's numpy model")
    ap.add_argument("--bench", type=float, default=0, help="no server: run N seconds and print timings")
    args = ap.parse_args()

    srv = None
    if not args.bench:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", args.port))
        srv.listen(1)
        log(f"listening on {args.port}")
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send(conn, {"type": "status", "text": "모델 로딩 중 (약 10초)…"})
    try:
        t0 = time.time()
        est = HandEstimator(args.dtype, args.conf, args.redetect, args.models, args.mano_out)
        load_s = time.time() - t0
        cam = Camera(args.camera, args.width, args.height, args.fps, args.video)
    except Exception as e:  # report to the studio instead of dying silently
        log("init failed:", repr(e))
        if srv:
            send(conn, {"type": "error", "text": str(e)})
        sys.exit(1)
    log(f"model loaded in {load_s:.1f}s")
    if srv:
        send(conn, {"type": "hello", "proto": PROTO, "faces": est.faces(), "load_s": round(load_s, 1),
                    "camera": args.camera, "device": est.torch.cuda.get_device_name(0)})

    seq, n, t_start = 0, 0, time.time()
    acc = {"detect": 0.0, "wilor": 0.0, "jpeg": 0.0}
    while True:
        frame, t_ns, seq = cam.latest(seq)
        if frame is None:
            break
        t1 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands, how = est.detect(rgb)
        t2 = time.perf_counter()
        res = est.estimate(rgb, hands)
        if not res and how == "track":   # lost while tracking: re-detect right away
            est.since_detect = 10 ** 9
        t3 = time.perf_counter()
        jpeg = b""
        if srv:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg])
            jpeg = buf.tobytes() if ok else b""
        t4 = time.perf_counter()
        acc["detect"] += t2 - t1
        acc["wilor"] += t3 - t2
        acc["jpeg"] += t4 - t3
        n += 1
        if srv:
            try:
                send(conn, {"type": "frame", "seq": seq, "t_ns": t_ns, "w": frame.shape[1], "h": frame.shape[0],
                            "hands": res, "how": how, "cam_fps": round(cam.fps, 1),
                            "ms": {"detect": round((t2 - t1) * 1000, 1), "wilor": round((t3 - t2) * 1000, 1),
                                   "total": round((t4 - t1) * 1000, 1)},
                            "lag_ms": round((time.perf_counter_ns() - t_ns) / 1e6, 1)}, jpeg)
            except OSError:
                log("client gone")
                break
        elif time.time() - t_start > args.bench:
            break
    cam.stop = True
    if args.bench:
        el = time.time() - t_start
        print(f"frames {n} in {el:.1f}s = {n / el:.1f} fps, cam {cam.fps:.1f} fps; per frame: "
              + ", ".join(f"{k} {v / max(n, 1) * 1000:.1f} ms" for k, v in acc.items()))


if __name__ == "__main__":
    main()
