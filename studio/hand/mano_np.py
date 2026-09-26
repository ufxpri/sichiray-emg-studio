"""MANO right hand in plain numpy, so the studio can draw a mesh for any pose
(e.g. one predicted from EMG) without PyTorch.

The parameter file is exported once by worker/hand_worker.py (--mano-out) from
the same smplx layer WiLoR uses (config.MANO_NPZ). Standard linear blend skinning,
rotation-matrix input, no pose mean added (as smplx.MANOLayer with pose2rot=False).
Joints come out in the 21-point OpenPose order WiLoR reports.
"""
from __future__ import annotations

import os

import numpy as np

from ..config import MANO_NPZ as PATH


def rodrigues(rv: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = rv / np.maximum(theta, 1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1).reshape(rv.shape[:-1] + (3, 3))
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + s * K + (1 - c) * (K @ K)


class Mano:
    def __init__(self, path: str = PATH) -> None:
        d = np.load(path)
        self.v_template = d["v_template"]
        self.shapedirs = d["shapedirs"]
        self.posedirs = d["posedirs"]
        self.J_regressor = d["J_regressor"]
        self.parents = d["parents"].astype(int)
        self.weights = d["lbs_weights"]
        self.faces = d["faces"].astype(np.int32)
        self.tips = d["extra_joints_idxs"].astype(int)
        self.joint_map = d["joint_map"].astype(int)

    @staticmethod
    def available(path: str = PATH) -> bool:
        return os.path.exists(path)

    def forward(self, global_orient: np.ndarray, hand_pose: np.ndarray,
                betas: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """global_orient (3,) and hand_pose (15, 3) as axis-angle. Returns
        vertices (778, 3) and joints (21, 3), in metres, model frame."""
        betas = np.zeros(10) if betas is None else np.asarray(betas, float)
        R = rodrigues(np.vstack([np.reshape(global_orient, (1, 3)), np.reshape(hand_pose, (15, 3))]))  # 16x3x3
        v_shaped = self.v_template + self.shapedirs @ betas
        J = self.J_regressor @ v_shaped
        pose_feat = (R[1:] - np.eye(3)).reshape(-1)
        v_posed = v_shaped + (pose_feat @ self.posedirs).reshape(-1, 3)
        # kinematic chain
        G = np.zeros((16, 4, 4))
        rel = J.copy()
        rel[1:] -= J[self.parents[1:]]
        for i in range(16):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R[i], rel[i]
            G[i] = T if i == 0 else G[self.parents[i]] @ T
        posed_J = G[:, :3, 3].copy()
        G[:, :3, 3] -= np.einsum("nij,nj->ni", G[:, :3, :3], J)
        A = np.einsum("vk,kij->vij", self.weights, G)
        verts = np.einsum("vij,vj->vi", A[:, :3, :3], v_posed) + A[:, :3, 3]
        joints = np.vstack([posed_J, verts[self.tips]])[self.joint_map]
        return verts, joints


_shared: Mano | None = None


def get_mano() -> Mano | None:
    """The one shared model, loaded on first use; None until the worker has exported it."""
    global _shared
    if _shared is None and Mano.available():
        _shared = Mano()
    return _shared
