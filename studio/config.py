"""Every path, port and environment variable the studio uses, in one place."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # emg_studio/

# runtime data (settings, pose dataset/model); not part of the source tree
DATA_DIR = os.environ.get("EMG_DATA_DIR", os.path.join(ROOT, "data"))
SETTINGS_PATH = os.path.join(DATA_DIR, "studio_settings.json")
POSE_DATA_PATH = os.path.join(DATA_DIR, "pose_dataset.npz")
POSE_MODEL_PATH = os.path.join(DATA_DIR, "pose_model.npz")

# the hand-pose worker runs in its own Python (PyTorch, numpy<2), kept outside the
# Google Drive project folder because it is several GB
HAND_HOME = os.path.join(os.path.expanduser("~"), ".emg_hand")
HAND_PY = os.environ.get("EMG_HAND_PY", os.path.join(HAND_HOME, "venv", "Scripts", "python.exe"))
HAND_MODELS = os.path.join(HAND_HOME, "models")
HAND_LOG = os.path.join(HAND_HOME, "worker.log")
HAND_PORT = int(os.environ.get("EMG_HAND_PORT", "8790"))
MANO_NPZ = os.path.join(HAND_HOME, "mano_right_np.npz")
WORKER = os.path.join(ROOT, "worker", "hand_worker.py")
HAND_PROTO = 1          # wire-format version shared with worker/hand_worker.py

TICK_HZ = 30            # hub clock
