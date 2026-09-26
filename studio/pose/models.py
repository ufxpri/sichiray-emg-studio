"""Regressors behind one interface, and the saved PoseModel.

A PoseModel is: standardiser + regressor + what it expects (sample rate, filter
signature). It is stored as plain arrays - no pickle - so a scikit-learn upgrade
cannot make a saved model unreadable (the MLP is evaluated with numpy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

MODEL_VERSION = 3   # 3: plain-array regressors, frozen filter signature


class Regressor(Protocol):
    kind: str

    @property
    def label(self) -> str: ...

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def state(self) -> dict[str, np.ndarray]: ...


class RidgeRegressor:
    kind = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.W: np.ndarray | None = None

    @property
    def label(self) -> str:
        return f"릿지 α={self.alpha:g}"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.W = np.linalg.solve(X.T @ X + self.alpha * np.eye(X.shape[1]), X.T @ Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W

    def state(self) -> dict[str, np.ndarray]:
        return {"alpha": np.array(self.alpha), "W": self.W}

    @classmethod
    def restore(cls, st: dict) -> "RidgeRegressor":
        r = cls(float(st["alpha"]))
        r.W = st["W"]
        return r


class MlpRegressor:
    kind = "mlp"
    HIDDEN = (128, 128)

    def __init__(self) -> None:
        self.Ws: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []

    @property
    def label(self) -> str:
        return "MLP " + "×".join(map(str, self.HIDDEN))

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor
        net = MLPRegressor(hidden_layer_sizes=self.HIDDEN, alpha=1e-3, max_iter=400,
                           early_stopping=True, random_state=0)
        net.fit(X, Y)
        self.Ws, self.bs = list(net.coefs_), list(net.intercepts_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = np.maximum(h @ W + b, 0.0)           # ReLU, as MLPRegressor's default
        return h @ self.Ws[-1] + self.bs[-1]         # identity output

    def state(self) -> dict[str, np.ndarray]:
        st = {"n": np.array(len(self.Ws))}
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            st[f"W{i}"], st[f"b{i}"] = W, b
        return st

    @classmethod
    def restore(cls, st: dict) -> "MlpRegressor":
        r = cls()
        n = int(st["n"])
        r.Ws = [st[f"W{i}"] for i in range(n)]
        r.bs = [st[f"b{i}"] for i in range(n)]
        return r


REGRESSORS = {"ridge": RidgeRegressor, "mlp": MlpRegressor}


@dataclass
class Standardizer:
    mu: np.ndarray
    sd: np.ndarray
    ymu: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "Standardizer":
        return cls(X.mean(axis=0), X.std(axis=0) + 1e-6, Y.mean(axis=0))

    def x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def y(self, Y: np.ndarray) -> np.ndarray:
        return Y - self.ymu


@dataclass
class PoseModel:
    regressor: Regressor
    scaler: Standardizer
    fs: float
    filter_sig: str
    filter_desc: str
    report: str = ""

    @classmethod
    def train(cls, regressor: Regressor, X: np.ndarray, Y: np.ndarray, fs: float,
              filter_sig: str, filter_desc: str) -> "PoseModel":
        sc = Standardizer.fit(X, Y)
        regressor.fit(sc.x(X), sc.y(Y))
        return cls(regressor, sc, fs, filter_sig, filter_desc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Features (..., 64) -> hand pose (..., 45)."""
        return self.regressor.predict(self.scaler.x(X)) + self.scaler.ymu

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        st = {f"r_{k}": v for k, v in self.regressor.state().items()}
        np.savez(path, version=MODEL_VERSION, kind=self.regressor.kind, fs=self.fs, mu=self.scaler.mu,
                 sd=self.scaler.sd, ymu=self.scaler.ymu, sig=self.filter_sig, filt=self.filter_desc,
                 report=self.report, **st)

    @classmethod
    def load(cls, path: str) -> "PoseModel | None":
        """None when missing or from an older format (retrain instead of guessing)."""
        try:
            d = np.load(path, allow_pickle=False)
            if int(d["version"]) != MODEL_VERSION:
                return None
            st = {k[2:]: d[k] for k in d.files if k.startswith("r_")}
            reg = REGRESSORS[str(d["kind"])].restore(st)
            return cls(reg, Standardizer(d["mu"], d["sd"], d["ymu"]), float(d["fs"]), str(d["sig"]),
                       str(d["filt"]), str(d["report"]))
        except (OSError, KeyError, ValueError):
            return None
