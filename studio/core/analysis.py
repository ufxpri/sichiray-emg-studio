"""Per-channel noise diagnostics for the spectrogram window.

Each metric is one number per channel plus a grade (ok / warn / bad / info) and a
plain-language explanation, because the point is to learn what the signal is made
of, not just to see a number. Thresholds come from what this rig has shown so far
(NOTES §13, §18, §19) and from textbook sEMG ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import welch

QUANT_RMS = 1 / np.sqrt(12)   # RMS of rounding to a 1-count step
EMG_LO, EMG_HI = 20.0, 250.0

OK, WARN, BAD, INFO = "ok", "warn", "bad", "info"


@dataclass
class Metric:
    key: str
    label: str
    unit: str
    help: str
    grade: Callable[[float, dict], str]
    fmt: str = "{:.1f}"


def _hi(warn: float, bad: float):
    return lambda v, ctx: INFO if not np.isfinite(v) else BAD if v >= bad else WARN if v >= warn else OK


METRICS: list[Metric] = [
    Metric("std_raw", "신호 유무", "cnt",
           "원시 값의 표준편차입니다. 0에 가까우면 값이 한 자리에 멈춰 있다는 뜻입니다. "
           "전극이 피부에서 떨어졌거나 채널이 죽은 경우입니다. 휴식 중에도 양자화 잡음 때문에 약간은 흔들려야 정상입니다.",
           lambda v, c: BAD if v < 0.05 else OK, "{:.2f}"),
    Metric("dc", "DC 오프셋", "cnt",
           "평균값이 중앙(HEX 127 / ASCII 2048)에서 얼마나 떨어져 있는지 나타냅니다. "
           "sEMG는 평균이 0인 신호라 원래 중앙에 있어야 합니다. 크게 벗어나면 전극 접촉 전위나 기준선 이동이 있는 것입니다. "
           "크기 자체보다 계속 움직이는지가 더 중요합니다.",
           lambda v, c: BAD if abs(v) > 0.3 * c["full"] / 2 else WARN if abs(v) > 0.1 * c["full"] / 2 else OK, "{:+.1f}"),
    Metric("clip", "포화", "%",
           "값이 0 또는 최댓값(255/4095)에 붙은 샘플의 비율입니다. 포화된 구간은 파형이 잘려 정보가 사라집니다. "
           "대개 큰 움직임 아티팩트나 접촉 불량 때문입니다.",
           _hi(0.1, 1.0), "{:.2f}"),
    Metric("rms", "RMS", "cnt",
           "신호 크기입니다. 휴식 때 작고 힘을 줄수록 커져야 근육 신호입니다. "
           "고정된 정상값은 없으므로 휴식 대비 비율(SNR)과 함께 보세요.",
           lambda v, c: INFO, "{:.2f}"),
    Metric("quant", "양자화 대비", "배",
           "RMS를 양자화 잡음(0.29 카운트)으로 나눈 값입니다. 1~2배이면 신호가 ADC 한 계단 안에 있어 "
           "어떤 필터나 모델로도 복원할 수 없습니다. HEX 8-bit의 휴식 바닥이 원래 이 수준입니다(§19). "
           "수축 중에도 이 값이면 전극 접촉을 의심하세요.",
           lambda v, c: BAD if v < 1.5 else WARN if v < 3 else OK, "{:.1f}"),
    Metric("snr", "휴식 대비", "dB",
           "지금 RMS가 보정 창에서 측정한 휴식 RMS보다 몇 dB 큰지 나타냅니다. 수축하면 +10 dB 이상 올라가야 "
           "근육을 잘 잡고 있는 채널입니다. 휴식 보정을 하기 전에는 비어 있습니다.",
           lambda v, c: INFO, "{:+.1f}"),
    Metric("lf", "저주파 <20Hz", "%",
           "전체 전력 중 20 Hz 아래가 차지하는 비율입니다. 근전도 전력은 대부분 20~250 Hz에 있습니다. "
           "20 Hz 아래는 전극이 피부 위에서 미끄러지는 움직임 아티팩트, 케이블 흔들림, 기준선 이동입니다. "
           "이 값이 크면 밴드를 더 단단히 고정하세요.",
           _hi(50, 80), "{:.0f}"),
    Metric("emg", "EMG 대역", "%",
           "20~250 Hz(ASCII 모드는 20~33 Hz) 대역의 전력 비율입니다. 근육 신호가 있어야 할 자리입니다. "
           "수축 중에 이 비율이 올라가면 좋은 채널입니다.",
           lambda v, c: INFO, "{:.0f}"),
    Metric("hum", "전원 험", "%",
           "50/60 Hz와 그 배수(가정용 전원)에 몰린 전력 비율입니다. 전원선, 충전 중인 노트북, 형광등에서 유도됩니다. "
           "노치 필터로 제거할 수 있지만, 크다면 전원에서 떨어져 앉는 것이 먼저입니다.",
           _hi(5, 20), "{:.1f}"),
    Metric("line", "협대역 간섭", "%",
           "한 주파수 빈에 전력이 몰린 뾰족한 선을 찾아 그 주파수와 비율을 보여줍니다. 근육 신호는 무작위라 이렇게 모이지 않습니다. "
           "이 장비에서는 충전기가 9.77 Hz에 전력의 46%를 넣은 적이 있습니다(§18). 15%를 넘으면 그 측정은 쓸 수 없습니다. "
           "측정할 때는 충전기를 뽑으세요.",
           _hi(5, 15), "{:.0f}"),
    Metric("mdf", "중앙주파수", "Hz",
           "EMG 대역 전력을 반으로 나누는 주파수입니다. 팔뚝 sEMG는 보통 50~150 Hz입니다. "
           "훨씬 낮으면 저주파 아티팩트가, 훨씬 높으면 고주파 잡음이 섞인 것입니다. "
           "오래 힘을 유지하면 피로로 서서히 내려갑니다.",
           lambda v, c: INFO if c["mode"] != "hex" or c["quant"] < 3 else WARN if (v < 40 or v > 180) else OK, "{:.0f}"),
    Metric("kurt", "첨도", "",
           "파형 분포의 뾰족함입니다(0.5초 조각별 값의 중앙값). 정규분포는 3입니다. 강하게 수축하면 많은 운동단위가 합쳐져 3에 가까워지고(§21), "
           "약한 수축은 스파이크가 튀어 커집니다. 10을 넘으면 튀는 잡음(정전기, 접촉 순간 끊김)일 가능성이 큽니다.",
           lambda v, c: WARN if v > 10 else INFO, "{:.1f}"),
    Metric("xcorr", "채널 간 상관", "",
           "이 채널과 다른 7개 채널의 평균 |상관계수|입니다. 근육 신호는 전극 위치마다 달라 서로 상관이 낮습니다. "
           "모든 채널에 똑같이 들어오는 신호(공통 모드: 전원, 충전기, 몸 전체의 움직임)는 상관이 높습니다. "
           "0.8을 넘으면 그 채널은 주로 공통 잡음을 보고 있는 것입니다.",
           _hi(0.5, 0.8), "{:.2f}"),
]


def compute(raw: np.ndarray, sig: np.ndarray, fs: float, ctx: dict,
            rest_rms: np.ndarray | None) -> tuple[dict[str, np.ndarray], dict[str, list[str]], list[tuple]]:
    """raw: values as received (for rails and offset); sig: the analysed source.
    Returns values, grades, and per-channel (freq, share, sharpness) of the line metric."""
    n, ch = sig.shape
    center, full = ctx["center"], ctx["full"]
    v: dict[str, np.ndarray] = {}
    v["std_raw"] = raw.std(axis=0)
    v["dc"] = raw.mean(axis=0) - center
    v["clip"] = ((raw <= 0) | (raw >= full)).mean(axis=0) * 100
    x = sig - sig.mean(axis=0)
    rms = np.sqrt((x ** 2).mean(axis=0))
    v["rms"] = rms
    v["quant"] = rms / QUANT_RMS
    v["snr"] = 20 * np.log10(np.maximum(rms, 1e-9) / rest_rms) if rest_rms is not None else np.full(ch, np.nan)

    nyq = fs / 2
    nper = min(n, 1024 if fs > 200 else 128)
    f, P = welch(x, fs=fs, nperseg=nper, axis=0)
    tot = np.maximum(P.sum(axis=0), 1e-12)

    def share(lo, hi):
        m = (f >= lo) & (f < hi)
        return P[m].sum(axis=0) / tot * 100

    v["lf"] = share(0, EMG_LO)
    v["emg"] = share(EMG_LO, min(EMG_HI, nyq + 1))
    hum = np.zeros(ch)
    for base in (50.0, 60.0):
        h = np.zeros(ch)
        for k in range(1, 6):
            if base * k < nyq:
                h += share(base * k - 1, base * k + 1)
        hum = np.maximum(hum, h)
    v["hum"] = hum

    Pn = P / tot
    lines, line_v = [], np.zeros(ch)
    for c in range(ch):
        best = (np.nan, 0.0, 0.0)
        if np.unique(sig[:, c]).size >= 8:
            for lo, hi in ((5, 15), (15, 30), (30, 70), (70, 130), (130, 250)):
                m = (f >= lo) & (f < min(hi, nyq))
                if m.sum() < 3 or Pn[m, c].sum() <= 0:
                    continue
                i = int(np.argmax(np.where(m, Pn[:, c], 0)))
                sharp = Pn[i, c] / Pn[m, c].sum()
                if sharp > 0.3 and Pn[i, c] > best[1]:
                    best = (float(f[i]), float(Pn[i, c]), float(sharp))
        lines.append(best)
        line_v[c] = best[1] * 100
    v["line"] = line_v

    m = (f >= EMG_LO) & (f < min(EMG_HI, nyq))
    mdf = np.full(ch, np.nan)
    if m.sum() > 2:
        cum = np.cumsum(P[m], axis=0)
        for c in range(ch):
            if cum[-1, c] > 0:
                mdf[c] = f[m][np.searchsorted(cum[:, c], cum[-1, c] / 2)]
    v["mdf"] = mdf

    # Kurtosis over the whole window would call a rest-then-squeeze window 'spiky'
    # just because the variance changed. Median over short windows keeps it local.
    seg = max(32, int(0.5 * fs))
    ks = []
    for i in range(0, n - seg + 1, seg):
        d = x[i:i + seg] - x[i:i + seg].mean(axis=0)
        var = (d ** 2).mean(axis=0)
        ks.append(np.where(var > 0, (d ** 4).mean(axis=0) / np.maximum(var, 1e-12) ** 2, np.nan))
    v["kurt"] = np.nanmedian(np.array(ks), axis=0) if ks else np.full(ch, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.corrcoef(x.T)
    cc = np.nan_to_num(np.abs(cc))
    np.fill_diagonal(cc, 0)
    v["xcorr"] = cc.sum(axis=0) / (ch - 1)

    grades: dict[str, list[str]] = {}
    for met in METRICS:
        row = []
        for c in range(ch):
            cctx = dict(ctx, quant=v["quant"][c])
            val = v[met.key][c]
            row.append(INFO if not np.isfinite(val) else met.grade(float(val), cctx))
        grades[met.key] = row
    return v, grades, lines
