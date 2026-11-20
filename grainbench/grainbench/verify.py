"""独立解码核对与指标计算。

所有指标都基于 **aomdec 独立解码**（而非编码器内部重建）得到的帧：

* 频段能量：对残差做环形波数分箱的 FFT 能量谱，区分低频布纹与高频颗粒；
* 纹理对比度：布片局部区域的微纹标准差；
* 沿运动轨迹取样：在物体局部坐标与屏幕固定坐标两套网格上分别计算帧间相关，
  用一组数字区分"纹理跟物体走"和"噪声跟屏幕走"；
* 同码量 PSNR/SSIM：两路码流字节数对齐后对原片的保真度。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import tools
from .config import ExperimentConfig
from .synth import read_i420_y


@dataclass
class FrameMetrics:
    psnr: float
    low_band_energy: float
    high_band_energy: float
    texture_contrast: float
    trail_object_corr: float       # 物体局部坐标下的帧间相关
    trail_screen_corr: float       # 屏幕固定坐标下的帧间相关


def decode(ivf: Path, cfg: ExperimentConfig, *, apply_grain: bool) -> np.ndarray:
    out = ivf.with_name(ivf.stem + (".grain.yuv" if apply_grain else ".base.yuv"))
    tools.decode_ivf(ivf, out, apply_grain=apply_grain,
                     width=cfg.width, height=cfg.height)
    return read_i420_y(out, cfg).astype(np.float32)


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float(99.0 if mse < 1e-9 else 20 * math.log10(255.0 / math.sqrt(mse)))


def band_energy(frame: np.ndarray, split_cycles: float = 0.18) -> tuple[float, float]:
    """环形 FFT 能量谱按归一化波数 split 分低/高两段。"""
    x = frame.astype(np.float64)
    x -= x.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    h, w = frame.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    radial = np.sqrt(fx ** 2 + fy ** 2)
    low = spec[radial <= split_cycles].sum() / spec.sum()
    high = 1.0 - low
    return float(low), float(high)


def texture_contrast(frame: np.ndarray, cx: float, cy: float,
                     radius: float) -> float:
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < radius * 0.8)
    if m.sum() < 64:
        return 0.0
    return float(frame[m].std())


def trail_correlations(frames: np.ndarray, geometry: list[dict],
                       cfg: ExperimentConfig) -> tuple[float, float]:
    """两套坐标的帧间相关，聚焦布片内部的高频细节。

    与分离器使用同一方法学：亚像素运动对齐 + 布片掩膜，避免背景渐变和
    静止区域主导统计量。

    object：把第 f 帧按速度做亚像素反向平移，与上一帧在同一物体坐标比较——
        依附物体的微纹在这里相关高，屏幕固定颗粒在这里相关低；
    screen：直接比较同屏坐标——运动物体上的微纹因位移而相关低，
        而屏幕固定颗粒的逐帧实现本就不相关（接近 0）。
    """
    from scipy.ndimage import map_coordinates
    n, h, w = frames.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def shift(img: np.ndarray, dx: float) -> np.ndarray:
        return map_coordinates(
            img, [yy, xx - dx], order=1, mode="reflect", prefilter=False)

    def cloth_mask(g: dict) -> np.ndarray:
        return np.sqrt((xx - g["cx"]) ** 2 +
                       (yy - g["cy"]) ** 2) < g["radius"] * 0.75

    obj, scr = [], []
    for f in range(1, n):
        vx = float(geometry[f]["vx"])
        m_prev = cloth_mask(geometry[f - 1])
        m_cur = cloth_mask(geometry[f])
        prev, cur = frames[f - 1], frames[f]
        # 物体坐标：当前帧亚像素反向平移回上一帧物体坐标。
        aligned = shift(cur, -vx)
        m_aligned = shift(m_cur.astype(np.float32), -vx) > 0.5
        m_obj = m_prev & m_aligned
        x, y = prev[m_obj], aligned[m_obj]
        denom = x.std() * y.std() + 1e-9
        obj.append(float(((x - x.mean()) * (y - y.mean())).mean() / denom))
        # 屏幕坐标：同一屏位置直接比较（取两帧布片交集）。
        m_scr = m_prev & m_cur
        x2, y2 = prev[m_scr], cur[m_scr]
        denom2 = x2.std() * y2.std() + 1e-9
        scr.append(float(((x2 - x2.mean()) * (y2 - y2.mean())).mean() / denom2))
    return float(np.mean(obj)), float(np.mean(scr))


def evaluate_stream(ivf: Path, ref_frames: np.ndarray,
                    geometry: list[dict], cfg: ExperimentConfig,
                    *, apply_grain: bool) -> dict:
    dec = decode(ivf, cfg, apply_grain=apply_grain)
    n = min(len(dec), len(ref_frames))
    per_frame: list[FrameMetrics] = []
    for f in range(n):
        low, high = band_energy(dec[f] - _local_mean(dec[f]))
        g = geometry[f]
        per_frame.append(FrameMetrics(
            psnr=psnr(ref_frames[f], dec[f]),
            low_band_energy=low,
            high_band_energy=high,
            texture_contrast=texture_contrast(dec[f], g["cx"], g["cy"],
                                              g["radius"]),
            trail_object_corr=0.0,
            trail_screen_corr=0.0,
        ))
    obj_corr, scr_corr = trail_correlations(dec, geometry, cfg)
    return {
        "apply_grain": apply_grain,
        "mean_psnr": float(np.mean([m.psnr for m in per_frame])),
        "mean_low_band": float(np.mean([m.low_band_energy for m in per_frame])),
        "mean_high_band": float(np.mean([m.high_band_energy for m in per_frame])),
        "mean_texture_contrast": float(
            np.mean([m.texture_contrast for m in per_frame])),
        "trail_object_corr": obj_corr,
        "trail_screen_corr": scr_corr,
        "per_frame_psnr": [m.psnr for m in per_frame],
        "per_frame_contrast": [m.texture_contrast for m in per_frame],
        "frames": dec,
    }


def _local_mean(frame: np.ndarray, k: int = 7) -> np.ndarray:
    """盒式滤波得到局部均值，残差用于频段能量分析。"""
    h, w = frame.shape
    pad = k // 2
    p = np.pad(frame, pad, mode="reflect")
    acc = np.zeros_like(frame, dtype=np.float64)
    for dy in range(k):
        for dx in range(k):
            acc += p[dy:dy + h, dx:dx + w]
    return (acc / (k * k)).astype(np.float32)


def route_change_map(decisions: list[dict], reports: list[dict],
                     cfg: ExperimentConfig) -> dict:
    """两路选择发生变化的位置：分段路由 + 触发变化的指标。"""
    changes = []
    for i, d in enumerate(decisions):
        if d["route"] != "grain-model":
            changes.append({
                "segment": i,
                "frame_range": [d["start"], d["end"]],
                "route": d["route"],
                "reason": d["reason"],
                "temporal_coherence": reports[i]["temporal_coherence"],
                "model_rms_error": d["model_error"],
                "error_bound": d["error_bound"],
            })
    return {"changes": changes,
            "segments": [{"index": i, "route": d["route"],
                          "frames": [d["start"], d["end"]]}
                         for i, d in enumerate(decisions)]}
