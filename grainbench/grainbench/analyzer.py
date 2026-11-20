"""分离器：从带噪素材与干净底图的残差中估计颗粒统计。

输出三类信息：

1. **按亮度分段的强度**：残差标准差关于局部亮度的分段函数，对应 AV1 film grain
   的 ``scaling_points_y``；
2. **空间相关性**：残差的自相关函数与 Yule-Walker 自回归系数，对应
   ``ar_coeffs_y``，决定解码器合成颗粒的空间纹理；
3. **时间相干性**：残差沿运动轨迹的帧间相关。逐帧独立颗粒在轨迹上不相关，而
   跟随物体移动的织物微纹高度相关——这是判定 AV1 颗粒模型能否表达该段的核心
   证据，因为解码器的颗粒合成固定在屏幕坐标、逐帧重新抽样。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates

from .config import ExperimentConfig

# AV1 film grain AR 模型在 lag=2 时共 12 个 Y 系数，对应 4 邻域阶数，
# 自相关需估计到滞后 4。
AR_ORDER = 4
NUM_AR_COEFFS = 12


def _subpixel_shift(img: np.ndarray, dx: float) -> np.ndarray:
    """水平方向亚像素平移（双线性）；dx>0 表示图像内容向右移动 dx。"""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.stack([yy, xx - dx])
    return map_coordinates(img, coords, order=1, mode="reflect", prefilter=False)


def _motion_highpass(img: np.ndarray) -> np.ndarray:
    """布片掩膜已隔离背景亮度渐变，残差直接参与时间相关，不做空间高通。

    实测任何水平均值高通都会同时削弱周期布纹的时间相关（布纹主能量在中频），
    反而压缩布纹与噪声的分离度，因此这里恒等返回。
    """
    return img


def _cloth_mask(geometry: dict, h: int, w: int, shrink: float = 0.75) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    radius = geometry["radius"] * shrink
    return np.sqrt((xx - geometry["cx"]) ** 2 +
                   (yy - geometry["cy"]) ** 2) < radius


@dataclass
class LumaBinStat:
    bin_index: int
    lo: int
    hi: int
    count: int
    strength: float                 # 残差标准差（灰度级）
    autocorr: list[float]           # 滞后 0..3 的空间自相关
    temporal_corr: float            # 沿运动轨迹的帧间相关
    fit_strength: float = 0.0       # 模型在该箱可复现的强度
    strength_rel_error: float = 0.0


@dataclass
class SegmentReport:
    start: int
    end: int
    luma_bins: list[LumaBinStat] = field(default_factory=list)
    ar_coeffs: list[float] = field(default_factory=list)
    ar_coeffs_q: list[int] = field(default_factory=list)
    scaling_points: list[tuple[int, int]] = field(default_factory=list)
    model_rms_error: float = 0.0          # 强度+相关性的综合相对误差
    temporal_coherence: float = 0.0       # 全段轨迹相关均值
    model_feasible: bool = False
    fallback_reason: str = ""
    # 带误差界限近似时记录保证上界（直接编码超预算场景）。
    approximation_bound: float | None = None


def _autocorr_2d(x: np.ndarray, max_lag: int = AR_ORDER) -> np.ndarray:
    """各向同性空间自相关：对水平/垂直两个方向求滞后乘积平均。"""
    x = x.astype(np.float64)
    x = x - x.mean()
    var = (x * x).mean() + 1e-12
    ac = [1.0]
    for lag in range(1, max_lag + 1):
        h = (x[:, lag:] * x[:, :-lag]).mean()
        v = (x[lag:, :] * x[:-lag, :]).mean()
        ac.append(float(((h + v) / 2.0) / var))
    return np.array(ac)


def _yule_walker(ac: np.ndarray, order: int) -> np.ndarray:
    """由自相关解 Yule-Walker 方程得到 AR(order) 系数。

    加入轻微岭正则：残差自相关在小块上估计时相关矩阵可能接近奇异，
    裸解会在相邻系数上交替爆到饱和值（±128 Q8），正则把系数拉回物理范围。
    ac 长度必须为 order+1（滞后 0..order）。
    """
    r = ac[1: order + 1].copy()
    mat = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            lag = abs(i - j)
            mat[i, j] = ac[lag] if lag <= order else 0.0
    mat += np.eye(order) * 1e-2
    return np.linalg.solve(mat, r)


def _motion_aligned_residual(res: np.ndarray, geometry: list[dict],
                             seg_start: int, seg_end: int,
                             width: int, height: int) -> float:
    """沿运动轨迹的时间相关（亚像素精确对齐，布片内部，运动轴高通后）。

    把第 f 帧残差按物体速度做亚像素反向平移，与 f-1 帧落在同一物体坐标后
    求相关。依附物体的布纹在该坐标下重合 → 高相关；逐帧独立颗粒无论怎样
    对齐都不相关。这是判定屏幕固定颗粒模型可达性的核心证据。
    """
    h, w = res.shape[1:]
    corrs = []
    for f in range(seg_start + 1, seg_end):
        vx = float(geometry[f]["vx"])
        prev = _motion_highpass(res[f - 1])
        aligned = _motion_highpass(_subpixel_shift(res[f], -vx))
        m_prev = _cloth_mask(geometry[f - 1], h, w)
        m_cur = _cloth_mask(geometry[f], h, w)
        # 对齐后两帧的布片掩膜也需对齐到同一坐标。
        m_cur_aligned = _subpixel_shift(m_cur.astype(np.float32), -vx) > 0.5
        m = m_prev & m_cur_aligned
        if m.sum() < 256:
            continue
        x, y = prev[m], aligned[m]
        denom = x.std() * y.std() + 1e-9
        corrs.append(float(((x - x.mean()) * (y - y.mean())).mean() / denom))
    return float(np.mean(corrs)) if corrs else 0.0


def analyze_segment(residual: np.ndarray, clean: np.ndarray,
                    geometry: list[dict], seg_start: int, seg_end: int,
                    cfg: ExperimentConfig) -> SegmentReport:
    h, w = residual.shape[1:]
    rep = SegmentReport(start=seg_start, end=seg_end)
    r_seg = residual[seg_start:seg_end]
    c_seg = clean[seg_start:seg_end]

    bins = cfg.luma_bins
    rep.temporal_coherence = _motion_aligned_residual(
        residual, geometry, seg_start, seg_end, w, h)

    all_ac, weights = [], []
    for bi in range(len(bins) - 1):
        lo, hi = bins[bi], bins[bi + 1]
        sel = (c_seg >= lo) & (c_seg < hi)
        count = int(sel.sum())
        if count < 256:
            rep.luma_bins.append(LumaBinStat(bi, lo, hi, count, 0.0,
                                             [1.0, 0, 0, 0, 0], 0.0))
            continue
        vals = r_seg[sel]
        strength = float(vals.std())
        # 空间自相关在帧内平坦区估计：取该亮度箱覆盖像素最多的中间帧块。
        mid = (seg_start + seg_end) // 2
        frame_sel = (clean[mid] >= lo) & (clean[mid] < hi)
        if frame_sel.sum() >= 256:
            ac = _autocorr_2d(residual[mid] * frame_sel, AR_ORDER)
        else:
            ac = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        # 箱级时间相关：只在该亮度掩膜内做运动对齐。
        tc = _bin_temporal_corr(residual, geometry, seg_start, seg_end,
                                clean, lo, hi, w, h)
        stat = LumaBinStat(bi, lo, hi, count, strength,
                           [float(v) for v in ac], tc)
        rep.luma_bins.append(stat)
        all_ac.append(ac)
        weights.append(count)

    valid = [s for s in rep.luma_bins if s.count >= 256 and s.strength > 0.5]
    if not valid:
        rep.fallback_reason = "残差能量过低，无有效颗粒样本"
        return rep

    # 全段 AR 系数：对各箱自相关按样本量加权后做 Yule-Walker。
    # AV1 lag=2 时系数总数 2*lag*(lag+1)=12，对应 4 邻域阶数，
    # 因此自相关需要滞后 0..4。
    w_arr = np.array(weights[: len(all_ac)], float)
    ac_avg = np.average(np.array(all_ac), axis=0, weights=w_arr)
    ar = _yule_walker(ac_avg, order=AR_ORDER)
    # AV1 系数排布含 4 个跳过位（上方不可达邻域），补零到 12 个。
    ar_full = np.zeros(NUM_AR_COEFFS)
    ar_full[:AR_ORDER] = ar
    rep.ar_coeffs = [float(v) for v in ar_full]
    # 量化到 aom 约定的 Q8 定点，裁剪到 OBU 合法范围 [-128,127]。
    rep.ar_coeffs_q = [int(np.clip(round(v * (1 << 8)), -128, 127))
                       for v in ar_full]

    # scaling points：以各亮度箱中心为 x、模型强度映射为 y。
    # 实测 aom 合成器（scaling_shift=11、高斯表 std≈512）下
    # 合成颗粒 sigma ≈ scaling/64，据此把残差标准差换成 OBU scaling 值。
    points = []
    for s in valid:
        x = int((s.lo + s.hi) // 2)
        y = int(np.clip(round(s.strength * 64.0), 0, 255))
        points.append((x, y))
        s.fit_strength = y / 64.0
        s.strength_rel_error = abs(s.fit_strength - s.strength) / s.strength
    rep.scaling_points = points[:14]

    # 综合拟合误差：强度相对误差 + 自相关形状误差。
    strength_err = np.mean([s.strength_rel_error for s in valid])
    ac_target = ac_avg[1:]
    ac_model = _ac_from_ar(ar, AR_ORDER)[1:]
    ac_err = float(np.sqrt(np.mean((ac_target - ac_model) ** 2)))
    rep.model_rms_error = float(0.7 * strength_err + 0.3 * ac_err)

    # 模型可达性判定。AV1 颗粒在屏幕坐标逐帧独立合成：
    # 时间相干性高说明信息依附在物体上，模型结构性地无法表达，必须直接编码。
    # 阈值 0.25 位于实测屏幕颗粒（|tc|<0.05）与跟随布纹（>0.3）的分离区间中部。
    if rep.temporal_coherence > 0.25:
        rep.model_feasible = False
        rep.fallback_reason = (
            f"轨迹时间相关 {rep.temporal_coherence:.2f} 过高："
            "微纹跟随物体，屏幕固定颗粒模型无法表达")
    elif rep.model_rms_error > cfg.fit_error_threshold:
        rep.model_feasible = False
        rep.fallback_reason = (
            f"颗粒统计拟合误差 {rep.model_rms_error:.2f} 超过阈值 "
            f"{cfg.fit_error_threshold:.2f}")
    else:
        rep.model_feasible = True
    return rep


def _bin_temporal_corr(residual: np.ndarray, geometry: list[dict],
                       seg_start: int, seg_end: int, clean: np.ndarray,
                       lo: int, hi: int, w: int, h: int) -> float:
    """亮度箱内的轨迹时间相关：亚像素对齐 + 运动轴高通 + 布片掩膜。"""
    corrs = []
    for f in range(seg_start + 1, seg_end):
        vx = float(geometry[f]["vx"])
        prev = _motion_highpass(residual[f - 1])
        aligned = _motion_highpass(_subpixel_shift(residual[f], -vx))
        m_prev = (clean[f - 1] >= lo) & (clean[f - 1] < hi) & \
            _cloth_mask(geometry[f - 1], h, w)
        clean_aligned = _subpixel_shift(clean[f], -vx)
        m_cur = (clean_aligned >= lo) & (clean_aligned < hi) & \
            (_subpixel_shift(
                _cloth_mask(geometry[f], h, w).astype(np.float32), -vx) > 0.5)
        m = m_prev & m_cur
        if m.sum() < 128:
            continue
        x, y = prev[m], aligned[m]
        denom = x.std() * y.std() + 1e-9
        corrs.append(float(((x - x.mean()) * (y - y.mean())).mean() / denom))
    return float(np.mean(corrs)) if corrs else 0.0


def _ac_from_ar(ar: np.ndarray, n_lag: int) -> np.ndarray:
    """由 AR 系数反推理论自相关（解 Yule-Walker 逆问题），用于形状误差评估。"""
    p = len(ar)
    mat = np.eye(p)
    rhs = np.zeros(p)
    for i in range(p):
        rhs[i] = ar[i]
        for j in range(p):
            if i > j:
                mat[i, j] -= ar[i - j - 1] if i - j - 1 < p else 0.0
    r = np.linalg.solve(mat, rhs)
    return np.concatenate([[1.0], r])[: n_lag + 1]
