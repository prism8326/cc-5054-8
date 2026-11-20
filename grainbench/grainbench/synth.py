"""合成两组同亮度、同运动的测试素材。

场景：一块带织物微纹的圆形布片沿水平方向往复运动，背景为静止渐变。

* ``adherent`` 组：织物微纹烘焙在布片局部坐标里，随布片一起平移（纹理跟物体走），
  再叠加逐帧独立的弱颗粒模拟采集噪声；
* ``screen`` 组：布片表面光滑（无织物微纹），只叠加逐帧独立、统计特性一致的颗粒
  （噪声跟屏幕走）。

两组的逐帧亮度直方图保持一致：screen 组在布片区域用与织物微纹同方差的随机颗粒
补偿被移除的微纹能量，因此"同亮度"在均值与方差两个层面都成立。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig


@dataclass
class FrameGeometry:
    cx: float           # 布片中心（浮点，亚像素运动）
    cy: float
    radius: float
    vx: float           # 该帧速度（像素/帧），供沿轨迹取样使用


def _object_motion(t: np.ndarray, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """水平正弦往复运动，返回中心 x 坐标与速度。"""
    period = cfg.frames
    amp = cfg.width * 0.22
    x0 = cfg.width * 0.5
    phase = 2.0 * np.pi * t / period
    cx = x0 + amp * np.sin(phase)
    vx = amp * 2.0 * np.pi / period * np.cos(phase)
    return cx, vx


def _weave_texture(uo: np.ndarray, vo: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """织物微纹：经纬两个方向的中频正弦交织，加少量低频起伏避免过于规则。

    周期取 9~15 像素：既明显高于编码块尺寸可被直接编码保留，又不至于细到
    被亚像素运动差分抵消。输入为布片局部坐标（随物体移动），输出零均值微纹。
    """
    warp = 4.2 * np.sin(2 * np.pi * uo / 9.0
                        + 1.2 * np.sin(2 * np.pi * vo / 31.0))
    weft = 3.7 * np.cos(2 * np.pi * vo / 11.0
                        + 1.0 * np.sin(2 * np.pi * uo / 27.0))
    # 交叉点压暗，模拟经纬线上下叠压。
    inter = -2.1 * np.sin(2 * np.pi * uo / 18.0) * np.cos(2 * np.pi * vo / 20.0)
    slow = 1.3 * np.sin(2 * np.pi * (uo + vo) / 41.0 + rng.uniform(0, 6))
    return warp + weft + inter + slow


def _spatially_correlated_grain(h: int, w: int, rng: np.random.Generator,
                                sigma: float) -> np.ndarray:
    """逐帧独立、带轻度空间相关的颗粒（高斯白噪经软化核）。

    核强度刻意取弱（相邻像素相关约 0.26）：既保留胶片颗粒的空间纹理感，
    又使 AV1 AR 系数落在 OBU 合法范围而不饱和。每一帧重新抽样，因此
    时间上完全不相关——这是"噪声跟屏幕走"的关键性质。
    """
    white = rng.standard_normal((h, w)).astype(np.float32)
    kernel = np.array([[0.5, 1.0, 0.5],
                       [1.0, 8.0, 1.0],
                       [0.5, 1.0, 0.5]], dtype=np.float32) / 14.0
    padded = np.pad(white, 1, mode="reflect")
    out = np.zeros_like(white)
    for dy in range(3):
        for dx in range(3):
            out += kernel[dy, dx] * padded[dy:dy + h, dx:dx + w]
    out *= sigma / (out.std() + 1e-9)
    return out


def generate(cfg: ExperimentConfig, out_dir: Path) -> dict[str, Path]:
    """生成干净底图、两组带颗粒素材与几何元数据。

    返回各 YUV/元数据文件路径的字典。全部为 8-bit I420，U/V 取中性 128。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w, n = cfg.height, cfg.width, cfg.frames
    t = np.arange(n, dtype=np.float64)
    cx_arr, vx_arr = _object_motion(t, cfg)
    cy = h * 0.5
    radius = h * 0.30

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 静止背景：覆盖暗到亮的垂直渐变 + 极弱固定底纹，跨越多个亮度箱。
    background = (48.0 + 150.0 * yy / h).astype(np.float32)
    background += 4.0 * np.sin(2 * np.pi * 0.01 * xx) * np.sin(2 * np.pi * 0.014 * yy)

    clean = np.empty((n, h, w), np.float32)
    masks = np.empty((n, h, w), np.bool_)
    geo: list[FrameGeometry] = []
    base_rng = np.random.default_rng(cfg.seed)
    weave_phase_rng = np.random.default_rng(cfg.seed + 1)

    for f in range(n):
        cx = float(cx_arr[f])
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # 软边缘（2px 抗锯齿），避免运动时边缘闪烁。
        alpha = np.clip((radius - dist) / 2.0, 0.0, 1.0)
        masks[f] = alpha > 0.5
        # 布片基色取中高亮度，并带随帧缓慢变化的照明起伏。
        cloth_base = 188.0 + 7.0 * np.sin(2 * np.pi * f / n * 2)
        plate = background * (1 - alpha) + cloth_base * alpha
        clean[f] = plate
        geo.append(FrameGeometry(cx=cx, cy=cy, radius=radius, vx=float(vx_arr[f])))

    # 织物微纹只存在于布片局部坐标，跟随物体移动，且只覆盖中间分段：
    # 两端分段布片光滑（模型可表达），中段整段有依附物体的布纹（必须直接编码），
    # 从而在同一素材内产生 model→direct→model 的段间路由切换。
    # 取中间分段的帧区间作为满强度窗，两侧留 4 帧余弦过渡。
    seg_bounds = [i for i in range(0, n, cfg.segment_frames)]
    mid = len(seg_bounds) // 2
    win_lo = seg_bounds[mid]
    win_hi = min(n, seg_bounds[mid] + cfg.segment_frames)
    fade = 4
    texture_window = np.zeros(n, np.float32)
    for f in range(n):
        if f < win_lo or f >= win_hi:
            texture_window[f] = 0.0
        elif f < win_lo + fade:
            texture_window[f] = 0.5 - 0.5 * np.cos(
                np.pi * (f - win_lo) / fade)
        elif f >= win_hi - fade:
            texture_window[f] = 0.5 + 0.5 * np.cos(
                np.pi * (f - (win_hi - fade)) / fade)
        else:
            texture_window[f] = 1.0

    adherent = np.empty_like(clean)
    texture_std = np.empty(n)
    for f in range(n):
        uo = xx - cx_arr[f]
        vo = yy - cy
        tex = _weave_texture(uo, vo, weave_phase_rng)
        tex = tex * masks[f] * texture_window[f]
        texture_std[f] = tex[masks[f]].std() if texture_window[f] > 0 else 0.0
        adherent[f] = clean[f] + tex

    # 颗粒：两组使用独立随机实现但同分布；screen 组在布片上补足微纹的方差。
    # sigma 取 2.5：对应 AV1 scaling≈160；布片补偿后最高箱 sigma≈3.5
    # （scaling≈224），全部落在 OBU 0..255 可表达区间，避免强度先于纹理触顶。
    grain_sigma_bg = 2.5
    grain_rng_a = np.random.default_rng(cfg.seed + 100)
    grain_rng_s = np.random.default_rng(cfg.seed + 200)
    screen = np.empty_like(clean)
    adherent_grain = np.empty_like(clean)
    screen_grain = np.empty_like(clean)

    for f in range(n):
        g_a = _spatially_correlated_grain(h, w, grain_rng_a, grain_sigma_bg)
        adherent_grain[f] = g_a
        adherent[f] += g_a

        g_s_bg = _spatially_correlated_grain(h, w, grain_rng_s, grain_sigma_bg)
        # 布片区域：颗粒标准差 = sqrt(微纹方差 + 背景颗粒方差)，
        # 使 screen 组布片总高频能量与 adherent 组（微纹+颗粒）对齐。
        cloth_sigma = float(np.sqrt(texture_std[f] ** 2 + grain_sigma_bg ** 2))
        g_s_cloth = _spatially_correlated_grain(h, w, grain_rng_s, cloth_sigma)
        g_s = np.where(masks[f], g_s_cloth, g_s_bg)
        screen_grain[f] = g_s
        screen[f] = clean[f] + g_s

    # 亮度校验：两组逐帧均值/方差差异写入元数据，证明"同亮度"。
    parity = []
    for f in range(n):
        parity.append({
            "frame": f,
            "mean_adherent": float(adherent[f].mean()),
            "mean_screen": float(screen[f].mean()),
            "std_adherent": float(adherent[f].std()),
            "std_screen": float(screen[f].std()),
            "mean_delta": float(abs(adherent[f].mean() - screen[f].mean())),
        })

    paths = {
        "clean": out_dir / "clean.yuv",
        "adherent": out_dir / "adherent.yuv",
        "screen": out_dir / "screen.yuv",
        "adherent_grain": out_dir / "adherent_grain.yuv",
        "screen_grain": out_dir / "screen_grain.yuv",
    }
    _write_i420(paths["clean"], clean)
    _write_i420(paths["adherent"], adherent)
    _write_i420(paths["screen"], screen)
    _write_i420(paths["adherent_grain"], adherent_grain)
    _write_i420(paths["screen_grain"], screen_grain)

    meta = {
        "width": w, "height": h, "frames": n, "fps": cfg.fps,
        "pix_fmt": "yuv420p",
        "grain_sigma_background": grain_sigma_bg,
        "texture_std_per_frame": [float(v) for v in texture_std],
        "geometry": [g.__dict__ for g in geo],
        "luma_parity": parity,
    }
    import json
    meta_path = out_dir / "source_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    paths["meta"] = meta_path
    return paths


def _write_i420(path: Path, y_seq: np.ndarray) -> None:
    """把 (n,h,w) float Y 序列写成 8-bit I420，色度平面填中性值。"""
    n, h, w = y_seq.shape
    y_u8 = np.clip(np.rint(y_seq), 0, 255).astype(np.uint8)
    c_h, c_w = h // 2, w // 2
    chroma = np.full((c_h, c_w), 128, np.uint8)
    with open(path, "wb") as fh:
        for f in range(n):
            fh.write(y_u8[f].tobytes())
            fh.write(chroma.tobytes())
            fh.write(chroma.tobytes())


def read_i420_y(path: Path, cfg: ExperimentConfig) -> np.ndarray:
    """读回 I420 的 Y 平面，返回 (n,h,w) uint8。"""
    h, w, n = cfg.height, cfg.width, cfg.frames
    frame_size = h * w + 2 * (h // 2) * (w // 2)
    raw = np.fromfile(path, dtype=np.uint8, count=n * frame_size)
    raw = raw.reshape(n, frame_size)
    return raw[:, : h * w].reshape(n, h, w).copy()
