"""全局配置：素材规格、分段长度、码率预算与外部工具路径。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("HOME", "/home/node"))

# 静态 ffmpeg（含 libaom 编码器与 libdav1d 解码器）和独立 aom 工具链。
# 允许通过环境变量覆盖，便于在已系统安装这些工具的机器上直接复用。
FFMPEG_DIR = Path(os.environ.get("GB_FFMPEG_DIR", HOME / "opt" / "ffmpeg"))
AOM_DIR = Path(os.environ.get("GB_AOM_DIR", HOME / "opt" / "aom" / "usr"))
AOM_LIB = Path(os.environ.get(
    "GB_AOM_LIB", HOME / "opt" / "aom" / "usr" / "lib" / "aarch64-linux-gnu"))

FFMPEG = os.environ.get("GB_FFMPEG", str(FFMPEG_DIR / "ffmpeg"))
FFPROBE = os.environ.get("GB_FFPROBE", str(FFMPEG_DIR / "ffprobe"))
AOMENC = os.environ.get("GB_AOMENC", str(AOM_DIR / "bin" / "aomenc"))
AOMDEC = os.environ.get("GB_AOMDEC", str(AOM_DIR / "bin" / "aomdec"))

# aomenc/aomdec 动态链接到用户目录下 libaom.so.3，运行时需要库路径。
LIB_PATH = f"{AOM_LIB}{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}"

# 素材规格：灰度（Y 平面承载全部信息）、I420 容器、小尺寸保证 CPU 可承受。
WIDTH = 320
HEIGHT = 256
FPS = 24
FRAMES = 48
PIX_FMT = "yuv420p"
SEGMENT_FRAMES = 12          # 每个时间分段 0.5 秒，颗粒参数与路径选择按段生效
LUMA_BINS = (32, 64, 96, 128, 160, 192, 224)   # 分离器的亮度分段边界

# 默认预算（kbps），两路编码使用同一目标码量做公平比较。
DEFAULT_BITRATE_KBPS = 220


@dataclass
class ExperimentConfig:
    width: int = WIDTH
    height: int = HEIGHT
    fps: int = FPS
    frames: int = FRAMES
    segment_frames: int = SEGMENT_FRAMES
    bitrate_kbps: int = DEFAULT_BITRATE_KBPS
    # 模型路径允许的最大相对拟合误差；超出则该段改走直接编码或带界近似。
    fit_error_threshold: float = 0.25
    # 带误差界限的近似：当直接编码超出预算时，允许的最大相对误差上界。
    approximation_bound: float = 0.40
    luma_bins: tuple[int, ...] = LUMA_BINS
    seed: int = 20261120
    extra: dict = field(default_factory=dict)

    @property
    def segments(self) -> list[tuple[int, int]]:
        return [
            (i, min(i + self.segment_frames, self.frames))
            for i in range(0, self.frames, self.segment_frames)
        ]
