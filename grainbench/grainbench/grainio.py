"""生成 aomenc ``--film-grain-table`` 可读的文本参数表。

格式（见 aom 源码 aom_dsp/grain_table.c）：

    filmgrn1
    E <start-time> <end-time> <apply-grain> <random-seed> <update-parms>
    \tp <ar_coeff_lag> <ar_coeff_shift> <grain_scale_shift> <scaling_shift>
           <chroma_scaling_from_luma> <overlap_flag>
           <cb_mult> <cb_luma_mult> <cb_offset>
           <cr_mult> <cr_luma_mult> <cr_offset>
    \tsY <num> <x0> <y0> ...
    \tsCb <num> ...
    \tsCr <num> ...
    \tcY <12 个 Y AR 系数>
    \tcCb <13 个>
    \tcCr <13 个>

时间戳单位是纳秒级时间基（time_base = 1/fps 秒，aom 内部以 1e7/秒换算），
与 noise_model 示例 ``frame * 10000000 / fps`` 保持一致。
"""
from __future__ import annotations

from pathlib import Path

from .analyzer import SegmentReport
from .config import ExperimentConfig

# 灰度素材：色度关闭颗粒，全部参数置中性。
_CHROMA_ZERO = [0] * 13


def _time_base(frame: int, fps: int) -> int:
    return frame * 10_000_000 // fps


def write_grain_table(path: Path, reports: list[SegmentReport],
                      cfg: ExperimentConfig,
                      decisions: list[str] | None = None) -> None:
    """按分段写表。decisions 中标记为 'direct' 的段 apply_grain=0。

    混合路径下直接编码段不合成颗粒（颗粒已被直接编入纹理），模型段才打开合成。
    """
    lines = ["filmgrn1"]
    seed = 7391
    for idx, rep in enumerate(reports):
        start = _time_base(rep.start, cfg.fps)
        end = (
            _time_base(rep.end, cfg.fps)
            if rep.end < cfg.frames
            else 9_223_372_036_854_775_807
        )
        apply_grain = 0 if (decisions and decisions[idx] == "direct") else 1
        update = 1
        lines.append(f"E {start} {end} {apply_grain} {seed} {update}")
        seed = 0
        # p 行 12 个标量：lag=2, AR shift=8, grain_scale=0, scaling_shift=11,
        # chroma_from_luma=0, overlap=1, 色度 6 个中性参数。
        lines.append(
            "\tp 2 8 0 11 0 1 0 0 0 0 0 0")
        pts = rep.scaling_points or [(128, 24)]
        sy = " ".join(f"{x} {y}" for x, y in pts)
        lines.append(f"\tsY {len(pts)} {sy}")
        lines.append("\tsCb 0")
        lines.append("\tsCr 0")
        ar_y = rep.ar_coeffs_q or [0] * 12
        ar_y = (ar_y + [0] * 12)[:12]
        lines.append("\tcY " + " ".join(str(v) for v in ar_y))
        lines.append("\tcCb " + " ".join(str(v) for v in _CHROMA_ZERO))
        lines.append("\tcCr " + " ".join(str(v) for v in _CHROMA_ZERO))
    path.write_text("\n".join(lines) + "\n")
