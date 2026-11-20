"""两条编码路径与分段预算决策。

路径 A（direct）：直接压缩含颗粒/微纹的原片，编码器必须为每帧的高频细节付码率。
路径 B（grain-model）：先估计干净底图，压缩底图，颗粒统计以 AV1 film grain 元数据
写入码流，接收端解码器按元数据合成颗粒。

分段决策：分离器判定模型不可行的段改走直接编码；若直接编码导致该段超出预算，则
退而求其次给出带误差界限的近似（保留模型路径并记录误差上界），保证总码量不破预算。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config as cfgmod
from . import tools
from .analyzer import SegmentReport
from .config import ExperimentConfig
from .grainio import write_grain_table


@dataclass
class SegmentDecision:
    index: int
    start: int
    end: int
    route: str                      # "grain-model" | "direct" | "approx-model"
    reason: str
    model_error: float
    error_bound: float | None
    direct_bytes: int
    model_bytes: int
    budget_bytes: int


def denoise(raw_yuv: Path, out_yuv: Path, cfg: ExperimentConfig,
            strength: float = 4.0) -> None:
    """用 ffmpeg hqdn3d 做时空去噪，估计干净底图。

    强度按素材颗粒 sigma 标定；hqdn3d 的时空窗对逐帧独立颗粒抑制充分，
    对跟随物体的微纹会产生拖影——该失真会在独立解码核对中量化。
    """
    s = strength
    tools.run([
        cfgmod.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", cfgmod.PIX_FMT,
        "-video_size", f"{cfg.width}x{cfg.height}", "-framerate", str(cfg.fps),
        "-i", str(raw_yuv),
        "-vf", f"hqdn3d={s}:{s}:{s * 1.5}:{s * 1.5}",
        "-f", "rawvideo", "-pix_fmt", cfgmod.PIX_FMT, str(out_yuv),
    ])


def _aomenc_common(cfg: ExperimentConfig, bitrate_kbps: int) -> list[str]:
    """全片流：CBR 固定目标码率，保证两路在同一码量下公平比较。"""
    return [
        cfgmod.AOMENC, "--codec=av1", f"--width={cfg.width}",
        f"--height={cfg.height}", f"--fps={cfg.fps}/1",
        f"--limit={cfg.frames}", "--i420", "--bit-depth=8",
        "--cpu-used=6", "--end-usage=cbr",
        f"--target-bitrate={bitrate_kbps}",
        "--lag-in-frames=12", "--auto-alt-ref=1", "--kf-max-dist=9999",
        "--threads=8",
    ]


def _aomenc_cq(seg_cfg: ExperimentConfig, cq_level: int = 20) -> list[str]:
    """段探测：约束质量编码，段字节反映该段真实纹理复杂度而非码率上限。

    预算决策需要知道"保住这段细节要花多少字节"，CBR 会主动降质适配而失真，
    因此探测流用 CQ；最终交付的全片流仍用 CBR 锁定同码量。
    """
    return [
        cfgmod.AOMENC, "--codec=av1", f"--width={seg_cfg.width}",
        f"--height={seg_cfg.height}", f"--fps={seg_cfg.fps}/1",
        f"--limit={seg_cfg.frames}", "--i420", "--bit-depth=8",
        "--cpu-used=6", "--end-usage=q", f"--cq-level={cq_level}",
        "--lag-in-frames=12", "--auto-alt-ref=1", "--kf-max-dist=9999",
        "--threads=8",
    ]


def encode_direct(src_y4m: Path, ivf_out: Path, cfg: ExperimentConfig,
                  bitrate_kbps: int) -> None:
    cmd = _aomenc_common(cfg, bitrate_kbps) + [
        "-o", str(ivf_out), str(src_y4m)]
    tools.run(cmd)


def encode_with_grain(base_y4m: Path, ivf_out: Path, table: Path,
                      cfg: ExperimentConfig, bitrate_kbps: int) -> None:
    cmd = _aomenc_common(cfg, bitrate_kbps) + [
        f"--film-grain-table={table}",
        "-o", str(ivf_out), str(base_y4m)]
    tools.run(cmd)


def plan_segments(reports: list[SegmentReport],
                  direct_sizes: list[int], model_sizes: list[int],
                  cfg: ExperimentConfig) -> list[SegmentDecision]:
    """按段选择路径并做预算约束。

    每段预算 = 总目标码量按帧数均摊。段级探测流各自带 IVF 容器与序列头的
    固定开销（短段占比可观），预算扣除统一的 ``SEGMENT_OVERHEAD`` 后再比较，
    避免把容器开销误判成编码超支。模型可行走模型；不可行优先直接编码，
    直接编码破预算时回退到带误差界限的模型近似。
    """
    total_budget = int(
        cfg.bitrate_kbps * 1000 / 8 * cfg.frames / cfg.fps)
    segment_overhead = 1200
    decisions: list[SegmentDecision] = []
    for i, rep in enumerate(reports):
        n_frames = rep.end - rep.start
        budget = total_budget * n_frames // cfg.frames
        d_bytes, m_bytes = direct_sizes[i], model_sizes[i]
        d_payload = max(0, d_bytes - segment_overhead)
        if rep.model_feasible:
            route, reason = "grain-model", "模型在误差阈值内可同时复现强度与相关性"
            bound = None
        elif d_payload <= budget:
            route, reason = "direct", rep.fallback_reason
            bound = None
        else:
            route = "approx-model"
            reason = (f"{rep.fallback_reason}；且直接编码 {d_payload}B 超预算 "
                      f"{budget}B，回退带界近似")
            bound = max(rep.model_rms_error, cfg.approximation_bound)
        decisions.append(SegmentDecision(
            index=i, start=rep.start, end=rep.end, route=route, reason=reason,
            model_error=rep.model_rms_error, error_bound=bound,
            direct_bytes=d_bytes, model_bytes=m_bytes, budget_bytes=budget))
    return decisions


def build_hybrid_table(reports: list[SegmentReport],
                       decisions: list[SegmentDecision],
                       table_path: Path, cfg: ExperimentConfig) -> None:
    """混合路径的 grain table：直接段关闭合成，模型段打开。"""
    flags = ["direct" if d.route == "direct" else "grain" for d in decisions]
    write_grain_table(table_path, reports, cfg, flags)


def run_encoding_pipeline(name: str, src_yuv: Path, base_yuv: Path,
                          reports: list[SegmentReport], work: Path,
                          cfg: ExperimentConfig) -> dict:
    """编排单组素材的完整编码实验，返回产物索引与决策。"""
    work.mkdir(parents=True, exist_ok=True)
    src_y4m, base_y4m = work / f"{name}.y4m", work / f"{name}_base.y4m"
    tools.write_raw_y4m(src_yuv, src_y4m, cfg.width, cfg.height, cfg.fps)
    tools.write_raw_y4m(base_yuv, base_y4m, cfg.width, cfg.height, cfg.fps)

    # 全片两条参考路径（用于同码量对比）。
    direct_ivf = work / f"{name}_direct.ivf"
    model_ivf = work / f"{name}_model.ivf"
    full_table = work / f"{name}_full.tbl"
    write_grain_table(full_table, reports, cfg)
    encode_direct(src_y4m, direct_ivf, cfg, cfg.bitrate_kbps)
    encode_with_grain(base_y4m, model_ivf, full_table, cfg, cfg.bitrate_kbps)

    # 分段码量探测：把每段单独编码以获得真实段级字节数（决策依据）。
    direct_sizes, model_sizes = [], []
    seg_dir = work / "segments"
    seg_dir.mkdir(exist_ok=True)
    for i, rep in enumerate(reports):
        d_ivf = seg_dir / f"{name}_s{i}_direct.ivf"
        m_ivf = seg_dir / f"{name}_s{i}_model.ivf"
        seg_cfg = _segment_cfg(cfg, rep)
        _encode_segment(src_y4m, d_ivf, seg_cfg, rep, grain=False)
        _encode_segment(base_y4m, m_ivf, seg_cfg, rep, grain=True,
                        table=full_table)
        direct_sizes.append(tools.file_size(d_ivf))
        model_sizes.append(tools.file_size(m_ivf))

    decisions = plan_segments(reports, direct_sizes, model_sizes, cfg)
    hybrid_table = work / f"{name}_hybrid.tbl"
    build_hybrid_table(reports, decisions, hybrid_table, cfg)
    hybrid_ivf = work / f"{name}_hybrid.ivf"
    encode_with_grain(base_y4m, hybrid_ivf, hybrid_table, cfg,
                      cfg.bitrate_kbps)

    manifest = {
        "name": name,
        "direct_ivf": str(direct_ivf),
        "model_ivf": str(model_ivf),
        "hybrid_ivf": str(hybrid_ivf),
        "grain_table": str(full_table),
        "hybrid_table": str(hybrid_table),
        "segment_direct_bytes": direct_sizes,
        "segment_model_bytes": model_sizes,
        "decisions": [d.__dict__ for d in decisions],
        "sizes": {
            "direct": tools.file_size(direct_ivf),
            "model": tools.file_size(model_ivf),
            "hybrid": tools.file_size(hybrid_ivf),
        },
    }
    (work / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _segment_cfg(cfg: ExperimentConfig, rep: SegmentReport) -> ExperimentConfig:
    return ExperimentConfig(
        width=cfg.width, height=cfg.height, fps=cfg.fps,
        frames=rep.end - rep.start, segment_frames=cfg.segment_frames,
        bitrate_kbps=cfg.bitrate_kbps,
        fit_error_threshold=cfg.fit_error_threshold,
        approximation_bound=cfg.approximation_bound,
        luma_bins=cfg.luma_bins, seed=cfg.seed)


def _encode_segment(y4m: Path, ivf: Path, seg_cfg: ExperimentConfig,
                    rep: SegmentReport, *, grain: bool,
                    table: Path | None = None) -> None:
    # aomenc 不支持帧裁切参数，先用 ffmpeg 从 Y4M 抽出该段，再做 CQ 探测编码。
    from . import config as c
    tmp_y4m = ivf.with_suffix(".y4m")
    tools.run([
        c.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(y4m), "-vf", f"select='between(n,{rep.start},{rep.end - 1})'",
        "-vsync", "0", "-fps_mode", "passthrough", str(tmp_y4m)])
    cmd = _aomenc_cq(seg_cfg)
    if grain:
        cmd.append(f"--film-grain-table={table}")
    cmd += ["-o", str(ivf), str(tmp_y4m)]
    tools.run(cmd)
    tmp_y4m.unlink(missing_ok=True)
