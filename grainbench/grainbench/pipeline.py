"""端到端编排：素材生成 → 分离 → 双路编码 → 独立解码核对 → 报告落盘。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import analyzer as az
from . import encoder as enc
from . import verify
from .config import ExperimentConfig
from .synth import generate, read_i420_y


def _save_png(frames: np.ndarray, out_dir: Path, prefix: str,
              sample_frames: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sample_frames:
        if f >= len(frames):
            continue
        img = Image.fromarray(frames[f].astype(np.uint8), mode="L")
        img.save(out_dir / f"{prefix}_f{f:03d}.png")


def run(out_root: Path, cfg: ExperimentConfig | None = None) -> Path:
    cfg = cfg or ExperimentConfig()
    out_root.mkdir(parents=True, exist_ok=True)
    src_dir = out_root / "sources"
    paths = generate(cfg, src_dir)
    meta = json.loads(paths["meta"].read_text())
    geometry = meta["geometry"]
    sample_frames = list(range(0, cfg.frames, 6))

    clean = read_i420_y(paths["clean"], cfg).astype(np.float32)
    report_index: dict[str, list[dict]] = {}
    all_results: dict[str, dict] = {}

    for name in ("adherent", "screen"):
        group_dir = out_root / name
        group_dir.mkdir(parents=True, exist_ok=True)
        src = read_i420_y(paths[name], cfg).astype(np.float32)
        residual = src - clean

        # 分离器：分段强度、空间/时间相关性与模型可达性。
        reports = []
        for seg_start, seg_end in cfg.segments:
            rep = az.analyze_segment(residual, clean, geometry,
                                     seg_start, seg_end, cfg)
            reports.append(rep)
        report_index[name] = [_report_to_dict(r) for r in reports]

        # 干净底图估计（外部去噪器），并量化其估计误差。
        base_yuv = group_dir / f"{name}_denoised.yuv"
        grain_sigma = meta["grain_sigma_background"]
        enc.denoise(paths[name], base_yuv, cfg, strength=grain_sigma + 1.0)
        base_frames = read_i420_y(base_yuv, cfg).astype(np.float32)
        base_psnr = verify.psnr(clean, base_frames)

        # 两条编码路径 + 分段预算决策的混合路径。
        manifest = enc.run_encoding_pipeline(
            name, paths[name], base_yuv, reports, group_dir, cfg)

        # 独立解码核对：每个码流分别取含颗粒合成与剥离颗粒两种输出。
        streams = {
            "direct": Path(manifest["direct_ivf"]),
            "model": Path(manifest["model_ivf"]),
            "hybrid": Path(manifest["hybrid_ivf"]),
        }
        metrics = {}
        decoded_store = {}
        for route, ivf in streams.items():
            grain_out = verify.evaluate_stream(
                ivf, src, geometry, cfg, apply_grain=True)
            base_out = verify.evaluate_stream(
                ivf, clean, geometry, cfg, apply_grain=False)
            decoded_store[route] = grain_out["frames"]
            metrics[route] = {
                "bytes": manifest["sizes"][route],
                "with_grain": _strip_frames(grain_out),
                "base_only": _strip_frames(base_out),
            }

        # 参考基准：原片与干净底图自身的统计。
        ref_obj, ref_scr = verify.trail_correlations(src, geometry, cfg)
        clean_obj, clean_scr = verify.trail_correlations(clean, geometry, cfg)
        metrics["reference"] = {
            "source_trail_object_corr": ref_obj,
            "source_trail_screen_corr": ref_scr,
            "clean_trail_object_corr": clean_obj,
            "clean_trail_screen_corr": clean_scr,
            "denoiser_psnr_to_true_clean": base_psnr,
        }
        metrics["route_change_map"] = verify.route_change_map(
            manifest["decisions"], report_index[name], cfg)

        # UI 用抽样帧 PNG（同码量三态切换的图像来源）。
        png_dir = group_dir / "frames"
        _save_png(src, png_dir, "source", sample_frames)
        _save_png(clean, png_dir, "clean", sample_frames)
        for route, frames in decoded_store.items():
            _save_png(frames, png_dir, route, sample_frames)

        all_results[name] = {
            "manifest": manifest,
            "metrics": metrics,
            "sample_frames": sample_frames,
        }

    summary = {
        "config": cfg.__dict__,
        "luma_parity": meta["luma_parity"],
        "separator": report_index,
        "results": _json_safe(all_results),
        "sample_frames": sample_frames,
    }
    # 去掉 numpy 标量，保证 JSON 可序列化。
    summary_path = out_root / "report.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary_path


def _report_to_dict(rep: az.SegmentReport) -> dict:
    return {
        "start": rep.start, "end": rep.end,
        "temporal_coherence": rep.temporal_coherence,
        "model_rms_error": rep.model_rms_error,
        "model_feasible": rep.model_feasible,
        "fallback_reason": rep.fallback_reason,
        "approximation_bound": rep.approximation_bound,
        "ar_coeffs": rep.ar_coeffs,
        "ar_coeffs_q8": rep.ar_coeffs_q,
        "scaling_points": rep.scaling_points,
        "luma_bins": [b.__dict__ for b in rep.luma_bins],
    }


def _strip_frames(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "frames"}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj
