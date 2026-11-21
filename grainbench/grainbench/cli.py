"""命令行入口：
    python -m grainbench.cli run /path/to/out [--quick]
    python -m grainbench.cli grainlab /path/to/out
    python -m grainbench.cli web /path/to/out [--port 8000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .grainlab import run as run_grainlab
from .pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description="胶片颗粒编解码实验台")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="运行完整实验管线")
    r.add_argument("out", type=Path, help="实验产物输出目录")
    r.add_argument("--quick", action="store_true",
                   help="快速冒烟模式：更少帧、更小尺寸")
    r.add_argument("--bitrate", type=int, default=None, help="目标码率 kbps")
    r.add_argument("--threshold", type=float, default=None,
                   help="模型拟合误差阈值")
    g = sub.add_parser("grainlab", help="平场+切镜的颗粒状态实验")
    g.add_argument("out", type=Path, help="实验产物输出目录")
    g.add_argument("--quick", action="store_true",
                   help="快速冒烟模式：更少帧")
    w = sub.add_parser("web", help="启动零依赖实验台界面")
    w.add_argument("out", type=Path, help="实验产物输出目录")
    w.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.cmd == "run":
        if args.quick:
            cfg = ExperimentConfig(width=176, height=144, frames=24,
                                   segment_frames=8)
        else:
            cfg = ExperimentConfig()
        if args.bitrate:
            cfg.bitrate_kbps = args.bitrate
        if args.threshold is not None:
            cfg.fit_error_threshold = args.threshold
        report = run(args.out, cfg)
        data = json.loads(report.read_text())
        print(f"报告已写入: {report}")
        for name in ("adherent", "screen"):
            m = data["results"][name]["metrics"]
            print(f"\n== {name} ==")
            for route in ("direct", "model", "hybrid"):
                mm = m[route]
                print(f"  {route:7s} {mm['bytes']:8d}B  "
                      f"PSNR={mm['with_grain']['mean_psnr']:5.2f}  "
                      f"高频能量={mm['with_grain']['mean_high_band']:.3f}  "
                      f"轨迹(物体/屏幕)={mm['with_grain']['trail_object_corr']:.2f}/"
                      f"{mm['with_grain']['trail_screen_corr']:.2f}")
            print(f"  去噪底图对真值 PSNR="
                  f"{m['reference']['denoiser_psnr_to_true_clean']:.2f}")

    elif args.cmd == "grainlab":
        cfg = ExperimentConfig(frames=24, width=320, height=256) \
            if args.quick else ExperimentConfig()
        summary = run_grainlab(args.out, cfg)
        print(f"颗粒状态实验已写入: {args.out / 'grainlab' / 'grainlab.json'}")
        a = summary["access"]["summary"]
        rs = summary["restart_slice"]
        print(f"  逐帧缓存对齐解码器: {a['cache_matches_decoder'][0]}/"
              f"{a['cache_matches_decoder'][1]} 帧")
        print(f"  序号推算种子分叉: {a['cache_vs_restart_diverged']}/"
              f"{a['frames']} 帧（最大差 {a['max_cache_vs_restart']} 灰阶）")
        print(f"  关键帧切片起播 vs 顺播: 底图一致={rs['base_identical']} "
              f"含颗粒一致={rs['grain_identical']}（{rs['frames']} 帧）")
        bs_ = summary["block_size_invariance"]
        print(f"  解码/编码块大小 64 vs 128 颗粒图案一致: "
              f"{bs_['grain_pattern_identical']}")
        tw = summary["twin"]
        print(f"  双片底图一致={tw['base_identical']} 切镜后颗粒强度 "
              f"P={tw['grain_std_segB_P']:.2f} Q={tw['grain_std_segB_Q']:.2f} "
              f"像素差均值={tw['pixel_diff_at_cut_mean']:.2f}")

    elif args.cmd == "web":
        import sys
        sys.argv = ["grainbench.web", str(args.out), str(args.port)]
        from . import web_server
        web_server.main()


if __name__ == "__main__":
    main()
