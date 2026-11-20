"""命令行入口：python -m grainbench.cli run /path/to/out [--quick]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
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
            print(f"  去噪底图对真值 PSNR={m['reference']['denoiser_psnr_to_true_clean']:.2f}")


if __name__ == "__main__":
    main()
