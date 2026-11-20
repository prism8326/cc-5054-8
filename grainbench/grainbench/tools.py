"""外部编解码工具的薄封装，统一注入用户目录下的动态库路径。"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from . import config


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = config.LIB_PATH
    return env


def run(cmd: list[str], *, cwd: Path | None = None,
        check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, env=_env(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"命令失败 ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return proc


def ffprobe_json(path: Path) -> dict:
    proc = run([
        config.FFPROBE, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    return json.loads(proc.stdout)


def file_size(path: Path) -> int:
    return Path(path).stat().st_size


def write_raw_y4m(raw_yuv: Path, y4m_path: Path, width: int, height: int,
                  fps: int, pix_fmt: str = config.PIX_FMT) -> None:
    """把裸 I420 流封装成 aomenc 可直接读取的 Y4M。"""
    run([
        config.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", pix_fmt,
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", str(raw_yuv), "-pix_fmt", pix_fmt, str(y4m_path),
    ])


def decode_ivf(ivf: Path, out_yuv: Path, *, apply_grain: bool,
               width: int, height: int) -> None:
    """用独立 aomdec 解码；apply_grain=False 时剥离颗粒只留干净底图。"""
    cmd = [config.AOMDEC, "--rawvideo", f"--i420",
           "-o", str(out_yuv), str(ivf)]
    if not apply_grain:
        cmd.append("--skip-film-grain")
    run(cmd)
