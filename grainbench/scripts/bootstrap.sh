#!/usr/bin/env bash
# 实验台环境引导：在无 root 的 Debian 12 (arm64) 容器内准备 Python 依赖、
# 静态 ffmpeg 与支持 film-grain-table 的独立 aomenc/aomdec。
# 已存在的工具会跳过，可重复执行。
set -euo pipefail

HOME_DIR="${HOME}"
OPT="${HOME_DIR}/opt"
mkdir -p "${OPT}" /tmp/gb-bootstrap
cd /tmp/gb-bootstrap

# 1) pip（PEP668 受管环境用 --break-system-packages 装到用户目录）。
if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "[1/4] 引导 pip ..."
  if [ ! -f get-pip.py ]; then
    curl -sSL -o get-pip.py https://bootstrap.pypa.io/get-pip.py
  fi
  python3 get-pip.py --user --break-system-packages --quiet
fi
PIP="${HOME_DIR}/.local/bin/pip3"

# 2) Python 库。
echo "[2/4] 安装 numpy/scipy/pillow/pytest ..."
"${PIP}" install --user --break-system-packages --quiet \
  numpy scipy pillow pytest

# 3) 静态 ffmpeg（含 libaom 封装与 libdav1d 解码器）。
if [ ! -x "${OPT}/ffmpeg/ffmpeg" ]; then
  echo "[3/4] 安装静态 ffmpeg ..."
  curl -sSL -o ffmpeg.tar.xz \
    https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz
  tar -xf ffmpeg.tar.xz -C "${OPT}"
  d=$(find "${OPT}" -maxdepth 1 -type d -name 'ffmpeg-*-static' | head -1)
  ln -sfn "${d}" "${OPT}/ffmpeg"
else
  echo "[3/4] ffmpeg 已存在，跳过"
fi

# 4) 独立 aom 3.6（bookworm，glibc 2.36 兼容；支持 --film-grain-table）。
if [ ! -x "${OPT}/aom/usr/bin/aomenc" ]; then
  echo "[4/4] 安装 aomenc/aomdec 3.6 ..."
  base="http://deb.debian.org/debian/pool/main"
  curl -sSL -O "${base}/a/aom/aom-tools_3.6.0-1+deb12u2_arm64.deb"
  curl -sSL -O "${base}/a/aom/libaom3_3.6.0-1+deb12u2_arm64.deb"
  curl -sSL -O "${base}/liby/libyuv/libyuv0_0.0~git20230123.b2528b0-1_arm64.deb"
  mkdir -p "${OPT}/aom"
  for f in *.deb; do dpkg-deb -x "${f}" "${OPT}/aom"; done
else
  echo "[4/4] aom 已存在，跳过"
fi

# 冒烟自检。
export LD_LIBRARY_PATH="${OPT}/aom/usr/lib/aarch64-linux-gnu"
"${OPT}/ffmpeg/ffmpeg" -version | head -1
"${OPT}/aom/usr/bin/aomenc" --help 2>&1 | grep -q film-grain-table \
  && echo "aomenc film-grain-table 支持 OK"
echo "环境就绪。运行："
echo "  python3 -m grainbench.cli run     <输出目录>   # 主实验（三路编码对照）"
echo "  python3 -m grainbench.cli grainlab <输出目录>   # 平场+切镜颗粒状态实验"
echo "  python3 -m grainbench.cli web     <输出目录>   # 启动实验台界面"
