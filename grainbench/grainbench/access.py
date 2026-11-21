"""两种访问模型：从重启点恢复状态 vs 保留逐帧状态缓存。

颗粒合成是逐帧无状态的，所以"从关键帧起播到第 f 帧"与"直接跳到第 f 帧"
在获得**同一份逐帧状态**（种子 + 参数）时必须得到逐像素相同的结果。反之，
若状态归属错误，则两条访问路径会分叉，表现为拖动后纹理换样或叠加两遍。

这里实现并对照三种获取第 f 帧状态的方法：

* ``restart``：按"从最近关键帧重新播放、编码器每帧 ``+3381``"的朴素规则
  推算种子。它在没有隐藏帧、没有分段的流上恰好正确，一旦流里含隐藏帧、
  show_existing 或按段切换的参数就会错位——正是题述"重启后纹理换样"的机理。
* ``naive_index``：按显示序号直接 ``7391 + 3381 * index``。忽略隐藏帧消耗。
* ``cache``：逐帧缓存从码流解析出的真实状态（``bitstream.explain``）。这是
  唯一在任何访问方式下都稳定复现的模型。

``compare_access`` 对同一码流分别用这三种状态来源合成所有帧，返回每帧的
逐像素差异，直接量化"重启恢复 vs 逐帧缓存"是否为同一确定性范围。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import bitstream
from .grainsynth import synth_luma

SEED_START = 7391
SEED_STEP = 3381


@dataclass
class AccessComparison:
    frame: int
    cache_vs_restart_maxdiff: int
    cache_vs_naive_maxdiff: int
    restart_vs_naive_maxdiff: int
    cache_seed: int | None
    restart_seed: int | None
    naive_seed: int | None
    cache_matches_decoder: bool | None
    cache_source: str = "shown"


def naive_seed(display_index: int) -> int:
    return (SEED_START + SEED_STEP * display_index) & 0xffff


def restart_seed(states: list[bitstream.DisplayState],
                 frame: int) -> int | None:
    """从第 frame 帧之前最近的关键帧起，按 +3381 逐帧推进得到的种子。"""
    key = max((s.display_index for s in states[:frame + 1]
               if s.source == "shown" and s.is_keyframe), default=0)
    steps = frame - key
    base_seed = states[key].seed if states[key].seed is not None else SEED_START
    return (base_seed + SEED_STEP * steps) & 0xffff


def _state_params(state: bitstream.DisplayState):
    if state.params is None or state.seed is None:
        return None
    p = dict(state.params)
    p["random_seed"] = state.seed
    return p


def synth_from_state(base: np.ndarray, state: bitstream.DisplayState,
                     override_seed: int | None = None) -> np.ndarray:
    """按某显示帧的状态对干净底图施加颗粒；override_seed 用于对照实验。"""
    p = _state_params(state)
    if p is None:
        return base.copy()
    if override_seed is not None:
        p = dict(p)
        p["random_seed"] = override_seed
    return synth_luma(base, p)


def compare_access(ivf: Path, bases: np.ndarray,
                   decoder: np.ndarray | None = None) -> list[AccessComparison]:
    """对码流所有显示帧比较三种状态来源的合成结果。

    bases 为按显示序排列的干净底图 (n,h,w)；decoder 若给出（aomdec 含颗粒
    输出），则用来核对逐帧缓存的正确性。
    """
    states = bitstream.extract(ivf)
    n = min(len(states), len(bases))
    rows: list[AccessComparison] = []
    for i in range(n):
        st = states[i]
        p_cache = _state_params(st)
        if p_cache is None:
            rows.append(AccessComparison(i, 0, 0, 0, st.seed, None,
                                         naive_seed(i), None, st.source))
            continue
        cache = synth_luma(bases[i], p_cache)
        # 朴素重启：种子按关键帧后 +3381；参数沿用该帧实际参数。
        r_seed = restart_seed(states, i)
        restart = synth_from_state(bases[i], st, override_seed=r_seed)
        n_seed = naive_seed(i)
        naive = synth_from_state(bases[i], st, override_seed=n_seed)
        decoded_ok = None
        if decoder is not None and i < len(decoder):
            decoded_ok = bool(np.array_equal(
                cache.astype(np.int16), decoder[i].astype(np.int16)))
        rows.append(AccessComparison(
            frame=i,
            cache_vs_restart_maxdiff=int(np.abs(
                cache.astype(np.int16) - restart.astype(np.int16)).max()),
            cache_vs_naive_maxdiff=int(np.abs(
                cache.astype(np.int16) - naive.astype(np.int16)).max()),
            restart_vs_naive_maxdiff=int(np.abs(
                restart.astype(np.int16) - naive.astype(np.int16)).max()),
            cache_seed=st.seed, restart_seed=r_seed, naive_seed=n_seed,
            cache_matches_decoder=decoded_ok, cache_source=st.source))
    return rows


def summary(rows: list[AccessComparison]) -> dict:
    """聚合分歧统计：多少帧上"重启恢复"与"逐帧缓存"不一致。"""
    n = len(rows)
    drift_restart = sum(r.cache_vs_restart_maxdiff > 0 for r in rows)
    drift_naive = sum(r.cache_vs_naive_maxdiff > 0 for r in rows)
    verified = [r for r in rows if r.cache_matches_decoder is not None]
    return {
        "frames": n,
        "cache_vs_restart_diverged": drift_restart,
        "cache_vs_naive_diverged": drift_naive,
        "cache_matches_decoder": (
            sum(bool(r.cache_matches_decoder) for r in verified),
            len(verified)),
        "max_cache_vs_restart": max(
            (r.cache_vs_restart_maxdiff for r in rows), default=0),
        "max_cache_vs_naive": max(
            (r.cache_vs_naive_maxdiff for r in rows), default=0),
    }
