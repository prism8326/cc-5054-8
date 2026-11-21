"""颗粒状态实验的确定性不变量测试。

覆盖：
* 逐帧无状态性：同状态在任何帧序/尺寸下合成一致；
* 环境实际跑通端到端实验（需 aom/ffmpeg，缺失则跳过）；
* 逐帧缓存对齐解码器、序号推算在隐藏帧后分叉；
* 关键帧切片起播与顺播逐像素一致（颗粒关/开）；
* 编码块大小不改变颗粒图案；
* 同强度双片：底图一致、强度相近、切镜后参数不同；
* 重复施加使颗粒幅度翻倍（错误调用位置的后果）。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grainbench import config as cfgmod
from grainbench.config import ExperimentConfig
from grainbench.grainsynth import synth_luma

HAVE_TOOLS = all(Path(p).exists()
                 for p in (cfgmod.AOMENC, cfgmod.AOMDEC, cfgmod.FFMPEG))


def _params(seed=7391, scaling=148, overlap=1, lag=2, coeffs=None):
    return dict(scaling_points_y=[(128, scaling)], ar_coeff_lag=lag,
                ar_coeffs_y=(coeffs or [0] * 12), ar_coeff_shift=8,
                grain_scale_shift=0, scaling_shift=11, overlap_flag=overlap,
                random_seed=seed)


def test_synthesis_is_stateless_per_frame():
    """同一种子/参数对同一底帧，无论前后帧如何，输出完全相同。"""
    rng = np.random.default_rng(1)
    base = rng.integers(60, 120, (256, 320), dtype=np.uint8)
    p = _params(seed=12345)
    a = synth_luma(base, p)
    # 中间插入别的合成调用，不应影响结果（无全局状态）。
    synth_luma(base, _params(seed=999))
    b = synth_luma(base, p)
    assert np.array_equal(a, b)
    # 不同种子必须给出不同图案。
    assert not np.array_equal(a, synth_luma(base, _params(seed=7391)))


def test_grain_is_clean_base_only_once():
    """重复施加使颗粒幅度约为单次的 sqrt(2)~2 倍，证明确实叠加了两遍。"""
    base = np.full((256, 320), 110, np.uint8)
    p = _params(seed=7391, scaling=148)
    once = synth_luma(base, p).astype(np.float64)
    twice = synth_luma(once.astype(np.uint8), p).astype(np.float64)
    s1 = (once - base).std()
    s2 = (twice - base).std()
    assert s2 > 1.6 * s1, (s1, s2)


@pytest.mark.skipif(not HAVE_TOOLS, reason="需要 aomenc/aomdec/ffmpeg")
def test_grainlab_end_to_end(tmp_path: Path):
    from grainbench.grainlab import run
    cfg = ExperimentConfig(frames=32, width=320, height=256)
    summary = run(tmp_path, cfg)

    # 逐帧缓存必须与 aomdec 含颗粒输出逐像素一致。
    a = summary["access"]["summary"]
    assert a["cache_matches_decoder"][0] == a["cache_matches_decoder"][1]

    # 真实关键帧切片起播与顺播逐像素一致（颗粒关、开皆然）。
    rs = summary["restart_slice"]
    assert rs["frames"] > 0
    assert rs["base_identical"] and rs["grain_identical"]
    assert rs["base_maxdiff"] == 0 and rs["grain_maxdiff"] == 0

    # 编码块大小不改变颗粒图案。
    assert summary["block_size_invariance"]["grain_pattern_identical"]

    # 颗粒开/关留下相同底图。
    assert summary["grain_switch"]["base_identical_on_vs_off"]

    # 双片：底图一致、切镜后参数不同、强度相近。
    tw = summary["twin"]
    assert tw["base_identical"]
    assert tw["params_at_cut_P"] != tw["params_at_cut_Q"]
    rel = abs(tw["grain_std_segB_P"] - tw["grain_std_segB_Q"]) / \
        max(tw["grain_std_segB_P"], tw["grain_std_segB_Q"])
    assert rel < 0.15, rel
    assert tw["pixel_diff_at_cut_mean"] > 0

    # 重复施加幅度显著更大。
    d = summary["double_apply"]
    assert d["grain_std_doubled"] > 1.6 * d["grain_std_correct"]
    assert d["decoder_vs_single_maxdiff"] == 0


@pytest.mark.skipif(not HAVE_TOOLS, reason="需要 aomenc/aomdec/ffmpeg")
def test_seed_attribution_diverges_only_after_hidden_frames(tmp_path: Path):
    """序号推算种子在真实码流里确实会在隐藏帧/show_existing 后分叉。"""
    from grainbench.access import compare_access
    from grainbench.bitstream import extract, trace_obus
    from grainbench.grainlab import _decode, _encode, _load_y, _table_text, \
        make_scene, _PARAM_A
    from grainbench import tools

    cfg = ExperimentConfig(frames=32, width=320, height=256)
    yuv = tmp_path / "scene.yuv"
    make_scene(cfg, yuv)
    y4m = tmp_path / "scene.y4m"
    tools.write_raw_y4m(yuv, y4m, cfg.width, cfg.height, cfg.fps)
    tbl = tmp_path / "a.tbl"
    tbl.write_text(_table_text(_PARAM_A, _PARAM_A, cfg.frames // 2, cfg.fps))
    ivf = tmp_path / "s.ivf"
    _encode(y4m, ivf, cfg, 40, tbl)

    states = extract(ivf)
    hidden = sum(1 for c in trace_obus(ivf)
                 if c.get("obu_type") in (3, 6)
                 and "show_existing_frame" in c
                 and c.get("show_existing_frame") == 0
                 and c.get("show_frame") == 0)
    base_yuv = tmp_path / "b.yuv"
    grain_yuv = tmp_path / "g.yuv"
    _decode(ivf, base_yuv, cfg, grain=False)
    _decode(ivf, grain_yuv, cfg, grain=True)
    base = _load_y(base_yuv, cfg)
    grain = _load_y(grain_yuv, cfg)
    rows = compare_access(ivf, base, grain)
    # 缓存始终正确。
    assert all(r.cache_matches_decoder for r in rows)
    if hidden:
        # 有隐藏帧时，序号推算应至少在一帧上分叉。
        assert any(r.cache_vs_naive_maxdiff > 0 for r in rows)
    assert len(states) == len(grain)