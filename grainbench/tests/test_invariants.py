"""核心不变量测试：合成对齐、模型可达性判别、grain table 格式与预算路由。

运行：PYTHONPATH=. python -m pytest tests/ -q
（无 pytest 时可直接 python tests/test_invariants.py）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grainbench import analyzer as az
from grainbench import grainio
from grainbench.config import ExperimentConfig
from grainbench.encoder import plan_segments
from grainbench.synth import generate, read_i420_y


def _small_cfg() -> ExperimentConfig:
    return ExperimentConfig(width=176, height=144, frames=24, segment_frames=8)


def _load(tmp_path: Path):
    cfg = _small_cfg()
    paths = generate(cfg, tmp_path)
    meta = json.loads(paths["meta"].read_text())
    clean = read_i420_y(paths["clean"], cfg).astype(np.float32)
    return cfg, paths, meta, clean


def test_luma_parity(tmp_path: Path):
    cfg, paths, meta, clean = _load(tmp_path)
    # 两组逐帧亮度均值差必须小于 1 个灰度级（"同亮度"要求）。
    assert max(p["mean_delta"] for p in meta["luma_parity"]) < 1.0
    # 两组全局标准差差异控制在 10% 内（高频能量对齐）。
    a = np.mean([p["std_adherent"] for p in meta["luma_parity"]])
    s = np.mean([p["std_screen"] for p in meta["luma_parity"]])
    assert abs(a - s) / max(a, s) < 0.10


def test_feasibility_separation(tmp_path: Path):
    cfg, paths, meta, clean = _load(tmp_path)
    for name, expect_feasible in (("screen", True), ("adherent", None)):
        src = read_i420_y(paths[name], cfg).astype(np.float32)
        residual = src - clean
        flags = []
        for s, e in cfg.segments:
            rep = az.analyze_segment(residual, clean, meta["geometry"], s, e, cfg)
            flags.append(rep.model_feasible)
        if name == "screen":
            # 纯屏幕颗粒：每段都应可被 AV1 颗粒模型表达。
            assert all(flags), flags
        else:
            # 含跟随布纹的中段必须判为不可行，光滑段可行。
            assert flags == [True, False, True], flags


def test_temporal_corr_signature(tmp_path: Path):
    cfg, paths, meta, clean = _load(tmp_path)
    src = read_i420_y(paths["screen"], cfg).astype(np.float32)
    residual = src - clean
    rep = az.analyze_segment(residual, clean, meta["geometry"], 0, 8, cfg)
    # 逐帧独立颗粒沿轨迹的时间相关必须接近 0。
    assert abs(rep.temporal_coherence) < 0.08


def test_ar_coeffs_in_obu_range(tmp_path: Path):
    cfg, paths, meta, clean = _load(tmp_path)
    src = read_i420_y(paths["screen"], cfg).astype(np.float32)
    residual = src - clean
    for s, e in cfg.segments:
        rep = az.analyze_segment(residual, clean, meta["geometry"], s, e, cfg)
        assert len(rep.ar_coeffs_q) == 12
        assert all(-128 <= q <= 127 for q in rep.ar_coeffs_q)
        # 标定后的颗粒不应有任何 scaling 点触顶 255。
        assert all(y < 255 for _, y in rep.scaling_points)


def test_grain_table_format(tmp_path: Path):
    cfg, paths, meta, clean = _load(tmp_path)
    src = read_i420_y(paths["screen"], cfg).astype(np.float32)
    residual = src - clean
    reports = [az.analyze_segment(residual, clean, meta["geometry"], s, e, cfg)
               for s, e in cfg.segments]
    tbl = tmp_path / "g.tbl"
    grainio.write_grain_table(tbl, reports, cfg)
    text = tbl.read_text()
    assert text.startswith("filmgrn1\n")
    # 每段一条 E 记录，且模型段 apply_grain=1。
    entries = [ln for ln in text.splitlines() if ln.startswith("E ")]
    assert len(entries) == len(reports)
    assert all(ln.split()[3] == "1" for ln in entries)
    # 混合表：direct 段 apply_grain=0。
    flags = ["grain", "direct", "grain"]
    grainio.write_grain_table(tbl, reports, cfg, flags)
    entries = [ln for ln in tbl.read_text().splitlines() if ln.startswith("E ")]
    assert [ln.split()[3] for ln in entries] == ["1", "0", "1"]


def test_budget_routing(tmp_path: Path):
    cfg, paths, meta, clean = _load(tmp_path)
    src = read_i420_y(paths["adherent"], cfg).astype(np.float32)
    residual = src - clean
    reports = [az.analyze_segment(residual, clean, meta["geometry"], s, e, cfg)
               for s, e in cfg.segments]
    # 充足预算：不可行段走 direct。
    decisions = plan_segments(
        reports, direct_sizes=[5000, 6000, 5000],
        model_sizes=[3000, 3000, 3000], cfg=cfg)
    assert [d.route for d in decisions] == ["grain-model", "direct", "grain-model"]
    # 紧预算：direct 超支的不可行段回退带界近似，并给出误差界限。
    tight = ExperimentConfig(**{**cfg.__dict__, "bitrate_kbps": 60})
    decisions = plan_segments(
        reports, direct_sizes=[5000, 6000, 5000],
        model_sizes=[3000, 3000, 3000], cfg=tight)
    assert decisions[1].route == "approx-model"
    assert decisions[1].error_bound is not None
    assert decisions[1].error_bound >= tight.approximation_bound


def _run_all():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for fn in (test_luma_parity, test_feasibility_separation,
                   test_temporal_corr_signature, test_ar_coeffs_in_obu_range,
                   test_grain_table_format, test_budget_routing):
            fn(root)
            print(f"PASS {fn.__name__}")


if __name__ == "__main__":
    _run_all()
