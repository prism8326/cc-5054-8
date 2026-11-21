"""静止平场 + 切镜的颗粒状态实验。

本模块回答题述现象：为什么"从头播放正常，拖到镜头中间或改一次解码块大小，
纹理就换样或被叠加两遍"。它固化四项可复核的事实：

1. **保存状态**：对每条码流保存干净底图、码流里的颗粒参数、以及每个显示帧
   实际采用的随机状态（种子 + 参数 + 来源），见 ``states/<stream>.json``。
2. **两种访问模型**：真实按关键帧切片起播（``bitstream.build_restart_stream``）
   与顺播逐帧解码，二者在颗粒关/开时都应逐像素一致；但**按显示序号推算种子**
   （``access.restart_seed`` / ``naive_seed``）会在隐藏帧与 show_existing 处分叉。
3. **颗粒开关测度**：平场+切镜素材在颗粒开/关下测底图差、空间方差、帧间相关。
4. **同强度双片**：两段素材使用相同视觉颗粒强度，却在切镜处选用不同 AR 参数，
   底图逐一相同、强度相等、参数不同。

产物布局（相对 ``out_root``）::

    grainlab/
      material/           素材 YUV/Y4M
      streams/            码流 <stream>.ivf
      decoded/            解码 YUV <stream>.{base,grain}.yuv
      states/             每显示帧状态 <stream>.json
      frames/<stream>/    逐帧 PNG（base/grain/replay）
      grainlab.json       指标 + 状态摘要 + 逐帧重演结论
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import access
from . import bitstream as bs
from . import config as cfgmod
from . import tools
from .config import ExperimentConfig
from .grainsynth import synth_luma
from .synth import _write_i420

# 两段素材：前半静止平场（亮度 96），后半切镜到运动物体 / 渐变。
_FLAT_LUMA = 96.0

# 三种颗粒参数。H 与 V 分别朝水平 / 垂直方向做 AR 预测，overlap 不同，
# 强度经 scaling 标定到与 A 接近（约 2.37 灰阶标准差）。
_PARAM_A = dict(scaling=148, lag=2, overlap=1, coeffs=[0] * 12)
_PARAM_H = dict(scaling=126, lag=2, overlap=0,
                coeffs=[0, 0, 0, 0, 0, 40, 0, 40, 0, 40, 60, 40])
_PARAM_V = dict(scaling=138, lag=2, overlap=1,
                coeffs=[40, 40, 40, 40, 40, 0, 0, 0, 0, 0, 0, 0])


def make_scene(cfg: ExperimentConfig, path: Path) -> None:
    """静止平场（前半）→ 运动亮斑 + 横向渐变（后半）。"""
    h, w, n = cfg.height, cfg.width, cfg.frames
    cut = n // 2
    yy, xx = np.mgrid[0:h, 0:w]
    y = np.empty((n, h, w), np.float32)
    for f in range(n):
        if f < cut:
            y[f] = _FLAT_LUMA
        else:
            cx = 50.0 + (f - cut) * 8.0
            blob = (xx - cx) ** 2 + (yy - h * 0.5) ** 2 < 55 ** 2
            plate = np.where(blob, 205.0, 150.0) * (0.85 + 0.15 * xx / w)
            y[f] = plate
    _write_i420(path, y)


def make_twin(cfg: ExperimentConfig, path: Path) -> None:
    """静止平场（前半）→ 垂直渐变（后半）；用于同强度双片对照。"""
    h, w, n = cfg.height, cfg.width, cfg.frames
    cut = n // 2
    grad = np.linspace(72, 168, h)[:, None].repeat(w, axis=1)
    y = np.empty((n, h, w), np.float32)
    for f in range(n):
        y[f] = _FLAT_LUMA if f < cut else grad
    _write_i420(path, y)


def _table_text(seg_a: dict, seg_b: dict, cut: int, fps: int) -> str:
    """两段 grain table：切镜时间戳处切换到第二组参数。"""
    cut_ts = cut * 10_000_000 // fps
    lines = ["filmgrn1"]

    def entry(start: int, end: int, seed: int, params: dict) -> None:
        lines.append(f"E {start} {end} 1 {seed} 1")
        lines.append(
            f"\tp {params['lag']} 8 0 11 0 {params['overlap']} 0 0 0 0 0 0")
        lines.append(f"\tsY 1 128 {params['scaling']}")
        lines.append("\tsCb 0")
        lines.append("\tsCr 0")
        lines.append("\tcY " + " ".join(str(v) for v in params["coeffs"]))
        lines.append("\tcCb " + " ".join(["0"] * 13))
        lines.append("\tcCr " + " ".join(["0"] * 13))

    entry(0, cut_ts, 7391, seg_a)
    entry(cut_ts, 9_223_372_036_854_775_807, 12345, seg_b)
    return "\n".join(lines) + "\n"


def _encode(y4m: Path, ivf: Path, cfg: ExperimentConfig, cq: int,
            table: Path | None) -> None:
    cmd = [
        cfgmod.AOMENC, "--codec=av1", f"--width={cfg.width}",
        f"--height={cfg.height}", f"--fps={cfg.fps}/1",
        f"--limit={cfg.frames}", "--i420", "--bit-depth=8",
        "--cpu-used=6", "--end-usage=q", f"--cq-level={cq}",
        "--lag-in-frames=16", "--auto-alt-ref=1", "--kf-max-dist=9999",
        "--threads=4",
    ]
    if table is not None:
        cmd.append(f"--film-grain-table={table}")
    cmd += ["-o", str(ivf), str(y4m)]
    tools.run(cmd)


def _decode(ivf: Path, out: Path, cfg: ExperimentConfig, *, grain: bool) -> None:
    tools.decode_ivf(ivf, out, apply_grain=grain,
                     width=cfg.width, height=cfg.height)


def _load_y(path: Path, cfg: ExperimentConfig) -> np.ndarray:
    h, w = cfg.height, cfg.width
    frame_size = h * w + 2 * (h // 2) * (w // 2)
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // frame_size
    return raw.reshape(n, frame_size)[:, : h * w].reshape(n, h, w).copy()


def _save_png(frame: np.ndarray, path: Path) -> None:
    Image.fromarray(frame.astype(np.uint8), mode="L").save(path)


def _states_json(states: list[bs.DisplayState]) -> list[dict]:
    return [s.to_dict() for s in states]


def _replay(base: np.ndarray, decoder: np.ndarray,
            states: list[bs.DisplayState]) -> list[dict]:
    """逐帧用缓存状态独立重演，与解码器含颗粒输出逐像素对照。"""
    out = []
    n = min(len(base), len(decoder), len(states))
    for i in range(n):
        st = states[i]
        if st.params is None or st.seed is None:
            replay_frame = base[i]
        else:
            p = dict(st.params)
            p["random_seed"] = st.seed
            replay_frame = synth_luma(base[i], p)
        diff = np.abs(replay_frame.astype(np.int16) -
                      decoder[i].astype(np.int16))
        out.append({
            "frame": i,
            "seed": st.seed,
            "source": st.source,
            "exact": bool(diff.max() == 0),
            "maxdiff": int(diff.max()),
            "mean_abs": float(diff.mean()),
        })
        out[-1]["_frame"] = replay_frame
    return out


def _spatial_variance(frames: np.ndarray) -> list[float]:
    return [float(f.astype(np.float64).var()) for f in frames]


def _interframe_corr(frames: np.ndarray) -> list[float]:
    corr = []
    for i in range(1, len(frames)):
        a = frames[i - 1].astype(np.float64).ravel()
        b = frames[i].astype(np.float64).ravel()
        denom = a.std() * b.std()
        corr.append(float(np.corrcoef(a, b)[0, 1]) if denom > 1e-9 else 1.0)
    return corr


def _grain_std(grain: np.ndarray, base: np.ndarray, lo: int,
               hi: int) -> float:
    vals = [(grain[f].astype(np.float64) - base[f].astype(np.float64)).std()
            for f in range(lo, min(hi, len(grain)))]
    return float(np.mean(vals))


def _double_apply_demo(base: np.ndarray, decoder: np.ndarray,
                       states: list[bs.DisplayState], frame: int) -> dict:
    """演示把颗粒再次施加到已含颗粒的画面上（错误调用位置）的后果。

    正确调用：合成一次作用在干净底图上，与 aomdec 输出逐像素一致。
    错误调用：把同一份颗粒再叠到已含颗粒的画面上，颗粒幅度明显偏大——
    这正是"被叠加两遍"的观感来源。
    """
    st = states[frame]
    p = dict(st.params)
    p["random_seed"] = st.seed
    base_f = base[frame].astype(np.float64)
    once = synth_luma(base[frame], p).astype(np.float64)
    twice = synth_luma(once.astype(np.uint8), p).astype(np.float64)
    dec_f = decoder[frame].astype(np.float64)
    return {
        "frame": frame,
        "grain_std_correct": float((once - base_f).std()),
        "grain_std_doubled": float((twice - base_f).std()),
        "decoder_vs_single_maxdiff": int(np.abs(dec_f - once).max()),
        "double_vs_single_maxdiff": int(np.abs(twice - once).max()),
        "double_vs_single_mean_abs": float(np.abs(twice - once).mean()),
    }


def _write_stream_frames(stream_dir: Path, base: np.ndarray, grain: np.ndarray,
                         replay_frames: list[np.ndarray]) -> None:
    stream_dir.mkdir(parents=True, exist_ok=True)
    for f in range(len(base)):
        _save_png(base[f], stream_dir / f"f{f:03d}_base.png")
        _save_png(grain[f], stream_dir / f"f{f:03d}_grain.png")
        if f < len(replay_frames):
            _save_png(replay_frames[f], stream_dir / f"f{f:03d}_replay.png")


def _stream_block(name: str, ivf: Path, cfg: ExperimentConfig,
                  work: Path, *, want_frames: bool = True) -> dict:
    """解码一条码流、导出逐帧状态与重演结论，按需写 PNG。"""
    base_yuv = work / "decoded" / f"{name}.base.yuv"
    grain_yuv = work / "decoded" / f"{name}.grain.yuv"
    base_yuv.parent.mkdir(parents=True, exist_ok=True)
    _decode(ivf, base_yuv, cfg, grain=False)
    _decode(ivf, grain_yuv, cfg, grain=True)
    base = _load_y(base_yuv, cfg)
    grain = _load_y(grain_yuv, cfg)
    states = bs.extract(ivf)
    state_path = work / "states" / f"{name}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(_states_json(states), indent=2))
    replay = _replay(base, grain, states)
    replay_frames = [r.pop("_frame") for r in replay]
    if want_frames:
        _write_stream_frames(work / "frames" / name, base, grain, replay_frames)
    return {
        "ivf": str(ivf),
        "base_yuv": str(base_yuv),
        "grain_yuv": str(grain_yuv),
        "states": str(state_path),
        "display_frames": len(states),
        "keyframes": [s.display_index for s in states
                      if s.source == "shown" and s.is_keyframe],
        "scene_cuts": bs.scene_cuts(states),
        "show_existing": [{"frame": s.display_index, "slot": s.slot}
                          for s in states if s.source == "show_existing"],
        "replay_exact_frames": sum(1 for r in replay if r["exact"]),
        "replay": replay,
        "base": base,
        "grain": grain,
    }


def run(out_root: Path, cfg: ExperimentConfig | None = None) -> dict:
    """执行平场+切镜实验，写 ``grainlab.json`` 并返回摘要（含数组以外字段）。"""
    cfg = cfg or ExperimentConfig()
    work = out_root / "grainlab"
    for sub in ("material", "streams", "decoded", "states", "frames"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    cut = cfg.frames // 2

    scene_yuv = work / "material" / "scene.yuv"
    twin_yuv = work / "material" / "twin.yuv"
    make_scene(cfg, scene_yuv)
    make_twin(cfg, twin_yuv)
    scene_y4m = work / "material" / "scene.y4m"
    twin_y4m = work / "material" / "twin.y4m"
    tools.write_raw_y4m(scene_yuv, scene_y4m, cfg.width, cfg.height, cfg.fps)
    tools.write_raw_y4m(twin_yuv, twin_y4m, cfg.width, cfg.height, cfg.fps)

    tbl_scene = work / "streams" / "scene_A.tbl"
    tbl_p = work / "streams" / "twin_P.tbl"
    tbl_q = work / "streams" / "twin_Q.tbl"
    tbl_scene.write_text(_table_text(_PARAM_A, _PARAM_A, cut, cfg.fps))
    tbl_p.write_text(_table_text(_PARAM_A, _PARAM_H, cut, cfg.fps))
    tbl_q.write_text(_table_text(_PARAM_A, _PARAM_V, cut, cfg.fps))

    grain_on_ivf = work / "streams" / "grain_on.ivf"
    grain_off_ivf = work / "streams" / "grain_off.ivf"
    twin_p_ivf = work / "streams" / "twin_P.ivf"
    twin_q_ivf = work / "streams" / "twin_Q.ivf"
    _encode(scene_y4m, grain_on_ivf, cfg, 40, tbl_scene)
    _encode(scene_y4m, grain_off_ivf, cfg, 40, None)
    _encode(twin_y4m, twin_p_ivf, cfg, 36, tbl_p)
    _encode(twin_y4m, twin_q_ivf, cfg, 36, tbl_q)

    grain_on = _stream_block("grain_on", grain_on_ivf, cfg, work)
    grain_off = _stream_block("grain_off", grain_off_ivf, cfg, work)
    twin_p = _stream_block("twin_P", twin_p_ivf, cfg, work)
    twin_q = _stream_block("twin_Q", twin_q_ivf, cfg, work)

    # 两种访问模型：逐帧缓存 vs 序号推算的种子归属。
    access_rows = access.compare_access(
        grain_on_ivf, grain_on["base"], grain_on["grain"])
    access_summary = access.summary(access_rows)

    # 真实按切镜关键帧切片起播，与顺播逐帧对照（颗粒关/开各一次）。
    states = bs.extract(grain_on_ivf)
    key_display = cut
    if not any(s.display_index == key_display and s.is_keyframe
               for s in states):
        key_display = next(s.display_index for s in states if s.is_keyframe)
    pkt = bs.packet_index_of_display(states, key_display)
    restart_ivf = work / "streams" / "grain_on_restart.ivf"
    bs.build_restart_stream(grain_on_ivf, restart_ivf, pkt)
    restart_base_yuv = work / "decoded" / "grain_on_restart.base.yuv"
    restart_grain_yuv = work / "decoded" / "grain_on_restart.grain.yuv"
    _decode(restart_ivf, restart_base_yuv, cfg, grain=False)
    _decode(restart_ivf, restart_grain_yuv, cfg, grain=True)
    restart_base = _load_y(restart_base_yuv, cfg)
    restart_grain = _load_y(restart_grain_yuv, cfg)
    seq_base = grain_on["base"][key_display:]
    seq_grain = grain_on["grain"][key_display:]
    m = min(len(restart_base), len(seq_base))
    restart_cmp = {
        "start_display": key_display,
        "frames": m,
        "base_identical": bool(np.array_equal(
            restart_base[:m], seq_base[:m])),
        "grain_identical": bool(np.array_equal(
            restart_grain[:m], seq_grain[:m])),
        "base_maxdiff": int(np.abs(
            restart_base[:m].astype(np.int16) -
            seq_base[:m].astype(np.int16)).max()),
        "grain_maxdiff": int(np.abs(
            restart_grain[:m].astype(np.int16) -
            seq_grain[:m].astype(np.int16)).max()),
    }

    # 颗粒开/关的底图差、空间方差、帧间相关。
    bo, bf = grain_on["base"], grain_off["base"]
    m2 = min(len(bo), len(bf))
    grain_switch = {
        "base_mean_abs_diff": [
            float(np.abs(bo[i].astype(np.int16) -
                         bf[i].astype(np.int16)).mean()) for i in range(m2)],
        "spatial_variance_on": _spatial_variance(grain_on["grain"]),
        "spatial_variance_off": _spatial_variance(grain_off["grain"]),
        "spatial_variance_base": _spatial_variance(grain_on["base"]),
        "interframe_corr_on": _interframe_corr(grain_on["grain"]),
        "interframe_corr_off": _interframe_corr(grain_off["grain"]),
        "base_identical_on_vs_off": bool(np.array_equal(
            bo[:m2], bf[:m2])),
    }

    # 两段素材使用**同一张平场/渐变底图**、相同视觉颗粒强度，只在切镜处选用
    # 不同 AR 参数。用两条流的实际状态对同一底图重演，直接对比像素与统计。
    twin_states = states_of(twin_p_ivf)
    twin_q_states = states_of(twin_q_ivf)
    seg_b = list(range(cut + 2, cfg.frames - 2))
    p_pix, p_std, q_std = [], [], []
    twin_diff_frames = []
    for f in seg_b:
        if f >= len(twin_states) or twin_states[f].params is None:
            continue
        sp = dict(twin_states[f].params)
        sp["random_seed"] = twin_states[f].seed
        sq = dict(twin_q_states[f].params)
        sq["random_seed"] = twin_q_states[f].seed
        gp = synth_luma(twin_p["base"][f], sp)
        gq = synth_luma(twin_q["base"][f], sq)
        b = twin_p["base"][f].astype(np.float64)
        p_pix.append(int(np.abs(gp.astype(np.int16) -
                                gq.astype(np.int16)).max()))
        p_std.append(float((gp - b).std()))
        q_std.append(float((gq - b).std()))
        twin_diff_frames.append(f)
    twin = {
        "base_identical": bool(np.array_equal(
            twin_p["base"][:min(len(twin_p["base"]), len(twin_q["base"]))],
            twin_q["base"][:min(len(twin_p["base"]), len(twin_q["base"]))])),
        "grain_std_segA_P": _grain_std(twin_p["grain"], twin_p["base"], 2, cut),
        "grain_std_segA_Q": _grain_std(twin_q["grain"], twin_q["base"], 2, cut),
        "grain_std_segB_P": float(np.mean(p_std)) if p_std else None,
        "grain_std_segB_Q": float(np.mean(q_std)) if q_std else None,
        "pixel_diff_at_cut_max": max(p_pix) if p_pix else None,
        "pixel_diff_at_cut_mean": float(np.mean(p_pix)) if p_pix else None,
        "segB_frames_compared": len(twin_diff_frames),
        "params_at_cut_P": _params_at(twin_states, cut),
        "params_at_cut_Q": _params_at(twin_q_states, cut),
    }

    double_demo = _double_apply_demo(
        grain_on["base"], grain_on["grain"], states,
        min(key_display + 3, len(states) - 1))

    # 颗粒图案与编码块大小无关：以同一干净底帧、同一状态合成，结果应逐像素
    # 相同。这里用 --sb-size 64/128 两条真实码流的状态做对照。
    block_size = _block_size_invariance(scene_y4m, tbl_scene, cfg,
                                        work, states)

    summary = {
        "config": {k: v for k, v in vars(cfg).items() if k != "extra"},
        "scene_cut_display": cut,
        "determinism_scope": (
            "给定干净底帧 + 码流内种子/参数 + 尺寸，亮度合成与 aomdec 3.6 "
            "逐像素一致，且与解码块大小、解码起点、显示顺序无关；"
            "种子必须取自码流（含隐藏帧推进与 show_existing 复用），"
            "不能用显示序号推算。"),
        "streams": {
            "grain_on": _strip(grain_on),
            "grain_off": _strip(grain_off),
            "twin_P": _strip(twin_p),
            "twin_Q": _strip(twin_q),
        },
        "access": {
            "summary": access_summary,
            "per_frame": [r.__dict__ for r in access_rows],
        },
        "restart_slice": restart_cmp,
        "block_size_invariance": block_size,
        "grain_switch": grain_switch,
        "twin": twin,
        "double_apply": double_demo,
        "frame_counts": {"frames": cfg.frames, "cut": cut},
    }
    (work / "grainlab.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def states_of(ivf: Path) -> list[bs.DisplayState]:
    return bs.extract(ivf)


def _block_size_invariance(y4m: Path, table: Path, cfg: ExperimentConfig,
                           work: Path, reference_states: list) -> dict:
    """编码块大小（--sb-size）改变码流，但同一状态在同一底帧上的颗粒图案不变。"""
    const = np.full((cfg.height, cfg.width), _FLAT_LUMA, np.uint8)
    samples = [f for f in (0, cfg.frames // 4, cfg.frames // 2, cfg.frames - 1)
               if f < len(reference_states)]
    identical = 0
    checked = 0
    for sb in (64, 128):
        ivf = work / "streams" / f"sb{sb}.ivf"
        cmd = [
            cfgmod.AOMENC, "--codec=av1", f"--width={cfg.width}",
            f"--height={cfg.height}", f"--fps={cfg.fps}/1",
            f"--limit={cfg.frames}", "--i420", "--bit-depth=8",
            "--cpu-used=6", "--end-usage=q", "--cq-level=40",
            "--lag-in-frames=16", "--auto-alt-ref=1", "--kf-max-dist=9999",
            f"--sb-size={sb}", f"--film-grain-table={table}",
            "-o", str(ivf), str(y4m),
        ]
        tools.run(cmd)
    states64 = bs.extract(work / "streams" / "sb64.ivf")
    states128 = bs.extract(work / "streams" / "sb128.ivf")
    seeds_equal = all(
        a.seed == b.seed for a, b in zip(states64, states128))
    for f in samples:
        if f >= len(states64) or f >= len(states128):
            continue
        p64 = dict(states64[f].params)
        p64["random_seed"] = states64[f].seed
        p128 = dict(states128[f].params)
        p128["random_seed"] = states128[f].seed
        checked += 1
        if np.array_equal(synth_luma(const, p64), synth_luma(const, p128)):
            identical += 1
    return {
        "sb_sizes": [64, 128],
        "seed_schedules_equal": seeds_equal,
        "grain_pattern_identical": identical == checked and checked > 0,
        "frames_checked": checked,
    }


def _params_at(states: list[bs.DisplayState], frame: int) -> dict | None:
    if frame >= len(states):
        return None
    st = states[frame]
    if st.params is None:
        return None
    return {"seed": st.seed,
            "scaling_points_y": st.params["scaling_points_y"],
            "overlap_flag": st.params["overlap_flag"],
            "ar_coeffs_y": st.params["ar_coeffs_y"]}


def _strip(block: dict) -> dict:
    return {k: v for k, v in block.items() if k not in ("base", "grain")}
