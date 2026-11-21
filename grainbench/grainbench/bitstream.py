"""从 AV1 码流头解析"每个显示帧实际使用的颗粒状态"。

颗粒合成在 libaom 里是逐显示帧无状态的纯函数：输出只由该帧 OBU 携带的
``grain_seed``、解析后的颗粒参数与帧尺寸决定。但**状态归属**容易出错：

* 编码器每编一个帧（含不显示、只作参考的隐藏帧）把种子 ``+= 3381``（回绕
  0→7391）；``show_existing_frame`` 重放某参考槽时不消耗新种子，而是连同
  颗粒参数一起复用被重放帧的状态。
* 因此种子 ≠ ``7391 + 3381 * 显示序号``。按显示序号推断、或在关键帧处把
  种子重置为 7391，都会在隐藏帧与切镜处错位——这是"拖动/切块后纹理换样"
  的根源。

本模块用 ffmpeg 的 ``trace_headers`` 逐 OBU 解析，维护 8 个参考槽，把每个
**显示帧**解析成 ``DisplayState``（种子 + 颗粒参数 + 来源），作为可验证的
逐帧状态缓存的权威来源。
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import config

_OBU_HEADER = "OBU header"
_FIELD_RE = re.compile(
    r"^\[trace_headers[^\]]*\]\s*\d+\s+([\w\[\]]+)\s+\S+\s*=\s*(-?\d+)\s*$")

# frame_type: 0=KEY_FRAME 1=INTER_FRAME 2=INTRA_ONLY_FRAME 3=SWITCH_FRAME
KEY_FRAME = 0


@dataclass
class DisplayState:
    """一个显示帧的颗粒状态（对齐 aomdec 的实际合成输入）。"""
    display_index: int
    seed: int | None              # apply_grain=0 时为 None
    params: dict | None           # 颗粒参数；None 表示该帧不合成颗粒
    source: str                   # "shown" | "show_existing"
    obu_index: int                # 在码流中的 OBU 序号（含隐藏帧）
    frame_type: int | None
    is_keyframe: bool
    refresh_flags: int
    slot: int | None = None       # show_existing 时被重放的参考槽

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_trace(stderr: str) -> list[dict]:
    chunks: list[dict] = []
    cur: dict | None = None
    for line in stderr.splitlines():
        if _OBU_HEADER in line:
            if cur is not None:
                chunks.append(cur)
            cur = {}
            continue
        m = _FIELD_RE.match(line)
        if m and cur is not None:
            cur[m.group(1)] = int(m.group(2))
    if cur is not None:
        chunks.append(cur)
    return chunks


def trace_obus(ivf: Path, ffmpeg: str | None = None) -> list[dict]:
    """用 trace_headers 位流过滤器把每个 OBU 的头部字段解析成字典列表。"""
    proc = subprocess.run(
        [ffmpeg or config.FFMPEG, "-hide_banner", "-i", str(ivf),
         "-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"],
        capture_output=True, text=True)
    return _parse_trace(proc.stderr)


def params_from_header(ch: dict) -> dict:
    """把某个帧头的 trace 字段映射为 ``grainsynth.synth_luma`` 的参数。"""
    n = ch.get("num_y_points", 0)
    pts = [(ch.get(f"point_y_value[{i}]", 0), ch.get(f"point_y_scaling[{i}]", 0))
           for i in range(n)]
    lag = ch.get("ar_coeff_lag", 0)
    n_coeff = 2 * lag * (lag + 1)
    ar = [ch.get(f"ar_coeffs_y_plus_128[{i}]", 128) - 128
          for i in range(n_coeff)]
    return {
        "scaling_points_y": pts,
        "ar_coeff_lag": lag,
        "ar_coeffs_y": (ar + [0] * 12)[:12],
        "ar_coeff_shift": ch.get("ar_coeff_shift_minus_6", 0) + 6,
        "grain_scale_shift": ch.get("grain_scale_shift", 0),
        "scaling_shift": ch.get("grain_scaling_minus_8", 0) + 8,
        "overlap_flag": ch.get("overlap_flag", 0),
    }


def resolve_display_states(chunks: list[dict]) -> list[DisplayState]:
    """按解码顺序还原每个显示帧的种子与参数，处理隐藏帧与 show_existing。

    参考槽保存的是 (seed, params)；``refresh_frame_flags`` 按位刷新。
    ``show_existing_frame`` 直接重放槽内状态（不产生新种子）。
    ``update_grain=0`` 时参数继承 ``film_grain_params_ref_idx`` 指向的槽，
    但种子仍用本帧的新值。
    """
    slots: list[tuple | None] = [None] * 8
    out: list[DisplayState] = []
    obu_index = 0
    for ch in chunks:
        if ch.get("obu_type") not in (3, 6):
            continue
        if "show_existing_frame" not in ch:
            continue
        obu_index += 1
        if ch["show_existing_frame"] == 1:
            idx = ch.get("frame_to_show_map_idx")
            if idx is None or slots[idx] is None:
                continue
            seed, params = slots[idx]
            out.append(DisplayState(
                display_index=len(out), seed=seed, params=params,
                source="show_existing", obu_index=obu_index,
                frame_type=None, is_keyframe=False,
                refresh_flags=0, slot=idx))
            continue

        frame_type = ch.get("frame_type")
        apply = ch.get("apply_grain", 0)
        seed = ch.get("grain_seed") if apply else None
        params = None
        if apply:
            if ch.get("update_grain", 1):
                params = params_from_header(ch)
            else:
                ridx = ch.get("film_grain_params_ref_idx")
                if ridx is not None and slots[ridx] is not None:
                    params = slots[ridx][1]
        rec = (seed, params)
        flags = ch.get("refresh_frame_flags", 0)
        for s in range(8):
            if flags & (1 << s):
                slots[s] = rec
        if ch.get("show_frame", 0) == 1:
            out.append(DisplayState(
                display_index=len(out), seed=seed, params=params,
                source="shown", obu_index=obu_index,
                frame_type=frame_type,
                is_keyframe=(frame_type == KEY_FRAME),
                refresh_flags=flags))
    return out


def extract(ivf: Path) -> list[DisplayState]:
    return resolve_display_states(trace_obus(ivf))


def scene_cuts(states: list[DisplayState]) -> list[int]:
    """显示序列中的切镜点：第一个关键帧之后再次出现关键帧的显示序号。"""
    cuts = []
    for st in states:
        if st.source == "shown" and st.is_keyframe and st.display_index > 0:
            cuts.append(st.display_index)
    return cuts


def encoded_seed_schedule(states: list[DisplayState]) -> list[int]:
    """由实际状态还原编码器"每编一帧 +3381"的种子序列（用于对照模型）。"""
    return [st.seed for st in states]


# ---------------------------------------------------------------------------
# IVF / OBU 切片：构造"从某关键帧起播"的真实码流，用于验证两种访问模型。
# ---------------------------------------------------------------------------

def read_ivf_packets(path: Path) -> tuple[dict, list[bytes]]:
    """解析 IVF 容器，返回文件头字段与每个时间单元（packet）的载荷。"""
    import struct
    raw = Path(path).read_bytes()
    if raw[:4] != b"DKIF":
        raise ValueError(f"不是 IVF 文件: {path}")
    (_, version, header_len, fourcc, width, height,
     fps_den, fps_num, n_frames, _unused) = struct.unpack(
        "<4sHH4sHHIIII", raw[:32])
    packets: list[bytes] = []
    off = header_len
    while off + 12 <= len(raw) and len(packets) < n_frames:
        size = struct.unpack("<I", raw[off:off + 4])[0]
        payload = raw[off + 12:off + 12 + size]
        packets.append(payload)
        off += 12 + size
    header = {"version": version, "fourcc": fourcc, "width": width,
              "height": height, "fps_den": fps_den, "fps_num": fps_num}
    return header, packets


def _obu_type(first_byte: int) -> int:
    return (first_byte >> 3) & 0x0F


def split_obus(payload: bytes) -> list[bytes]:
    """把 packet 载荷拆成独立 OBU（低开销位流，含 leb128 size）。"""
    out = []
    p = 0
    n = len(payload)
    while p < n:
        start = p
        header = payload[p]
        p += 1
        if header & 0x04:            # obu_extension_flag
            p += 1
        if not (header & 0x02):      # 无 size，直到 packet 末尾
            out.append(payload[start:])
            break
        size = 0
        shift = 0
        while p < n:
            byte = payload[p]
            p += 1
            size |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        p += size
        out.append(payload[start:p])
    return out


def first_sequence_header(packets: list[bytes]) -> bytes | None:
    for pkt in packets:
        for obu in split_obus(pkt):
            if obu and _obu_type(obu[0]) == 1:   # OBU_SEQUENCE_HEADER
                return obu
    return None


def _ivf_header_for(header: dict, n_frames: int) -> bytes:
    import struct
    return struct.pack(
        "<4sHH4sHHIIII", b"DKIF", header["version"], 32,
        header["fourcc"], header["width"], header["height"],
        header["fps_den"], header["fps_num"], n_frames, 0)


def build_restart_stream(src: Path, out: Path, start_packet: int) -> Path:
    """构造从 ``start_packet`` 起播的码流：序列头 + 该包及之后的全部载荷。

    关键帧是帧内编码且刷新全部参考槽，其后编码顺序的帧只向前引用，
    因此从关键帧的分组边界切片得到的码流可独立解码。
    """
    header, packets = read_ivf_packets(src)
    if not 0 <= start_packet < len(packets):
        raise ValueError(f"start_packet 越界: {start_packet}/{len(packets)}")
    seq = first_sequence_header(packets)
    if seq is None:
        raise ValueError("源码流不含序列头 OBU")
    keep = packets[start_packet:]
    body = bytearray()
    for i, pkt in enumerate(keep):
        # 序列头只前置一次；后续 packet 原样保留（含各自的时间分隔符）。
        payload = (seq + pkt) if i == 0 else pkt
        body += struct_packet(payload, i)
    header_bytes = _ivf_header_for(header, len(keep))
    Path(out).write_bytes(header_bytes + bytes(body))
    return Path(out)


def struct_packet(payload: bytes, timestamp: int) -> bytes:
    import struct
    return struct.pack("<IQ", len(payload), timestamp) + payload


def packet_index_of_display(states: list[DisplayState], display_index: int) -> int:
    """显示序号 → IVF packet 序号。每个帧 OBU 占一个 packet，OBU 序号从 1 起。"""
    if not 0 <= display_index < len(states):
        raise IndexError(display_index)
    return states[display_index].obu_index - 1
