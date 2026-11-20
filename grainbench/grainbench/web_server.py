"""零依赖 Web 界面：标准库 http.server 提供报告 JSON 与帧图。

界面能力（全部在相同码量下对照）：

* 三态切换：无颗粒底图 / 解码器合成输出 / 基准原片；
* 沿运动轨迹取样：物体坐标与屏幕坐标的帧间相关曲线，区分纹理与颗粒的附着系；
* 频段能量与纹理对比度核对图；
* 两路选择变化位置的分段时间线。
"""
from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "web" / "static"


def make_handler(out_root: Path) -> type[BaseHTTPRequestHandler]:
    report_path = out_root / "report.json"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, ctype: str | None = None) -> None:
            if not path.is_file():
                self._send(404, b"not found", "text/plain")
                return
            ctype = ctype or mimetypes.guess_type(str(path))[0] \
                or "application/octet-stream"
            self._send(200, path.read_bytes(), ctype)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._file(STATIC / "index.html", "text/html; charset=utf-8")
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if path == "/app.js":
                return self._file(STATIC / "app.js", "application/javascript")
            if path == "/style.css":
                return self._file(STATIC / "style.css", "text/css")
            if path == "/api/report":
                return self._file(report_path, "application/json")
            if path.startswith("/frames/"):
                # /frames/<group>/<file>.png → <out>/<group>/frames/<file>.png
                parts = path.removeprefix("/frames/").split("/", 1)
                if len(parts) != 2:
                    return self._send(404, b"not found", "text/plain")
                group, fname = parts
                target = (out_root / group / "frames" / fname).resolve()
                if not str(target).startswith(str(out_root.resolve())):
                    return self._send(403, b"forbidden", "text/plain")
                return self._file(target, "image/png")
            self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python -m grainbench.web <实验输出目录> [端口]")
        sys.exit(1)
    out_root = Path(sys.argv[1]).resolve()
    if not (out_root / "report.json").is_file():
        print(f"未找到 {out_root}/report.json，请先运行 run。")
        sys.exit(1)
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(out_root))
    print(f"实验台界面: http://localhost:{port}  (数据目录 {out_root})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
