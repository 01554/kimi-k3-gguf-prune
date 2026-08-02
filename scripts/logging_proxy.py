#!/usr/bin/env python3
"""Transparent logging proxy for llama-server: record every chat request body.

  scripts/logging_proxy.py --listen 8091 --upstream 8090 --dump-dir out/reqdump_rerun

Used for failure attribution: rerun the failed SWE-Lancer tasks with the agent
pointed at the proxy, so every turn's full request (system prompt, tool schemas,
conversation so far) is captured. Those texts then go through a per-document
calibration pass on the unpruned source to measure which experts the *full*
router wanted, and how much of that demand fell outside the pruned keep set.

Streaming responses are passed through unbuffered; only request bodies are
persisted.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

COUNTER = {"n": 0}
LOCK = threading.Lock()


def make_handler(upstream: str, dump_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _proxy(self, body: bytes | None):
            url = f"{upstream}{self.path}"
            req = Request(url, data=body, method=self.command)
            for k in ("Content-Type", "Authorization", "Accept"):
                if self.headers.get(k):
                    req.add_header(k, self.headers[k])
            try:
                with urlopen(req, timeout=14400) as r:
                    self.send_response(r.status)
                    ctype = r.headers.get("Content-Type", "application/json")
                    self.send_header("Content-Type", ctype)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = r.read(16384)
                        if not chunk:
                            break
                        self.wfile.write(f"{len(chunk):X}\r\n".encode())
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
            except Exception as e:
                try:
                    self.send_error(502, str(e)[:100])
                except Exception:
                    pass

        def do_GET(self):
            self._proxy(None)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else None
            if body and "/chat/completions" in self.path:
                with LOCK:
                    COUNTER["n"] += 1
                    i = COUNTER["n"]
                try:
                    parsed = json.loads(body)
                    (dump_dir / f"req_{i:04d}.json").write_text(
                        json.dumps(parsed, ensure_ascii=False, indent=1))
                except Exception:
                    (dump_dir / f"req_{i:04d}.raw").write_bytes(body)
                print(f"[proxy] captured request #{i} ({n:,} bytes)", flush=True)
            self._proxy(body)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8091)
    ap.add_argument("--upstream", default="http://127.0.0.1:8090")
    ap.add_argument("--dump-dir", required=True)
    a = ap.parse_args()
    dump = Path(a.dump_dir)
    dump.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", a.listen), make_handler(a.upstream, dump))
    print(f"[proxy] {a.listen} -> {a.upstream}, dumping to {dump}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
