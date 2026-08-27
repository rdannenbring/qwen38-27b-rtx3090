#!/usr/bin/env python3
"""Transparent proxy in front of the vLLM server that records what each request
actually carries.

The question it exists to answer: when an agent harness spawns a subagent, does
the subagent's request still carry the parent's MCP tool schemas? Cumulative
server metrics cannot tell you -- they only give a total. This logs the `tools`
array per request, so parent and subagent turns are distinguishable.

  python3 tool_tap.py            # listen on 18021, forward to 18020
  python3 tool_tap.py --report   # summarise the captured log and exit
"""
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = int(os.environ.get("LISTEN", "18021"))
UPSTREAM = os.environ.get("UPSTREAM", "http://localhost:18020")
LOG = os.environ.get("TAP_LOG", "tool_tap_log.jsonl")

_lock = threading.Lock()
_seq = [0]


def record(entry):
    with _lock:
        _seq[0] += 1
        entry["seq"] = _seq[0]
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        t = entry.get("n_tools")
        print(f"  #{entry['seq']:>3} {entry['path']:<24} tools={t} "
              f"msgs={entry.get('n_messages')} "
              f"mcp={entry.get('n_mcp_tools')} "
              f"model={entry.get('model')}", flush=True)


def classify(names):
    """MCP tools carry a server prefix; built-ins are bare verbs."""
    mcp, builtin = [], []
    for n in names:
        if not n:
            continue
        if n.startswith("mcp__") or "__" in n:
            mcp.append(n)
        else:
            builtin.append(n)
    return mcp, builtin


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence default access log
        pass

    def _proxy(self, body):
        url = UPSTREAM + self.path
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "connection")}
        req = urllib.request.Request(url, data=body, headers=hdrs,
                                     method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=3600)
        except Exception as e:  # noqa: BLE001
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy error: {e}".encode())
            return
        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        try:
            d = json.loads(body)
            names = [((t.get("function") or {}).get("name") or t.get("name"))
                     for t in (d.get("tools") or [])]
            mcp, builtin = classify(names)
            msgs = d.get("messages") or []
            sysmsg = next((m for m in msgs if m.get("role") == "system"), None)
            record({"ts": time.time(), "path": self.path,
                    "model": d.get("model"),
                    "n_tools": len(names), "n_mcp_tools": len(mcp),
                    "n_builtin_tools": len(builtin),
                    "mcp_tools": mcp, "builtin_tools": builtin,
                    "n_messages": len(msgs),
                    "system_chars": len(sysmsg.get("content") or "") if sysmsg else 0,
                    "first_user": next((str(m.get("content"))[:120] for m in msgs
                                        if m.get("role") == "user"), None)})
            if os.environ.get("TAP_DUMP"):
                with open(f"tap_body_{_seq[0]}.json", "w") as f:
                    json.dump(d, f, indent=2)
        except Exception as e:  # noqa: BLE001
            record({"ts": time.time(), "path": self.path,
                    "parse_error": f"{type(e).__name__}: {e}"})
        self._proxy(body)

    def do_GET(self):
        self._proxy(None)


def report():
    rows = [json.loads(l) for l in open(LOG) if l.strip()]
    print(f"{'#':>4} {'tools':>6} {'mcp':>5} {'builtin':>8} {'msgs':>5}  first user turn")
    print("-" * 100)
    for r in rows:
        if "parse_error" in r:
            continue
        print(f"{r['seq']:>4} {r['n_tools']:>6} {r['n_mcp_tools']:>5} "
              f"{r['n_builtin_tools']:>8} {r['n_messages']:>5}  "
              f"{(r.get('first_user') or '')[:70]}")
    if rows:
        mcpcounts = {r["n_mcp_tools"] for r in rows if "n_mcp_tools" in r}
        print(f"\ndistinct MCP tool counts across requests: {sorted(mcpcounts)}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        sys.exit(0)
    print(f"tool_tap listening on {LISTEN}, forwarding to {UPSTREAM}")
    print(f"logging to {os.path.abspath(LOG)}")
    ThreadingHTTPServer(("127.0.0.1", LISTEN), Handler).serve_forever()
