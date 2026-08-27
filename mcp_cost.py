#!/usr/bin/env python3
"""Measure what each MCP server costs in prompt context.

Connects to every configured MCP server, pulls its tool list, serialises the
tools the way an OpenAI-compatible API receives them, and writes the result for
tokenisation. Server definitions are read from ~/.claude.json, which is also
what omp inherits.

Toggling servers and re-measuring a live agent tells you the total; this tells
you the per-server breakdown in one pass, without restarting anything.

  python3 mcp_cost.py            # all enabled servers
  python3 mcp_cost.py --only github,playwright
"""
import json
import os
import subprocess
import sys
import threading
import urllib.request

OUT = "/tmp/claude-1000/-home-rdannenbring-Development-qwen38-27b-rtx3090/e7c7425b-194c-460a-815d-7e4004309bb4/scratchpad/mcp_tools.json"
TIMEOUT = 90

PROTO = {"protocolVersion": "2024-11-05",
         "capabilities": {},
         "clientInfo": {"name": "mcp-cost", "version": "1.0"}}


def _reader(stream, sink):
    for line in stream:
        sink.append(line)


def probe_stdio(name, cfg):
    env = dict(os.environ)
    for k, v in (cfg.get("env") or {}).items():
        env[k] = os.path.expandvars(v)
    args = [os.path.expandvars(a) for a in cfg.get("args", [])]
    cmd = [cfg["command"]] + args
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)

    def send(obj):
        p.stdin.write(json.dumps(obj) + "\n")
        p.stdin.flush()

    lines = []
    t = threading.Thread(target=_reader, args=(p.stdout, lines), daemon=True)
    t.start()
    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": PROTO})
        # wait for the initialize reply before asking for tools
        got = _await_id(lines, 1)
        if got is None:
            return None, "no initialize response"
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        res = _await_id(lines, 2)
        if res is None:
            return None, "no tools/list response"
        return (res.get("result") or {}).get("tools", []), None
    finally:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            p.kill()


def _await_id(lines, want, timeout=TIMEOUT):
    import time
    deadline = time.time() + timeout
    seen = 0
    while time.time() < deadline:
        while seen < len(lines):
            raw = lines[seen].strip()
            seen += 1
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == want:
                return msg
        time.sleep(0.05)
    return None


def probe_http(name, cfg):
    url = cfg["url"].rstrip("/")
    hdrs = {"Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"}
    for k, v in (cfg.get("headers") or {}).items():
        hdrs[k] = os.path.expandvars(v)

    def rpc(obj, session=None):
        h = dict(hdrs)
        if session:
            h["Mcp-Session-Id"] = session
        req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers=h)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            sid = r.headers.get("Mcp-Session-Id")
            body = r.read().decode("utf-8", "replace")
        # streamable HTTP may answer as SSE
        if body.lstrip().startswith("event:") or "\ndata: " in body:
            for line in body.splitlines():
                if line.startswith("data: "):
                    try:
                        return json.loads(line[6:]), sid
                    except json.JSONDecodeError:
                        continue
            return None, sid
        try:
            return json.loads(body), sid
        except json.JSONDecodeError:
            return None, sid

    init, sid = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": PROTO})
    if init is None:
        return None, "no initialize response"
    try:
        rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    except Exception:  # noqa: BLE001
        pass
    res, _ = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                  "params": {}}, sid)
    if res is None:
        return None, "no tools/list response"
    return (res.get("result") or {}).get("tools", []), None


def serialise(tools):
    """The shape an OpenAI-compatible server actually receives."""
    return [{"type": "function",
             "function": {"name": t.get("name"),
                          "description": t.get("description") or "",
                          "parameters": t.get("inputSchema") or {}}}
            for t in tools]


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))

    cfgpath = os.path.expanduser("~/.claude.json")
    servers = json.load(open(cfgpath)).get("mcpServers", {})

    out = {}
    for name, cfg in sorted(servers.items()):
        if only and name not in only:
            continue
        kind = cfg.get("type") or ("http" if cfg.get("url") else "stdio")
        print(f"probing {name} ({kind}) ...", flush=True)
        try:
            tools, err = (probe_http(name, cfg) if kind in ("http", "sse")
                          else probe_stdio(name, cfg))
        except Exception as e:  # noqa: BLE001
            tools, err = None, f"{type(e).__name__}: {e}"
        if tools is None:
            print(f"  FAILED: {err}", flush=True)
            out[name] = {"error": err}
            continue
        payload = json.dumps(serialise(tools), separators=(",", ":"))
        out[name] = {"n_tools": len(tools), "chars": len(payload),
                     "payload": payload,
                     "tool_names": [t.get("name") for t in tools]}
        print(f"  {len(tools)} tools, {len(payload)} chars", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
