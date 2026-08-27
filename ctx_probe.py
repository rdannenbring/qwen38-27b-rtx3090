#!/usr/bin/env python3
"""Long-context probe: what a given serve config actually costs at length.

For each requested context length it builds a prompt of roughly that many
tokens from a fixed corpus, streams a generation, and reports the real prompt
size (from the server's usage), TTFT and decode rate. Decode is measured from
the first streamed token onward, so prefill is not averaged in.

  python3 ctx_probe.py <tag> --ctx 4000,32000,100000 [--max-tokens 256]
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BASE", "http://localhost:18020/v1")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")


def _auth_headers():
    """The server requires a key once VLLM_API_KEY is set in .env."""
    h = {"Content-Type": "application/json"}
    key = os.environ.get("VLLM_API_KEY")
    if not key:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
        if os.path.exists(p):
            key = open(p).read().strip()
    if key:
        h["Authorization"] = "Bearer " + key
    return h
CORPUS = os.path.expanduser(os.environ.get("CORPUS", "~/bench/longctx_corpus.txt"))
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


CTXS = [int(x) for x in arg("--ctx", "4000,32000,100000").split(",")]
MAXTOK = int(arg("--max-tokens", "256"))
CHARS_PER_TOK = 3.6  # measured on this corpus; the real count is read back from usage

TEXT = open(CORPUS, encoding="utf-8", errors="ignore").read()

QUESTION = ("Above is a section of a Python codebase. In three sentences, describe "
            "what kinds of modules appear in it and what they have in common. "
            "Do not quote the text.")


def probe(ctx_tokens):
    n_chars = int(ctx_tokens * CHARS_PER_TOK)
    if n_chars > len(TEXT):
        return {"ctx_requested": ctx_tokens, "error": f"corpus too short "
                f"({len(TEXT)} chars, need {n_chars})"}
    # Each length reads from its own offset. Slicing every prompt from char 0
    # would make the short probes a literal prefix of the long ones, and with
    # PREFIX_CACHE=1 the long TTFT would then be measured against a warm cache.
    span = len(TEXT) - n_chars
    off = (ctx_tokens * 7919) % span if span > 0 else 0
    body = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": TEXT[off:off + n_chars] + "\n\n" + QUESTION}],
        "max_tokens": MAXTOK,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers=_auth_headers(),
    )
    t0 = time.perf_counter()
    ttft = None
    last = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                p = line[6:]
                if p == "[DONE]":
                    break
                chunk = json.loads(p)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        last = time.perf_counter()
    except urllib.error.HTTPError as e:
        return {"ctx_requested": ctx_tokens,
                "error": f"HTTP {e.code}: {e.read()[:400].decode('utf-8', 'replace')}"}
    except Exception as e:  # noqa: BLE001
        return {"ctx_requested": ctx_tokens, "error": f"{type(e).__name__}: {e}"}

    if ttft is None or usage is None:
        return {"ctx_requested": ctx_tokens, "error": "no tokens streamed"}
    comp = usage.get("completion_tokens") or 0
    decode_s = max(last - (t0 + ttft), 1e-6)
    return {
        "ctx_requested": ctx_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": comp,
        "ttft_s": round(ttft, 2),
        "decode_tok_s": round((comp - 1) / decode_s, 1) if comp > 1 else None,
        "total_s": round(time.perf_counter() - t0, 2),
    }


def main():
    rows = []
    print(f"=== ctx_probe [{TAG}] max_tokens={MAXTOK} ===", flush=True)
    for c in CTXS:
        r = probe(c)
        rows.append(r)
        if "error" in r:
            print(f"  ctx~{c:>7}  ERROR  {r['error'][:160]}", flush=True)
        else:
            print(f"  ctx~{c:>7}  prompt={r['prompt_tokens']:>7}  "
                  f"TTFT={r['ttft_s']:>7.2f}s  decode={r['decode_tok_s']:>6.1f} tok/s",
                  flush=True)
    out = f"ctx_probe_{TAG}.json"
    with open(out, "w") as f:
        json.dump({"tag": TAG, "rows": rows}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
