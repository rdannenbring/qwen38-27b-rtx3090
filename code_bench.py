#!/usr/bin/env python3
"""Speculative-decoding benchmark on real coding work.

Acceptance depends on what the model is writing, and code is not chat. A block
drafter looks brilliant on tasks that reproduce their input and mediocre on
tasks that invent new text, so a single average hides the answer. This splits
the prompts into four families and reports each one separately:

  reproduce  emit a file back with a small edit          (maximum copying)
  refactor   restructure existing code                   (heavy copying)
  generate   write new code from a description           (no copying)
  explain    prose about code                            (no copying, prose)

Decode rate is measured from the first streamed token, so prefill is excluded.
Tokens per step comes from the server's own spec-decode counters, sampled
around each request, so it is the engine's number and not an inference.

  python3 code_bench.py <tag> [--max-tokens 700] [--repeats 1]
"""
import json
import os
import re
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://localhost:18020")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MAXTOK = int(arg("--max-tokens", "700"))
REPEATS = int(arg("--repeats", "1"))

REPO = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(REPO, "ctx_probe.py")) as f:
    FILE_PY = f.read()
with open(os.path.join(REPO, "single-user/start_qwen.sh")) as f:
    FILE_SH = f.read()[:6000]

PROMPTS = [
    # ---- reproduce: the answer is mostly text already in the prompt ----
    dict(family="reproduce", id="rewrite_py_const",
         prompt="Here is a Python file:\n\n```python\n" + FILE_PY +
                "\n```\n\nChange the default of CHARS_PER_TOK from 3.6 to 3.82 and "
                "nothing else. Output the complete file, unchanged apart from that "
                "one value. No commentary."),
    dict(family="reproduce", id="rewrite_sh_var",
         prompt="Here is a shell script:\n\n```bash\n" + FILE_SH +
                "\n```\n\nChange the default PORT from 18020 to 18030 and output the "
                "whole script back, otherwise identical. No commentary."),
    dict(family="reproduce", id="add_docstring",
         prompt="Here is a Python file:\n\n```python\n" + FILE_PY +
                "\n```\n\nAdd a one-line docstring to the `probe` function saying "
                "\"Run one probe at the requested context length.\" Output the "
                "complete file with no other changes."),

    # ---- refactor: structure preserved, wording changes ----
    dict(family="refactor", id="extract_helper",
         prompt="Here is a Python file:\n\n```python\n" + FILE_PY +
                "\n```\n\nRefactor the streaming loop inside `probe` into a separate "
                "module-level function `_stream(req)` that returns "
                "(ttft, last, usage). Output the full updated file."),
    dict(family="refactor", id="add_error_handling",
         prompt="Here is a Python file:\n\n```python\n" + FILE_PY +
                "\n```\n\nAdd a --retries N option that retries a failed probe up to "
                "N times with a 2 second pause, defaulting to 0. Output the full "
                "updated file."),

    # ---- generate: new code, nothing to copy ----
    dict(family="generate", id="lru_cache",
         prompt="Write a Python class `TTLCache` implementing an LRU cache with "
                "per-entry time-to-live: get, set, __len__, and eviction of expired "
                "entries on access. Use only the standard library. Include type "
                "hints and a short docstring on each method."),
    dict(family="generate", id="sql_migration",
         prompt="Write a Python script that connects to PostgreSQL with psycopg, "
                "finds every table lacking a primary key, and emits the ALTER TABLE "
                "statements to add a bigint identity primary key to each. Handle "
                "schema qualification and print a dry-run plan unless --apply."),
    dict(family="generate", id="rust_ratelimit",
         prompt="Write a Rust module implementing a token-bucket rate limiter that "
                "is safe to share across threads. Provide `new(capacity, refill_per_sec)` "
                "and `try_acquire(n) -> bool`, use std::sync primitives only, and "
                "include unit tests."),

    # ---- explain: prose about code ----
    dict(family="explain", id="explain_script",
         prompt="Here is a shell script:\n\n```bash\n" + FILE_SH +
                "\n```\n\nExplain what the CTX and SPEC variables control and how they "
                "interact, in a few paragraphs of prose. Do not quote the script."),
    dict(family="explain", id="review_py",
         prompt="Here is a Python file:\n\n```python\n" + FILE_PY +
                "\n```\n\nReview it as a senior engineer would: what would you change "
                "and why? Prose only, no code blocks."),
]

def _auth_headers():
    """The server requires a key once VLLM_API_KEY is set in .env."""
    h = {"Content-Type": "application/json"}
    key = os.environ.get("VLLM_API_KEY")
    if not key:
        p = os.path.join(REPO, "api_key.txt")
        if os.path.exists(p):
            key = open(p).read().strip()
    if key:
        h["Authorization"] = "Bearer " + key
    return h


SPEC_KEYS = ("vllm:spec_decode_num_drafts_total",
             "vllm:spec_decode_num_draft_tokens_total",
             "vllm:spec_decode_num_accepted_tokens_total")


def spec_metrics():
    try:
        req = urllib.request.Request(BASE + "/metrics", headers=_auth_headers())
        raw = urllib.request.urlopen(req, timeout=30).read().decode()
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for line in raw.splitlines():
        for k in SPEC_KEYS:
            if line.startswith(k + "{"):
                m = re.search(r"\}\s+([0-9.eE+-]+)$", line)
                if m:
                    out[k] = float(m.group(1))
    return out if len(out) == len(SPEC_KEYS) else None


def run(prompt, max_tokens):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=_auth_headers())
    before = spec_metrics()
    t0 = time.perf_counter()
    ttft = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
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
    after = spec_metrics()
    comp = (usage or {}).get("completion_tokens") or 0
    decode_s = max((last or t0) - (t0 + (ttft or 0)), 1e-6)

    tps = accept = None
    if before and after:
        drafts = after[SPEC_KEYS[0]] - before[SPEC_KEYS[0]]
        dtok = after[SPEC_KEYS[1]] - before[SPEC_KEYS[1]]
        acc = after[SPEC_KEYS[2]] - before[SPEC_KEYS[2]]
        if drafts > 0:
            tps = round(1 + acc / drafts, 2)      # bonus token per step + accepted
        if dtok > 0:
            accept = round(acc / dtok, 3)         # fraction of drafted tokens kept
    return dict(prompt_tokens=(usage or {}).get("prompt_tokens"),
                completion_tokens=comp,
                ttft_s=round(ttft or 0, 2),
                decode_tok_s=round((comp - 1) / decode_s, 1) if comp > 1 else None,
                tok_per_step=tps, accept_rate=accept)


def main():
    rows = []
    print(f"=== code_bench [{TAG}] max_tokens={MAXTOK} repeats={REPEATS} ===",
          flush=True)
    for rep in range(REPEATS):
        for p in PROMPTS:
            r = run(p["prompt"], MAXTOK)
            r.update(family=p["family"], id=p["id"], rep=rep)
            rows.append(r)
            print(f"  {p['family']:<9} {p['id']:<18} out={r['completion_tokens']:>4} "
                  f"decode={r['decode_tok_s']:>6.1f} tok/s  "
                  f"tok/step={r['tok_per_step']}  accept={r['accept_rate']}",
                  flush=True)

    print(f"\n--- {TAG}: by family ---")
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    for fam, rs in fams.items():
        n = len(rs)
        d = sum(x["decode_tok_s"] for x in rs) / n
        ts = [x["tok_per_step"] for x in rs if x["tok_per_step"]]
        ac = [x["accept_rate"] for x in rs if x["accept_rate"]]
        print(f"  {fam:<10} decode {d:>6.1f} tok/s   "
              f"tok/step {sum(ts)/len(ts) if ts else float('nan'):>5.2f}   "
              f"accept {sum(ac)/len(ac) if ac else float('nan'):>5.3f}   (n={n})")
    alld = sum(r["decode_tok_s"] for r in rows) / len(rows)
    print(f"  {'ALL':<10} decode {alld:>6.1f} tok/s")

    out = f"code_bench_{TAG}.json"
    with open(out, "w") as f:
        json.dump({"tag": TAG, "rows": rows}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
