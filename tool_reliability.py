#!/usr/bin/env python3
"""Tool-call reliability test for the local Qwen3.8-27B W4A16 server.

The question this answers: how often does the INT8-activation quantised model
emit a tool call that an agent harness would reject?

It fires a fixed set of cases through /v1/chat/completions with a realistic
coding-agent tool schema and classifies every response into exactly one bucket.
Anything that is not `ok` is a call the harness could not have executed.

Buckets
  ok                 usable call (or correctly declined to call)
  http_error         server returned non-200 / connection died
  truncated          hit max_tokens before finishing the call
  no_tool_call       a call was required, none was emitted
  parser_leak        raw <tool_call>/JSON blob left in content, tool_calls empty
  unexpected_call    called a tool when the turn needed a plain answer
  unknown_tool       invented a tool name not in the schema
  wrong_tool         picked the wrong tool from the schema
  bad_json           arguments string is not parseable JSON
  schema_violation   arguments parse but fail the tool's JSON Schema
                     (missing required key, wrong type, bad enum)
  wrong_value        schema-valid but the values contradict the request
                     (e.g. wrote to a different path than the one asked for)

Usage
  python3 tool_reliability.py                  # greedy + sampled sweeps
  MODE=greedy python3 tool_reliability.py      # temperature 0 only
  REPEATS=10 python3 tool_reliability.py       # more samples per case
  BASE=http://host:18020/v1 python3 tool_reliability.py
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from jsonschema import Draft7Validator

BASE = os.environ.get("BASE", "http://localhost:18020/v1")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
REPEATS = int(os.environ.get("REPEATS", "5"))
MODE = os.environ.get("MODE", "both")  # greedy | sampled | both
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
OUT = os.environ.get("OUT", "tool_reliability_results.json")

# --------------------------------------------------------------------------
# A tool schema shaped like a real coding harness: overlapping read tools that
# force a genuine choice, plus integer / boolean / enum / array / nested-object
# arguments, which are where quantised models usually break first.
# --------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a single file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository-relative file path."},
                    "start_line": {"type": "integer", "description": "First line to read, 1-indexed."},
                    "end_line": {"type": "integer", "description": "Last line to read, inclusive."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite a file with new contents. Creates the file if missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "create_parents": {
                        "type": "boolean",
                        "description": "Create missing parent directories.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the entries of a directory. Does not read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "max_depth": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the repository for a regular expression and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression."},
                    "path": {"type": "string", "description": "Directory to search under."},
                    "file_glob": {"type": "string", "description": "Glob filter, e.g. *.py"},
                    "case_sensitive": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the repository root and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_issue",
            "description": "Open a tracker issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "assignee": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "notify": {"type": "boolean"},
                        },
                        "required": ["username"],
                        "additionalProperties": False,
                    },
                },
                "required": ["title", "priority"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Perform an HTTP request against a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    },
                    "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                    "json_body": {"type": "object"},
                },
                "required": ["url", "method"],
                "additionalProperties": False,
            },
        },
    },
]

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


TOOLS_BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}
VALIDATORS = {n: Draft7Validator(f["parameters"]) for n, f in TOOLS_BY_NAME.items()}

SYSTEM = (
    "You are a coding agent operating on a repository. When the user asks for "
    "something that a tool can do, call exactly one tool. Do not guess file "
    "contents. If the request needs no tool, answer directly in plain text."
)

# --------------------------------------------------------------------------
# Cases. `expect` is the required tool name, or None meaning "no call at all".
# `args_exact` are values that must match, `args_present` keys that must exist,
# `args_absent` keys that must not appear.
# `followup` turns the case into a two-step exchange: the first call is
# validated, a canned tool result is fed back, and the SECOND call is the one
# scored (this is where multi-turn agent loops actually break).
# --------------------------------------------------------------------------
CASES = [
    dict(id="read_simple",
         prompt="Show me what's in src/server/handlers.py",
         expect="read_file", args_exact={"path": "src/server/handlers.py"}),
    dict(id="read_line_range",
         prompt="Read lines 120 through 168 of src/server/handlers.py.",
         expect="read_file",
         args_exact={"path": "src/server/handlers.py", "start_line": 120, "end_line": 168}),
    dict(id="list_vs_read",
         prompt="What files are in the tests/integration directory?",
         expect="list_directory", args_exact={"path": "tests/integration"}),
    dict(id="list_recursive_depth",
         prompt="List everything under src/, recursively, but stop at 3 levels deep.",
         expect="list_directory",
         args_exact={"path": "src", "recursive": True, "max_depth": 3}),
    dict(id="search_basic",
         prompt="Find every place we call connect_timeout in the codebase.",
         expect="search_code", args_present=["pattern"]),
    dict(id="search_glob_limit",
         prompt="Search only Python files under src/ for the regex ^async def "
                "handle_, case sensitive, and cap it at 25 results.",
         expect="search_code",
         args_exact={"case_sensitive": True, "max_results": 25},
         args_present=["pattern", "file_glob"]),
    dict(id="run_tests",
         prompt="Run the unit tests with pytest -q and give them 600 seconds.",
         expect="run_command", args_exact={"timeout_seconds": 600},
         args_present=["command"]),
    dict(id="run_git",
         prompt="What's the current git status of the repo?",
         expect="run_command", args_present=["command"]),
    dict(id="write_simple",
         prompt="Create a file called .editorconfig at the repo root containing "
                "exactly: root = true",
         expect="write_file", args_exact={"path": ".editorconfig"},
         args_present=["content"]),
    dict(id="write_nested_bool",
         prompt="Write a file at config/staging/db.yml with the single line "
                "'pool: 12'. The config/staging directory does not exist yet, so "
                "make sure parent directories get created.",
         expect="write_file",
         args_exact={"path": "config/staging/db.yml", "create_parents": True},
         args_present=["content"]),
    dict(id="issue_enum",
         prompt="Open a critical-priority issue titled 'Connection pool exhausted "
                "under load'.",
         expect="create_issue",
         args_exact={"title": "Connection pool exhausted under load", "priority": "critical"}),
    dict(id="issue_array_labels",
         prompt="File a medium priority issue called 'Flaky auth test' and tag it "
                "with the labels flaky, tests and auth.",
         expect="create_issue",
         args_exact={"priority": "medium"},
         args_present=["title", "labels"]),
    dict(id="issue_nested_object",
         prompt="Open a high priority issue titled 'Rotate expired signing key', "
                "assign it to the user dmitri, and make sure they get notified.",
         expect="create_issue",
         args_exact={"priority": "high"},
         args_present=["title", "assignee"]),
    dict(id="http_enum_method",
         prompt="Send a DELETE to https://api.internal/v2/sessions/8831",
         expect="http_request",
         args_exact={"url": "https://api.internal/v2/sessions/8831", "method": "DELETE"}),
    dict(id="http_headers_body",
         prompt="POST to https://api.internal/v2/jobs with the header "
                "X-Trace-Id set to abc123 and a JSON body of {\"kind\": \"reindex\"}.",
         expect="http_request",
         args_exact={"url": "https://api.internal/v2/jobs", "method": "POST"},
         args_present=["headers", "json_body"]),
    dict(id="no_tool_concept",
         prompt="In one sentence, what is the difference between a B-tree index "
                "and a hash index?",
         expect=None),
    dict(id="no_tool_opinion",
         prompt="Briefly, should I use tabs or spaces in a new Python project? "
                "Just tell me, don't touch the repo.",
         expect=None),
    dict(id="omit_optional",
         prompt="Read the file README.md. Don't restrict it to a line range.",
         expect="read_file", args_exact={"path": "README.md"},
         args_absent=["start_line", "end_line"]),
    dict(id="followup_after_search",
         prompt="Find where MAX_RETRIES is defined, then read the file it's in.",
         expect="search_code", args_present=["pattern"],
         followup=dict(
             tool_result=json.dumps({
                 "matches": [{"path": "src/net/retry.py", "line": 14,
                              "text": "MAX_RETRIES = 5"}]}),
             next_prompt=None,
             expect="read_file", args_exact={"path": "src/net/retry.py"})),
    dict(id="followup_after_read",
         prompt="Read src/net/retry.py.",
         expect="read_file", args_exact={"path": "src/net/retry.py"},
         followup=dict(
             tool_result="MAX_RETRIES = 5\nBACKOFF = 0.25\n",
             next_prompt="MAX_RETRIES should be 8. Update the file, keeping the "
                         "BACKOFF line as it is.",
             expect="write_file", args_exact={"path": "src/net/retry.py"},
             args_present=["content"])),
]

LEAK_RE = re.compile(r"<tool_call|<function|\"name\"\s*:\s*\"(?:read_file|write_file|"
                     r"list_directory|search_code|run_command|create_issue|http_request)\"")


def call_api(messages, temperature, top_p, seed=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
    }
    if top_p is not None:
        body["top_p"] = top_p
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers=_auth_headers(),
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}", time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - any transport failure is a bucket
        return None, f"{type(e).__name__}: {e}", time.perf_counter() - t0
    return payload, None, time.perf_counter() - t0


def classify(payload, expect, args_exact, args_present, args_absent):
    """Return (bucket, detail, parsed_args)."""
    choice = payload["choices"][0]
    msg = choice.get("message") or {}
    calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""

    if not calls:
        if choice.get("finish_reason") == "length":
            return "truncated", "hit max_tokens with no tool call", None
        if expect is None:
            return "ok", "declined to call, as required", None
        if LEAK_RE.search(content):
            return "parser_leak", content[:200], None
        return "no_tool_call", content[:200], None

    if expect is None:
        return "unexpected_call", f"called {calls[0]['function']['name']}", None

    fn = calls[0]["function"]
    name = fn.get("name")
    if name not in TOOLS_BY_NAME:
        return "unknown_tool", f"invented tool {name!r}", None
    if name != expect:
        return "wrong_tool", f"called {name}, expected {expect}", None

    raw = fn.get("arguments")
    if isinstance(raw, str):
        try:
            args = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            return "bad_json", f"{e}: {raw[:200]}", None
    elif isinstance(raw, dict):
        args = raw
    else:
        return "bad_json", f"arguments of type {type(raw).__name__}", None

    errs = sorted(VALIDATORS[name].iter_errors(args), key=lambda e: list(e.path))
    if errs:
        e = errs[0]
        loc = "/".join(str(p) for p in e.path) or "(root)"
        return "schema_violation", f"{loc}: {e.message}"[:200], args

    for k, v in (args_exact or {}).items():
        if k not in args:
            return "wrong_value", f"missing {k!r} (wanted {v!r})", args
        if args[k] != v:
            return "wrong_value", f"{k}={args[k]!r}, wanted {v!r}", args
    for k in (args_present or []):
        if k not in args or args[k] in (None, "", [], {}):
            return "wrong_value", f"{k!r} missing or empty", args
    for k in (args_absent or []):
        if k in args:
            return "wrong_value", f"{k!r} should have been omitted, got {args[k]!r}", args

    return "ok", "", args


def run_case(case, temperature, top_p, rep):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": case["prompt"]}]
    payload, err, dt = call_api(messages, temperature, top_p)
    base = dict(case_id=case["id"], rep=rep, temperature=temperature,
                latency_s=round(dt, 3), step=1)
    if err:
        return {**base, "bucket": "http_error", "detail": err}

    bucket, detail, args = classify(
        payload, case.get("expect"), case.get("args_exact"),
        case.get("args_present"), case.get("args_absent"))

    fup = case.get("followup")
    if bucket != "ok" or not fup:
        return {**base, "bucket": bucket, "detail": detail}

    # second leg: feed the tool result back and score the follow-up call
    call = payload["choices"][0]["message"]["tool_calls"][0]
    messages.append({
        "role": "assistant",
        "content": payload["choices"][0]["message"].get("content") or "",
        "tool_calls": [{"id": call.get("id", "call_0"), "type": "function",
                        "function": {"name": call["function"]["name"],
                                     "arguments": call["function"]["arguments"]}}],
    })
    messages.append({"role": "tool", "tool_call_id": call.get("id", "call_0"),
                     "content": fup["tool_result"]})
    if fup.get("next_prompt"):
        messages.append({"role": "user", "content": fup["next_prompt"]})

    payload2, err2, dt2 = call_api(messages, temperature, top_p)
    base2 = dict(base, step=2, latency_s=round(dt + dt2, 3))
    if err2:
        return {**base2, "bucket": "http_error", "detail": err2}
    bucket2, detail2, _ = classify(
        payload2, fup.get("expect"), fup.get("args_exact"),
        fup.get("args_present"), fup.get("args_absent"))
    return {**base2, "bucket": bucket2, "detail": detail2}


def sweep(label, temperature, top_p, repeats):
    jobs = [(c, r) for c in CASES for r in range(repeats)]
    results = []
    print(f"\n=== sweep {label}: {len(jobs)} requests "
          f"(temperature={temperature}, top_p={top_p}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(run_case, c, temperature, top_p, r) for c, r in jobs]
        for i, f in enumerate(futs, 1):
            r = f.result()
            r["sweep"] = label
            results.append(r)
            flag = "  " if r["bucket"] == "ok" else "!!"
            print(f"{flag} [{i}/{len(jobs)}] {r['case_id']}#{r['rep']} "
                  f"step{r['step']} {r['bucket']} {r['detail'][:90]}", flush=True)
    return results


# A call the harness physically cannot execute (or never received) is a
# different class of problem from a call it executes fine that simply is not
# the one the task needed. Both are failures; only the first is a quantisation
# smoking gun, so they are counted separately.
HARNESS_BREAKING = {"http_error", "truncated", "parser_leak", "unknown_tool",
                    "bad_json", "schema_violation"}
BEHAVIOURAL = {"no_tool_call", "unexpected_call", "wrong_tool", "wrong_value"}


def report(results):
    total = len(results)
    buckets = {}
    for r in results:
        buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
    ok = buckets.get("ok", 0)
    hard = sum(n for b, n in buckets.items() if b in HARNESS_BREAKING)
    soft = sum(n for b, n in buckets.items() if b in BEHAVIOURAL)
    print(f"\n{'-' * 62}")
    print(f"requests            {total}")
    print(f"usable calls        {ok}  ({100.0 * ok / total:.1f}%)")
    print(f"FAILURE RATE        {total - ok}/{total}  "
          f"({100.0 * (total - ok) / total:.2f}%)")
    print(f"  malformed         {hard}/{total}  ({100.0 * hard / total:.2f}%)"
          f"   <- harness cannot execute these")
    print(f"  wrong choice      {soft}/{total}  ({100.0 * soft / total:.2f}%)"
          f"   <- executable, but not what was asked")
    print("\nbucket breakdown")
    for b, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        if b == "ok":
            continue
        kind = "malformed" if b in HARNESS_BREAKING else "choice"
        print(f"  {b:<18} {n:>4}  ({100.0 * n / total:.2f}%)  [{kind}]")
    bad_cases = {}
    for r in results:
        if r["bucket"] != "ok":
            bad_cases.setdefault(r["case_id"], []).append(r["bucket"])
    if bad_cases:
        print("\nfailing cases")
        for cid, bs in sorted(bad_cases.items(), key=lambda kv: -len(kv[1])):
            n = len([r for r in results if r["case_id"] == cid])
            print(f"  {cid:<22} {len(bs)}/{n}  {sorted(set(bs))}")
    else:
        print("\nno failing cases")


def main():
    sweeps = []
    if MODE in ("greedy", "both"):
        sweeps.append(("greedy", 0.0, None, 1))
    if MODE in ("sampled", "both"):
        sweeps.append(("sampled", 0.6, 0.95, REPEATS))

    all_results = []
    for label, temp, top_p, reps in sweeps:
        rs = sweep(label, temp, top_p, reps)
        all_results.extend(rs)
        print(f"\n--- {label} ---")
        report(rs)

    if len(sweeps) > 1:
        print(f"\n{'=' * 62}\nCOMBINED")
        report(all_results)

    with open(OUT, "w") as f:
        json.dump({"model": MODEL, "base": BASE, "results": all_results}, f, indent=2)
    print(f"\nwrote {OUT}")
    return 0 if all(r["bucket"] == "ok" for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
