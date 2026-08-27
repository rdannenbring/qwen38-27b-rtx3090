# Handoff: Qwen3.8-27B on RTX 3090 via syv-ai/qwen38-27b-rtx3090

Status as of 2026-08-26. Items 1-5 are done and measured; item 6 is done except
the two steps that need root.

## READ THIS BEFORE UPGRADING vLLM

The 15 patches in `patches/` are pinned to **vLLM 0.27.1** and are reapplied by
hand. A `pip install -U vllm` or a rebuilt image without reapplying them gives a
server that starts and is quietly wrong or slower. `bash verify.sh` reports
which patches are live — it currently says `OK (0 failures)`. The KVarN port
(`kvarn/install.sh`) is a second, separate set of hunks on top of those, and its
installer cannot distinguish "already applied" from "does not apply" — both exit
non-zero — so it counts `port(kvarn-v2)` markers per file instead. Re-run both
and re-check after any vLLM change.

## Hardware / OS

- Arch Linux, kernel `7.1.8-zen1-3-zen`, Hyprland
- MSI PRO Z890-A WIFI (MS-7E32), Core Ultra 7 265K, 64 GB RAM
- RTX 3090 (ASUS TUF Gaming OC, 2.9-slot, 2x 8-pin), 24576 MiB
- Displays run on the Intel iGPU, so the 3090 is dedicated to compute
- PSU: Lian Li SX1000P, 1000W Platinum
- Driver 610.57.04, `nvidia-open-dkms`, built for both zen and lts kernels
- `autoinstall_all_kernels="yes"` set in `/etc/dkms/framework.conf`
- Power limit 350W (all numbers below are at 350W; repo reference is 250W)

## Current serve config

`.env` next to `docker-compose.yml` drives everything (never committed):

```
PREFIX_CACHE=1
CTX=long
VLLM_API_KEY=<48 hex chars, also in api_key.txt, chmod 600>
```

- fp8 KV via FlashInfer, `max_model_len` 150,000, KV pool 176,020 tokens
- MTP-3 speculative decoding, prefix caching on, auth enforced (401 without key)
- Endpoints: `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic)
- Still binds 0.0.0.0 — now key-protected, but consider 127.0.0.1 if nothing
  off-box needs it

## Results

### 1. Prefix caching — done

On, and it is what makes long context usable. A 250k-token prompt cold: 235.2 s.
The same prompt again: **16.7 s**. Prefill is the real long-context cost
(~900-1,100 tok/s in every config), not decode.

### 2. Tool-call reliability — done, it holds up

`python3 tool_reliability.py` — 20 cases, 7-tool schema, 11 failure buckets,
separating *malformed* (harness cannot execute) from *wrong choice* (executable,
not what was asked).

**1 malformed call in 440 requests (0.23%)** — a single `end_end_line` argument
key. Wrong-choice was 1.8%, mostly the model reading a file before overwriting
it, which a real agent loop handles fine. Stressed integers, booleans, enums,
string arrays, nested objects, optional-arg omission, two must-not-call cases,
and two multi-turn cases where a tool result is fed back. INT8 activations did
not degrade any of it. Re-verified on the dflash2 V2 runner: 0 malformed in 120.

No baseline: a fair one needs the same model at a different quantisation.

### 3. Context — 262,144 is reachable

| config | max ctx | KV pool | 4k | 32k | 100k |
|---|---|---|---|---|---|
| `CTX=fast` bf16 | 65,536 | 73,246 | **114.9** | 99.0 | won't fit |
| `CTX=long` fp8 (current) | 150,000 | 176,020 | 86-92 | 81.3 | **72.0** |
| `CTX=long MAX_LEN=172000` | 172,000 | 178,198 | — | — | needle OK at 155k |
| `CTX=huge` KVarN | **262,144** | 295,141 | 82.3 | 63.6 | 37.6 |

Verified by retrieval, not just by fitting: a planted fact was recovered from a
250,139-token prompt under KVarN and from 155,369 tokens under fp8.

`CTX=long` costs 25% of short-prompt decode for 2.3x the context. KVarN doubles
context again but halves decode at 100k — take it only when a request would not
otherwise fit.

**Do not chase vLLM's `--kv-cache-memory` suggestion.** `docs/gotchas.md` items 4
and 16: 0.93 utilisation is deliberate (MTP's DeltaNet workspace grows past the
startup profile and kills the engine mid-generation at 0.95+), and the
suggestion is inflated on a cold compile cache. Confirmed here — the same config
reported a 176,020-token pool cold and 202,040 warm. At 202k pool, `MAX_LEN`
could go to ~195k on fp8.

#### 3a. The one long-context failure found — position, not length

Needle-in-a-haystack passes everywhere. A harder test does not: take a document
the model has memorised, change one fact, and ask about the changed fact. If it
answers from training instead of from the document, retrieval lost.

Method: RFC 9110 (`~/bench/rfc9110.txt`, 123,666 tokens), `OWASP` replaced with
`VELTRIX-SEC` in memory per request, asked which organisation Section 17.16
cites. `OWASP` in the answer means it ignored the document.

| prompt tokens | where the altered fact sits | answer |
|---|---|---|
| 244 | at end | Veltrix ✓ |
| 10,865 | at end | Veltrix ✓ |
| 34,452 | at end | Veltrix ✓ |
| 70,358 | at end | Veltrix ✓ |
| **119,523** | **at end** (front-padded to length) | **Veltrix ✓** |
| 123,758 | 82% depth, ~21k tokens follow it | "Open Web Application Security Project" ✗ |

**It is not raw length.** 119,523 tokens with the fact near the end is fine; the
same model fails at 123,758 only because ~21k tokens of text sit *after* the
fact. Length and depth were confounded until the front-padded row separated them.

The failure needs two things at once: the fact buried with content following it,
**and** a strong competing prior. The earlier needle test used an invented fact
with no prior ("Yusuf Ardan, 37 days") and passed at 250,139 tokens, 75% depth.
Neither test alone finds this.

The failure mode is specific: in the failing case the model's *reasoning trace
contains VELTRIX-SEC* — it retrieves correctly, then overrides its own retrieval
with the memorised name in the final answer. Any pass/fail check must therefore
classify on the answer only; scoring answer-plus-reasoning together turns these
into false passes (it did, for two rounds, before being caught).

Practical consequence: reading a **fork** of something well known — a patched
vLLM, a modified stdlib, a company variant of a public spec — is exactly the
shape that breaks. Put the modified section near the end of the prompt, or ask
about it directly rather than burying it mid-document.

Also learned here: `@file` in omp does **not** test long context. The read tool
truncates a 10,785-line file and the model greps instead — a turn that looked
like a 123k-token test used 32,588 input tokens. Use `ctx_probe.py`, or post the
document in a single request, when the point is to exercise context.

### 4. Speculative decoding — depends on prompt size, and yours are large

`python3 code_bench.py <tag>` — 10 coding prompts in four families. Decode tok/s:

| family | mtp-3 | dflash2 k=7 | dflash2 k=15 |
|---|---|---|---|
| reproduce (edit a file, emit it back) | 118.2 | 217.3 | **268.9** |
| refactor | 103.5 | 162.9 | **170.3** |
| generate (new code) | 93.1 | **151.3** | 134.6 |
| explain (prose) | 94.2 | **149.1** | 132.2 |
| **all** | **102.9** | 173.0 | **181.5** |
| max context | **150,000** | 131,072 | 114,000 |

Those prompts are 100-2,151 tokens. At realistic agentic sizes it reverses:

| prompt tokens | mtp-3 decode / TTFT | dflash2 k=7 decode / TTFT |
|---|---|---|
| 32,724 | 81.3 / 28.7 s | 86.5 / **41.2 s** |
| 94,596 | **72.0** / **109.6 s** | 60.6 / 222.9 s |

Decode advantage gone by 32k; prefill always worse (1.4x at 32k, 2.0x at 100k).
**`SPEC=mtp CTX=long` stays the default.** Switch to `SPEC=dflash2
DFLASH_TOKENS=7` only if prompts stay under ~10k.

Gotcha found: `DFLASH_TOKENS=15` will not start at the default context — it wants
5.73 GiB of pool against a pinned 5.2 GiB and crash-loops. Fix is
`DFLASH_MAX_LEN=114000`; note that path reads `DFLASH_MAX_LEN`, **not** `MAX_LEN`,
which is silently ignored. The script's own comment says k>7 at `CTX=long` is
untested upstream.

Numbers are deterministic: the k=7 run reproduced 173.0 → 173.1 tok/s.

### 5. omp wired up — done

`~/.omp/agent/models.yml`: provider `qwen38-local` → `http://localhost:18020/v1`,
model `qwen3.8-27b`, `contextWindow: 150000`, cost 0.
`~/.omp/agent/config.yml`: `smol`, `task`, `commit`, `tiny` → local; `plan` and
`slow` stay on Claude; `advisor` deliberately untouched.
Backups: `*.bak-20260826-*` in `~/.omp/agent/`.

**The context-bloat theory was wrong.** omp does *not* inline MCP tool schemas —
it mounts them as routes (`"brave_web_search" → xd://mcp__brave_search_...`) and
fetches schemas on demand. All 125 tools across 8 servers cost **1,313 tokens**.
Disabling MCP servers saves almost nothing. (The 30,423-token figure from
`mcp_cost.py` is what those same servers cost *Claude Code*, which does inline
them — real, but a different harness.)

What the 34,109-token startup overhead actually is:

| component | tokens | % |
|---|---|---|
| built-in tool schemas (11 tools) | **14,169** | 41.5% |
| Skills & Rules | **8,820** | 25.9% |
| "Additional devices" docs | 3,748 | 11.0% |
| MCP Tool Routes (all 125 tools) | 1,313 | 3.8% |
| everything else | ~6,059 | 17.8% |

The lever that works:

```
omp --tools read,edit,write,glob,grep,bash,todo --no-skills
```

**13,578 tokens instead of 34,109 — 60% off**, taking startup from 22.7% of the
150k window to 9.1%.

Still open: whether subagents inherit the parent's tools. Since MCP is never
inlined there is nothing to propagate, so the original worry is moot, but I never
observed an actual subagent request — my scout test produced one conversation,
not a parent plus a child. `tool_tap.py` is the instrument if you want certainty.

### 6. Housekeeping — done except root

- **API key: done.** 48 hex chars in `.env` and `api_key.txt` (chmod 600).
  Verified: 401 without, 401 with a wrong key, 200 with the right one. omp
  updated. `ctx_probe.py`, `code_bench.py` and `tool_reliability.py` now read
  `api_key.txt` (or `$VLLM_API_KEY`) automatically.
- **Persistence mode: needs you.** Unit written and syntax-checked:

      sudo cp nvidia-gpu-tuning.service /etc/systemd/system/
      sudo systemctl daemon-reload
      sudo systemctl enable --now nvidia-gpu-tuning.service

  Verify after reboot:
  `nvidia-smi --query-gpu=persistence_mode,power.limit --format=csv`
- **Power limit: needs you, and I would leave it at 350W.** The unit pins 350W.
  The 250W case is lower transient draw on a marginal PSU; a 1000W Platinum
  feeding one 3090 is not marginal. Every number in this file is at 350W, so if
  you do drop it, re-run `python3 code_bench.py pl250` before trusting them.

## Tools added

| file | what it does |
|---|---|
| `tool_reliability.py` | tool-call failure rate, 11 buckets, malformed vs wrong-choice |
| `ctx_probe.py` | TTFT and decode at a given context length |
| `code_bench.py` | coding-prompt decode by family, with tokens/step from server metrics |
| `mcp_cost.py` | per-MCP-server tool-schema token cost |
| `tool_tap.py` | logging proxy on 18021; records the `tools` array of every request |
| `nvidia-gpu-tuning.service` | persistence mode + pinned power limit |

Corpora for the context work:

- `~/bench/longctx_corpus.txt` — 1.4 MB, ~370k tokens, built from local Python
  stdlib source. Varied, no strong priors: good for needle tests and for padding
  a prompt to a target length.
- `~/bench/rfc9110.txt` — 491 KB, 123,666 tokens, HTTP Semantics. Technical, full
  of precisely checkable facts, and *memorised by the model* — which is what
  makes it the right corpus for the altered-fact test in 3a. Unmodified on disk.

## The purchase decision

Both blockers cleared. Tool calling is 0.23% malformed; context went from 65k to
150k in use and 262k verified. On the original terms — "if tool calling holds up
and context can be pushed past 65K, I don't buy the card" — **the second 3090 is
not needed.**

## How I want you to work

- Lists over paragraphs, sentence-case bullets, no em dashes
- Lead with the concrete next action; number multi-step work
- Don't summarize at the end
- One clarifying question max when details are missing
- Measure before and after any change — I want numbers, not assurances
