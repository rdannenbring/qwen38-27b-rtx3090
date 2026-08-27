# PLAN

Real map of this codebase: a fork of `syv-ai/qwen38-27b-rtx3090` (vLLM 0.27.1 +
15 patches, serving Qwen3.8-27B W4A16 on one RTX 3090), plus a local deployment
and validation layer added on top of it.

Every `[x]` below was checked against source or a produced artifact, cited on the
`tech:` line. See `## decisions` for what could not be verified.

---

## Launch and tune the single-GPU server {#serve}
tech: single-user/start_qwen.sh (550 lines), batch/start_qwen.sh, docker-compose.yml
needs: #patches, #prepare
links: #kvarn, #measure, #deploy
files: single-user/**, batch/**, docker/**, docker-compose.yml, Dockerfile

The hot surface: 61 commits on `single-user/` alone, last 2026-08-25. One shell
script resolves CTX x SPEC into a vLLM command line and rejects bad pairs.

- [x] `CTX=fast|long|huge` picks KV dtype, context length and draft count
      tech: single-user/start_qwen.sh:79,90-104
- [x] `SPEC=mtp|dflash2` picks the drafter; incompatible pairs fall back to mtp
      tech: single-user/start_qwen.sh:87,105-125
- [x] `PREFIX_CACHE=1` turns on prefix caching
      tech: single-user/start_qwen.sh:335
- [x] `.env` passed straight into the container as the single knob surface
      tech: docker-compose.yml:28-29
- [x] `single` / `batch` compose profiles off one image
      tech: docker-compose.yml:65-73, batch/start_qwen.sh
- [x] stale `/dev/shm/vllm_offload_*.mmap` regions unlinked before boot
      tech: single-user/start_qwen.sh:47-58
- [ ] `DFLASH_TOKENS=15` will not boot at default context; needs `DFLASH_MAX_LEN=114000`,
      and that branch ignores `MAX_LEN` entirely
      from: agent
      tech: single-user/start_qwen.sh:240 (`MAX_LEN=${DFLASH_MAX_LEN:-131072}`)

## Keep the 15 vLLM 0.27.1 patches applied and provable {#patches}
tech: verify.sh (171 lines), patches/_check_applied.py, patches/*.patch (3,778 lines)
needs: —
links: #serve, #kvarn
files: patches/**, verify.sh

Patches are reapplied by hand against a pinned vLLM. `verify.sh` is the only
thing standing between an upgrade and a server that is quietly wrong.

- [x] 15 patches present, content-checked rather than exit-code-checked
      tech: verify.sh:52, patches/_check_applied.py
- [x] full verify passes on the running container
      tech: `verify: OK (0 failures)` from `bash verify.sh` in-container, 2026-08-27
- [x] API-key presence is part of verify
      tech: verify.sh:142-145
- [ ] reapply and re-verify after any vLLM version change — nothing enforces this
      from: agent

## Port the 4/2-bit KV cache for 262k context {#kvarn}
tech: kvarn/files/vllm/v1/attention/backends/kvarn_attn.py, kvarn/install.sh
needs: #patches
links: #serve, #measure
files: kvarn/**

Huawei CSL's KVarN backported from its vLLM 0.23 fork onto 0.27.1. Two-stage
install; the second stage counts `port(kvarn-v2)` markers because `patch
--forward` cannot distinguish a rerun from a bad apply.

- [x] `kvarn_k4v2_g128` registered as a cache dtype, KVARN backend selected
      tech: kvarn/files/.../quantization/kvarn/config.py; engine log `Using KVARN attention backend`
- [x] V2-runner port present, so `SPEC=dflash2` + `CTX=huge` is available
      tech: kvarn/kvarn-v2-runner.patch; entrypoint `PASS kvarn-v2-runner.patch applied`
- [x] boots at the model's full 262,144 with a 295,141-token pool
      tech: ctx_probe_ctxhuge.json, engine log 2026-08-26
- [ ] raise fp8 `MAX_LEN` toward ~195k now the warm pool measures 202,040 tokens
      from: agent

## Quantize and assemble the served model artifacts {#prepare}
tech: docker/prepare.sh, prepare/quant_lm_head.py, drafter/gptq_lm_head.py
needs: —
links: #serve, #patches
files: prepare/**, drafter/**, docker/prepare.sh

Idempotent, resumable pipeline: it computes remaining steps from what is already
on disk. Stable — `prepare/` has 2 commits, `drafter/` 5, both last touched
2026-08-21 — so the drafter training code folds in here rather than standing alone.

- [x] int8 lm_head, embeddings and MTP module
      tech: docker/prepare.sh:49-51, prepare/quant_lm_head.py, prepare/quant_embed.py, prepare/quant_mtp.py
- [x] 40k-token draft head from a shipped id list
      tech: docker/prepare.sh:52-54, prepare/build_draft_vocab.py, prepare/draft_vocab_ids.json
- [x] int4-GPTQ "fast" variant, auto-selected by the launcher when present
      tech: prepare/fetch_fast_variant.py; single-user/start_qwen.sh:64-67
- [x] optional W4A16 DFlash2 block drafter, failure is non-fatal
      tech: docker/prepare.sh:57-60, prepare/fetch_dflash2.py
- [x] GPTQ requantisation of lm_head and MTP calibrated on hidden states
      tech: drafter/gptq_lm_head.py, drafter/requant_mtp_gptq.py, drafter/gptq_utils.py

## Measure decode, context and tool-call correctness {#measure}
tech: ctx_probe.py, code_bench.py, tool_reliability.py, bench/run_benchmarks.sh
needs: #serve
links: #kvarn, #harness, #docs
files: bench/**, ctx_probe.py, code_bench.py, tool_reliability.py, *_results.json, ctx_probe_*.json, code_bench_*.json

Two harnesses: upstream's `bench/` (27 commits, unused locally) and a local set
written for the questions upstream does not answer. Folded into one card because
one agent owns "does this server actually perform" for a session.

- [x] context/TTFT/decode probe with per-length offsets so prefix caching cannot flatter it
      tech: ctx_probe.py:56-63, ctx_probe_ctxfast.json, ctx_probe_ctxlong.json, ctx_probe_ctxhuge.json
- [x] coding-prompt decode split by family, tokens/step from server spec-decode counters
      tech: code_bench.py, code_bench_mtp_long.json, code_bench_dflash2_k7.json, code_bench_dflash2_k15.json
- [x] tool-call failure rate, 11 buckets, malformed separated from wrong-choice
      tech: tool_reliability.py, tool_reliability_results.json (440 requests, 1 malformed), tool_reliability_dflash2.json
- [x] upstream harness for throughput, acceptance and quality batteries
      tech: bench/run_benchmarks.sh, bench/labd_bench.py, bench/quality_battery.py
- [ ] `bench/labd_bench.py` hard-codes `~/qwen-serving/api_key.txt` and cannot run here
      from: agent
      tech: bench/labd_bench.py:31
- [ ] none of the local probes are committed; all five are untracked
      from: agent

## Route agent work to the local model without drowning it in context {#harness}
tech: ~/.omp/agent/models.yml, ~/.omp/agent/config.yml, mcp_cost.py, tool_tap.py
needs: #serve, #deploy
links: #measure
files: mcp_cost.py, tool_tap.py

omp (`can1357/oh-my-pi`) points at the local server for the high-frequency roles.
Context budget was the open question; it was measured off the wire, not estimated.

- [x] `qwen38-local` provider at localhost:18020, contextWindow pinned to served 150000
      tech: ~/.omp/agent/models.yml:8-24
- [x] `smol`/`task`/`commit`/`tiny` on local, `plan`/`slow` on Claude, `advisor` untouched
      tech: ~/.omp/agent/config.yml:5-10
- [x] per-MCP-server schema cost measured with the served tokenizer
      tech: mcp_cost.py (125 tools, 8 servers)
- [x] request tap proving omp mounts MCP as routes, not inlined schemas (1,313 tokens, not 30,423)
      tech: tool_tap.py; captured `tools` array showed 11 built-ins, 0 MCP
- [ ] make the lean profile the default — `--tools read,edit,write,glob,grep,bash,todo --no-skills`
      cuts startup from 34,109 to 13,578 tokens
      from: agent
- [ ] settle whether subagents inherit the parent's tool set; never observed a real subagent request
      from: agent

## Pin the local deployment so it survives a reboot {#deploy}
tech: nvidia-gpu-tuning.service, .env, api_key.txt
needs: #serve
links: #harness, #docs
files: nvidia-gpu-tuning.service, .env, api_key.txt

- [x] API key enforced: 401 without, 401 with a wrong key, 200 with the right one
      tech: .env, api_key.txt (chmod 600), verify.sh `PASS API key configured`
- [x] persistence-mode unit installed and running
      tech: nvidia-gpu-tuning.service; `systemctl is-enabled` → enabled, `is-active` → active
- [x] power limit pinned in the unit rather than left to the last manual call
      tech: nvidia-gpu-tuning.service `ExecStart=/usr/bin/nvidia-smi -pl 350`
- [x] secrets excluded from git
      tech: .gitignore (`api_key.txt`, `.env`)
- [ ] confirm persistence survives an actual reboot; only observed post-`enable --now`
      from: agent
- [ ] server still binds 0.0.0.0; narrow to 127.0.0.1 if nothing off-box needs it
      from: agent
- [ ] evaluate the 250W reference power limit — every number here was taken at 350W
      from: roadmap

## Keep the decision log ahead of the prose {#docs}
tech: docs/gotchas.md (41 entries), docs/long-context.md, qwen38-vllm-handoff.md
needs: #measure
links: #serve, #kvarn, #deploy
files: docs/**, README.md, qwen38-vllm-handoff.md, */README.md

33 commits on `docs/`, last 2026-08-25. `docs/gotchas.md` is the real decision
log and is load-bearing: it is what stops the `--kv-cache-memory` and
`gpu-memory-utilization` traps from being re-entered.

- [x] 41 numbered gotchas, each with the failure mode and the reason
      tech: docs/gotchas.md
- [x] long-context tradeoffs with measured tables per KV mode
      tech: docs/long-context.md
- [x] local handoff carrying this fork's measurements and an upgrade warning
      tech: qwen38-vllm-handoff.md:6 (upgrade warning), :90 (§3a long-context failure)
- [ ] `qwen38-vllm-handoff.md` and `PLAN.md` are untracked; nothing is committed yet
      from: agent

---

## decisions

**No plan convention was found.** The task named an "agenttrail plan convention"
section in `CLAUDE.md` or `AGENTS.md`. Neither file exists — not in the repo, not
in `.claude/` (absent), not at user level (`~/.claude/*.md` empty). `git ls-files
'*.md'` returns 11 files, none of them agent-instruction files. The only
`agenttrail` string on disk is this session's own transcript. Structure here
follows the spec given in the task text; if a real convention exists elsewhere,
this file should be re-cut against it.

**Trust order applied.** Directory layout and source first, then `git log` (churn
by directory over 40 commits: single-user 31, docs 20, root 17, bench 12, batch
10), then `docs/gotchas.md` and `qwen38-vllm-handoff.md` for in-flight state, then
`README.md` last. The README claims nothing forward-looking — no roadmap, TODO or
"planned" language — so open intent came almost entirely from the handoff doc,
which is why most open items carry `from: agent` and only one carries
`from: roadmap`.

**Component count: 8.** Two folds. `drafter/` (5 commits, last 2026-08-21) folded
into #prepare — it produces the same artifacts and is stable plumbing. `bench/`
folded into #measure with the local probes; they are different codebases but one
session's work, and upstream's harness is unused here. `patches/` and `kvarn/`
stayed separate despite both being vLLM surgery: #kvarn is a two-stage install
with its own failure mode and its own doneness.

**Nothing is marked `[~]`.** No component is being actively built right now; the
last session ended at a stopping point. Marking anything in-progress would be
false.

**Unverified, therefore unchecked.** Three ticks were withheld for lack of
evidence:
- Reboot survival of the persistence unit. `is-enabled`/`is-active` both pass, but
  that state was created by `enable --now` in the same session. The unit orders
  itself after `systemd-modules-load.service`, which is an assumption about driver
  readiness that no reboot has tested.
- Subagent tool inheritance. Agent frontmatter shows explicit `tools:` allowlists
  (`scout` names five built-ins; `task` names none and claims full access), which
  implies the answer, but no subagent request was ever captured by `tool_tap.py`.
  Frontmatter is not observation.
- The 15-patch count. `verify.sh` reports `OK (0 failures)` and `patches/` holds 15
  `.patch` files, but verify's own output does not enumerate one line per patch, so
  "all 15 verified individually" is asserted from the file count, not from output.

**One claim in the handoff doc was corrected during this pass** and the corrected
figure is what appears above: MCP tool schemas cost omp 1,313 tokens, not the
30,423 originally recorded. The larger number is what the same eight servers cost
Claude Code, which inlines schemas; omp mounts them as `xd://` routes. Evidence is
the captured request body, not inference.
