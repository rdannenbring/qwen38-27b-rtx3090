# PLAN

A fork of `syv-ai/qwen38-27b-rtx3090` — Qwen3.8-27B running on one RTX 3090 via
vLLM 0.27.1 with 15 hand-applied patches — plus the deployment, measurement and
agent-wiring layer added on this machine.

Every `[x]` was checked against source or a produced artifact and cites it on a
`tech:` line. `## decisions` records what could not be verified.

---

## Start the model server and choose speed against context {#serve}
tech: single-user/start_qwen.sh (550 lines), batch/start_qwen.sh, docker-compose.yml, Dockerfile
needs: [patches, prepare]
links: [kvarn, measure, deploy]
files: [single-user/**, batch/**, docker/**, docker-compose.yml, Dockerfile]

One shell script turns two knobs into a vLLM command line and refuses the
combinations that do not work. The busiest surface in the repo: 61 commits on
`single-user/`, last upstream change 2026-08-25.

- [x] Pick short-and-fast or long-and-slower with one setting {#serve-ctx}
      tech: CTX=fast|long|huge at single-user/start_qwen.sh:79,90-104
- [x] Pick which drafter guesses ahead, and reject bad pairings {#serve-spec}
      tech: SPEC=mtp|dflash2 at single-user/start_qwen.sh:87,105-125
- [x] Reuse a conversation's prefix instead of re-reading it every turn {#serve-prefix}
      tech: PREFIX_CACHE=1 at single-user/start_qwen.sh:335
- [x] Change any setting by editing one file, no rebuild {#serve-env}
      tech: env_file at docker-compose.yml:28-29
- [x] Run either a low-latency server or a throughput server from one image {#serve-profiles}
      tech: profiles [single]/[batch] at docker-compose.yml:65-73, batch/start_qwen.sh
- [x] Survive a dead engine without needing a manual cleanup {#serve-shm}
      tech: stale /dev/shm/vllm_offload_*.mmap unlinked, single-user/start_qwen.sh:47-58
- [ ] Make the 15-token drafter start without hand-editing a second variable {#serve-dflash15}
      from: agent
      tech: that branch reads DFLASH_MAX_LEN and ignores MAX_LEN — single-user/start_qwen.sh:240

## Prove the custom engine changes are still in place {#patches}
tech: verify.sh (171 lines), patches/_check_applied.py, 15 files in patches/ (3,778 lines)
needs: []
links: [serve, kvarn, prepare]
files: [patches/**, verify.sh]

The patches are reapplied by hand against a pinned vLLM. This is the only thing
between an upgrade and a server that looks fine and is quietly wrong.

- [x] Confirm all 15 engine changes are live, one line each {#patches-all}
      by: claude
      tech: verify.sh enumerates 15 PASS lines, dflash2-backport through xgrammar-spec-terminated, 2026-08-27
- [x] Catch a patch that applied partially, not just one that errored {#patches-content}
      tech: content check at verify.sh:52, patches/_check_applied.py
- [x] Check the whole stack end to end, not just the files {#patches-live}
      tech: 38 PASS / 0 FAIL including /health 200 and a real completion, 2026-08-27
- [ ] Re-apply and re-verify after any vLLM version change {#patches-upgrade}
      from: agent
      tech: nothing enforces this; the warning lives at qwen38-vllm-handoff.md:6

## Fit a book-length conversation in the card's memory {#kvarn}
tech: kvarn/files/vllm/v1/attention/backends/kvarn_attn.py, kvarn/install.sh, kvarn/kvarn-v2-runner.patch
needs: [patches]
links: [serve, measure]
files: [kvarn/**]

Huawei CSL's KVarN cache backported from its vLLM 0.23 fork onto 0.27.1. Trades
decode speed for roughly double the conversation length.

- [x] Store the conversation in 4/2-bit so far more of it fits {#kvarn-backend}
      tech: kvarn_k4v2_g128 registered, `Using KVARN attention backend` in engine log
- [x] Let the block drafter run at the longest context too {#kvarn-v2}
      tech: kvarn-v2-runner.patch applied — verify.sh PASS, 2026-08-27
- [x] Hold the model's full 262,144-token limit {#kvarn-full}
      by: claude
      tech: booted with a 295,141-token pool; ctx_probe_ctxhuge.json
- [x] Show it still finds a fact buried at that length {#kvarn-recall}
      by: claude
      tech: planted fact recovered from a 250,139-token prompt, qwen38-vllm-handoff.md:76
- [ ] Take the spare room the warm pool leaves — 202,040 tokens measured, 150,000 served {#kvarn-maxlen}
      from: agent

## Build the shrunken model files the server loads {#prepare}
tech: docker/prepare.sh, prepare/quant_lm_head.py, prepare/build_draft_vocab.py, drafter/gptq_lm_head.py
needs: []
links: [serve, patches]
files: [prepare/**, drafter/**, docker/prepare.sh]

Resumable pipeline that works out what is still missing from what is on disk.
Stable: `prepare/` has 2 commits, `drafter/` 5, both last touched 2026-08-21.

- [x] Shrink the parts of the model that dominate memory {#prepare-quant}
      tech: int8 lm_head, embeddings and MTP module — verify.sh PASS x3, docker/prepare.sh:49-51
- [x] Build the 40k-word shortlist the drafter guesses from {#prepare-vocab}
      tech: verify.sh `40k-token draft head present`, prepare/build_draft_vocab.py
- [x] Pick up the faster variant automatically when it exists {#prepare-fast}
      tech: verify.sh `fast variant present`, selected at single-user/start_qwen.sh:64-67
- [x] Treat the optional block drafter as optional {#prepare-dflash2}
      tech: verify.sh `DFlash2 drafter present, W4A16`, non-fatal at docker/prepare.sh:57-60
- [x] Confirm the downloaded model is complete and its tokenizer loads {#prepare-integrity}
      tech: verify.sh `8 safetensors shards present`, `<think> -> [248068]` for both variants

## Measure how fast and how correct this server really is {#measure}
tech: ctx_probe.py, code_bench.py, tool_reliability.py, tool_tap.py, bench/run_benchmarks.sh
needs: [serve]
links: [kvarn, harness, docs]
files: [bench/**, ctx_probe.py, code_bench.py, tool_reliability.py, tool_tap.py, ctx_probe_*.json, code_bench_*.json, tool_reliability_*.json]

Upstream's `bench/` answers throughput and quality. These answer the questions
that decided whether this stack was usable at all.

- [x] Find out how often a tool call comes back unusable {#measure-tools}
      by: claude
      tech: 1 malformed in 440 requests; tool_reliability.py, tool_reliability_results.json
- [x] Separate a broken call from a merely different choice {#measure-buckets}
      by: claude
      tech: 11 buckets, HARNESS_BREAKING vs BEHAVIOURAL in tool_reliability.py
- [x] Measure speed at a given conversation length without fooling ourselves {#measure-ctx}
      by: claude
      tech: per-length corpus offsets so short probes are not a prefix of long ones, ctx_probe.py:56-63
- [x] Show which coding tasks the faster drafter actually helps {#measure-code}
      by: claude
      tech: four task families, tokens/step from server counters; code_bench_mtp_long.json, code_bench_dflash2_k7.json, code_bench_dflash2_k15.json
- [x] See exactly what a request carries before it reaches the model {#measure-tap}
      by: claude
      tech: tool_tap.py; captured body showed 11 built-in tools and 0 inlined MCP schemas
- [ ] Make upstream's long-context benchmark runnable here {#measure-labd}
      from: agent
      tech: bench/labd_bench.py:31 hard-codes ~/qwen-serving/api_key.txt

## Send my coding agent's cheap work to the local model {#harness}
tech: ~/.omp/agent/models.yml, ~/.omp/agent/config.yml, mcp_cost.py
needs: [serve, deploy]
links: [measure]
files: [mcp_cost.py]

omp (`can1357/oh-my-pi`) routes its high-frequency roles here and keeps the
expensive reasoning on Claude.

- [x] Point the agent at the local server with the right context limit {#harness-provider}
      by: claude
      tech: ~/.omp/agent/models.yml:8-24, contextWindow 150000 matching what is served
- [x] Give the local model the frequent cheap jobs and keep planning on Claude {#harness-roles}
      by: claude
      tech: ~/.omp/agent/config.yml:5-10; advisor deliberately not routed
- [x] Find out what each connected tool server costs in context {#harness-mcpcost}
      by: claude
      tech: mcp_cost.py, 125 tools across 8 servers, tokenised with the served model
- [x] Settle whether tool schemas are the context problem {#harness-mcptruth}
      by: claude
      tech: they are not — 1,313 tokens as xd:// routes, not 30,423 inlined; qwen38-vllm-handoff.md:180
- [ ] Make the lean profile the default: 34,109 tokens of startup down to 13,578 {#harness-lean}
      from: agent
- [ ] Find out whether sub-agents inherit the parent's tools {#harness-subagents}
      from: agent

## Keep the server private and running after a reboot {#deploy}
tech: nvidia-gpu-tuning.service, .env, api_key.txt, .gitignore
needs: [serve]
links: [harness, docs]
files: [nvidia-gpu-tuning.service, .env, api_key.txt]

- [x] Stop anyone on the network from using the server {#deploy-key}
      by: claude
      tech: 401 without a key, 401 with a wrong one, 200 with the right one; verify.sh `API key configured`
- [x] Keep the GPU driver resident so the first request is not slow {#deploy-persist}
      by: claude
      tech: nvidia-gpu-tuning.service; systemctl is-enabled → enabled, is-active → active
- [x] Fix the power limit instead of leaving it to the last manual command {#deploy-power}
      by: claude
      tech: `-pl 350` in the unit; verify.sh reports `power limit 350.00 W`
- [x] Keep the key and settings out of git {#deploy-secrets}
      by: claude
      tech: .gitignore:2,11; pushed tree contains neither api_key.txt nor .env
- [x] Stop a GPU-starved start from looping forever {#deploy-noloop}
      by: claude
      tech: restart policy removed (0df3d35); live container set restart=no. It had
            looped 155 times hitting the 21.91 GiB guard 414 times on 2026-08-28
- [x] Stop the two GPU servers racing each other at boot {#deploy-arbiter}
      by: claude
      tech: unsloth-studio.service disabled, its autostart .desktop and generated
            unit gone; ~/Development/gpu-tray is now the only thing that starts either
- [ ] Confirm the GPU settings survive an actual reboot {#deploy-reboot}
      from: agent
      tech: also the first real test of the arbiter — nothing but the tray should come up
- [ ] Stop listening on every network interface {#deploy-bind}
      from: agent
- [ ] Decide whether to drop to the 250W reference limit {#deploy-250w}
      from: roadmap
      tech: every number recorded here was taken at 350W

## Write down what we learned so it is not rediscovered {#docs}
tech: docs/gotchas.md (41 entries), docs/long-context.md, qwen38-vllm-handoff.md, PLAN.md
needs: [measure]
links: [serve, kvarn, deploy]
files: [docs/**, README.md, qwen38-vllm-handoff.md, PLAN.md, */README.md]

`docs/gotchas.md` is the real decision log — 33 commits, last upstream change
2026-08-25. It is what stops the memory-tuning traps being re-entered.

- [x] Keep a numbered log of every trap and why it bites {#docs-gotchas}
      tech: 41 entries in docs/gotchas.md
- [x] Record what each memory mode costs at length {#docs-longctx}
      tech: measured tables per KV mode, docs/long-context.md
- [x] Warn loudly that an engine upgrade silently breaks this {#docs-upgrade}
      by: claude
      tech: first section of qwen38-vllm-handoff.md:6
- [x] Record the long-context failure we found, and its shape {#docs-3a}
      by: claude
      tech: qwen38-vllm-handoff.md:90 — positional, not length-bound
- [x] Get this work into version control {#docs-commit}
      by: claude
      tech: dfd3425, 0ecb818, 359ae0e on branch local/3090-validation, pushed to the fork
- [ ] Send the two upstream-worthy bugs back to syv-ai {#docs-upstream}
      from: agent
      tech: #serve-dflash15 and #measure-labd are both upstream defects, not local ones

---

## decisions

**The convention exists now; the previous pass improvised.** `CLAUDE.md` and
`AGENTS.md` (identical, 2.1 KB) were created at 10:28 today, after the first
version of this file was written and committed. That earlier version guessed a
format and said so. This rewrite follows the real convention, which differs in
four ways: `needs:`/`links:`/`files:` take bracketed lists; every task carries its
own `{#id}`; tasks record `by:` attribution; and titles must be plain-language for
the owner, with engineer phrasing moved to `tech:`. Titles were rewritten
accordingly — "Port the 4/2-bit KV cache" became "Fit a book-length conversation
in the card's memory".

**The eight component ids are unchanged.** The convention says ids are stable and
may only be added or removed. `serve`, `patches`, `kvarn`, `prepare`, `measure`,
`harness`, `deploy`, `docs` were committed in 359ae0e and carry over verbatim
despite every title changing. Task ids are new, since the previous version had
none.

**Trust order.** Source and layout first; then `git log` — the three commits at
the top are this machine's work, everything from 60daef8 down is upstream, last
2026-08-25; then `docs/gotchas.md` and `qwen38-vllm-handoff.md` for in-flight
state; `README.md` last. The README still contains no roadmap, TODO, or "planned"
language, so it contributed no open items. That is why ten of the eleven open
tasks are `from: agent` and only `#deploy-250w` is `from: roadmap`.

**One tick previously withheld is now granted.** `#patches-all` was left unchecked
last pass because verify's summary line did not enumerate patches. Running
`verify.sh` with full output does: 15 individual PASS lines, `dflash2-backport`
through `xgrammar-spec-terminated`, plus 38 PASS / 0 FAIL overall including a live
completion. That is per-patch evidence, so the tick is now honest.

**Two ticks remain withheld.**
- `#deploy-reboot`. `is-enabled` and `is-active` both pass, but that state was
  created by `enable --now` in the same session. The unit orders itself after
  `systemd-modules-load.service`, an assumption about driver readiness no reboot
  has tested. Verifiable only by rebooting.
- `#harness-subagents`. Agent frontmatter shows explicit `tools:` allowlists
  (`scout` names five built-ins; `task` names none and claims full access), which
  implies the answer, but `tool_tap.py` never captured a real sub-agent request —
  the one attempt produced a single growing conversation, not a parent and a
  child. Frontmatter is not observation.

**Nothing is `[~]` or `[!]`.** No task is under way at this moment and none is
blocked. The convention wants `[~]` set before starting and flipped on
completion; since this session's work was finished before the convention existed,
those tasks are recorded as `[x]` with `by: claude` under the "graduate completed
todos" rule rather than being retroactively marked in progress.

**GPU arbitration moved out of this repo.** On 2026-08-28 Unsloth Studio
autostarted holding ~20.6 GiB and this server crash-looped 155 times against its
21.91 GiB startup guard. Two servers cannot share a 24 GiB card, and neither
belongs to the other, so the arbiter is a separate project —
`~/Development/gpu-tray` — and this repo only carries the half that is its own:
no restart policy, so a GPU-starved start fails once and stays down
(`#deploy-noloop`). The tray is now the only thing that starts this container.

**A figure in the handoff doc was corrected and the corrected one is used here.**
MCP tool schemas cost omp 1,313 tokens, not 30,423. The larger number is what the
same eight servers cost a harness that inlines schemas; omp mounts them as `xd://`
routes. Evidence is a captured request body, not inference — `#harness-mcptruth`.
