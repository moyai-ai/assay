# Verification-scaling benchmark

Independent coding candidates → blind LLM verification/pivot tournament → locked
winner → official Harbor grading. Uses the ten tasks and dataset revision in
`tasks.json`, Harbor 0.23.0, and Assay's actual `runCodingAgent` and `selectCandidate`.
This is a fixed smoke subset, not a full-benchmark leaderboard result.

## Research goal: three arms, two comparisons

The target study compares **vanilla DeepSeek-V4.1-Flash**, **vanilla GPT-6 Astra**
(using the model ID available to the configured provider/account), and **DeepSeek
with verification scaling**. For a DeepSeek-only result, configure DeepSeek for
both generation and judging; the external baseline never selects its candidates.

- **A — generator uplift:** selected-solution pass rate minus vanilla DeepSeek's.
- **B — external-model gap:** selected-solution pass rate minus Astra's, with
  task-paired wins, losses, ties and disclosed inference budgets/costs.

These are hypotheses, not promised outcomes. Predeclare the primary candidate/
verification budget before a confirmatory evaluation; do not pick the best-looking
arm after seeing official labels. Reports preserve negative results and missing
labels. Paired p-values are exploratory, unadjusted for multiple arms, and cannot
establish equivalence. A tied small-subset score is only an observed tie—not a
general SOTA claim. A larger evaluation and a prespecified parity tolerance are
needed for a defensible Astra-level performance claim.

## Run

Requires Node.js 22+, `uv`, Git, and Docker with Compose. Run from the repository
root. The root `.env` stays on the host; existing environment variables take precedence.

```sh
npm ci
npm run benchmark:prepare
npm run benchmark:test                     # Local mocks, no paid calls
npm run benchmark:scale                    # Plan only: no containers/model calls

# Small paid end-to-end pilot: two independent candidates on one task.
npm run benchmark:scale -- --execute --task regex-log \
  --candidate-counts 1,2 --verifier-repetitions 2 \
  --verifier-reasoning-effort none --verifier-output-tokens 8192 --name my-pilot

# Full study: 40 generations across 10 tasks. Reuse the same pools for
# N=1/2/4 and verifier repetition budgets 2/4. This makes paid model calls.
npm run benchmark:scale -- --execute --candidate-counts 1,2,4 \
  --verifier-repetitions 2,4 --concurrency 2 --name my-study \
  --verifier-reasoning-effort none --verifier-output-tokens 8192 \
  --max-evidence-bytes 40000 --allow-evidence-truncation

npm run benchmark:report -- benchmark/runs/my-study
```

Named runs never overwrite existing directories; plans are not resumed. Use a new
name for execution. There are no automatic generation, selection, or trial retries.
Keep laptops powered with lids open. `caffeinate -i` prevents macOS idle sleep, not
lid-closure sleep. Pools can pipeline across generation, selection, and grading;
the selection-before-grading barrier still applies separately to each task.
Global budgets—not per-task multipliers—bound generation, baseline, judging,
setup, snapshots, and grading. Whole pools reserve container capacity before
admission, including paused candidates, so partial pools cannot deadlock each other.
The runner checks task compatibility
before model calls: Linux, single-step, shared grading environment, and no custom
Compose services. All ten pinned tasks satisfy these constraints. Other Harbor
lifecycle types are rejected rather than silently graded in reconstructed environments.

### Aggressive 16/32 profile

```sh
npm run benchmark:scale -- --execute --name parallel-study \
  --baseline-model YOUR_AVAILABLE_ASTRA_MODEL_ID --baseline-api responses \
  --candidate-counts 1,2,4 --verifier-repetitions 2,4 \
  --task-concurrency 4 --concurrency 16 --baseline-concurrency 4 \
  --provider-concurrency 32 --judge-concurrency 32 --verifier-concurrency 16 \
  --pair-concurrency 8 --arm-concurrency 4 --max-live-containers 32 \
  --setup-concurrency 4 --snapshot-concurrency 2 --grading-concurrency 4 \
  --prepare-grader --verifier-reasoning-effort none --verifier-output-tokens 8192 \
  --max-evidence-bytes 40000 --allow-evidence-truncation \
  --build-task filter-js-from-html --chromedriver-path /usr/bin/chromedriver
```

This admits up to four five-candidate pools (20 original containers), with up to
16 DeepSeek generation agents and four separately scheduled baseline agents.
Judging can use 32 concurrent requests across selector processes; each process
reserves its local ceiling (16 here), and parallel pairs share that ceiling.
The combined DeepSeek generation/judging ceiling is also 32. A generation agent
conservatively reserves one provider slot through its tool execution, not just
while an HTTP request is in flight. These are ceilings, not promised utilization
or speedups. More concurrency can increase latency or rate-limit failures.

Check Docker's memory allocation, not just host RAM. The live-container cap does
not guarantee that every possible task fits in memory. Do not change resource
settings or code in an active study; use a new named run. `progress.json` updates
every 15 seconds with stage information and global active/peak/queued budgets.
For just the primary three-way comparison, predeclare `--candidate-counts 1,4`
and `--verifier-repetitions 4`; this omits the secondary scaling curves, not candidates
from the primary pool. Never remove failed arms after seeing their results.

## Experimental protocol

1. **Independent generation.** Every candidate gets a new container, conversation,
   and coding-agent process. All candidates share the task and budgets, not answers.
   Before generation, fingerprint the initial `/app` files and image filesystem/
   runtime configuration. Different initial states cannot enter the same tournament.
2. **Freeze evidence.** Pause each original Docker container after generation.
   Collect `/app` changes on the host without extracting or executing archive contents.
   Include a bounded, labeled tail of agent-initiated shell results—not official
   tests and not independently authored test assertions. Worker logs stay outside
   container-writable Harbor log mounts.
3. **Blind selection.** The selector receives only the instruction, anonymized
   candidate evidence, and fixed settings. It has no tools, grader paths, official
   rewards, tests, or reference solutions. It uses `PairwiseVerifier` and the pivot
   tournament, not a replacement heuristic. A paid synthetic capability probe runs
   before generation to require usable score-token logprobs.
4. **Lock decisions.** Every requested N/repetition/pivot configuration uses the
   same frozen pool. Write an exclusive `decisions.json`, including evidence hashes,
   before releasing any official grader for that task. Failed selections remain
   errors; there is no fallback to a convenient or previously passing candidate.
5. **Grade unchanged candidates.** Unpause the original containers and let Harbor
   run the official tests. No filesystem reconstruction, solution editing, or
   grader modification. All available candidates are graded after the lock, so
   non-winner labels can provide optional oracle/random-selection diagnostics.
   Those labels never influence selection.

Nested prefixes define candidate-count scaling: N=2 uses c001/c002 from the same
pool used at N=4. The **vanilla baseline is always c001**, chosen before rewards.
Vary `--verifier-repetitions` and/or `--pivot-counts` at fixed N to isolate verifier
compute. More verification is not guaranteed to improve the selected reward.

A turn limit is a recorded `turn_limit` stop, not successful completion; the partial
implementation remains gradeable. Harbor time limits likewise retain their warning
and gradeable state. Provider/transport/parse failures are not disguised as successful
generations. Unavailable candidates invalidate affected prefixes without replacement.

## Configuration

| Setting | Default / source |
|---|---|
| Generator model / URL | `GENERATOR_MODEL` / `GENERATOR_BASE_URL`, otherwise `VERIFIER_*` |
| Generator key | `GENERATOR_API_KEY`, then `NEBIUS_API_KEY`, then `VERIFIER_API_KEY` |
| Selector | Explicit `VERIFIER_MODEL`, `VERIFIER_BASE_URL`, `VERIFIER_API_KEY`, `VERIFIER_CONTEXT_TOKENS` |
| Selector provider options | `VERIFIER_EXTRA_BODY` JSON; cannot override scoring protocol fields. Explicit `--verifier-reasoning-effort` overrides its reasoning setting. |
| Candidate counts | `--candidate-counts 1,2,4`; max 16, vanilla N=1 always included |
| Verification budgets | `--verifier-repetitions 2,4`, `--pivot-counts 2`; per-selector `--verifier-concurrency 16` |
| Scheduling | `--task-concurrency 1`, global generation `--concurrency 2`, separate `--baseline-concurrency 2` |
| Judge/provider caps | Global `--judge-concurrency 32`, combined `--provider-concurrency 32`, `--pair-concurrency 8`, `--arm-concurrency 4` |
| Containers/setup | `--max-live-containers 32`, `--setup-concurrency 4`, `--snapshot-concurrency 2`, `--grading-concurrency 4` |
| Generation | `--max-turns 60`, `--max-output-tokens 32768`, `--reasoning-effort none` |
| Request deadlines | `--request-timeout-sec 900`; SDK and Undici header/body deadlines aligned |
| Harbor deadlines | `--agent-timeout-multiplier 2`, `--verifier-timeout-multiplier 8` |
| Selector | `--verifier-output-tokens 32768`, `--selection-timeout-sec 3600`, `--min-captured-mass 0` |
| Evidence | `--max-evidence-bytes 48000`; oversized evidence fails unless `--allow-evidence-truncation` is explicit |
| Snapshot bound | `--snapshot-byte-limit 268435456` (256 MiB); read only, no host extraction |
| Ring seed | `--seed 0`; controls tournament scheduling, not independent model sampling |

Shell commands remain limited to 32,768 characters, 1–120 seconds, and the last
16 KiB per output stream. Tool inputs are validated locally; server strict mode is
off. Token-truncated responses are rejected before partial tool calls execute.

The DeepSeek examples explicitly disable scorer reasoning: provider-default
reasoning exhausted a 4,096-token pilot budget, then a 32,768-token pilot encountered
HTTP 504 despite some valid scores. A synthetic full tournament with `none`
completed all 12 scoring calls with no reasoning tokens. This is an operational
configuration choice, not evidence that lower reasoning improves judge quality.
The failed pilots are retained; there is no automatic retry or silent fallback.

Evidence excludes `.git`, virtualenvs, dependency/cache directories; binaries and
large files have metadata/hashes rather than readable contents. These omissions
are explicit. Outside-`/app` installation changes are not a complete source diff.
Official grading still uses the full original environment. Network access is not
a hard anti-contamination boundary; models are instructed not to seek benchmark
solutions/tests online. Use a trusted Docker daemon without sensitive mounts.

### Optional separate-model vanilla baseline

Set only `OPENAI_API_KEY` in the host's gitignored `.env`; the baseline defaults
to `https://api.openai.com/v1`. No base URL is required. `BASELINE_BASE_URL` remains
an optional override for an explicitly chosen compatible endpoint. The former
`BASELINE_API_KEY` name is no longer used for authentication. Never paste keys
into chat or commit them. Then add:

```sh
npm run benchmark:scale -- --execute --name with-baseline \
  --baseline-model YOUR_AVAILABLE_ASTRA_MODEL_ID --baseline-api responses \
  --verifier-reasoning-effort none --verifier-output-tokens 8192 \
  --max-evidence-bytes 40000 --allow-evidence-truncation
```

This adds **one independent baseline generation per task** (50 total generations
with the default four-candidate pools). The baseline never enters Assay's tournament.
It uses the same coding scaffold, tools, task, turn/token limits and phase deadlines,
but its recorded provider/API/reasoning settings may differ. Defaults are Responses
API and provider-default reasoning; Chat Completions is also supported. No provider
model ID or availability is assumed. Baseline credentials never fall back to the
generator's credentials. Responses runs request and carry opaque encrypted
reasoning context between stateless tool turns. This measures the models in
Assay's shared coding scaffold, not a provider's proprietary coding product or
its strongest published agent configuration.

### Grader health and native images

`--prepare-grader` seeds hash-verified **uv 0.9.5** binaries and prewarms the pinned
generic pytest/runtime dependencies declared by these ten tasks, before generation.
Each sandbox has an independent cache. No official tests or reference solutions
are uploaded in this phase; no test assertions or installer scripts are patched.
The host downloads wheels from PyPI with source-pinned SHA-256 checksums, reads only
named binary members, and never executes their code. Public package-download
retries are bounded; this does not enable model, selection, or trial retries.
The dependency profile, architecture, package pins, and wheel hash are recorded in
`grading-dependencies.json` and included in the initial-condition fingerprint.

This addresses the observed GitHub installer download failure: even if that step
fails, the original grader can still use the already seeded `uvx` and cache.
Preparation fails closed before that candidate's model calls if its tools cannot
be installed. It is not a blanket guarantee against network outages, missing
packages, or grader bugs. The initial software environment changes are declared
and apply to all candidates and baselines; don't pool results across configurations.

Before spending generation tokens on HTML filtering, the runner checks that a real
headless browser can execute a synthetic alert. An unavailable driver must not look
like a passing XSS test. Reports also inspect grader test counts and known browser/
collection errors; suspect labels retain their raw official reward but are excluded
from a complete quality score. These checks are not a proof of grader completeness.

On ARM machines, an incompatible prebuilt Chromium image may require building the
unchanged pinned Dockerfile natively:

```sh
npm run benchmark:scale -- --execute --name native-html \
  --build-task filter-js-from-html --chromedriver-path /usr/bin/chromedriver
```

`--build-task` is repeatable; `--force-build` applies to all tasks. The optional
`--chromedriver-path` sets the HTML container's non-secret `SE_CHROMEDRIVER`, bypassing
Selenium Manager's driver discovery. Native ARM Chromium with this explicit driver
passed the synthetic browser check locally. No official test assertions are patched. Builds/downloads may take time, and image tags/dependencies
are not fully pinned by the task Git revision.

## Reports and accounting

`benchmark/runs/<name>/` is private and Git-ignored:

- `manifest.json`: fixed arms, budgets, source hashes, model IDs, and evidence policy.
- `progress.json`: live stage information and global resource-budget counters.
- `tasks/<task>/candidates/<id>/`: private worker logs, before/after evidence,
  stop reasons, usage and provenance.
- `tasks/<task>/selection/<arm>/`: exact blind input, raw verifier responses,
  expected score distributions, captured mass, pairwise evaluations and rankings.
- `tasks/<task>/decisions.json`: pre-grading selection lock and evidence hashes.
- `tasks/<task>/trials/`: Harbor's original official results and grader logs.
- `summary.json` / `summary.md`: selected rewards, per-arm pass rates, paired wins/
  losses against vanilla and the external baseline, percentage-point uplift/gap,
  and exploratory paired tests. Refreshable during a run.

Headline pass rate is selected passes / **all planned tasks**, and is only presented
as a complete quality score when every task has a trustworthy selected grade.
Errors, pending tasks, raw official rewards and budget stops remain visible.
Verifier scores are relative expected scores/preferences, **not calibrated correctness
probabilities**. Oracle-any-pass and random-choice diagnostics are not selected-solution
accuracy. Ten tasks provide exploratory evidence, not a statistically strong claim.

Token counts include all recorded responses, including unsuccessful experiments.
Per-arm costs count its candidate prefix plus its selection, even though the study
reuses pools physically. Actual whole-study usage is reported separately. Optional
`--prices prices.json` accepts user-supplied flat USD/million-token estimates:

```json
{"generator":{"input":1,"cached_input":0.1,"output":2},
 "verifier":{"input":1,"cached_input":0.1,"output":2}}
```

These are **illustrative numbers, not provider prices**. Add a `baseline` entry when
needed. Estimates exclude pricing tiers, grading, infrastructure and shared probe
overhead; missing usage/prices produces null, not a guessed bill. Generation timing,
selection timing and total task wall time are recorded separately. New runs record
UTC start/end timestamps and wall-clock elapsed time (including machine sleep),
plus a separate monotonic-clock duration. Concurrent task times must not be summed
and called whole-study elapsed time. Incomplete arms
exit nonzero; ordinary graded task failures alone do not.

## Local validation and generation-only mode

```sh
npm run check && npm test && npm run benchmark:test
# Real Docker + Harbor + SDK + tournament + grader, synthetic local model server:
ASSAY_DOCKER_TEST=1 uv run --frozen --project benchmark \
  python -m unittest benchmark.tests.test_scaling.DockerPipelineTests \
  benchmark.tests.test_dependencies.DockerToolsTests -v

npm run benchmark:run                       # Older generation-only plan
npm run benchmark:run -- --execute          # No tournament; paid independent attempts
```

The Docker fixture deliberately makes vanilla fail and selection pass. This checks
wiring, isolation, selection ordering, partial-budget grading and separate baseline
credentials; **it is not measured model-quality uplift**. No large live scaling
study has yet established an Assay quality or cost advantage.

**Live validation, September 24, 2026:** four real DeepSeek candidates on
`regex-log`; vanilla and all four N/repetition configurations passed official
Harbor grading, with no missing/suspect grades. The tournament used 162 scoring
calls (plus a capability probe), and every original container was graded after
the selection lock. This verifies the workflow, not uplift: vanilla also passed.
Evidence is retained privately in `benchmark/runs/verification-scaling-smoke/`.
The ten-task three-model study plans 40 DeepSeek generations plus 10 external
baseline generations. Its September 24 attempt was stopped on September 25 after
lid-closure sleep, verifier-format/alignment errors, and grading-tool download
failures. Both vanilla models passed its first task; no trustworthy scaled outcome
was obtained there. Original errors and partial candidates are retained.

**Parallel validation, September 25, 2026:** a real Docker/Harbor test exercised
20 isolated containers and 16 simultaneous synthetic generator requests, preserving
each task's decision lock before grading. A seeded-tool test graded a synthetic
case offline after a simulated installer failure. The scorer's v2 protocol uses a
concise assessment, mandatory-verdict reminders, and byte-aware UTF-8 alignment;
missing/malformed verdicts still fail without retries or fabricated scores.
A live DeepSeek synthetic probe completed all 16 then all 32 concurrent judge calls
(48 calls total, no benchmark data), in about 8.5 and 24.6 seconds respectively.
The larger batch was not proportionally faster; these checks establish operation,
not benchmark uplift or provider throughput guarantees.
