# RESULTS — 2026-10-03 (run 2)

Environment: this VM (no GPU, no torch/transformers — stdlib + numpy only).
All suites are deterministic (seeded) and run in seconds.

---

## Test 1 — Bayesian Dialectical Argumentation (arXiv:2610.02005, AGENT 5/5)

Reproduced the paper's core mechanism: council deliberation as typed moves
(propose / challenge / concede) observed through a classical annotator model
with per-agent reliabilities, EM inference, reliability-weighted posteriors.

**Scenario A — 3-of-8 agents collude on one fixed wrong answer (K=3):**

| Method | Accuracy | Brier (calibration) |
|---|---|---|
| BDA | **0.953** | **0.047** |
| Majority vote (agreement fraction as confidence) | 0.740 | 0.180 |

Per-agent reliability recovery (true → estimated):
`[0.85, 0.80, 0.78, 0.72, 0.68] → [0.83, 0.76, 0.76, 0.70, 0.68]`;
adversaries `[0.15, 0.12, 0.20] → [0.24, 0.22, 0.27]` — all estimated below
0.5, i.e. **inverted rather than merely outvoted** (correlation true-vs-est
0.999).

**Scenario B — clean council (no adversaries):** BDA 0.983 vs vote 0.927
accuracy; Brier 0.017 vs 0.108.

**Verdict: SUPPORTED.** Better calibration than the zero-cost baseline in
both settings, robust under a persistent adversarial coalition, competitive
in the clean setting, and unreliable agents are identified and inverted —
the paper's three headline claims at mechanism level.

---

## Test 2 — Causal Memory Policy (arXiv:2610.02070, AGENT 5/5)

Synthetic memory store (60 memories, 40 queries, 20 required); 55% of required
memories suppressed to never-retrieve (paper: 54% on LongMemEval).

| Estimator | never-retrieved required (truth 0.144) | AUC required vs non-required |
|---|---|---|
| Naive store-level intervention | **0.000** (positivity violation) | 0.450 |
| CMP (retrieval intervention + SNIPW) | **0.131** (≈ unbiased) | **0.818** |

Other groups: retrieved-required truth 0.146 → naive 0.487 (overestimates
when retrieved because removal also shifts the remaining ranking);
non-required truth 0.026 → naive 0.026, CMP 0.023.

**Verdict: SUPPORTED.** The positivity violation is exact (naive attributes
precisely zero utility to never-retrieved memories), CMP restores
identification (bias 0.144→0.013), and discrimination improves 0.450→0.818
(paper: 0.54→0.66).

**Honest limitations:**
- Task performance is linear (sum of utilities), so the true ATE is known
  exactly — real agent-task performance is non-linear and noisy, which is
  why the paper needs the self-normalized variant and variance analysis.
- The paper's further claim — that identified per-query utility does not
  aggregate to predict value on unseen queries — is not tested here.

---

## Test 3 — FERPO policy fitting (arXiv:2610.02198, TUNE 5/5)

Target = 3-mode Gaussian mixture (entropy/KL-regularized improvement target);
rollout = broad Gaussian; actor = 2-component mixture (capacity below modes).

**Claim A — KL limit keeps SNIS weights well behaved** (ESS of importance
weights vs target temperature α):

| α (higher = target closer to rollout) | 0.25 | 0.50 | 1.00 | 2.00 |
|---|---|---|---|---|
| ESS (% of 200k samples) | 6.8% | 14.2% | 28.1% | 39.9% |

Monotone and substantial (≈6×). **SUPPORTED.**

**Claim B — forward-KL covers modes, reverse-KL favors a subset:**

| Fit (K=2) | components | modes covered |
|---|---|---|
| forward-KL (weighted EM on SNIS) | μ=[−3.03, 1.47], one wide σ=1.92 | **2/3** |
| reverse-KL (MC Adam) | both μ≈−3 (dominant mode) | **1/3** |

**Verdict: SUPPORTED.** With capacity below the number of modes, reverse-KL
collapses onto the dominant mode while forward-KL spreads mass across
multiple high-value modes.

**Honest limitations:**
- E[log target] is *higher* for the reverse-KL fit (−0.41 vs −1.53): sitting
  on the tallest mode maximizes the static objective. The paper's argument
  for forward-KL is exploration across training, which a single-step fit
  cannot demonstrate — coverage, not return, is the tested quantity.
- MuJoCo/ManiSkill sample-efficiency claims need real RL loops (GPU track).

---

## Fixes / repo notes

- **No fix needed in the digest pipeline today.** The AGENTS.md-documented
  quirk is live again: `submittedDate:[...TO...]` range queries returned
  HTTP 500; newest-first pull + local Atom `published` filtering works.
- One quirk of this run's own setup: `submittedDate` sorts by *submission*
  while arXiv announces 1–2 days later, so all 200 pulled papers fell inside
  the 3-day window (no older stragglers to drop).

## Queued for the GPU track (Colab T4 / GitHub Action)

- 2610.02092 TESS (meta-network for data selection) — needs LLM training loops.
- 2610.01026 FloWright (workflow co-evolution) — needs trained workflow agents.
- 2610.01382 Gacha Decoding — needs an instruction-following LM (the core
  "planning with dice" mechanism is LM-bound, not simulable on CPU).
- 2610.02200 VISTA, 2610.02204 RPG — need multimodal models / robot simulators.
- 2610.01471 cross-model review — needs multiple reviewer models.

## Slated for a future CPU run

- 2610.01349 PACE — the provenance/capability mediation rule is testable on
  synthetic tool-call graphs (poisoned metadata steering vs enforcement).
- 2610.01256 DeFA — dependency-graph failure attribution testable on synthetic
  agent trajectories (DAG + planted decisive errors).
