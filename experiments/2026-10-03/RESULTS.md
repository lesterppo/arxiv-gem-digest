# RESULTS — 2026-10-03

Environment: this VM (no GPU, no torch/transformers — stdlib + numpy only).
Both suites are deterministic (seeded) and run in seconds.

---

## Test 1 — Mingbird harness mechanisms (arXiv:2610.02001, AGENT 5/5)

Reproduced the paper's three representative mechanisms; each tested against a
scripted policy exhibiting the failure form the mechanism targets (harness
varied, policy fixed — the paper's controlled-comparison design, toy scale).

| Mechanism vs failure form | OFF (naive harness) | ON (mechanism) |
|---|---|---|
| Byte-level net-zero prefill budget vs tool-prefill context overflow | prefill **6660 B** — overflows the 4096 B toy context | prefill **600 B** — exactly at budget |
| Finish gate (re-reads task) vs silent abandonment | premature "done" **accepted**, artifact score **0.50** | premature "done" **rejected**, missing criterion named (`status_artifact`) |
| Signature-level loop detection vs tool-demonstration loops | **60/60** steps burned | loop broken after **2** executed steps |

**Verdict: SUPPORTED.** Each mechanism fixes its targeted failure form in the
toy environment. This is a mechanism-level reproduction, not a replication of
the paper's LRAB/τ²-bench numbers (which need real small models + tasks).

---

## Test 2 — Kontoyiannis entropy-rate filter (arXiv:2610.01493, TUNE 5/5)

Implemented h_k = 1 / mean(L_i / log2(i+1)) from word-level match-length
statistics, pure Python (C-speed substring search, ~2.5 s for the whole suite).

**2a — calibration on known-entropy sources:**

| Source (truth) | measured h_k |
|---|---|
| Bernoulli p=0.5 (1.000 bits/word) | **0.982** |
| Bernoulli p=0.1 (0.469 bits/word) | **0.401** |
| Periodic, period 2 / 50 (~0) | **0.055 / 0.056** |

Monotone in true entropy and within ±0.1 of ground truth. **SUPPORTED.**

**2b — six-generation collapse simulation** (each generation resampled from a
temperature-sharpened copy of the last, mimicking synthetic-data fine-tuning):

| gen | T | h_k | unique trigrams | repetition |
|---|---|---|---|---|
| 0 | 1.00 | 6.483 | 2653 | 0.115 |
| 1 | 0.85 | 4.498 | 1949 | 0.350 |
| 2 | 0.70 | 1.786 | 521 | 0.826 |
| 3 | 0.55 | 0.340 | 35 | 0.988 |
| 4 | 0.40 | 0.056 | 4 | 0.999 |
| 5 | 0.25 | 0.054 | 1 | 1.000 |

h_k declines monotonically, tracking diversity collapse. **SUPPORTED.**

**2c — filtering a 40-doc pool (20 diverse + 20 phrase-collapsed), pooled-corpus metrics:**

| Filter (keep 50%) | diverse kept | pooled uniq. trigrams | pooled repetition |
|---|---|---|---|
| h_k top-50% | 20/20 | 9534 | 0.205 |
| random 50% | 9/20 | 8011 | 0.332 |
| length top-50% | 20/20 | 9539 | 0.205 |
| unique-unigram top-50% | 20/20 | 9533 | 0.205 |

h_k vs random: **+19% pooled unique trigrams, −38% pooled repetition.**
**SUPPORTED** as a model-free filter.

**Honest limitations:**
- On this synthetic mixture, cheap heuristics (unigram-vocab count) separate
  the classes about as well as h_k. The paper's headline edge is specifically
  vs **logprob-based** filtering (which showed no significant benefit,
  p > 0.23) — reproducing that comparison needs model access this VM lacks.
- The paper's real validation is a six-generation **QLoRA run on
  Llama-3.1-8B**; ours is a temperature-sharpening simulation. Same signature
  (h_k tracks collapse), different substrate.
- What IS validated here, and is the paper's core practical claim: a
  **model-free** entropy-rate proxy that is calibrated (2a), tracks collapse
  (2b), and filters for diversity (2c) — in ~2.5 s of CPU.

---

## Fixes / repo notes

- **No fix needed in the digest pipeline today.** The AGENTS.md-documented
  quirk is live: `submittedDate:[...TO...]` range queries on
  `export.arxiv.org`/`arxiv.org` returned HTTP 500 on all retries/hosts
  (2026-10-03); the script's newest-first + local date filtering path works
  fine. Verified the fallback the repo relies on.
- Fixed one bug in today's own test code (missing `score` key on the
  gate-rejection path in `test_mingbird.py`) — caught by the run itself.

## Queued for the GPU track (Colab T4 / GitHub Action)

- 2610.01509 Sharpening Tax — needs base vs post-trained rollouts to measure
  pass@1 vs pass@K coverage.
- 2610.02140 Finetuning with Sampling — needs reference-model likelihoods +
  an SFT loop for the MCMC data transform.
- 2610.02039 CARM — the masking rule is CPU-testable on synthetic log-ratios;
  slated for the next CPU run.
