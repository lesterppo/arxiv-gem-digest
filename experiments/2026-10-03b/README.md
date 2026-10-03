# Experiments — 2026-10-03 (run 2)

Second same-day manual run of the read-and-test task. The first run
(`experiments/2026-10-03/`) covered the window's earliest picks; this run
screens the **remaining 190 papers** from the same 3-day window
(**2026-09-30 → 2026-10-03**, 200 newest cs.AI pulled via the arXiv API;
`submittedDate:` range queries returned HTTP 500 again today, so the window
was filtered locally by Atom `published` date — the repo's documented
fallback) and tests a fresh set of picks. Deduped against run 1's picks
(2610.02001, 2610.01493, 2610.01509, 2610.02140, 2610.02039).

Scored with the digest's own criteria (AGENT = local agent-harness value,
TUNE = small-model fine-tune value on a free Colab T4).

## Picks

| arXiv | Title | AGENT | TUNE | Test |
|---|---|---|---|---|
| 2610.02005 | Counting Moves, Weighing Voices: Bayesian Dialectical Argumentation for Calibrated Multi-LLM Councils under Persistent Adversaries | 5/5 | 2/5 | ✅ `test_bda.py` — typed-move annotator model + EM on synthetic councils |
| 2610.02070 | Causal Memory Policy: Making Memory Utility Identifiable by Intervening on Retrieval | 5/5 | 1/5 | ✅ `test_cmp.py` — positivity violation + SNIPW retrieval intervention |
| 2610.02198 | FERPO: Forward Entropy-Regularized Policy Optimization | 1/5 | 5/5 | ✅ `test_ferpo.py` — SNIS weight health + forward vs reverse-KL mode coverage |

Also screened (AGENT/TUNE 3–4, not tested today): 2610.01349 PACE
(provenance-aware capability enforcement for tool agents), 2610.01256 DeFA
(dependency-guided failure attribution), 2610.01415 PoS (explicit belief
states for long-horizon agents), 2610.01045 Empty Commitments, 2610.01042 ICR
(multi-agent communication audit), 2610.01140 ReSolve (candidate-reasoning
reuse), 2610.01382 Gacha Decoding (diversity via instruction following),
2610.01766 VideoEvolve (evolving agent harnesses).

## What ran today

- `python3 test_bda.py` — **ALL CHECKS PASSED**
- `python3 test_cmp.py` — **ALL CHECKS PASSED**
- `python3 test_ferpo.py` — **ALL CHECKS PASSED**

See `RESULTS.md` for measured numbers, verdicts, and honest limitations.

## Files

| File | Purpose |
|---|---|
| `bda_council.py` | Typed-move annotator model, EM inference, council simulator |
| `test_bda.py` | Adversarial-coalition + clean-council scenarios vs majority vote |
| `cmp_memory.py` | Synthetic memory store, naive store-intervention, CMP balanced design |
| `test_cmp.py` | Positivity-violation demo, bias check, AUC comparison |
| `ferpo_modes.py` | Regularized target, SNIS, forward-KL (weighted EM), reverse-KL (MC Adam) |
| `test_ferpo.py` | ESS-vs-temperature curve, mode-coverage comparison |
| `RESULTS.md` | Measured results and verdicts |
