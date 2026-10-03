# Experiments — 2026-10-03

Daily read-and-test run against the arXiv cs.AI window **2026-09-30 → 2026-10-03**
(200 newest papers pulled; `submittedDate:` range queries returned HTTP 500
today, so the window was filtered locally by Atom `published` date — the same
fallback `arxiv_gem_daily.py` uses).

Scored with the digest's own criteria (AGENT = local agent-harness value,
TUNE = small-model fine-tune value on a free Colab T4).

## Picks

| arXiv | Title | AGENT | TUNE | Test |
|---|---|---|---|---|
| 2610.02001 | Mingbird: A Local-First Agent Harness Enabling Small Open Models to Complete Real Tasks | 5/5 | 2/5 | ✅ `test_mingbird.py` — the 3 representative harness mechanisms reproduced on CPU |
| 2610.01493 | No Model Required: Text Entropy Rate Filtering Mitigates Iterative Fine-Tuning Collapse | 2/5 | 5/5 | ✅ `test_entropy.py` — Kontoyiannis h_k estimator + collapse simulation + filtering, all CPU |
| 2610.01509 | Sharpening Tax in Post-Training | 4/5 | 3/5 | ⏳ queued — needs base vs post-trained model rollouts; no GPU/models on this VM. Candidate for the Colab-T4 track |
| 2610.02140 | Finetuning with Sampling: SFT Learns Better Than You Think | 2/5 | 5/5 | ⏳ queued — needs reference-model likelihoods + SFT loop; Colab-T4 track |
| 2610.02039 | CARM: Cancellation-Aware Response Masking for LLM RL | 1/5 | 4/5 | ⏳ queued — the masking rule itself is CPU-testable (synthetic log-ratios); next run |

Also screened (AGENT/TUNE 3–4, not tested today): 2610.02002 Mem++
(non-destructive agent memory), 2610.01618 Agents Are Systems Not Models
(agentic eval), 2610.01787 component routing for self-improving GUI agents,
2610.01896 GMC-GRPO async post-training, 2610.02015 language drift in RLVR.

## What ran today

- `python3 test_mingbird.py` — **ALL CHECKS PASSED**
- `python3 test_entropy.py` — **ALL CHECKS PASSED**

See `RESULTS.md` for measured numbers, verdicts, and honest limitations.

## Files

| File | Purpose |
|---|---|
| `mingbird_mechanisms.py` | PrefillBudget, FinishGate, LoopDetector |
| `test_mingbird.py` | Toy env + scripted failure-mode policies + assertions |
| `entropy_rate_filter.py` | Kontoyiannis h_k via word-level match-length statistics (stdlib only) |
| `test_entropy.py` | Calibration, 6-generation collapse sim, filtering vs baselines |
| `RESULTS.md` | Measured results and verdicts |
