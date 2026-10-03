# arXiv cs.AI Daily Digest — Gemini

Daily GitHub Actions cron job that screens the latest **arXiv cs.AI** papers,
summarizes the relevant ones and emails **recommendations**, using **Gemini
Flash + Extended Thinking** (gemini-webapi, zero API key — browser-cookie
auth) as the reader.

Recommendation criteria scored per paper (1–5 each):
- **AGENT** — applicability to building/improving a *local AI-agent harness*
  (tool calling, memory, agent loops, multi-agent orchestration, evals,
  web/software agents, code gen).
- **TUNE** — applicability to *small open-weights model fine-tune/training*
  that fits a single **free Google Colab T4** (16 GB VRAM) or small local GPU.

## How it works

```
fetch cs.AI (arXiv API, 2-day back-window)
        │   arXiv date-range filter unsupported → pull newest 200/page,
        │   drop outside window, normalize id (strip vN)
        ▼
dedup vs state/seen_ids.json   (nothing double-emailed; window overlap
        │                       means nothing missed between runs)
        ▼
Gemini Flash+Extended reads title+abstract batch → per-paper AGENT/TUNE
scores + a RECOMMENDED section with concrete takeaways / minimal experiments
        ▼
HTML email via Gmail SMTP
```

Reasons a paper is *not* recommended are respected: a method that needs a
GPU cluster is flagged TUNE-low rather than forced into a T4 recipe.

## Files

| File | Purpose |
|------|---------|
| `arxiv_gem_daily.py` | The whole pipeline (stdlib + gemini-webapi) |
| `gemini.py` | gemini-webapi CLI backend (cookie auth) |
| `urllib_session.py` | HTTP session helper for gemini.py |
| `.github/workflows/daily.yml` | Daily 02:00 UTC cron + manual dispatch |
| `requirements.txt` | Python deps |

## Setup

1. Create a GitHub repo and add these **secrets**:

| Secret | Value |
|--------|-------|
| `GEMINI_SID` | Firefox `__Secure-1PSID` (Gemini login) |
| `GEMINI_TS`  | Firefox `__Secure-1PSIDTS` |
| `SMTP_USER`  | Gmail sender (e.g. you@gmail.com) |
| `SMTP_PASS`  | Gmail **app password** |
| `RECIPIENT`  | Destination inbox |

2. The daily cron fires the workflow automatically.

### Refreshing Gemini cookies

Google Gemini session cookies (~`__Secure-1PSID`/`__Secure-1PSIDTS`) expire
roughly monthly. Refresh them locally from a signed-in Firefox and push to
the GitHub secrets with:

```bash
python3 refresh_gh_secrets.py    # reads ~/.gemini-cli/auth.json -> GH secrets
```

(or run `gemini.py --init` to write fresh cookies, then re-run it).

## Manual run

```bash
# local (uses ~/.gemini-cli/auth.json cookies)
python3 arxiv_gem_daily.py

# override window / candidate cap
RUN_BACK_DAYS=5 MAX_CANDIDATES=30 python3 arxiv_gem_daily.py
```

On GitHub: **Actions → arXiv cs.AI Daily Digest → Run workflow** (optional
`back_days`, `max_candidates` inputs).

## Token / cost profile

- arXiv fetch: 1–2 HTTP calls (stdlib).
- Gemini: one Flash+Extended call per run (batched candidates, `MAX 22`
  papers = ~1 longer prompt). Free gemini-webapi quota is ample for a daily run.
- Email: one Gmail SMTP send.

## Read-and-test experiments

A daily agent job reads the latest cs.AI window with this digest's AGENT/TUNE
criteria, picks the most relevant **testable** papers, and reproduces their
core claims as small CPU-runnable experiments (GPU-bound ones are queued for
the Colab-T4 track). Results land in `experiments/YYYY-MM-DD/` with measured
numbers, verdicts, and honest limitations.

- `experiments/2026-10-03/` — Mingbird harness mechanisms (prefill budget,
  finish gate, loop detection — all checks passed) and the Kontoyiannis
  entropy-rate filter from "No Model Required" (calibrated vs known rates,
  tracks a 6-generation collapse sim, +19% pooled unique trigrams vs random).
- `experiments/2026-10-03b/` — second same-day run: BDA council calibration
  under adversarial coalitions (Brier 0.047 vs 0.180, adversaries inverted),
  Causal Memory Policy retrieval intervention (AUC 0.45→0.82, positivity
  violation exactly reproduced), FERPO forward-KL mode coverage (2/3 vs 1/3
  modes) + SNIS weight health vs KL limit. All checks passed.
