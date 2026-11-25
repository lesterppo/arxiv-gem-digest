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

Overflow beyond the 22-paper batch is deferred to the next run — but only
inside the 2-day window (capped at 200 newest). Out-of-window overflow is
dropped: newest-first ordering means it would never be picked, and carrying
it forever grew the deferred file to 11 MB by 2026-11-25.
```

Reasons a paper is *not* recommended are respected: a method that needs a
GPU cluster is flagged TUNE-low rather than forced into a T4 recipe.

## Staying alive (GitHub's 60-day rule)

GitHub disables scheduled workflows after 60 days without a push to the
repo — the daily runs alone do not count, which is why this workflow was
warned on 2026-11-25 after the last push (2026-10-03). Two mechanisms
keep it alive, both added 2026-11-25:

- **State commit**: every successful run commits `state/seen_ids.json` +
  `state/deferred_ids.json` to the repo (force-added — the files stay
  gitignored locally so a dev machine never commits them by accident).
  Each run is therefore its own keepalive push. The commit is gated on
  the digest step succeeding: if the email fails, the script exits
  non-zero and the state is NOT committed, so those papers are retried
  next run instead of being marked seen-and-lost.
- **Keepalive workflow** (`.github/workflows/keepalive.yml`): checks
  weekly (Sundays 03:00 UTC) and pushes a heartbeat commit if the repo
  has had no push in the last 7 weeks (49 days) — guaranteeing a push at
  least once per 8-week window, with a week of margin before GitHub's
  60-day cutoff. Backstop only; the daily state commit normally fires
  first.

## Files

| File | Purpose |
|------|---------|
| `arxiv_gem_daily.py` | The whole pipeline (stdlib + gemini-webapi) |
| `gemini.py` | gemini-webapi CLI backend (cookie auth) |
| `urllib_session.py` | HTTP session helper for gemini.py |
| `.github/workflows/daily.yml` | Daily 02:00 UTC cron + manual dispatch |
| `.github/workflows/keepalive.yml` | Weekly self re-enable push (8-week window) |
| `test_review_fixes.py` | Offline tests for the 2026-11-25 review fixes |
| `requirements.txt` | Python deps (exact pins — see the comment there) |

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

A dry run (`dry_run: 1`) exercises the whole pipeline but sends no email —
and, since 2026-11-25, it does NOT mark papers seen either, so the next
real run still reports them. An email failure now fails the run (non-zero
exit) instead of silently reporting success.

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
