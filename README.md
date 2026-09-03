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
fetch cs.AI (arXiv API, ~3-day back-window)
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
