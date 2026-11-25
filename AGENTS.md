# arxiv-gem-digest

Repository for the **arXiv cs.AI Gemini daily digest** — a scheduled GitHub
Actions job (02:00 UTC) that reads the latest arXiv cs.AI papers, uses Gemini
Flash + Extended Thinking to summarize and recommend papers relevant to
(A) local AI-agent harness engineering and (B) small-model fine-tune/training
on a free Google Colab T4, then emails an HTML report.

## For AI coding agents

If you are an AI agent told to modify or operate this repo, read the
`README.md` first for architecture and setup. Key operational facts:

- **Backend model**: Gemini Flash + Extended Thinking via `gemini.py`
  (gemini-webapi, browser-cookie auth — NO API key). Cookies come from the
  `GEMINI_SID` / `GEMINI_TS` GitHub secrets, written to `~/.gemini-cli/auth.json`.
- **No-miss guarantee**: `arxiv_gem_daily.py` fetches a `RUN_BACK_DAYS=2`
  window so a skipped/late daily run never drops a paper, and dedups on
  normalized arXiv id via `state/seen_ids.json` against double-reporting.
- **arXiv API quirk**: `submittedDate:[...TO...]` range filters return HTTP
  500 on the plain cat query — the script instead pulls newest pages then
  filters locally by the Atom `published` date. Don't reintroduce a date-range
  search_query.
- **Fetch resilience (three tiers)**: the API call runs a retry ladder
  (429/5xx + transient socket errors, exponential backoff, honours
  `Retry-After`) across two hosts (`export.arxiv.org` → `arxiv.org`). Sources
  are then tried in order — `api` → `rss` → `oai` — and the first that yields
  in-window papers wins; the email subject is tagged with the fallback that
  served it (e.g. `(OAI fallback)`).
  - `rss` = `rss.arxiv.org/rss/cs.AI`. Announcement-fresh, but legitimately
    **empty on Sat/Sun** (arXiv skips those announcement days).
  - `oai` = `oaipmh.arxiv.org/oai` (`set=cs:cs`, `metadataPrefix=arXiv`),
    full abstracts, announcement-dated, available at weekends. Records whose
    `created` date is >30 days old are metadata churn and are dropped.
  The API windows on *submission* time while arXiv announces 1-2 days later,
  so a 0-paper API window falls through to the next source instead of
  reporting a silent "nothing new".
- **Never fail silently**: if no source yields papers *and* any source
  errored, the run sends a WARN email and exits 2 — a green run always means
  a digest was really produced (or the window was genuinely empty everywhere).
- **Verification switches**: `ARXIV_FETCH_SOURCE=api|rss|oai` pins the first
  source tried, `DIGEST_DRY_RUN=1` exercises everything but suppresses email.
  Workflow inputs `fetch_source` / `dry_run` map to them, so a tier can be
  proven on CI without emailing.
- **Dedup state cache**: the `actions/cache` key must stay ROLLING
  (`arxiv-seen-${{ github.run_id }}` + `restore-keys: arxiv-seen-`). A fixed
  key always hits, a hit skips the save, and the dedup state then freezes —
  every run re-reports the whole window.
- **States**: only `state/seen_ids.json` (+ `state/deferred_ids.json`) is
  stateful. Both stay gitignored locally, but CI force-commits them after
  every successful run — the commit is what resets GitHub's 60-day
  scheduled-workflow clock (see README "Staying alive"). The cache is the
  fast path; the committed copy is the durable fallback if the cache is
  evicted. Never commit them by hand from a dev machine.
- **Deferred overflow is window-bounded**: `state/deferred_ids.json` keeps
  only un-analysed papers inside the current 2-day window, capped at 200
  newest. It used to grow without bound (11 MB, ~6,000 papers by
  2026-11-25) because newest-first ordering never drains it — and the
  'screened' count in the email was inflated by that stale backlog
  (6,131 on the 2026-11-25 dry run). Out-of-window overflow is dropped,
  not carried forever.
- **Email honesty**: `send_email` returns False on failure and `main()`
  exits 1 — a green run always means a digest was really delivered. Seen
  is marked ONLY after a confirmed send, and never in dry-run mode.
- **Local dev run** uses `~/.gemini-cli/auth.json`; CI writes it from secrets.
- **Cookie refresh**: run `gemini.py --init` (or re-login Firefox) then
  `python3 refresh_gh_secrets.py owner/repo`.
- Keep `gemini.py` + `urllib_session.py` byte-identical to the upstream
  `lesterppo/hermes-gem-cli` copy so the `-p`/`--thinking extended` contract
  stays valid.

Sensitive values live only as GitHub secrets; never commit `SMTP_PASS`,
`GEMINI_SID`, `GEMINI_TS`, real cookies, or `state/seen_ids.json`.
