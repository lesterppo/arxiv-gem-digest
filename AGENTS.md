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
- **Fetch resilience (two tiers)**: the API call runs a retry ladder
  (429/5xx + transient socket errors, exponential backoff, honours
  `Retry-After`) across two hosts (`export.arxiv.org` → `arxiv.org`). If the
  API is still unusable — GitHub runners share egress IPs and get IP-level
  429s — the run falls back to the arXiv **RSS feed**
  (`rss.arxiv.org/rss/cs.AI`, a different service) and tags the email subject
  `(RSS fallback)`. This also covers the API's index lag: the API windows on
  *submission* time while arXiv announces 1-2 days later, so a 0-paper API
  window triggers the same RSS cross-check instead of a silent "nothing new".
- **States**: only `state/seen_ids.json` is stateful (gitignored); everything
  else is idempotent.
- **Local dev run** uses `~/.gemini-cli/auth.json`; CI writes it from secrets.
- **Cookie refresh**: run `gemini.py --init` (or re-login Firefox) then
  `python3 refresh_gh_secrets.py owner/repo`.
- Keep `gemini.py` + `urllib_session.py` byte-identical to the upstream
  `lesterppo/hermes-gem-cli` copy so the `-p`/`--thinking extended` contract
  stays valid.

Sensitive values live only as GitHub secrets; never commit `SMTP_PASS`,
`GEMINI_SID`, `GEMINI_TS`, real cookies, or `state/seen_ids.json`.
