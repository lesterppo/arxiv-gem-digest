#!/usr/bin/env python3
"""
arXiv cs.AI Daily Digest — fetch, summarize, recommend, email.

Pipeline:
  1. Fetch recent cs.AI papers via arXiv API (windowed on last N days,
     deduped against a state file so no paper is double-reported or missed
     between two runs).
  2. Batch paper title+abstracts to Gemini Flash + Extended Thinking
     (gemini-webapi, no API key — cookie auth) -> markdown digest that
     summarizes each candidate and recommends ones applicable to:
       (A) local AI-agent harness engineering
       (B) small-model fine-tune/training on a free Google Colab T4.
  3. Email the HTML report via Gmail SMTP.

Any mis-run gaps the previous window: every run covers [now - run_back_days],
so overlapping sleeps never drop a paper.

Auth (Gemini): GEMINI_SID / GEMINI_TS env vars (GitHub secrets) written to
~/.gemini-cli/auth.json by the CI workflow.

Environment secrets expected on GitHub Action:
  GEMINI_SID, GEMINI_TS, SMTP_USER, SMTP_PASS, RECIPIENT
"""

import json
import os
import re
import smtplib
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Shared infographic generator (also used by yt-gem-daily). Import works when
# the script lives in ~/.hermes/scripts/ or in the repo checkout.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
for _cand in (os.path.expanduser("~/.hermes/scripts"),):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
try:
    import digest_infographic as infographic
except ImportError:
    infographic = None  # infographics degrade gracefully

# ── Configuration ──────────────────────────────────────────────────────────

ARXIV_API_HOSTS = ("export.arxiv.org", "arxiv.org")
ARXIV_API_PATH = "/api/query"
ARXIV_RSS = "https://rss.arxiv.org/rss/{cat}"
ARXIV_PAGE_SLEEP = 3.0            # arXiv etiquette: >=3s between API calls
ARXIV_HTTP_TRIES = 4              # retry ladder per URL (429/5xx/transient)
ARXIV_UA = os.environ.get("ARXIV_UA") or (
    "arxiv-gem-digest/1.0 (+https://github.com/lesterppo/arxiv-gem-digest)")
CATEGORY = "cs.AI"
QUERY_TERM = f'cat:{CATEGORY} AND submittedDate:[{{start}}T000000 TO {{end}}T235959]'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "state", "seen_ids.json")
GEMINI_SCRIPT = os.path.join(SCRIPT_DIR, "gemini.py")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = os.environ.get("SMTP_USER", "")   # sender email (GitHub secret / env)
RECIPIENT = os.environ.get("RECIPIENT", SMTP_USER)

# Timing
RUN_BACK_DAYS = int(os.environ.get("RUN_BACK_DAYS", "2"))  # 2-day screen window
MAX_ABSTRACT_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "22"))
ABSTRACT_CHARS = 1200            # arXiv abstract char budget sent to Gemini per paper
GEMINI_MODEL = "flash"           # Gemini Flash
GEMINI_THINKING = "extended"     # Extended-thinking tier
GEMINI_TIMEOUT = 280             # s per batch call
MAX_CONCURRENT_GEM = 3
GEMINI_RETRIES = 2
TOTAL_TIMEOUT = 1200             # script hard timeout (CI timeout-minutes 25)

# ── arXiv API ──────────────────────────────────────────────────────────────

NS = {"a": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}


def _retry_after(err) -> float | None:
    """Honour a server-supplied Retry-After (seconds form) when present."""
    try:
        ra = err.headers.get("Retry-After") if err.headers else None
        if ra:
            return max(1.0, min(60.0, float(str(ra).strip())))
    except Exception:  # noqa: BLE001
        pass
    return None


def _http_get(url: str, timeout: int = 45, tries: int = ARXIV_HTTP_TRIES,
              label: str = "arxiv") -> bytes:
    """GET with a retry ladder. arXiv rate-limits by IP (HTTP 429) — GitHub
    runners share egress IPs, so a single unlucky request must not kill the
    whole digest run. Retries 429/5xx and transient socket errors with
    exponential backoff, honouring Retry-After when the server sends it."""
    last = None
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={
            "User-Agent": ARXIV_UA,
            "Accept": "*/*",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:  # noqa: PERF203
            last = e
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            wait = _retry_after(e) or min(60.0, 5.0 * (3 ** attempt))  # 5/15/45s
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            if attempt == tries - 1:
                raise
            wait = min(60.0, 4.0 * (2 ** attempt))  # 4/8/16s
        log(f"{label}: transient failure ({last}) — retry "
            f"{attempt + 1}/{tries - 1} in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"{label}: exhausted {tries} attempts ({last})")


def _api_url(host: str, start_page: int) -> str:
    return (f"https://{host}{ARXIV_API_PATH}?search_query=" +
            urllib.parse.quote(f"cat:{CATEGORY}") +
            f"&start={start_page}&max_results=200" +
            "&sortBy=submittedDate&sortOrder=descending")


def _api_page(start_page: int):
    """One API page, trying each arXiv host before giving up."""
    errs = []
    for host in ARXIV_API_HOSTS:
        try:
            return ET.fromstring(
                _http_get(_api_url(host, start_page),
                          label=f"arxiv-api[{host}]"))
        except Exception as e:  # noqa: BLE001
            errs.append(f"{host}: {e}")
            log(f"arXiv API host {host} failed: {e}")
    raise RuntimeError("all arXiv API hosts failed — " + "; ".join(errs))


def fetch_papers(start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Fetch recent cs.AI papers via the arXiv API, newest-first, then filter
    to those actually published (submitted) within [start_dt, end_dt].

    arXiv API does not support a submittedDate range filter on the plain
    cat query (returns HTTP 500), so we pull `max_results` newest by date and
    drop anything older than the window. ~160-170 papers land per day on
    cs.AI, so max_results covers several days of submissions."""
    papers = []
    start_page = 0
    while True:
        try:
            root = _api_page(start_page)
        except Exception:
            if start_page == 0:
                raise  # nothing usable -> caller falls back to RSS
            log(f"arXiv page {start_page} unavailable — using {len(papers)} "
                "papers already collected")
            break
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        hit_old = False
        for entry in entries:
            aid = entry.find("a:id", NS).text
            aid = aid.rstrip("/").split("/abs/")[-1]
            aid = re.sub(r"v\d+$", "", aid)  # normalize away version suffix
            published = _dt(entry.findtext("a:published", "", NS))
            if published is None or published < start_dt:
                hit_old = True
                continue
            primary = entry.find("arxiv:primary_category", NS)
            if primary is not None:
                pcat = primary.get("term")
                if pcat and pcat != CATEGORY:
                    continue
            authors = [a.findtext("a:name", "", NS)
                       for a in entry.findall("a:author", NS)]
            papers.append({
                "id": aid,
                "abs_url": f"https://arxiv.org/abs/{aid}",
                "title": _clean(entry.findtext("a:title", "", NS)),
                "authors": authors,
                "summary": _clean(entry.findtext("a:summary", "", NS)),
                "published": published,
            })
        if hit_old or len(entries) < 200:
            break
        start_page += 200
        if start_page >= 800:  # safety against pathological pagination
            break
        time.sleep(ARXIV_PAGE_SLEEP)  # be polite between pages
    return papers


def fetch_papers_rss(start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Fallback source: the arXiv RSS feed (rss.arxiv.org), which is a
    different service from the API and keeps working when the API is
    IP-rate-limited. Same output shape as fetch_papers()."""
    raw = _http_get(ARXIV_RSS.format(cat=CATEGORY), label="arxiv-rss")
    root = ET.fromstring(raw)
    items = root.findall(".//item")
    if not items:
        raise RuntimeError("arXiv RSS feed contained no items")
    papers = []
    undated = 0
    for item in items:
        link = (item.findtext("link") or "").strip()
        m = re.search(r"/abs/([^/]+)$", link)
        if not m:
            continue
        aid = re.sub(r"v\d+$", "", m.group(1))
        desc = item.findtext("description") or ""
        desc = _clean(desc)
        # RSS description is "arXiv:NNNN.NNNNNvN Announce Type: new\nAbstract: ..."
        desc = re.sub(r"^arXiv:\S+\s*|^Announce Type:\s*\S+\s*", "", desc)
        desc = re.sub(r"^Abstract:\s*", "", desc)
        pub = None
        try:
            from email.utils import parsedate_to_datetime
            pub = parsedate_to_datetime(item.findtext("pubDate") or "")
        except Exception:  # noqa: BLE001
            pub = None
        if pub is None:
            undated += 1
            pub = end_dt
        creators = item.findtext("{http://purl.org/dc/elements/1.1/}creator") or ""
        announce = item.findtext("{http://arxiv.org/schemas/atom}announce_type") or ""
        papers.append({
            "id": aid,
            "abs_url": f"https://arxiv.org/abs/{aid}",
            "title": _clean(item.findtext("title") or ""),
            "authors": [a.strip() for a in creators.split(",") if a.strip()],
            "summary": desc,
            "published": pub,
            "announce_type": announce.strip(),
        })
    in_window = [p for p in papers if p["published"] >= start_dt]
    if not in_window:
        # Feed is a rolling "latest announcement" list; if its dates fall
        # outside our window keep them anyway rather than dropping the day.
        log(f"WARN arXiv RSS: {len(papers)} items but none inside the window "
            f"— using them unfiltered (no-miss preference)")
        in_window = papers
    log(f"arXiv RSS fallback: {len(in_window)}/{len(papers)} items "
        f"({undated} undated)")
    return in_window


def _dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# Gemini backend ─────────────────────────────────────────────────────────────


def setup_auth_from_env():
    """Write GEMINI_SID/TS env cookies into ~/.gemini-cli/auth.json."""
    sid = os.environ.get("GEMINI_SID")
    ts = os.environ.get("GEMINI_TS")
    if not sid or not ts:
        return  # rely on local auth.json (dev run)
    auth_dir = os.path.expanduser("~/.gemini-cli")
    os.makedirs(auth_dir, exist_ok=True)
    with open(os.path.join(auth_dir, "auth.json"), "w") as f:
        json.dump({"__Secure-1PSID": sid, "__Secure-1PSIDTS": ts}, f)


def gemini_analyze(candidates: list[dict], batch_label: str) -> dict:
    """Send one batch of candidate title+abstract to Gemini Flash Extended.
    Returns dict {ok, text|err, out_file}."""
    prompt = _build_prompt(candidates)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    args = [
        sys.executable, GEMINI_SCRIPT, "-m", GEMINI_MODEL,
        "--thinking", GEMINI_THINKING, "--raw", "--brief",
        "-o", os.path.join(SCRIPT_DIR, "state", f"digest_{batch_label}.md"),
        "-p", prompt,
    ]
    for attempt in range(1, GEMINI_RETRIES + 2):
        try:
            proc = subprocess.run(args, capture_output=True, text=True,
                                  env=env, timeout=GEMINI_TIMEOUT)
            out = proc.stdout.strip()
            try:
                parsed = json.loads(out) if out else {}
            except json.JSONDecodeError:
                parsed = {}  # some fallback modes print md not json
            if parsed.get("ok"):
                f = parsed.get("f")
                if f and os.path.exists(f):
                    with open(f, encoding="utf-8") as fh:
                        return {"ok": True, "text": fh.read(), "out": f}
                return {"ok": True, "text": parsed.get("text", ""), "out": ""}
            err = parsed.get("err") or proc.stderr.strip() or out[:400]
            if attempt <= GEMINI_RETRIES:
                log(f"  attempt {attempt}: gemini err {err} — retry")
                time.sleep(8 * attempt)
                continue
            return {"ok": False, "err": err}
        except subprocess.TimeoutExpired:
            if attempt <= GEMINI_RETRIES:
                log(f"  attempt {attempt}: gemini timeout — retry")
                time.sleep(8 * attempt)
                continue
            return {"ok": False, "err": "TIMEOUT"}
    return {"ok": False, "err": "unknown"}


def _build_prompt(candidates: list[dict]) -> str:
    lines = [
        "You run a daily arXiv screening for an AI engineer building: "
        "(A) a local AI-agent harness (tool calling, memory, agent loops, "
        "multi-agent orchestration, evals, web/software agents, code gen), and "
        "(B) small open-weights model fine-tune/training that fits on a FREE "
        "Google Colab T4 (16GB VRAM) or small local GPU.",
        "",
        f"Below are {len(candidates)} recent {CATEGORY} arXiv papers "
        "(title + author + abstract snippet + arXiv id).",
        "",
        "For EVERY paper output one short scored line in EXACTLY this format "
        "(keep the title SHORT — it must appear on the line so the reader can "
        "judge relevance without clicking):",
        "  arXiv-ID — Short title — AGENT n/5 — TUNE n/5",
        "  - 'AGENT' 1-5: applicability to local AI-agent harness work (5 = directly reusable method/idea)",
        "  - 'TUNE'  1-5: applicability to small-model fine-tune/training on free Colab T4 (5 = practical/implementable there)",
        "  Do NOT give both low unless truly irrelevant.",
        "",
        "Then a 'RECOMMENDED' section, papers with AGENT >=4 OR TUNE >=4, best first. "
        "For each: a 1-3 sentence plain-English summary (what it does, the key idea), "
        "WHY it matters for (A)/(B), and a concrete takeaway or minimal experiment "
        "idea an engineer could try with local tools / a T4.",
        "",
        "Rules: be concrete not generic. A method that needs 100s of GPUs is NOT TUNE-applicable "
        "— say so, or suggest a scaled-down variant only if realistic. No hallucinated claims "
        "about the method. Respect each abstract's actual claims. If two papers overlap, cross-note.",
        "",
        "Papers:",
    ]
    for p in candidates:
        ab = p.get("summary", "")[:ABSTRACT_CHARS]
        pub = p.get("published")
        pub_s = pub.strftime("%Y-%m-%d") if pub else "?"
        lines.append(f"\n### arxiv:{p['id']}  ({pub_s})")
        lines.append(f"Title: {p['title']}")
        if p.get("authors"):
            lines.append("Authors: " + ", ".join(p["authors"][:6]) +
                         (" et al." if len(p["authors"]) > 6 else ""))
        lines.append(f"Abstract: {ab}")
    return "\n".join(lines)


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# State / dedup ──────────────────────────────────────────────────────────────


def _load_seen() -> dict[str, str]:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_seen(seen: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(seen, f)


def _prune_seen(seen: dict[str, str], keep_days: int = 7) -> dict[str, str]:
    cutoff = (datetime.now(timezone.utc) -
              timedelta(days=keep_days)).strftime("%Y-%m-%d")
    return {k: v for k, v in seen.items() if v >= cutoff}


# Email ─────────────────────────────────────────────────────────────────────


def _load_smtp_pass() -> str:
    p = os.environ.get("SMTP_PASS")
    if p:
        return p
    for cand in (os.path.expanduser("~/.hermes/secrets/smtp_app_password.txt"),
                 os.path.join(SCRIPT_DIR, "..", "smtp_pass.txt")):
        try:
            with open(cand) as f:
                return f.read().strip()
        except FileNotFoundError:
            continue
    return ""


def send_email(subject: str, html_body: str,
               image_paths: list[str] | str | None = None,
               cids: list[str] | None = None) -> None:
    """Send HTML email with optional inline CID images (chart, gemini banner)."""
    try:
        with open("/tmp/arxiv_digest_email.html", "w") as f:
            f.write(html_body)
    except Exception:
        pass

    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_paths = image_paths or []
    cids = cids or [f"infographic{i}" for i in range(len(image_paths))]
    pus = SMTP_USER
    pas = _load_smtp_pass()
    if not pas:
        log("ERROR: no SMTP_PASS — cannot send email")
        return
    if image_paths:
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("Digest delivered as HTML — enable HTML view.",
                            "plain", "utf-8"))
        alt.attach(MIMEText(html_body, "html", "utf-8"))
        msg = MIMEMultipart("related")
        msg.attach(alt)
        for i, ip in enumerate(image_paths):
            if not ip or not os.path.exists(ip):
                continue
            with open(ip, "rb") as fh:
                img = MIMEImage(fh.read())
            cid = cids[i] if i < len(cids) else f"infographic{i}"
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline",
                           filename=os.path.basename(ip))
            msg.attach(img)
    else:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(html_body, "html", "utf-8"))
    msg["Subject"] = subject
    msg["From"] = pus
    msg["To"] = RECIPIENT
    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=30)
        server.login(pus, pas)
        server.sendmail(pus, [RECIPIENT], msg.as_string())
        server.quit()
        log(f"Email sent to {RECIPIENT} ({len(html_body)} chars"
            + (f", {len(image_paths)} image(s)" if image_paths else "") + ")")
    except Exception as e:  # noqa: BLE001
        log(f"ERROR sending email: {e}")


# HTML render ─────────────────────────────────────────────────────────────


def md_to_htmlish(text: str) -> str:
    """Markdown -> email-safe HTML: headings (#..####), **bold**, *italic*,
    `code`, bullets (- and *), bare arXiv ids linkified. Escapes HTML first."""
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _inline(s: str) -> str:
        # arXiv ids: bare '2609.02885' first (creates anchors), then the
        # 'arxiv:ID' / 'arxiv ID' prefix form — running prefix-form first
        # leaves the bare id inside the generated href, which the bare pass
        # would then wrap again (nested <a>).
        s = re.sub(r"(?<![\d/.])(\d{4}\.\d{4,5})(?![\d/.])",
                   r'<a href="https://arxiv.org/abs/\1">\1</a>', s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)      # bold
        s = re.sub(r"(?<!\*)\*([^*\s][^*]*)\*(?!\*)", r"<i>\1</i>", s)  # italic
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)      # code
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
        return s

    lines = []
    in_ul = False
    for raw in esc.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if in_ul:
                lines.append("</ul>")
                in_ul = False
            lines.append("<br/>")
            continue
        # headings: #/## -> h3, ###/#### -> h4
        m = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if m:
            if in_ul:
                lines.append("</ul>")
                in_ul = False
            tag = "h3" if len(m.group(1)) <= 2 else "h4"
            lines.append(f"<{tag}>{_inline(m.group(2))}</{tag}>")
            continue
        # bullets: '- ' or '* '
        m = re.match(r"^[-*]\s+(.*)", stripped)
        if m:
            if not in_ul:
                lines.append("<ul>")
                in_ul = True
            lines.append(f"<li>{_inline(m.group(1))}</li>")
            continue
        if in_ul:
            lines.append("</ul>")
            in_ul = False
        # table-ish score lines and plain text
        lines.append(_inline(stripped))
    if in_ul:
        lines.append("</ul>")
    return "<br/>".join(lines)


def render_email_digest(date_label: str, gemini_text: str,
                        infographic_cids: list[str] | None = None,
                        img_sources: list[str] | None = None) -> str:
    css = """
    <style>
      body{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;
           color:#1f2328;line-height:1.55;max-width:760px;margin:0 auto;padding:16px;}
      h1{font-size:22px;border-bottom:2px solid #d0d7de;padding-bottom:8px;}
      h3{color:#116329;margin-bottom:2px;}
      h4{color:#0969da;margin-bottom:2px;}
      li{margin:2px 0;}
      .meta{color:#57606a;font-size:13px;}
      a{color:#0969da;}
      ul{padding-left:20px;margin:4px 0 10px;}
      .infographic{width:100%;border-radius:10px;margin:14px 0 4px;}
      .cap{color:#57606a;font-size:12px;margin-bottom:12px;}
</style>"""
    info_html = ""
    labels = ["Data-viz score chart (rendered locally from digest data: "
              "AGENT = agent-harness applicability, TUNE = Colab-T4 "
              "trainability)",
              "Dashboard infographic generated by Gemini NotebookLM from "
              "the attached paper sources"]
    if infographic_cids:
        info_html = "<h4>📌 Today at a glance</h4>"
        for i, cid in enumerate(infographic_cids):
            cap = labels[i] if i < len(labels) else ""
            info_html += (f"<img class='infographic' src='cid:{cid}' "
                          f"alt='digest infographic {i+1}'/>")
            if cap:
                info_html += f"<div class='cap'>{cap}</div>"
        info_html += "<hr/>"
    return (f"<html><head><meta charset='utf-8'>{css}</head><body>"
            f"<h1>arXiv cs.AI — Gemini Daily Digest</h1>"
            f"<div class='meta'>Generated {date_label}, model: Gemini Flash + "
            f"Extended Thinking (gemini-webapi). "
            f"Criteria: agent-harness applicability & small-model "
            f"fine-tune-on-Colab-T4 applicability.</div><hr/>"
            f"{md_to_htmlish(gemini_text)}"
            f"<hr/>{info_html}"
            f"<div class='meta'>Automated filter: not every arXiv paper is "
            f"agent/fine-tune relevant — scores reflect that. Full listing: "
            f"<a href='https://arxiv.org/list/cs.AI/recent'>arxiv.org/list/cs.AI</a>."
            f"</div></body></html>")


# Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    start_all = time.time()
    log("=== arXiv cs.AI daily digest starting ===")
    setup_auth_from_env()

    now = datetime.now(timezone.utc)
    back = RUN_BACK_DAYS
    start_dt = now - timedelta(days=back)
    date_label = now.strftime("%Y-%m-%d %H:%M UTC")

    # 1. Fetch (API first; RSS fallback when the API IP-rate-limits us)
    source = "api"
    try:
        raw = fetch_papers(start_dt, now)
        log(f"Fetched {len(raw)} raw entries [{start_dt.date()} .. {now.date()}]")
    except Exception as e:  # noqa: BLE001
        log(f"arXiv API fetch failed ({e}) — trying RSS fallback")
        try:
            raw = fetch_papers_rss(start_dt, now)
            source = "rss"
            log(f"Fetched {len(raw)} raw entries via RSS "
                f"[{start_dt.date()} .. {now.date()}]")
        except Exception as e2:  # noqa: BLE001
            log(f"FATAL arXiv fetch failed (api: {e}; rss: {e2})")
            traceback.print_exc()
            send_email(f"[WARN] arXiv cs.AI digest — fetch failed {date_label}",
                       f"<pre>arXiv API error: {e}\n\narXiv RSS error: {e2}\n\n"
                       f"{traceback.format_exc()}</pre>")
            return 2

    if not raw:
        # The API window is measured against *submission* time, but arXiv
        # announces papers a day or two later and the API index lags; a
        # 0-paper API window therefore doesn't mean "nothing new". The RSS
        # feed is announcement-fresh — use it before declaring a quiet day.
        log("arXiv API returned nothing inside the window — checking RSS")
        try:
            raw = fetch_papers_rss(start_dt, now)
            source = "rss"
            log(f"Fetched {len(raw)} raw entries via RSS")
        except Exception as e:  # noqa: BLE001
            log(f"RSS cross-check failed: {e}")

    if not raw:
        log("No entries in window — nothing to do.")
        return 0

    # 2. Dedup (skip already-analysed ids)
    seen = _load_seen()
    unknown = [p for p in raw if p["id"] not in seen]
    if not unknown:
        log("All fetched papers already seen — no new candidates.")
        return 0

    # Sort newest first, cap candidates
    unknown.sort(key=lambda p: p["published"] or start_dt, reverse=True)
    candidates = unknown[:MAX_ABSTRACT_CANDIDATES]
    log(f"{len(unknown)} new papers; analysing top {len(candidates)} by date")

    # 3. Mark candidates seen immediately (so a crash mid-run still blocks
    #    re-dispatch; window overlap already protects, this is extra safety)
    today_key = now.strftime("%Y-%m-%d")
    for c in candidates:
        seen[c["id"]] = today_key
    _save_seen(_prune_seen(seen))

    # 4. Gemini analysis
    batch_label = now.strftime("%Y%m%d-%H%M")
    res = gemini_analyze(candidates, batch_label)
    if not res.get("ok"):
        log(f"Gemini analysis FAILED: {res.get('err')}")
        return 1

    text = res.get("text", "")
    log(f"Gemini digest produced {len(text)} chars")

    # 5. Concluding infographic via Gemini NotebookLM: dedicated notebook per
    #    digest, paper sources attached (ids/titles/scores/summaries),
    #    infographic generated grounded in those sources, CDN image
    #    downloaded. matplotlib chart kept as the immediate primary image.
    img_paths: list[str] = []
    sources: list[str] = []
    if infographic is not None:
        try:
            score_lines = [ln.strip() for ln in text.splitlines()
                           if re.search(r"AGENT\s*\d\s*/\s*5", ln, re.I)]
            n_rec = len(re.findall(r"RECOMMENDED", text, re.I))
            # 5a. matplotlib chart — precise data visualization
            chart = infographic.render_arxiv_chart(
                score_lines, n_recommended=n_rec,
                filedate=now.strftime("%Y-%m-%d"))
            if chart:
                img_paths.append(chart)
                sources.append("matplotlib")
            # 5b. NotebookLM infographic (grounded in attached sources)
            try:
                papers_ctx = infographic.parse_score_lines(score_lines)
                nb_title = f"arXiv cs.AI Daily Digest — {now.strftime('%Y-%m-%d')}"
                nb = infographic.nlm_ensure_notebook(nb_title)
                rec_text = text[text.find("RECOMMENDED"):] if "RECOMMENDED" in text else ""
                n = infographic.nlm_add_text_sources(
                    nb, infographic.nlm_arxiv_sources(
                        papers_ctx[:6], notes=rec_text))
                log(f"NotebookLM: notebook {nb}, {n} sources")
                inst = infographic.arxiv_notebook_context(papers_ctx[:6], rec_text[:1200])
                url = infographic.nlm_generate_infographic(nb, inst)
                if url:
                    out = os.path.join(infographic.OUT_ROOT, "arxiv",
                                       f"nlm_{now.strftime('%Y%m%d-%H%M')}.png")
                    infographic.nlm_download_image(url, out)
                    img_paths.append(out)
                    sources.append("notebooklm")
            except Exception as ne:  # noqa: BLE001
                log(f"NotebookLM infographic failed (digest continues): {ne}")
            log(f"Infographics ({'+'.join(sources) or 'none'}): {len(img_paths)} image(s)")
        except Exception as e:  # noqa: BLE001
            log(f"Infographic skipped: {e}")

    # 6. Compose + email (chart and gemini banner get separate CIDs)
    html = render_email_digest(date_label, text,
                               infographic_cids=[f"infographic{i}" for i in range(len(img_paths))],
                               img_sources=sources)
    tag = "" if source == "api" else " (RSS fallback)"
    subject = (f"[arXiv cs.AI digest] {now.strftime('%Y-%m-%d')} — "
               f"{len(candidates)} new papers{tag}")
    send_email(subject, html, image_paths=img_paths)

    log(f"Done in {time.time()-start_all:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 — last-ditch report
        err = traceback.format_exc()
        try:
            send_email(f"[FAIL] arXiv cs.AI digest {datetime.now():%Y-%m-%d %H:%M}",
                       f"<pre>{err}</pre>")
        except Exception:  # noqa: BLE001
            pass
        print(err)
        sys.exit(1)
