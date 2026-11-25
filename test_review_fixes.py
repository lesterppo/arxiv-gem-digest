#!/usr/bin/env python3
"""Offline tests for the 2026-11-25 arxiv-gem-digest review fixes.

Covers: dry run must not consume papers (seen state untouched); email
failure fails the run and leaves papers unmarked; successful send marks
seen; NotebookLM run-relative budget; workflow shape (state commit,
conditional NLM steps, timeout, keepalive); exact requirements pins.

Run:  python3 test_review_fixes.py   (from the repo root; needs stdlib
only — matplotlib is NOT required: the infographic module is stubbed out)
Exit 0 = all pass.
"""
import json
import os
import re
import smtplib
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DIGEST_DRY_RUN", "")
import arxiv_gem_daily as g

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} :: {extra}")


def _fake_papers():
    now = datetime.now(timezone.utc)
    return [
        {"id": "2511.00001", "abs_url": "https://arxiv.org/abs/2511.00001",
         "title": "Fake agent paper", "authors": ["A One"],
         "summary": "A fake abstract about tool calling.",
         "published": now - timedelta(hours=5)},
        {"id": "2511.00002", "abs_url": "https://arxiv.org/abs/2511.00002",
         "title": "Fake tune paper", "authors": ["B Two"],
         "summary": "A fake abstract about LoRA on a T4.",
         "published": now - timedelta(hours=3)},
    ]


def _fake_gemini(candidates, batch_label):
    lines = ["2511.00001 — Fake agent paper — AGENT 4/5 — TUNE 2/5",
             "2511.00002 — Fake tune paper — AGENT 2/5 — TUNE 4/5",
             "", "RECOMMENDED", "Both fake papers are recommended."]
    return {"ok": True, "text": "\n".join(lines), "out": ""}


class _FakeSMTP:
    """Records sendmail calls; optionally raises on login."""
    instances = []

    def __init__(self, *a, **k):
        self.sent = []
        self.fail_login = _FakeSMTP.fail_login
        _FakeSMTP.instances.append(self)

    def login(self, u, p):
        if self.fail_login:
            raise smtplib.SMTPAuthenticationError(535, b"bad login")

    def sendmail(self, frm, to, msg):
        self.sent.append((frm, to, msg))

    def quit(self):
        pass


_FakeSMTP.fail_login = False


def _run_main(monkey, env, seed_deferred=None):
    """Run g.main() with fetch/gemini stubbed and temp state files.
    seed_deferred: raw on-disk deferred records (ISO published strings)
    written before the run, to simulate a legacy backlog file."""
    tmp = tempfile.mkdtemp(prefix="agd-test-")
    monkey_state = os.path.join(tmp, "seen_ids.json")
    monkey_def = os.path.join(tmp, "deferred_ids.json")
    g.STATE_FILE = monkey_state
    g.DEFERRED_FILE = monkey_def
    if seed_deferred is not None:
        with open(monkey_def, "w") as f:
            json.dump(seed_deferred, f)
    g.infographic = None
    g._have_web_auth = lambda: True
    g.fetch_papers = lambda s, e: _fake_papers()
    g.gemini_analyze = _fake_gemini
    old_env = dict(os.environ)
    os.environ.update(env)
    os.environ.pop("GEMINI_SID", None)
    os.environ.pop("GEMINI_TS", None)
    os.environ["GEMINI_API_KEY"] = ""
    os.environ["SMTP_USER"] = "sender@example.com"
    os.environ["RECIPIENT"] = "rcpt@example.com"
    os.environ["SMTP_PASS"] = "fake-pass"
    try:
        rc = g.main()
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    seen = {}
    if os.path.exists(monkey_state):
        with open(monkey_state) as f:
            seen = json.load(f)
    deferred = []
    if os.path.exists(monkey_def):
        with open(monkey_def) as f:
            deferred = json.load(f)
    return rc, seen, _FakeSMTP.instances, deferred


print("== send_email: dry run suppresses and reports success ==")
os.environ["DIGEST_DRY_RUN"] = "1"
os.environ["SMTP_PASS"] = "x"
check("dry-run send_email returns True",
      g.send_email("subj", "<p>hi</p>") is True)
check("dry-run writes the QA html artifact",
      os.path.exists("/tmp/arxiv_digest_email.html"))
os.environ.pop("DIGEST_DRY_RUN", None)

print("== send_email: SMTP failure returns False ==")
_FakeSMTP.fail_login = True
_real_smtp = smtplib.SMTP_SSL
smtplib.SMTP_SSL = _FakeSMTP
try:
    check("failed send_email returns False",
          g.send_email("subj", "<p>hi</p>") is False)
finally:
    smtplib.SMTP_SSL = _real_smtp
    _FakeSMTP.fail_login = False

print("== main(): dry run does NOT consume papers ==")
_FakeSMTP.instances.clear()
rc, seen, _, _ = _run_main(None, {"DIGEST_DRY_RUN": "1"})
check("dry run exits 0", rc == 0, f"rc={rc}")
check("dry run marks nothing seen", seen == {}, str(seen))

print("== main(): successful send marks seen ==")
_FakeSMTP.instances.clear()
smtplib.SMTP_SSL = _FakeSMTP
try:
    rc, seen, inst, _ = _run_main(None, {})
finally:
    smtplib.SMTP_SSL = _real_smtp
check("real run exits 0", rc == 0, f"rc={rc}")
check("real run marks both candidates seen",
      set(seen) == {"2511.00001", "2511.00002"}, str(seen))
check("real run actually sent one email",
      len(inst) == 1 and len(inst[0].sent) == 1,
      f"instances={len(inst)}")

print("== main(): failed send fails the run, papers unmarked ==")
_FakeSMTP.instances.clear()
_FakeSMTP.fail_login = True
smtplib.SMTP_SSL = _FakeSMTP
try:
    rc, seen, inst, _ = _run_main(None, {})
finally:
    smtplib.SMTP_SSL = _real_smtp
    _FakeSMTP.fail_login = False
check("failed send exits non-zero", rc != 0, f"rc={rc}")
check("failed send marks nothing seen", seen == {}, str(seen))
check("failed send attempted SMTP (login tried)",
      len(inst) == 1, f"instances={len(inst)}")

print("== deferred queue is bounded (window + cap) ==")
_now = datetime.now(timezone.utc)
_cut = _now - timedelta(days=2)
_tmpd = tempfile.mkdtemp(prefix="agd-def-")
g.DEFERRED_FILE = os.path.join(_tmpd, "deferred_ids.json")
_old = [{"id": f"2501.{i:05d}", "title": "old", "authors": [], "summary": "s",
         "published": _now - timedelta(days=10)} for i in range(300)]
_new = [{"id": f"2511.{i:05d}", "title": "new", "authors": [], "summary": "s",
         "published": _now - timedelta(hours=2)} for i in range(5)]
g._save_deferred(_old + _new, _cut)
_loaded = g._load_deferred(_cut)
check("save drops out-of-window papers", len(_loaded) == 5,
      f"len={len(_loaded)}")
check("load drops stale entries",
      all(p["published"] >= _cut for p in _loaded))
_many = [{"id": f"2511.{i:05d}", "title": "n", "authors": [], "summary": "s",
         "published": _now - timedelta(hours=1)} for i in range(500)]
g._save_deferred(_many, _cut)
_loaded = g._load_deferred(_cut)
check("save caps the queue at MAX_DEFERRED",
      len(_loaded) == g.MAX_DEFERRED, f"len={len(_loaded)}")
check("cap keeps the newest",
      all(p["id"].startswith("2511.") for p in _loaded))

print("== main(): stale deferred backlog is drained, not carried ==")
_FakeSMTP.instances.clear()
smtplib.SMTP_SSL = _FakeSMTP
try:
    # 300 stale (10 days old) + 25 fresh overflow at distinct ages 6..30h.
    # Unknown = 27, cap 22 -> candidates = 2 fetched + 20 newest seeded;
    # deferred remainder = the 5 oldest seeded. Stale must vanish.
    _seed = ([{"id": f"2501.{i:05d}", "abs_url": "", "title": "stale",
               "authors": [], "summary": "s",
               "published": (_now - timedelta(days=10)).isoformat()}
             for i in range(300)]
            + [{"id": f"2511.900{i:02d}", "abs_url": "", "title": "fresh",
                "authors": [], "summary": "s",
                "published": (_now - timedelta(hours=6 + i)).isoformat()}
               for i in range(25)])
    rc, seen, inst, def_after = _run_main(None, {}, seed_deferred=_seed)
finally:
    smtplib.SMTP_SSL = _real_smtp
check("stale-backlog run exits 0", rc == 0, f"rc={rc}")
check("stale backlog dropped (only in-window overflow kept)",
      len(def_after) == 5, f"len={len(def_after)}")
check("kept deferred are the 5 oldest in-window overflow",
      {p["id"] for p in def_after} == {f"2511.900{i:02d}" for i in range(20, 25)},
      str([p["id"] for p in def_after]))
check("run analysed the fresh candidates (cap 22)",
      len(seen) == 22 and "2511.00001" in seen and "2511.00002" in seen,
      f"seen={len(seen)}")

print("== NotebookLM run-relative budget ==")
check("over budget (700s > 600s) skips NLM",
      g._nlm_over_budget(time.time() - 700))
check("under budget (100s) keeps NLM",
      not g._nlm_over_budget(time.time() - 100))

print("== workflow shape (daily.yml + keepalive.yml) ==")
daily = open(".github/workflows/daily.yml").read()
check("daily: timeout 45 min", "timeout-minutes: 45" in daily)
check("daily: contents write permission",
      "contents: write" in daily)
check("daily: checkout full history (rebase before push)",
      "fetch-depth: 0" in daily)
check("daily: NLM setup conditional on the secret (via job env — "
      "secrets is not usable in if:)",
      "if: ${{ env.NLM_STORAGE_STATE_GZ != '' }}" in daily
      and "secrets.NLM_STORAGE_STATE_GZ != ''" not in daily
      and "NLM_STORAGE_STATE_GZ: ${{ secrets.NLM_STORAGE_STATE_GZ }}" in daily)
check("daily: commit digest state step present",
      "Commit digest state" in daily)
check("daily: commit gated on digest success",
      "steps.digest.outcome == 'success'" in daily)
check("daily: force-adds gitignored state files",
      "git add -f state/seen_ids.json state/deferred_ids.json" in daily)
check("daily: run step has id for the gate", "id: digest" in daily)
check("daily: inputs passed independently",
      'export RUN_BACK_DAYS=' in daily and 'export MAX_CANDIDATES=' in daily)
keep = open(".github/workflows/keepalive.yml").read()
check("keepalive: weekly Sunday schedule",
      '"0 3 * * 0"' in keep)
check("keepalive: 49-day (7-week) quiet threshold",
      "-lt 49" in keep)
check("keepalive: contents write permission",
      "contents: write" in keep)
check("keepalive: heartbeat file under .github/",
      ".github/keepalive.md" in keep)

print("== requirements: exact pins ==")
req = open("requirements.txt").read()
for pin in ("gemini-webapi==2.1.1", "curl_cffi==0.16.3",
            "loguru==0.7.3", "browser-cookie3==0.20.1",
            "matplotlib==3.11.2", "notebooklm-py==0.8.6"):
    check(f"pinned {pin}", pin in req)
check("no loose >= pins left",
      ">=" not in req and re.search(r"(?m)^(gemini-webapi|curl_cffi|loguru|browser-cookie3|matplotlib|notebooklm-py)[^=]", req) is None,
      req)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
