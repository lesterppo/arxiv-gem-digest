#!/usr/bin/env python3
"""
Refresh GEMINI_SID/GEMINI_TS GitHub Actions secrets from ~/.gemini-cli/auth.json.

Run locally after gemini.py --init (or after re-login in Firefox) so the daily
GitHub Actions digest keeps receiving valid Gemini cookies. Requires:
  pynacl (pip install pynacl)
  gh CLI authenticated as the repo owner
Usage:
  python3 refresh_gh_secrets.py [owner/repo]
"""
import base64
import json
import os
import subprocess
import sys

import nacl.encoding
import nacl.public

REPO = os.environ.get("GITHUB_REPOSITORY", "") or \
    (sys.argv[1] if len(sys.argv) > 1 else "lesterppo/arxiv-gem-digest")
AUTH_JSON = os.path.expanduser("~/.gemini-cli/auth.json")
SECRETS = ["GEMINI_SID", "GEMINI_TS"]
AUTH_KEYS = {"GEMINI_SID": "__Secure-1PSID", "GEMINI_TS": "__Secure-1PSIDTS"}
API_BASE = f"https://api.github.com/repos/{REPO}/actions/secrets"


def gh_token() -> str:
    out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError(f"gh auth token failed: {out.stderr.strip()}")
    return out.stdout.strip()


def api(method: str, path: str, data: dict | None = None, token: str = "") -> dict | None:
    import urllib.error
    import urllib.request
    url = f"{API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = json.dumps(data).encode() if data else None
    if body:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GitHub {method} {path}: HTTP {e.code} — {e.read().decode()[:400]}")


def encrypt(public_key_b64: str, value: str) -> str:
    pk = nacl.public.PublicKey(public_key_b64.encode(), nacl.encoding.Base64Encoder)
    return base64.b64encode(nacl.public.SealedBox(pk).encrypt(value.encode())).decode()


def main() -> int:
    if not os.path.exists(AUTH_JSON):
        print(f"ERROR: {AUTH_JSON} not found", file=sys.stderr)
        return 1
    with open(AUTH_JSON) as f:
        auth = json.load(f)
    token = gh_token()
    pub = api("GET", "/public-key", token=token)
    key_id, key = pub["key_id"], pub["key"]
    for name in SECRETS:
        val = auth.get(AUTH_KEYS[name], "")
        if not val:
            print(f"WARNING: {AUTH_KEYS[name]} missing — skipping {name} ({REPO})")
            continue
        enc = encrypt(key, val)
        api("PUT", f"/{name}", data={"encrypted_value": enc, "key_id": key_id}, token=token)
        print(f"  {name} updated for {REPO} ({len(val)} chars)")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
