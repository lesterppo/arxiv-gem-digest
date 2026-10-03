"""
Mingbird (arXiv:2610.02001) — minimal harness-mechanism reproduction.

The paper argues a substantial share of small-open-model agent failures is
attributable to the HARNESS, not the model, and introduces Mingbird, a
local-first agent harness with (among ten) three representative mechanisms:

  1. byte-level net-zero prefill budget   (tool prefill overflows the context)
  2. a finish gate that re-reads the task (tasks silently abandoned)
  3. signature-level loop detection       (tool demonstrations loop)

This file implements those three mechanisms in a tiny, dependency-free agent
harness and tests each against a deterministic scripted "small model" policy
that exhibits the corresponding failure mode — mirroring the paper's
controlled comparison (machine/model/budget fixed, harness varied) on a toy
task set with deterministic artifact scoring.
"""

import hashlib
import json


# ── Mechanism 1: byte-level net-zero prefill budget ──────────────────────────
class PrefillBudget:
    """Enforce a byte budget on the harness prefill (system text + tool
    schemas). If the uncompressed prefill exceeds the budget, tool schemas are
    progressively compressed (drop examples, then descriptions) until the
    prefill fits — the marginal prefill cost stays at/below the fixed budget
    ("net-zero" relative to the budgeted baseline)."""

    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes

    @staticmethod
    def _schema_bytes(schema: dict, level: int) -> str:
        """level 0 = full, 1 = drop examples, 2 = drop descriptions+examples."""
        d = dict(schema)
        if level >= 1:
            d.pop("examples", None)
            d.get("parameters", {}).pop("examples", None)
        if level >= 2:
            d.pop("description", None)
        return json.dumps(d, separators=(",", ":"))

    def build_prefill(self, system_text: str, tool_schemas: list[dict]):
        parts = [system_text]
        for s in tool_schemas:
            parts.append(self._schema_bytes(s, 0))
        full = "\n".join(parts).encode("utf-8")
        if len(full) <= self.budget_bytes:
            return full.decode("utf-8"), len(full), False
        # compress: raise compression level for all tools until it fits
        for level in (1, 2):
            parts = [system_text] + [self._schema_bytes(s, level) for s in tool_schemas]
            blob = "\n".join(parts).encode("utf-8")
            if len(blob) <= self.budget_bytes:
                return blob.decode("utf-8"), len(blob), True
        # last resort: truncate system text (never exceed budget)
        head = system_text.encode("utf-8")[: self.budget_bytes // 2]
        blob = head + b"\n" + b"\n".join(
            self._schema_bytes(s, 2).encode("utf-8") for s in tool_schemas
        )[: self.budget_bytes - len(head) - 1]
        return blob.decode("utf-8", "ignore"), len(blob), True


# ── Mechanism 2: finish gate that re-reads the task ──────────────────────────
class FinishGate:
    """Before accepting a model's 'task complete' claim, re-read the original
    task and require explicit evidence for every success criterion. Prevents
    silent abandonment (model says done, artifacts incomplete)."""

    def __init__(self, task_text: str, criteria: list):
        """"criteria": list of (name, check_fn(env) -> bool)."""
        self.task_text = task_text
        self.criteria = criteria

    def review(self, env) -> tuple[bool, list[str]]:
        """Re-read task, check each criterion against the CURRENT environment
        (not the model's claim). Returns (accepted, failed_criteria)."""
        failed = [name for name, check in self.criteria if not check(env)]
        return (len(failed) == 0), failed


# ── Mechanism 3: signature-level loop detection ──────────────────────────────
class LoopDetector:
    """Fingerprint every action as (tool, canonical-arg-signature). If the same
    signature repeats K times consecutively — or a short signature cycle
    repeats — flag a loop and halt the step budget burn."""

    def __init__(self, repeat_k: int = 3, max_cycle: int = 4):
        self.repeat_k = repeat_k
        self.max_cycle = max_cycle
        self.history: list[str] = []

    @staticmethod
    def signature(tool: str, args: dict) -> str:
        canon = json.dumps(args, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(f"{tool}|{canon}".encode()).hexdigest()[:12]

    def observe(self, tool: str, args: dict) -> str | None:
        sig = self.signature(tool, args)
        self.history.append(sig)
        h = self.history
        # exact-repeat streak
        if len(h) >= self.repeat_k and len(set(h[-self.repeat_k:])) == 1:
            return f"loop: signature {sig} repeated {self.repeat_k}x"
        # cycle of length 2..max_cycle repeated twice
        for cyc in range(2, self.max_cycle + 1):
            if len(h) >= 2 * cyc and h[-2 * cyc:-cyc] == h[-cyc:]:
                return f"loop: {cyc}-step cycle repeated"
        return None
