"""Causal Memory Policy (CMP) — arXiv:2610.02070.

Core mechanism from the paper: memory-utility estimates that intervene only at
the *store* level (remove a memory, measure the performance change) suffer a
retrieval-level positivity violation — a memory that is never retrieved has
estimated utility 0 no matter how useful it would be. CMP restores
identification by intervening on *retrieval itself*: a fixed number of context
slots are reserved for memories sampled with known propensities, and utility
is estimated by (self-normalized) inverse-propensity weighting under a
balanced assignment design.

This module implements a synthetic memory store + the two estimators.
Task performance is the sum of per-query memory utilities in context (linear,
so the true average treatment effect of including memory m on query q is
exactly true_u[m, q] — a clean calibration target). Deterministic (seeded),
numpy only.
"""
import numpy as np


class MemoryStore:
    def __init__(self, rng, n_mem=60, n_queries=40, n_required=20,
                 frac_never_retrieved=0.55, top_k=8):
        self.n_mem, self.n_queries, self.top_k = n_mem, n_queries, top_k
        self.true_u = np.zeros((n_mem, n_queries))
        self.required = np.arange(n_required)
        for m in self.required:
            qs = rng.choice(n_queries, size=6, replace=False)
            self.true_u[m, qs] = rng.uniform(0.6, 1.0, size=6)
        self.true_u += rng.uniform(0, 0.05, size=(n_mem, n_queries))
        # noisy retrieval scores; suppress a share of required memories ->
        # retrieval-level positivity violation (paper: 54% on LongMemEval)
        self.scores = self.true_u + rng.normal(0, 0.25, size=(n_mem, n_queries))
        n_supp = int(round(frac_never_retrieved * n_required))
        self.suppressed = self.required[:n_supp]
        self.scores[self.suppressed, :] = -10.0

    def retrieve(self, q, k):
        return np.argsort(-self.scores[:, q])[:k]

    def perf(self, mem_set, q):
        ctx = [m for m in self.retrieve(q, self.top_k) if m in mem_set]
        return float(self.true_u[ctx, q].sum()) if ctx else 0.0


def naive_store_intervention(store):
    """Remove each memory from the store; mean perf change over queries where
    it was retrieved. Never-retrieved memories score exactly 0 (the positivity
    violation: identification fails)."""
    n = store.n_mem
    est = np.zeros(n)
    for m in range(n):
        d, cnt = 0.0, 0
        for q in range(store.n_queries):
            if m in store.retrieve(q, store.top_k):
                d += store.perf(set(range(n)), q) - store.perf(set(range(n)) - {m}, q)
                cnt += 1
        est[m] = d / cnt if cnt else 0.0
    return est


def cmp_balanced_design(store, rng, reserved=2, n_rep=80):
    """CMP: per query keep top-(K-R) score slots; each remaining candidate is
    independently included with known propensity 0.5 (balanced assignment).
    Utility = SNIPW estimate of E[perf | included] - E[perf | excluded]."""
    M, Q = store.n_mem, store.n_queries
    num_in = np.zeros((M, Q)); den_in = np.zeros((M, Q))
    num_out = np.zeros((M, Q)); den_out = np.zeros((M, Q))
    for q in range(Q):
        base = list(store.retrieve(q, store.top_k - reserved))
        base_set = set(base)
        cand = [m for m in range(M) if m not in base_set]
        for _ in range(n_rep):
            mask = rng.random(len(cand)) < 0.5
            ctx = base + [cand[j] for j in range(len(cand)) if mask[j]]
            perf = float(store.true_u[ctx, q].sum())
            for j, m in enumerate(cand):
                if mask[j]:
                    num_in[m, q] += perf / 0.5
                    den_in[m, q] += 1 / 0.5
                else:
                    num_out[m, q] += perf / 0.5
                    den_out[m, q] += 1 / 0.5
    valid = (den_in > 0) & (den_out > 0)
    ate = np.full((M, Q), np.nan)
    ate[valid] = num_in[valid] / den_in[valid] - num_out[valid] / den_out[valid]
    return ate, valid


def auc(scores, labels):
    order = np.argsort(-np.asarray(scores, dtype=float))
    tp = np.cumsum(labels[order]); fp = np.cumsum(1 - labels[order])
    return float(np.trapz(tp / tp[-1], fp / fp[-1]))
