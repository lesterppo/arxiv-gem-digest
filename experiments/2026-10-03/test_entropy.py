"""
Test: Kontoyiannis entropy-rate filter (arXiv:2610.01493) on synthetic text.

The paper's claims, reproduced WITHOUT a model (CPU-only, stdlib):
  2a. h_k is a calibrated entropy-rate proxy on sources with known rates.
  2b. In a simulated single-lineage "generational collapse" (each generation
      resampled from a sharpened copy of the last, mimicking synthetic-data
      fine-tuning), h_k declines monotonically, tracking diversity loss.
  2c. Filtering a mixed pool by h_k preserves phrase diversity (unique
      trigrams) far better than random/length baselines — the paper's headline
      metric (+42% unique trigrams, -19% repetition in their QLoRA run).

Run:  python3 test_entropy.py
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entropy_rate_filter import (
    entropy_rate, unique_trigrams, repetition_rate,
    bernoulli_words, repetitive_words, vocab_words,
)

RNG = random.Random(20261003)


def shannon(p):
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0


def exp_2a_calibration():
    print("== 2a: calibration on known-entropy sources ==")
    rows = []
    for name, words, truth in [
        ("bernoulli p=0.5 (truth 1.000 b/w)", bernoulli_words(0.5, 4000, RNG), shannon(0.5)),
        ("bernoulli p=0.1 (truth 0.469 b/w)", bernoulli_words(0.1, 4000, RNG), shannon(0.1)),
        ("periodic period=2 (truth ~0 b/w)", repetitive_words(2, 4000), 0.0),
        ("periodic period=50 (truth ~0 b/w)", repetitive_words(50, 4000), 0.0),
    ]:
        h = entropy_rate(words, sample_stride=2)
        rows.append((name, truth, h))
        print(f"  {name:38s} h_k={h:.3f}")
    # checks: high-entropy source >> repetitive source; Bernoulli(0.5) near 1.0
    assert rows[0][2] > 0.7, "p=0.5 should estimate near 1.0 bit/word"
    assert rows[0][2] > rows[1][2] > rows[2][2], "monotonic in true entropy"
    assert rows[2][2] < 0.15 and rows[3][2] < 0.15, "periodic text must score ~0"
    assert abs(rows[0][2] - 1.0) < 0.25, "p=0.5 calibration tolerance"
    print("  PASS: h_k is monotone and calibrated vs known rates\n")


def sharpened_resample(words, temperature, n, rng):
    from collections import Counter
    counts = Counter(words)
    items = list(counts.items())
    ws = [c ** (1.0 / temperature) for _, c in items]
    return rng.choices([w for w, _ in items], weights=ws, k=n)


def exp_2b_collapse():
    print("== 2b: six-generation collapse simulation ==")
    temps = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]
    gen = vocab_words(2000, 3000, RNG)
    hs, uts, reps = [], [], []
    for g, T in enumerate(temps):
        h = entropy_rate(gen, sample_stride=3)
        ut, rr = unique_trigrams(gen), repetition_rate(gen)
        hs.append(h); uts.append(ut); reps.append(rr)
        print(f"  gen {g}: T={T:4.2f}  h_k={h:.3f}  uniq_trigrams={ut:5d}  repet={rr:.3f}")
        if g < len(temps) - 1:
            gen = sharpened_resample(gen, temps[g + 1], 3000, RNG)
    # monotone decline of h_k and diversity
    assert all(hs[i] >= hs[i + 1] - 1e-9 for i in range(5)), "h_k must decline"
    assert uts[-1] < 0.6 * uts[0], "unique trigrams must collapse"
    assert reps[-1] > reps[0] + 0.05, "repetition must rise"
    print(f"  PASS: h_k {hs[0]:.3f} -> {hs[-1]:.3f} tracks collapse "
          f"(uniq_tri {uts[0]} -> {uts[-1]})\n")
    return hs, uts, reps


def exp_2c_filtering():
    print("== 2c: h_k filtering vs baselines (phrase-level collapse) ==")
    print("   collapsed docs reuse 40 fixed SENTENCES (phrase repetition),")
    print("   so unigram-vocab counting is a weak signal here — as in real collapse.")
    docs = []
    for i in range(20):
        docs.append(("diverse", vocab_words(1500, 600, RNG)))
    for i in range(20):
        sents = [" ".join(vocab_words(1500, 15, RNG)) for _ in range(40)]
        doc = " ".join(RNG.choices(sents, k=40)).split()  # 600 words, phrase-level repeats
        docs.append(("collapsed", doc))
    scored = [(entropy_rate(w, sample_stride=3), kind, w) for kind, w in docs]

    def pool_metrics(doclist):
        pool = [tok for _, w in doclist for tok in w]
        return unique_trigrams(pool), repetition_rate(pool), len(set(pool))

    cand = {}
    cand["h_k top-50%"] = sorted(scored, key=lambda t: -t[0])[:20]
    cand["random 50%"] = RNG.sample(scored, 20)
    cand["length top-50%"] = sorted(scored, key=lambda t: -len(t[2]))[:20]
    cand["uniq-unigram top-50%"] = sorted(
        scored, key=lambda t: -len(set(t[2])))[:20]

    base = None
    for name, sel in cand.items():
        ut, rr, vocab = pool_metrics([(k, w) for _, k, w in sel])
        n_div = sum(1 for _, k, _ in sel if k == "diverse")
        print(f"  {name:22s} diverse_kept={n_div:2d}/20  "
              f"pool_uniq_tri={ut:6d}  pool_repet={rr:.3f}  pool_vocab={vocab:5d}")
        if name == "random 50%":
            base = (ut, rr, vocab)
    uh = pool_metrics([(k, w) for _, k, w in cand["h_k top-50%"]])
    gain_ut = (uh[0] - base[0]) / base[0] * 100
    drop_rr = (base[1] - uh[1]) / base[1] * 100
    print(f"  h_k vs random: +{gain_ut:.0f}% pooled unique trigrams, "
          f"-{drop_rr:.0f}% pooled repetition")
    n_div_kept = sum(1 for _, k, _ in cand["h_k top-50%"] if k == "diverse")
    assert n_div_kept >= 18, "h_k must prefer diverse docs"
    assert uh[0] > base[0] and uh[1] < base[1], "h_k must beat random on both metrics"
    print("  PASS: model-free h_k filtering preserves phrase diversity")
    print("  NOTE: on this synthetic mixture cheap heuristics (unigram vocab) also")
    print("  separate the classes; the paper's edge is specifically vs LOGPROB-based")
    print("  filtering, which needs model access we do not have on this CPU-only VM.")


def main():
    exp_2a_calibration()
    exp_2b_collapse()
    exp_2c_filtering()
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
