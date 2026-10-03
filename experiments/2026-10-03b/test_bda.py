"""Tests for Bayesian Dialectical Argumentation (arXiv:2610.02005).

Paper claims under test (mechanism level, from the abstract):
  1. BDA yields calibrated posterior probabilities over candidate answers
     ("best calibration among zero-cost council aggregation methods").
  2. It stays robust under persistent adversarial coalitions (agents that are
     systematically wrong), while remaining competitive in clean settings.
  3. Persistently unreliable agents are identified (reliability < 0.5) and
     thereby *inverted* rather than merely outvoted.

Two scenarios: (A) 3-of-8 agents form a colluding adversarial coalition that
proposes one fixed wrong answer; (B) a clean council with no adversaries.
BDA is compared against the zero-cost majority-vote baseline. All seeded and
deterministic.
"""
import numpy as np

from bda_council import simulate_council, bda_fit, majority_vote, brier_score


def evaluate(seed, n_answers, true_rho, collude_answer, n_cases=300):
    rng = np.random.default_rng(seed)
    truths, traces = simulate_council(rng, n_cases, n_answers, true_rho,
                                      collude_answer=collude_answer)
    bda_ok = vote_ok = 0
    bda_cal, vote_cal = [], []
    rho_hats = []
    for y, obs in zip(truths, traces):
        post, rho = bda_fit(n_answers, obs)
        rho_hats.append(rho)
        yh = int(post.argmax())
        bda_ok += (yh == y)
        bda_cal.append((float(post[yh]), yh == y))
        yv, conf = majority_vote(n_answers, obs)
        vote_ok += (yv == y)
        vote_cal.append((conf, yv == y))
    rho_hats = np.array(rho_hats)
    return {
        'bda_acc': bda_ok / n_cases,
        'vote_acc': vote_ok / n_cases,
        'bda_brier': brier_score(bda_cal),
        'vote_brier': brier_score(vote_cal),
        'rho_corr': float(np.corrcoef(true_rho, rho_hats.mean(0))[0, 1]),
        'rho_mean': rho_hats.mean(0),
    }


def test_adversarial_coalition():
    """3 of 8 agents collude on one fixed wrong answer (K=3)."""
    true_rho = np.array([0.85, 0.80, 0.78, 0.72, 0.68, 0.15, 0.12, 0.20])
    r = evaluate(seed=1, n_answers=3, true_rho=true_rho, collude_answer=1)
    print('adversarial coalition: BDA acc=%.3f vote acc=%.3f | '
          'Brier BDA=%.4f vote=%.4f | rho-corr=%.3f'
          % (r['bda_acc'], r['vote_acc'], r['bda_brier'], r['vote_brier'],
             r['rho_corr']))
    print('  rho true:', np.round(true_rho, 2))
    print('  rho est :', np.round(r['rho_mean'], 2))
    # claim 2: robustness under adversarial coalition
    assert r['bda_acc'] >= r['vote_acc'] + 0.05, 'BDA should beat majority vote under coalition'
    assert r['bda_acc'] >= 0.85
    # claim 1: better calibration than the zero-cost baseline
    assert r['bda_brier'] < r['vote_brier'] - 0.03, 'BDA should be better calibrated'
    # claim 3: unreliable agents identified and inverted (est < 0.5)
    assert np.all(r['rho_mean'][5:] < 0.5), 'adversaries must be estimated below 0.5'
    assert np.all(r['rho_mean'][:5] > 0.5), 'reliable agents must stay above 0.5'
    assert r['rho_corr'] > 0.95


def test_clean_council():
    """No adversaries: BDA must remain competitive and well calibrated."""
    true_rho = np.array([0.85, 0.80, 0.78, 0.72, 0.68, 0.62, 0.58, 0.55])
    r = evaluate(seed=2, n_answers=3, true_rho=true_rho, collude_answer=None)
    print('clean council: BDA acc=%.3f vote acc=%.3f | '
          'Brier BDA=%.4f vote=%.4f | rho-corr=%.3f'
          % (r['bda_acc'], r['vote_acc'], r['bda_brier'], r['vote_brier'],
             r['rho_corr']))
    assert r['bda_acc'] >= r['vote_acc'] - 0.02, 'BDA competitive in clean setting'
    assert r['bda_acc'] >= 0.95
    assert r['bda_brier'] < r['vote_brier'], 'BDA better calibrated in clean setting'
    assert r['rho_corr'] > 0.95


if __name__ == '__main__':
    test_adversarial_coalition()
    test_clean_council()
    print('ALL BDA CHECKS PASSED')
