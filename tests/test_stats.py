"""Tests for swimlab.stats.

Design principle (per TASKS.md task 5): every function is pinned against a
**published worked example or a reference implementation**, never against
synthetic data from this project. Standard statistics are independently
checkable, and a subtly wrong ICC(2,1) would silently invalidate Q3.

Reference packages used ONLY here (dev/test dependencies -- the shipped
``swimlab.stats`` imports only numpy and scipy):

* ``diptest``      -- reference Hartigan dip statistic
* ``scikit-learn`` -- reference Cohen kappa and ROC AUC
* ``pingouin``     -- reference ICC(2,1) point estimate and CI
"""

from __future__ import annotations

import numpy as np
import pytest

from swimlab import stats as S


# --------------------------------------------------------------------------- #
# Q1 -- Hartigan dip test
# --------------------------------------------------------------------------- #
def test_dip_statistic_matches_diptest_reference() -> None:
    """The dip statistic matches the reference ``diptest`` package to machine
    precision on continuous samples (the study's data domain)."""
    diptest = pytest.importorskip("diptest")
    rng = np.random.default_rng(20240501)
    max_abs_err = 0.0
    for _ in range(50):
        n = int(rng.integers(15, 300))
        if rng.integers(0, 2):
            x = rng.normal(size=n)
        else:
            x = np.concatenate([rng.normal(-3, 0.5, n // 2), rng.normal(3, 0.5, n - n // 2)])
        x = x + rng.normal(0, 1e-9, size=n)  # ensure distinct support
        mine = S.dip_test(x, n_boot=1).dip_statistic
        ref = diptest.diptest(np.asarray(x, float))[0]
        max_abs_err = max(max_abs_err, abs(mine - ref))
    assert max_abs_err < 1e-9, f"dip statistic diverged from diptest: {max_abs_err}"


def test_dip_unimodal_is_not_significant() -> None:
    """A clearly unimodal (Gaussian) sample does not trip the p < 0.05 gate."""
    x = np.random.default_rng(1).normal(size=300)
    res = S.dip_test(x, n_boot=500, seed=0)
    assert res.p_value > 0.05


def test_dip_bimodal_is_significant() -> None:
    """A clearly bimodal sample trips the p < 0.05 gate (Q1)."""
    x = np.concatenate(
        [np.random.default_rng(2).normal(-3, 0.4, 150), np.random.default_rng(3).normal(3, 0.4, 150)]
    )
    res = S.dip_test(x, n_boot=500, seed=0)
    assert res.p_value < 0.05


def test_dip_pvalue_is_reproducible() -> None:
    """Same seed -> identical bootstrap p-value."""
    x = np.random.default_rng(4).normal(size=120)
    a = S.dip_test(x, n_boot=200, seed=42)
    b = S.dip_test(x, n_boot=200, seed=42)
    assert a.dip_statistic == b.dip_statistic
    assert a.p_value == b.p_value


# --------------------------------------------------------------------------- #
# Q2 -- Cohen's kappa
# --------------------------------------------------------------------------- #
def test_cohen_kappa_wikipedia_2x2() -> None:
    """Published worked example: Wikipedia "Cohen's kappa" 2x2 table.

    50 items, confusion [[20, 5], [10, 15]] gives p_o = 0.70, p_e = 0.50,
    kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40.
    Source: https://en.wikipedia.org/wiki/Cohen%27s_kappa (worked example).
    """
    # rater A: 25 "yes" then 25 "no"; rater B chosen to realise the 2x2 counts
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    res = S.cohen_kappa(a, b)
    assert res.kappa == pytest.approx(0.40, abs=1e-9)
    assert np.array_equal(res.confusion_matrix, np.array([[15, 10], [5, 20]]))
    assert res.accuracy == pytest.approx(0.70, abs=1e-9)


def test_cohen_kappa_matches_sklearn_3x3() -> None:
    """Cross-check kappa against sklearn on a 3-category example."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    ra = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2]
    rb = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 1, 2, 0, 2, 1]
    res = S.cohen_kappa(ra, rb)
    assert res.kappa == pytest.approx(sklearn_metrics.cohen_kappa_score(ra, rb), abs=1e-12)


def test_cohen_kappa_confusion_orientation() -> None:
    """Rows index rater_a, columns index rater_b."""
    res = S.cohen_kappa(["L", "L", "R"], ["L", "R", "R"], labels=["L", "R"])
    # a=L,b=L ; a=L,b=R ; a=R,b=R
    assert np.array_equal(res.confusion_matrix, np.array([[1, 1], [0, 1]]))


# --------------------------------------------------------------------------- #
# Q4 -- ROC / AUC / Youden threshold + bootstrap CI
# --------------------------------------------------------------------------- #
def test_auc_matches_sklearn_small_example() -> None:
    """AUC equals sklearn.metrics.roc_auc_score on a fixed example (=0.875)."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.55])
    labels = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    res = S.roc_analysis(scores, labels, n_boot=200, seed=0)
    assert res.auc == pytest.approx(sklearn_metrics.roc_auc_score(labels, scores), abs=1e-12)
    assert res.auc == pytest.approx(0.875, abs=1e-12)


def test_auc_matches_sklearn_with_ties() -> None:
    """Tie-averaged AUC still matches sklearn."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    scores = rng.integers(0, 5, size=40).astype(float)  # heavy ties
    labels = rng.integers(0, 2, size=40)
    res = S.roc_analysis(scores, labels, n_boot=50, seed=0)
    assert res.auc == pytest.approx(sklearn_metrics.roc_auc_score(labels, scores), abs=1e-12)


def test_youden_threshold_maximises_j() -> None:
    """The returned threshold maximises sensitivity + specificity - 1 over all
    candidate thresholds (brute-force check)."""
    rng = np.random.default_rng(6)
    scores = rng.normal(size=60)
    labels = (scores + rng.normal(0, 0.5, 60) > 0).astype(int)
    res = S.roc_analysis(scores, labels, n_boot=1, seed=0)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    best_j = -np.inf
    for thr in np.unique(scores):
        pred = scores >= thr
        sens = ((labels == 1) & pred).sum() / n_pos
        spec = 1 - ((labels == 0) & pred).sum() / n_neg
        best_j = max(best_j, sens + spec - 1)
    assert res.youden_j == pytest.approx(best_j, abs=1e-12)
    # And the reported operating point is self-consistent.
    assert res.youden_j == pytest.approx(res.sensitivity + res.specificity - 1, abs=1e-12)


def test_roc_bootstrap_reproducible_and_brackets_point() -> None:
    """Bootstrap CIs are reproducible across seeded runs and bracket the point
    estimates."""
    rng = np.random.default_rng(7)
    scores = np.concatenate([rng.normal(0, 1, 40), rng.normal(2, 1, 40)])
    labels = np.array([0] * 40 + [1] * 40)
    a = S.roc_analysis(scores, labels, n_boot=500, seed=123)
    b = S.roc_analysis(scores, labels, n_boot=500, seed=123)
    assert a.auc_ci == b.auc_ci
    assert a.threshold_ci == b.threshold_ci
    # CI brackets the point estimate.
    assert a.auc_ci[0] <= a.auc <= a.auc_ci[1]
    assert a.threshold_ci[0] <= a.youden_threshold <= a.threshold_ci[1]
    # A different seed generally gives a different interval.
    c = S.roc_analysis(scores, labels, n_boot=500, seed=999)
    assert c.auc_ci != a.auc_ci


# --------------------------------------------------------------------------- #
# Q3 -- ICC(2,1)
# --------------------------------------------------------------------------- #
# Shrout & Fleiss (1979), Table 1: 6 subjects x 4 judges.
_SHROUT_FLEISS = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


def test_icc_2_1_shrout_fleiss_published_value() -> None:
    """Published worked example: Shrout & Fleiss (1979), Table 1.

    Their ICC(2,1) (two-way random, single measure, ABSOLUTE agreement) for the
    6x4 data is 0.29. We match to 2 decimals.
    """
    res = S.icc_2_1(_SHROUT_FLEISS)
    assert res.icc == pytest.approx(0.29, abs=0.005)


def test_icc_2_1_matches_pingouin() -> None:
    """Cross-check point estimate and CI against pingouin's ICC(A,1)."""
    pg = pytest.importorskip("pingouin")
    pd = pytest.importorskip("pandas")
    rows = [
        {"subj": s, "judge": j, "score": _SHROUT_FLEISS[s, j]}
        for s in range(_SHROUT_FLEISS.shape[0])
        for j in range(_SHROUT_FLEISS.shape[1])
    ]
    table = pg.intraclass_corr(data=pd.DataFrame(rows), targets="subj", raters="judge", ratings="score")
    row = table[table["Type"] == "ICC(A,1)"].iloc[0]
    ref_icc = float(row["ICC"])
    ref_ci = row["CI95"]

    res = S.icc_2_1(_SHROUT_FLEISS)
    assert res.icc == pytest.approx(ref_icc, abs=1e-9)
    # pingouin exposes the CI only to 2 decimals; match at that precision.
    assert round(res.ci_low, 2) == pytest.approx(float(ref_ci[0]), abs=1e-9)
    assert round(res.ci_high, 2) == pytest.approx(float(ref_ci[1]), abs=1e-9)


def test_icc_is_absolute_agreement_not_consistency() -> None:
    """ICC(2,1) is the absolute-agreement form: for data with a systematic
    between-rater offset it must be *lower* than the consistency form.

    Adding a constant column offset leaves consistency unchanged but must drop
    absolute agreement -- a guard against accidentally implementing ICC(3,1).
    """
    base = _SHROUT_FLEISS.copy()
    offset = base + np.array([0.0, 5.0, 10.0, 15.0])  # rater-specific bias
    assert S.icc_2_1(offset).icc < S.icc_2_1(base).icc


# --------------------------------------------------------------------------- #
# Q3 -- SEM and MDC95
# --------------------------------------------------------------------------- #
def test_sem_hand_worked() -> None:
    """Hand-worked: SD_pooled = 10, ICC = 0.75 -> SEM = 10*sqrt(0.25) = 5.0."""
    assert S.sem(10.0, 0.75) == pytest.approx(5.0, abs=1e-12)


def test_mdc95_hand_worked() -> None:
    """MDC95 = 1.96 * sqrt(2) * SEM. For SEM = 5 -> 13.8593..."""
    assert S.mdc95(5.0) == pytest.approx(1.96 * np.sqrt(2) * 5.0, abs=1e-12)
    assert S.mdc95(5.0) == pytest.approx(13.85929, abs=1e-4)


def test_sem_perfect_reliability_is_zero() -> None:
    """ICC = 1 -> no measurement error."""
    assert S.sem(7.3, 1.0) == 0.0


# --------------------------------------------------------------------------- #
# Bland-Altman
# --------------------------------------------------------------------------- #
# Bland & Altman (1986), Lancet: PEFR by large Wright vs mini Wright meter,
# first measurement, 17 subjects. Published bias ~ -2.1 L/min, SD of
# differences ~ 38.8 L/min.
_PEFR_LARGE = np.array(
    [494, 395, 516, 434, 476, 557, 413, 442, 650, 433, 417, 656, 267, 478, 178, 423, 427],
    dtype=float,
)
_PEFR_MINI = np.array(
    [512, 430, 520, 428, 500, 600, 364, 380, 658, 445, 432, 626, 260, 477, 259, 350, 451],
    dtype=float,
)


def test_bland_altman_pefr_published_example() -> None:
    """Published worked example: Bland & Altman (1986) PEFR data.

    With d = large - mini, bias ~ -2.1 L/min and SD(diff) ~ 38.8 L/min, giving
    limits of agreement of roughly -78 to +74.
    """
    res = S.bland_altman(_PEFR_LARGE, _PEFR_MINI)
    assert res.bias == pytest.approx(-2.1, abs=0.1)
    assert res.sd_diff == pytest.approx(38.8, abs=0.1)
    assert res.loa_lower == pytest.approx(res.bias - 1.96 * res.sd_diff, abs=1e-12)
    assert res.loa_upper == pytest.approx(res.bias + 1.96 * res.sd_diff, abs=1e-12)


def test_bland_altman_hand_example() -> None:
    """Tiny hand-computed check. x-y = [1, -1, 1, -1] -> bias 0, SD(ddof=1) of
    differences = sqrt(sum(d^2)/(n-1)) = sqrt(4/3)."""
    x = np.array([2.0, 2.0, 4.0, 4.0])
    y = np.array([1.0, 3.0, 3.0, 5.0])
    res = S.bland_altman(x, y)
    assert res.bias == pytest.approx(0.0, abs=1e-12)
    assert res.sd_diff == pytest.approx(np.sqrt(4.0 / 3.0), abs=1e-12)
    assert res.loa_upper == pytest.approx(1.96 * np.sqrt(4.0 / 3.0), abs=1e-12)
    assert res.loa_lower == pytest.approx(-1.96 * np.sqrt(4.0 / 3.0), abs=1e-12)


# --------------------------------------------------------------------------- #
# Production purity guard
# --------------------------------------------------------------------------- #
def test_production_module_has_no_reference_package_dependency() -> None:
    """swimlab.stats must not *import* the dev-only cross-check packages.

    Parses the module's import statements (docstring mentions are fine).
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(S))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("pingouin", "sklearn", "diptest", "statsmodels"):
        assert banned not in imported, f"stats.py must not import {banned}"
    assert imported <= {"__future__", "dataclasses", "numpy", "scipy"}
