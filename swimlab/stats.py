"""Statistical tests for the swimlab study endpoints.

This module is **pure statistics on arrays**. It has no dependency on the rest
of the pipeline (synth/calibrate/events/metrics) and imports only numpy and
scipy. Each public function maps onto one of the study questions in
``CLAUDE.md``:

============  ==========================================================
study Q       function
============  ==========================================================
Q1 bimodal    :func:`dip_test`        Hartigan dip test (p < 0.05 gate)
Q2 agreement  :func:`cohen_kappa`     Cohen's kappa (kappa >= 0.70 gate)
Q4 detection  :func:`roc_analysis`    ROC / AUC / Youden threshold + CI
Q3 repeat     :func:`icc_2_1`         ICC(2,1) (ICC >= 0.75 gate)
Q3 repeat     :func:`sem`,            SEM and MDC95 for a metric
              :func:`mdc95`
(agreement)   :func:`bland_altman`    bias + 95% limits of agreement
============  ==========================================================

All angles/metrics fed to these functions are already in study units
(degrees, dimensionless ratios, seconds); this module is unit-agnostic and
treats every input as a plain real-valued array.

Determinism: every resampling routine (:func:`dip_test`, :func:`roc_analysis`)
takes an explicit ``seed`` and draws only from ``numpy.random.default_rng``, so
results are reproducible run to run.

References
----------
Hartigan, J. A. & Hartigan, P. M. (1985). The Dip Test of Unimodality.
    *Annals of Statistics* 13(1), 70-84.
Shrout, P. E. & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. *Psychological Bulletin* 86(2), 420-428.
McGraw, K. O. & Wong, S. P. (1996). Forming inferences about some intraclass
    correlation coefficients. *Psychological Methods* 1(1), 30-46.
Bland, J. M. & Altman, D. G. (1986). Statistical methods for assessing
    agreement between two methods of clinical measurement. *Lancet* 1, 307-310.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import f as _f_dist

__all__ = [
    "DipResult",
    "KappaResult",
    "RocResult",
    "IccResult",
    "BlandAltmanResult",
    "dip_test",
    "cohen_kappa",
    "roc_analysis",
    "icc_2_1",
    "sem",
    "mdc95",
    "bland_altman",
]


# --------------------------------------------------------------------------- #
# Q1 -- Hartigan dip test of unimodality
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DipResult:
    """Result of :func:`dip_test`.

    Attributes
    ----------
    dip_statistic : float
        Hartigan's dip statistic D (dimensionless, in [0, 0.25]). Larger means
        stronger departure from unimodality.
    p_value : float
        Bootstrap p-value under the least-favourable unimodal (uniform) null.
        ``p_value < 0.05`` is the Q1 bimodality gate.
    n : int
        Sample size the statistic was computed on.
    n_boot : int
        Number of uniform null replicates used for the p-value.
    """

    dip_statistic: float
    p_value: float
    n: int
    n_boot: int


def _gcm(cdf: NDArray[np.float64], idx: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Greatest convex minorant of ``cdf`` sampled at support points ``idx``.

    Returns the minorant evaluated at every support point and the indices of
    the touch points (vertices). Port of the Hartigan & Hartigan (1985)
    construction (Bauer 2015 implementation, github.com/tatome/dip_test).
    """
    work_cdf = cdf
    work_idx = idx
    gcm = [work_cdf[0]]
    touch = [0]
    while len(work_cdf) > 1:
        distances = work_idx[1:] - work_idx[0]
        slopes = (work_cdf[1:] - work_cdf[0]) / distances
        min_i = int(np.where(slopes == slopes.min())[0][0]) + 1
        gcm.extend(work_cdf[0] + distances[:min_i] * slopes.min())
        touch.append(touch[-1] + min_i)
        work_cdf = work_cdf[min_i:]
        work_idx = work_idx[min_i:]
    return np.asarray(gcm, dtype=float), np.asarray(touch, dtype=np.int64)


def _lcm(cdf: NDArray[np.float64], idx: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Least concave majorant, obtained by reflecting the GCM."""
    g, t = _gcm(1 - cdf[::-1], idx.max() - idx[::-1])
    return 1 - g[::-1], len(cdf) - 1 - t[::-1]


def _dip_statistic(values: NDArray[np.float64]) -> float:
    """Hartigan's dip statistic for a 1-D sample.

    Exact (to machine precision) for samples with distinct support, which is
    the case for the study's continuous angle metrics; verified against the
    reference ``diptest`` package in the test-suite. Tied inputs are handled
    via the unique-support empirical CDF formulation.
    """
    x = np.sort(np.asarray(values, dtype=float))
    # Collapse to unique support with cumulative empirical mass.
    idx, counts = np.unique(x, return_counts=True)
    if idx.size <= 4 or idx[0] == idx[-1]:
        return 0.0
    hist = counts.astype(float) / counts.sum()
    cdf = np.cumsum(hist)

    work_idx = idx.astype(float)
    work_hist = hist
    work_cdf = cdf
    big_d = 0.0
    left = [0]
    right = [1]

    while True:
        left_part, left_touch = _gcm(work_cdf - work_hist, work_idx)
        right_part, right_touch = _lcm(work_cdf, work_idx)

        d_left = float(np.abs(right_part[left_touch] - left_part[left_touch]).max())
        left_diffs = np.abs(right_part[left_touch] - left_part[left_touch])
        d_right = float(np.abs(right_part[right_touch] - left_part[right_touch]).max())
        right_diffs = np.abs(right_part[right_touch] - left_part[right_touch])

        if d_right > d_left:
            xr = int(right_touch[right_diffs == d_right][-1])
            xl = int(left_touch[left_touch <= xr][-1])
            d = d_right
        else:
            xl = int(left_touch[left_diffs == d_left][0])
            xr = int(right_touch[right_touch >= xl][0])
            d = d_left

        left_diff = float(np.abs(left_part[: xl + 1] - work_cdf[: xl + 1]).max())
        right_diff = float(np.abs(right_part[xr:] - work_cdf[xr:] + work_hist[xr:]).max())

        if d <= big_d or xr == 0 or xl == len(work_cdf):
            the_dip = max(
                float(np.abs(cdf[: len(left)] - left).max()),
                float(np.abs(cdf[-len(right) - 1 : -1] - right).max()),
            )
            return the_dip / 2.0

        big_d = max(big_d, left_diff, right_diff)
        work_cdf = work_cdf[xl : xr + 1]
        work_idx = work_idx[xl : xr + 1]
        work_hist = work_hist[xl : xr + 1]
        left[len(left) :] = left_part[1 : xl + 1].tolist()
        right[:0] = right_part[xr:-1].tolist()


def dip_test(x: ArrayLike, n_boot: int = 2000, seed: int = 0) -> DipResult:
    """Hartigan dip test of unimodality (study question Q1).

    Parameters
    ----------
    x : array-like
        1-D sample (e.g. ``d_pitch_breath`` or ``roll_pitch_ratio`` across
        participants), in study units. Order does not matter.
    n_boot : int, default 2000
        Number of Uniform[0, 1] null replicates for the p-value. The uniform
        distribution is the least-favourable unimodal null, per Hartigan &
        Hartigan (1985).
    seed : int, default 0
        Seed for ``numpy.random.default_rng``; makes the p-value reproducible.

    Returns
    -------
    DipResult
        ``dip_statistic`` (dimensionless) and bootstrap ``p_value``. A
        ``p_value < 0.05`` rejects unimodality -- the Q1 gate separating
        "lifters" from "rotators".
    """
    data = np.asarray(x, dtype=float).ravel()
    n = data.size
    if n < 4:
        return DipResult(dip_statistic=0.0, p_value=1.0, n=n, n_boot=n_boot)

    observed = _dip_statistic(data)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_boot):
        if _dip_statistic(rng.uniform(size=n)) >= observed:
            ge += 1
    # +1/+1 keeps the p-value strictly positive.
    p_value = (ge + 1) / (n_boot + 1)
    return DipResult(dip_statistic=float(observed), p_value=float(p_value), n=n, n_boot=n_boot)


# --------------------------------------------------------------------------- #
# Q2 -- Cohen's kappa
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KappaResult:
    """Result of :func:`cohen_kappa`.

    Attributes
    ----------
    kappa : float
        Cohen's kappa. Chance-corrected agreement; ``kappa >= 0.70`` is the Q2
        gate for IMU-vs-coach agreement.
    confusion_matrix : numpy.ndarray
        ``(L, L)`` integer counts, rows = ``rater_a``, cols = ``rater_b``,
        ordered by :attr:`labels`.
    labels : numpy.ndarray
        The label order indexing the confusion matrix rows/columns.
    accuracy : float
        Raw observed agreement (trace / total) -- the study also gates on
        ``accuracy >= 0.85``.
    """

    kappa: float
    confusion_matrix: NDArray[np.int64]
    labels: NDArray[np.generic]
    accuracy: float


def cohen_kappa(rater_a: ArrayLike, rater_b: ArrayLike, labels: ArrayLike | None = None) -> KappaResult:
    """Cohen's kappa for two raters over categorical labels (study Q2).

    Parameters
    ----------
    rater_a, rater_b : array-like
        Paired categorical ratings of equal length (e.g. IMU classification vs
        a blinded coach rating). Labels may be ints or strings.
    labels : array-like, optional
        Explicit label ordering for the confusion matrix. Defaults to the
        sorted union of observed labels.

    Returns
    -------
    KappaResult
        ``kappa``, the ``confusion_matrix`` (rows ``rater_a``, cols
        ``rater_b``), the ``labels`` order, and raw ``accuracy``.

    Notes
    -----
    ``kappa = (p_o - p_e) / (1 - p_e)`` with ``p_o`` observed agreement and
    ``p_e`` chance agreement from the marginal products. Matches
    ``sklearn.metrics.cohen_kappa_score``.
    """
    a = np.asarray(rater_a).ravel()
    b = np.asarray(rater_b).ravel()
    if a.shape != b.shape:
        raise ValueError("rater_a and rater_b must have the same length")
    if a.size == 0:
        raise ValueError("ratings must be non-empty")

    if labels is None:
        lab = np.unique(np.concatenate([a, b]))
    else:
        lab = np.asarray(labels).ravel()
    index = {v: i for i, v in enumerate(lab.tolist())}

    n_lab = lab.size
    cm = np.zeros((n_lab, n_lab), dtype=np.int64)
    for va, vb in zip(a.tolist(), b.tolist(), strict=True):
        cm[index[va], index[vb]] += 1

    total = cm.sum()
    p_o = np.trace(cm) / total
    row_marg = cm.sum(axis=1) / total
    col_marg = cm.sum(axis=0) / total
    p_e = float(np.dot(row_marg, col_marg))
    kappa = 1.0 if p_e == 1.0 else (p_o - p_e) / (1 - p_e)

    return KappaResult(
        kappa=float(kappa),
        confusion_matrix=cm,
        labels=lab,
        accuracy=float(p_o),
    )


# --------------------------------------------------------------------------- #
# Q4 -- ROC / AUC / Youden threshold with bootstrap CI
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RocResult:
    """Result of :func:`roc_analysis`.

    Attributes
    ----------
    auc : float
        Area under the ROC curve. Q4 gate is ``auc >= 0.75``.
    youden_threshold : float
        Score threshold maximising Youden's J = sensitivity + specificity - 1;
        the operating point predicts positive when ``score >= threshold``.
    youden_j : float
        Youden's J at :attr:`youden_threshold`.
    sensitivity, specificity : float
        Sensitivity and specificity at :attr:`youden_threshold`.
    auc_ci : tuple[float, float]
        Percentile bootstrap CI for the AUC.
    threshold_ci : tuple[float, float]
        Percentile bootstrap CI for the Youden-optimal threshold.
    ci_level : float
        Two-sided coverage of both CIs (e.g. 0.95).
    n_boot : int
        Number of bootstrap resamples that contributed (both classes present).
    """

    auc: float
    youden_threshold: float
    youden_j: float
    sensitivity: float
    specificity: float
    auc_ci: tuple[float, float]
    threshold_ci: tuple[float, float]
    ci_level: float
    n_boot: int


def _auc(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """AUC via the rank (Mann-Whitney U) identity, ties averaged.

    Equal to ``sklearn.metrics.roc_auc_score``.
    """
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    sorted_scores = scores[order]
    # Average ranks within tie groups (ranks are 1-based).
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _youden(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> tuple[float, float, float, float]:
    """Return (threshold, J, sensitivity, specificity) maximising Youden's J.

    Candidate thresholds are the distinct scores; predict positive when
    ``score >= threshold``. On ties in J the lowest threshold is returned.
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    candidates = np.unique(scores)
    best = (-np.inf, np.inf, 0.0, 0.0, 0.0)  # (J, thr, sens, spec)
    for thr in candidates:
        pred = scores >= thr
        tp = int(((labels == 1) & pred).sum())
        fp = int(((labels == 0) & pred).sum())
        sens = tp / n_pos if n_pos else 0.0
        spec = 1.0 - (fp / n_neg if n_neg else 0.0)
        j = sens + spec - 1.0
        # Prefer larger J; break ties toward the lower threshold.
        if j > best[0] or (j == best[0] and thr < best[1]):
            best = (j, float(thr), sens, spec)
    j, thr, sens, spec = best
    return thr, j, sens, spec


def roc_analysis(
    scores: ArrayLike,
    labels: ArrayLike,
    n_boot: int = 2000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> RocResult:
    """ROC analysis with Youden-optimal threshold and bootstrap CIs (study Q4).

    Parameters
    ----------
    scores : array-like
        Continuous decision scores (e.g. an accelerometer breath-hold score).
        Higher scores should indicate the positive class.
    labels : array-like
        Binary ground-truth labels, coercible to {0, 1} (1 = positive).
    n_boot : int, default 2000
        Number of paired bootstrap resamples for the CIs.
    seed : int, default 0
        Seed for ``numpy.random.default_rng`` -- makes the CIs reproducible.
    ci_level : float, default 0.95
        Two-sided coverage for the percentile CIs.

    Returns
    -------
    RocResult
        Point ``auc`` and ``youden_threshold`` plus percentile-bootstrap
        ``auc_ci`` and ``threshold_ci``. Q4 gate: ``auc >= 0.75``.
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel().astype(int)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same length")
    uniq = np.unique(y)
    if not np.isin(uniq, (0, 1)).all():
        raise ValueError("labels must be binary {0, 1}")
    if uniq.size < 2:
        raise ValueError("labels must contain both classes")

    auc = _auc(s, y)
    thr, j, sens, spec = _youden(s, y)

    rng = np.random.default_rng(seed)
    n = s.size
    boot_auc: list[float] = []
    boot_thr: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue  # degenerate resample: only one class present
        boot_auc.append(_auc(s[idx], ys))
        boot_thr.append(_youden(s[idx], ys)[0])

    lo = 100 * (1 - ci_level) / 2
    hi = 100 * (1 + ci_level) / 2
    if boot_auc:
        auc_ci = (float(np.percentile(boot_auc, lo)), float(np.percentile(boot_auc, hi)))
        thr_ci = (float(np.percentile(boot_thr, lo)), float(np.percentile(boot_thr, hi)))
    else:
        auc_ci = (float("nan"), float("nan"))
        thr_ci = (float("nan"), float("nan"))

    return RocResult(
        auc=float(auc),
        youden_threshold=float(thr),
        youden_j=float(j),
        sensitivity=float(sens),
        specificity=float(spec),
        auc_ci=auc_ci,
        threshold_ci=thr_ci,
        ci_level=ci_level,
        n_boot=len(boot_auc),
    )


# --------------------------------------------------------------------------- #
# Q3 -- ICC(2,1): two-way random, single measure, absolute agreement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IccResult:
    """Result of :func:`icc_2_1`.

    Attributes
    ----------
    icc : float
        ICC(2,1) -- two-way random effects, single measurement, absolute
        agreement (Shrout & Fleiss 1979; = McGraw & Wong ICC(A,1)).
        ``icc >= 0.75`` is the Q3 repeatability gate.
    ci_low, ci_high : float
        Bounds of the ``ci_level`` CI (McGraw & Wong 1996).
    ci_level : float
        Two-sided coverage of the CI.
    msb, msj, mse : float
        Between-subjects, between-raters, and residual mean squares from the
        two-way ANOVA (exposed for SEM computation and auditing).
    n_subjects, n_raters : int
        ANOVA dimensions.
    """

    icc: float
    ci_low: float
    ci_high: float
    ci_level: float
    msb: float
    msj: float
    mse: float
    n_subjects: int
    n_raters: int


def _anova_mean_squares(ratings: NDArray[np.float64]) -> tuple[float, float, float]:
    """Two-way (subjects x raters) ANOVA mean squares, no replication.

    Returns ``(MSB, MSJ, MSE)`` = between-subjects, between-raters, residual.
    """
    n, k = ratings.shape
    grand = ratings.mean()
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)
    ss_total = float(((ratings - grand) ** 2).sum())
    ss_b = float(k * ((row_means - grand) ** 2).sum())
    ss_j = float(n * ((col_means - grand) ** 2).sum())
    ss_e = ss_total - ss_b - ss_j
    msb = ss_b / (n - 1)
    msj = ss_j / (k - 1)
    mse = ss_e / ((n - 1) * (k - 1))
    return msb, msj, mse


def icc_2_1(ratings: ArrayLike, ci_level: float = 0.95) -> IccResult:
    """ICC(2,1): two-way random, single measure, absolute agreement (study Q3).

    Parameters
    ----------
    ratings : array-like, shape (n_subjects, n_raters)
        Balanced matrix of measurements; rows are subjects (participants),
        columns are the raters or repeated sessions being compared for
        repeatability. No missing values.
    ci_level : float, default 0.95
        Two-sided coverage for the confidence interval.

    Returns
    -------
    IccResult
        ``icc`` with ``ci_low``/``ci_high``. Q3 passes when ``icc >= 0.75``.

    Notes
    -----
    Implements the Shrout & Fleiss (1979) ICC(2,1) point estimate

    ``ICC = (MSB - MSE) / (MSB + (k-1)*MSE + k*(MSJ - MSE)/n)``

    and the McGraw & Wong (1996) exact confidence interval for the
    absolute-agreement single-measure form (their ICC(A,1)). This is the
    *absolute-agreement* variant -- it penalises systematic between-rater
    differences (the ``MSJ`` term) and is deliberately not the
    consistency form ICC(C,1)/ICC(3,1).
    """
    r = np.asarray(ratings, dtype=float)
    if r.ndim != 2:
        raise ValueError("ratings must be a 2-D subjects x raters array")
    n, k = r.shape
    if n < 2 or k < 2:
        raise ValueError("need >= 2 subjects and >= 2 raters")
    if not np.isfinite(r).all():
        raise ValueError("ratings must not contain NaN/inf (balanced data required)")

    msb, msj, mse = _anova_mean_squares(r)
    icc = (msb - mse) / (msb + (k - 1) * mse + k * (msj - mse) / n)

    # McGraw & Wong (1996) CI for ICC(A,1).
    alpha = 1 - ci_level
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    fj = msj / mse
    vn = df2 * (k * icc * fj + n * (1 + (k - 1) * icc) - k * icc) ** 2
    vd = df1 * k**2 * icc**2 * fj**2 + (n * (1 + (k - 1) * icc) - k * icc) ** 2
    v = vn / vd
    f_upper = _f_dist.ppf(1 - alpha / 2, df1, v)
    f_lower = _f_dist.ppf(1 - alpha / 2, v, df1)
    ci_low = n * (msb - f_upper * mse) / (f_upper * (k * msj + (k * n - k - n) * mse) + n * msb)
    ci_high = n * (f_lower * msb - mse) / (k * msj + (k * n - k - n) * mse + n * f_lower * msb)

    return IccResult(
        icc=float(icc),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=ci_level,
        msb=float(msb),
        msj=float(msj),
        mse=float(mse),
        n_subjects=n,
        n_raters=k,
    )


# --------------------------------------------------------------------------- #
# Q3 -- Standard Error of Measurement and Minimal Detectable Change
# --------------------------------------------------------------------------- #
def sem(sd_pooled: float, icc: float) -> float:
    """Standard Error of Measurement.

    ``SEM = SD_pooled * sqrt(1 - ICC)``.

    Parameters
    ----------
    sd_pooled : float
        Pooled between-subjects SD of the metric, in the metric's own units
        (e.g. degrees for ``d_pitch_breath``).
    icc : float
        Reliability coefficient, typically :attr:`IccResult.icc`.

    Returns
    -------
    float
        SEM in the same units as ``sd_pooled``. The measurement-error noise
        floor of the metric.
    """
    if not 0.0 <= icc <= 1.0:
        raise ValueError("icc must be in [0, 1]")
    if sd_pooled < 0:
        raise ValueError("sd_pooled must be non-negative")
    return float(sd_pooled * np.sqrt(1.0 - icc))


def mdc95(sem_value: float) -> float:
    """Minimal Detectable Change at 95% confidence.

    ``MDC95 = 1.96 * sqrt(2) * SEM``. A change smaller than this between two
    measurements is within measurement noise.

    Parameters
    ----------
    sem_value : float
        Standard Error of Measurement, e.g. from :func:`sem`.

    Returns
    -------
    float
        MDC95 in the same units as ``sem_value``.
    """
    if sem_value < 0:
        raise ValueError("sem_value must be non-negative")
    return float(1.96 * np.sqrt(2.0) * sem_value)


# --------------------------------------------------------------------------- #
# Bland-Altman limits of agreement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlandAltmanResult:
    """Result of :func:`bland_altman`.

    Attributes
    ----------
    bias : float
        Mean difference ``mean(x - y)`` (systematic offset), in the metric's
        units.
    sd_diff : float
        Sample SD (ddof=1) of the differences.
    loa_lower, loa_upper : float
        95% limits of agreement, ``bias +/- 1.96 * sd_diff``.
    n : int
        Number of paired observations.
    mean_pairs : numpy.ndarray
        Per-pair means ``(x + y) / 2`` (the Bland-Altman x-axis).
    diffs : numpy.ndarray
        Per-pair differences ``x - y`` (the Bland-Altman y-axis).
    """

    bias: float
    sd_diff: float
    loa_lower: float
    loa_upper: float
    n: int
    mean_pairs: NDArray[np.float64] = field(repr=False)
    diffs: NDArray[np.float64] = field(repr=False)


def bland_altman(x: ArrayLike, y: ArrayLike) -> BlandAltmanResult:
    """Bland-Altman agreement analysis between two paired measurements.

    Parameters
    ----------
    x, y : array-like
        Paired measurements of the same quantity by two methods/sessions
        (e.g. test vs retest ``d_pitch_breath``), same length and units.

    Returns
    -------
    BlandAltmanResult
        ``bias`` = mean(x - y), ``sd_diff`` = SD of differences, and 95%
        limits of agreement ``bias +/- 1.96 * sd_diff``.

    Notes
    -----
    Follows Bland & Altman (1986): the differences are assumed approximately
    normal; the limits of agreement are ``bias +/- 1.96 * SD(diff)``.
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.shape != ya.shape:
        raise ValueError("x and y must have the same length")
    if xa.size < 2:
        raise ValueError("need at least 2 paired observations")

    diffs = xa - ya
    means = (xa + ya) / 2.0
    bias = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1))
    return BlandAltmanResult(
        bias=bias,
        sd_diff=sd_diff,
        loa_lower=bias - 1.96 * sd_diff,
        loa_upper=bias + 1.96 * sd_diff,
        n=xa.size,
        mean_pairs=means,
        diffs=diffs,
    )
