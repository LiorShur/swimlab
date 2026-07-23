"""Metric table for a calibrated swimlab trial (CLAUDE.md "Metric definitions").

This module turns a *calibrated* trial (``pitch_deg`` / ``roll_deg`` from
:func:`swimlab.calibrate.apply`) plus the detected/exclusion-marked breath
table (from :mod:`swimlab.events`) into the study's metrics:

* a **per-breath** table -- one row per detected breath window, and
* a **per-trial summary** -- one row aggregating the *valid* (non-excluded)
  breaths.

Everything here is a **pure function**: dataframe(s) in, dataframe out, no I/O
(beyond an optional ``config.yaml`` read that this module does not actually
need -- see below), no globals mutated.

Sign / unit conventions (inherited from calibrate.py, restated per CLAUDE.md)
---------------------------------------------------------------------------
* ``pitch_deg`` -- head pitch, degrees, **positive = nose up (head lift)**,
  zero at the T0b prone reference.
* ``roll_deg`` -- head roll, degrees, **positive = face to the swimmer's
  right**. ``side`` ``"R"`` = positive roll, ``"L"`` = negative.
* Magnetometer and yaw are never referenced (hard constraint 2).

No hardcoded study thresholds (hard constraint 4)
-------------------------------------------------
Every metric here is a pure arithmetic combination of the signal and the
breath timing -- there is **no study threshold baked into this file**. The one
free parameter, the length of the "preceding non-breath cycle" baseline
window, is *derived from the data* (the median inter-breath interval), not
hardcoded, and can be overridden by the caller. The tiny epsilon used to guard
``roll_pitch_ratio`` against a divide-by-zero is a numerical safety, not a
study parameter (a *detected* breath window already clears the config roll
threshold by construction, so a near-zero ``peak_roll_breath`` should never
occur in practice).

Never silently drop data (hard constraint 5)
--------------------------------------------
Every breath the detector produced stays in the per-breath table. The
``excluded`` / ``exclusion_reason`` columns from :func:`events.apply_exclusions`
are carried through untouched, and any metric-level problem (empty baseline
window, degenerate roll) is recorded as a reason code in ``metric_flags`` --
never by removing the row. The **summary** aggregates only the valid
(non-excluded) breaths, and says so.

The ``d_pitch_breath`` baseline window
--------------------------------------
``d_pitch_breath`` = peak pitch in the breath window minus the median pitch
over the *preceding non-breath cycles*. The preceding window is taken as::

    [ max(previous_breath_end, t_start - L),  t_start )

where ``L`` is **one stroke-cycle** -- a cycle is two strokes, i.e. exactly one
period of the body-roll (and head-pitch) oscillation. ``L`` is estimated
*from the data* as the dominant period of the roll signal (the first peak of
the roll autocorrelation; see :func:`_estimate_cycle_s`), not hardcoded and not
read from any generator parameter. Using one whole oscillation period means the
baseline median is taken over an integer number of stroke-cycle pitch
oscillations, so it is not biased by a phase-dependent partial slice (a
non-integer window such as the 3-stroke inter-breath interval is 1.5 cycles and
*is* so biased). Clipping the lower edge to the previous breath's end keeps the
baseline strictly inside non-breath pitch; the ``t_start - L`` cap keeps it to
the one cycle immediately before the breath (reaching further, e.g. two cycles,
lets the window overlap the previous breath's pitch bump).

Recovery of the synthetic ``true_d_pitch_deg`` (zero noise): the trial mean is
within ~0.5 deg for breaths whose roll excursion is modest, but a systematic,
*roll-dependent* positive residual remains for large-roll breaths. That
residual is **not** a baseline-window artifact -- it does not depend on the
baseline window length or phase, and it scales with the breath's peak roll
(~+0.04 deg of d_pitch over-read per degree of peak roll). It comes from the
gravity-tilt pitch/roll decode in :func:`swimlab.calibrate.apply`: at the
breath apex head pitch and head roll are simultaneously large, and the tilt
equations then cross-talk a little roll into the reported pitch. Because the
synthetic ground truth is an intrinsic-Euler pitch, tilt-decoded pitch reads a
bit high exactly where roll is largest. This is a property of the measurement
model (calibrate + the metric definition), documented and quantified in
``tests/test_metrics.py``, not something the baseline window can remove.

``pitch_drift_100m`` is defined for the T9 drift protocol, not T7; it is
implemented here as a per-length-median-pitch slope that runs on any
multi-length trial but is only *meaningful* on T9.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = [
    "REASON_NO_BASELINE",
    "REASON_ROLL_DEGENERATE",
    "per_breath_metrics",
    "trial_summary",
    "pitch_drift_per_length",
]

# Metric-level reason codes (accumulate in ``metric_flags``; never raise, never
# drop -- hard constraint 5).
REASON_NO_BASELINE = "NO_BASELINE_SAMPLES"
REASON_ROLL_DEGENERATE = "ROLL_NEAR_ZERO"

# Numerical divide-by-zero guard for roll_pitch_ratio (degrees). This is a
# floating-point safety, *not* a study threshold: a detected breath window
# already exceeds the config roll threshold, so peak_roll_breath is many orders
# of magnitude above this in any real row.
_ROLL_EPS_DEG = 1e-9


def _require_columns(df: pl.DataFrame, cols: tuple[str, ...], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: dataframe is missing column(s) {missing}")


def _estimate_cycle_s(t: np.ndarray, roll: np.ndarray) -> float:
    """Estimate one stroke-cycle (s) as the dominant period of the roll signal.

    A front-crawl stroke-cycle is two strokes and the body rolls through exactly
    one full ``+/-`` oscillation per cycle, so the fundamental period of
    ``roll_deg`` *is* the cycle length. It is recovered as the first peak of the
    (biased) autocorrelation of the mean-removed roll, searched in a plausible
    physiological band (0.8-5 s). This is fully data-driven -- no hardcoded
    cycle length and nothing read from the synthetic generator.

    The search band edges are not study thresholds: they only bracket "a
    human swim stroke-cycle" so the autocorrelation peak-finder ignores the
    zero-lag spike and any implausibly long lag. Returns ``nan`` when the
    signal is too short to have a period in band; callers then fall back to a
    look-back-to-trial-start baseline.
    """
    if t.size < 4:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    fs = 1.0 / dt
    x = roll - float(np.mean(roll))
    n = x.size
    ac = np.correlate(x, x, mode="full")[n - 1 :]
    if ac[0] == 0:
        return float("nan")
    ac = ac / ac[0]
    lo = int(0.8 * fs)
    hi = min(int(5.0 * fs), n - 1)
    if hi <= lo + 1:
        return float("nan")
    d = np.diff(ac)
    for k in range(lo, hi):
        if d[k - 1] > 0.0 and d[k] <= 0.0:  # first local maximum after lo
            return k / fs
    # no clear peak: fall back to the largest autocorrelation lag in band
    return (lo + int(np.argmax(ac[lo:hi]))) / fs


def per_breath_metrics(
    trial: pl.DataFrame,
    breaths: pl.DataFrame,
    *,
    baseline_window_s: float | None = None,
) -> pl.DataFrame:
    """Compute the per-breath metric table (one row per detected breath).

    Parameters
    ----------
    trial:
        A *calibrated* trial dataframe: the canonical schema plus ``pitch_deg``
        and ``roll_deg`` (degrees) from :func:`swimlab.calibrate.apply`. Only
        ``t``, ``pitch_deg`` and ``roll_deg`` are read here.
    breaths:
        The breath table from :func:`swimlab.events.detect_breath_windows`,
        preferably after :func:`swimlab.events.apply_exclusions` (so it carries
        ``excluded`` / ``exclusion_reason`` / ``flags``). Needs at least
        ``t_start``, ``t_end`` and ``side``. Rows are processed in the given
        (time) order; the baseline of breath *i* is clipped to end of breath
        *i-1*, so the table must be time-sorted (``events`` returns it sorted).
    baseline_window_s:
        Override the length ``L`` (s) of the preceding-non-breath baseline
        window. When ``None`` (default) it is derived from the data as one
        stroke-cycle -- the dominant period of the roll signal
        (:func:`_estimate_cycle_s`).

    Returns
    -------
    polars.DataFrame
        One row per breath, in the input order, with (in addition to the
        pass-through breath columns ``t_start``, ``t_end``, ``side``,
        ``peak_roll_deg``, ``baseline_roll_deg``, ``excluded``,
        ``exclusion_reason``, ``flags`` where present):

        * ``breath_index`` -- 0-based position in the table.
        * ``breath_duration`` -- window duration, ``t_end - t_start`` (s).
        * ``peak_pitch_deg`` -- max ``pitch_deg`` in ``[t_start, t_end]`` (deg,
          positive = nose up).
        * ``baseline_pitch_deg`` -- median ``pitch_deg`` over the preceding
          non-breath window (deg).
        * ``d_pitch_breath`` -- ``peak_pitch_deg - baseline_pitch_deg`` (deg).
          The **primary gate** metric.
        * ``peak_roll_breath`` -- max ``|roll_deg|`` in the window (deg, >= 0).
        * ``roll_pitch_ratio`` -- ``d_pitch_breath / peak_roll_breath``
          (dimensionless); ``None`` (with reason ``ROLL_NEAR_ZERO``) when
          ``peak_roll_breath`` is numerically ~0.
        * ``metric_flags`` -- ``list[str]`` of metric-level reason codes
          (empty when clean). Never used to drop a row.

    Notes
    -----
    Breaths whose true roll never reaches the config threshold (many ``FLAT``
    breaths) are *not* in ``breaths`` at all -- the roll detector cannot window
    them (see :func:`events.detect_breath_windows`). They are undetectable from
    roll alone, not dropped here; their absence shows up as a low valid-breath
    count / ``INSUFFICIENT_CYCLES`` in the summary. This function cannot
    reconstruct a window the detector never produced.
    """
    _require_columns(trial, ("t", "pitch_deg", "roll_deg"), "per_breath_metrics(trial)")
    _require_columns(breaths, ("t_start", "t_end", "side"), "per_breath_metrics(breaths)")

    t = trial["t"].to_numpy()
    pitch = trial["pitch_deg"].to_numpy()
    roll = trial["roll_deg"].to_numpy()

    t_start = breaths["t_start"].to_numpy()
    t_end = breaths["t_end"].to_numpy()

    L = baseline_window_s
    if L is None:
        L = _estimate_cycle_s(t, roll)

    n = breaths.height
    breath_duration = np.empty(n, dtype=np.float64)
    peak_pitch = np.empty(n, dtype=np.float64)
    baseline_pitch = np.empty(n, dtype=np.float64)
    d_pitch = np.empty(n, dtype=np.float64)
    peak_roll = np.empty(n, dtype=np.float64)
    ratio: list[float | None] = []
    metric_flags: list[list[str]] = []

    for i in range(n):
        ts = float(t_start[i])
        te = float(t_end[i])
        flags_i: list[str] = []

        in_window = (t >= ts) & (t <= te)
        w_pitch = pitch[in_window]
        w_roll = roll[in_window]

        breath_duration[i] = te - ts
        peak_pitch[i] = float(w_pitch.max()) if w_pitch.size else float("nan")
        peak_roll[i] = float(np.abs(w_roll).max()) if w_roll.size else float("nan")

        # Preceding non-breath baseline: [max(prev_end, ts - L), ts).
        lo = ts - L if np.isfinite(L) else -np.inf
        if i > 0:
            lo = max(lo, float(t_end[i - 1]))
        base_mask = (t >= lo) & (t < ts)
        if not base_mask.any():
            # Nothing between the previous breath and this one -- fall back to a
            # one-cycle lookback (or, failing that, all pitch before ts). Flag
            # it rather than silently substituting.
            fallback = (t >= ts - L) & (t < ts) if np.isfinite(L) else (t < ts)
            base_mask = fallback if fallback.any() else (t < ts)
            flags_i.append(REASON_NO_BASELINE)
        base_pitch_vals = pitch[base_mask]
        baseline_pitch[i] = (
            float(np.median(base_pitch_vals)) if base_pitch_vals.size else float("nan")
        )

        d_pitch[i] = peak_pitch[i] - baseline_pitch[i]

        if not np.isfinite(peak_roll[i]) or peak_roll[i] < _ROLL_EPS_DEG:
            ratio.append(None)
            flags_i.append(REASON_ROLL_DEGENERATE)
        else:
            ratio.append(float(d_pitch[i] / peak_roll[i]))

        metric_flags.append(flags_i)

    result = breaths.with_columns(
        pl.Series("breath_index", np.arange(n, dtype=np.int64)),
        pl.Series("breath_duration", breath_duration),
        pl.Series("peak_pitch_deg", peak_pitch),
        pl.Series("baseline_pitch_deg", baseline_pitch),
        pl.Series("d_pitch_breath", d_pitch),
        pl.Series("peak_roll_breath", peak_roll),
        pl.Series("roll_pitch_ratio", pl.Series(ratio, dtype=pl.Float64)),
        pl.Series("metric_flags", metric_flags, dtype=pl.List(pl.Utf8)),
    )
    return result


def _valid_mask(per_breath: pl.DataFrame) -> np.ndarray:
    """Boolean mask of breaths that count toward the summary aggregates.

    A breath is *valid* when it is not protocol-excluded. When the table has no
    ``excluded`` column (i.e. exclusions were never applied) every breath is
    treated as valid, and the caller is responsible for having done so
    knowingly.
    """
    if "excluded" in per_breath.columns:
        excluded = per_breath["excluded"].fill_null(False).to_numpy().astype(bool)
        return ~excluded
    return np.ones(per_breath.height, dtype=bool)


def trial_summary(per_breath: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the per-breath table into a one-row per-trial summary.

    All aggregate metrics are computed over the **valid** (non-excluded)
    breaths only; excluded breaths are counted but never averaged in. This
    matches the synthetic ground truth's ``summary`` (which is over
    ``n_valid_breaths``).

    Parameters
    ----------
    per_breath:
        Output of :func:`per_breath_metrics`.

    Returns
    -------
    polars.DataFrame
        A single row with:

        * ``n_breaths`` -- total detected breaths in the table.
        * ``n_valid`` -- non-excluded breaths (used for the aggregates).
        * ``n_excluded`` -- protocol-excluded breaths.
        * ``mean_d_pitch_breath`` / ``median_d_pitch_breath`` -- deg.
        * ``pitch_variability`` -- SD of ``d_pitch_breath`` across valid
          breaths (deg; sample SD, ``None`` when < 2 valid breaths).
        * ``mean_peak_roll_breath`` -- deg.
        * ``mean_roll_pitch_ratio`` -- dimensionless (over valid breaths whose
          ratio is defined).
        * ``mean_breath_duration`` -- s.
        * ``asymmetry_index`` -- ``(mean_left - mean_right) / mean_all`` on
          valid ``d_pitch_breath``, split by breathing ``side``. Negative when
          the right-side breath lifts more than the left. ``None`` when either
          side is unrepresented among valid breaths.
        * ``n_valid_left`` / ``n_valid_right`` -- valid breath counts per side.
        * ``flags`` -- sorted unique union of the pass-through ``flags`` across
          all breaths (e.g. ``INSUFFICIENT_CYCLES``), ``[]`` when none.
        * ``metric_flags`` -- sorted unique union of metric-level reason codes.
    """
    _require_columns(per_breath, ("d_pitch_breath", "side"), "trial_summary")

    valid = _valid_mask(per_breath)
    n_total = per_breath.height
    n_valid = int(valid.sum())
    n_excluded = n_total - n_valid

    d = per_breath["d_pitch_breath"].to_numpy()
    side = np.asarray(per_breath["side"].to_list(), dtype=object)

    dv = d[valid]
    sv = side[valid]

    def _mean(x: np.ndarray) -> float | None:
        return float(np.mean(x)) if x.size else None

    mean_d = _mean(dv)
    median_d = float(np.median(dv)) if dv.size else None
    pitch_var = float(np.std(dv, ddof=1)) if dv.size >= 2 else None

    left = dv[sv == "L"]
    right = dv[sv == "R"]
    if left.size and right.size and dv.size:
        asymmetry_index: float | None = float(
            (np.mean(left) - np.mean(right)) / np.mean(dv)
        )
    else:
        asymmetry_index = None

    peak_roll = per_breath["peak_roll_breath"].to_numpy()[valid]
    mean_peak_roll = _mean(peak_roll)

    dur = per_breath["breath_duration"].to_numpy()[valid]
    mean_dur = _mean(dur)

    ratio = per_breath["roll_pitch_ratio"].to_numpy()[valid]
    ratio_defined = ratio[np.isfinite(ratio)]
    mean_ratio = _mean(ratio_defined)

    def _union(col: str) -> list[str]:
        if col not in per_breath.columns:
            return []
        acc: set[str] = set()
        for row in per_breath[col].to_list():
            if row:
                acc.update(row)
        return sorted(acc)

    row = {
        "n_breaths": n_total,
        "n_valid": n_valid,
        "n_excluded": n_excluded,
        "n_valid_left": int(left.size),
        "n_valid_right": int(right.size),
        "mean_d_pitch_breath": mean_d,
        "median_d_pitch_breath": median_d,
        "pitch_variability": pitch_var,
        "mean_peak_roll_breath": mean_peak_roll,
        "mean_roll_pitch_ratio": mean_ratio,
        "mean_breath_duration": mean_dur,
        "asymmetry_index": asymmetry_index,
        "flags": _union("flags"),
        "metric_flags": _union("metric_flags"),
    }
    schema = {
        "n_breaths": pl.Int64,
        "n_valid": pl.Int64,
        "n_excluded": pl.Int64,
        "n_valid_left": pl.Int64,
        "n_valid_right": pl.Int64,
        "mean_d_pitch_breath": pl.Float64,
        "median_d_pitch_breath": pl.Float64,
        "pitch_variability": pl.Float64,
        "mean_peak_roll_breath": pl.Float64,
        "mean_roll_pitch_ratio": pl.Float64,
        "mean_breath_duration": pl.Float64,
        "asymmetry_index": pl.Float64,
        "flags": pl.List(pl.Utf8),
        "metric_flags": pl.List(pl.Utf8),
    }
    return pl.DataFrame([row], schema=schema)


def pitch_drift_per_length(
    trial: pl.DataFrame,
    pushoffs: pl.DataFrame,
) -> pl.DataFrame:
    """Per-length median pitch and its slope across the trial (drift).

    This is the ``pitch_drift_100m`` metric. It is **defined for the T9 drift
    protocol**, not T7 -- on a 4x25 m T7 trial it is computed but not
    physiologically meaningful. It runs on any multi-length trial: lengths are
    delimited by the detected push-offs, the median ``pitch_deg`` of each length
    is taken, and the slope (deg per length) of that per-length median series is
    the drift.

    Parameters
    ----------
    trial:
        Calibrated trial dataframe (needs ``t`` and ``pitch_deg``).
    pushoffs:
        Output of :func:`swimlab.events.detect_pushoffs`; each ``t_start`` marks
        the beginning of a new length. Lengths run from one push-off to the
        next (and from the last push-off to trial end).

    Returns
    -------
    polars.DataFrame
        One row per length with ``length_index`` (0-based), ``t_start``,
        ``t_end`` and ``median_pitch_deg``, plus a repeated
        ``pitch_drift_per_length`` column holding the ordinary-least-squares
        slope (deg per length) of ``median_pitch_deg`` vs ``length_index``.
        The slope is ``None`` (and the frame may still list the lengths) when
        fewer than two lengths carry pitch samples.
    """
    _require_columns(trial, ("t", "pitch_deg"), "pitch_drift_per_length(trial)")

    t = trial["t"].to_numpy()
    pitch = trial["pitch_deg"].to_numpy()
    if t.size == 0:
        return pl.DataFrame(
            [],
            schema={
                "length_index": pl.Int64,
                "t_start": pl.Float64,
                "t_end": pl.Float64,
                "median_pitch_deg": pl.Float64,
                "pitch_drift_per_length": pl.Float64,
            },
        )

    if pushoffs.height:
        starts = list(pushoffs["t_start"].to_numpy())
    else:
        starts = [float(t[0])]
    # Ensure the first length starts at (or before) the trial start.
    bounds = sorted({float(t[0]), *[float(s) for s in starts]})
    edges = bounds + [float(t[-1]) + 1e-9]  # right-open final edge covers t[-1]

    rows = []
    medians: list[float] = []
    for li in range(len(edges) - 1):
        lo, hi = edges[li], edges[li + 1]
        mask = (t >= lo) & (t < hi)
        seg = pitch[mask]
        med = float(np.median(seg)) if seg.size else None
        rows.append(
            {
                "length_index": li,
                "t_start": lo,
                "t_end": hi,
                "median_pitch_deg": med,
            }
        )
        if med is not None:
            medians.append(med)

    idx = np.array([r["length_index"] for r in rows if r["median_pitch_deg"] is not None])
    vals = np.array(medians)
    slope: float | None
    if idx.size >= 2:
        slope = float(np.polyfit(idx, vals, 1)[0])
    else:
        slope = None

    for r in rows:
        r["pitch_drift_per_length"] = slope

    schema = {
        "length_index": pl.Int64,
        "t_start": pl.Float64,
        "t_end": pl.Float64,
        "median_pitch_deg": pl.Float64,
        "pitch_drift_per_length": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)
