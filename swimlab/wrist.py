"""Wrist (forearm) placement module: calibrate -> events -> metrics, per arm.

A DOT on the wrist measures the forearm. In front crawl each arm cycles through
**catch -> pull (under water) -> exit -> recovery (over water) -> entry**, and the
gravity-referenced forearm **pitch** tracks the hand rising and falling through
that cycle: zero at the calibrated forward-horizontal reference, **negative when
the hand drops below the forward line (catch/pull, under water)** and positive on
the over-water recovery. One pull lobe = one stroke for that arm.

This module analyses **one** forearm sensor (left *or* right); the L/R
comparison is a fusion step, provided by :func:`symmetry` over the two arms'
outputs (the platform's first genuine two-sensor metric, docs/platform-design.md
section 7).

It inherits the platform rules verbatim: gravity-referenced only (magnetometer /
yaw never touched), no hardcoded thresholds (all from ``config.yaml``), never
silently drop data (excluded strokes stay with a reason code), pure functions.

Sign conventions (forearm, gravity-referenced, zero at arm forward-horizontal)
------------------------------------------------------------------------------
* ``pitch_deg`` -- **+ = hand above the forward line (recovery, over water),
  - = hand below (catch/pull, under water).**
* ``roll_deg``  -- forearm pronation/supination.
* yaw -- unused.

Calibration poses are **arm forward-horizontal + arm at the side** (~90 deg
apart). The tempting "overhead + at side" pair is ~180 deg apart and cannot span
a frame -- see :mod:`swimlab.placements`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.spatial.transform import Rotation

from swimlab import calibrate, events

__all__ = [
    "CALIB_POSE_SUSPECT",
    "REASON_PUSHOFF",
    "REASON_END",
    "calibrate_transform",
    "pose_check",
    "apply",
    "detect_events",
    "metrics",
    "symmetry",
]

CALIB_POSE_SUSPECT = calibrate.CALIB_POSE_SUSPECT
REASON_PUSHOFF = events.REASON_PUSHOFF
REASON_END = events.REASON_END

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_config(config_path: str | Path | None) -> dict:
    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    with open(path) as handle:
        return yaml.safe_load(handle)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` half-open index pairs of each ``True`` run in ``mask``."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


# --------------------------------------------------------------------------- #
# Calibration (gravity-only; forward-horizontal + arm-at-side, ~90 deg apart)
# --------------------------------------------------------------------------- #


def calibrate_transform(t0a_df: pl.DataFrame, t0b_df: pl.DataFrame) -> Rotation:
    """Recover the sensor->forearm rotation from the two arm poses.

    Same gravity-only two-pose fit as the head (:func:`swimlab.calibrate.fit_transform`);
    the forearm poses (forward-horizontal + at-side) are ~90 deg apart, so the
    identical construction applies with forearm sign conventions.
    """
    return calibrate.fit_transform(t0a_df, t0b_df)


def pose_check(
    t0a_df: pl.DataFrame,
    t0b_df: pl.DataFrame,
    *,
    config_path: str | Path | None = None,
) -> str | None:
    """Flag ``CALIB_POSE_SUSPECT`` if the two arm poses are not ~90 deg apart."""
    return calibrate.pose_check(t0a_df, t0b_df, config_path=config_path)


def apply(df: pl.DataFrame, transform: Rotation) -> pl.DataFrame:
    """Add gravity-referenced forearm ``pitch_deg`` / ``roll_deg`` to a trial."""
    return calibrate.apply(df, transform)


# --------------------------------------------------------------------------- #
# Events: per-arm strokes (pull lobes) with catch / exit / recovery timing
# --------------------------------------------------------------------------- #


def detect_events(
    calibrated: pl.DataFrame,
    trial: pl.DataFrame | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, pl.DataFrame]:
    """Detect one arm's strokes from its forearm pitch.

    A **stroke** is one under-water pull: a contiguous run where the forearm
    pitch drops ``wrist_pull_pitch_min_deg`` below its baseline (hand below the
    forward line). The run's start is the **catch** (entry into the pull), its end
    the **exit**; the gap to the next catch is the over-water **recovery**.

    Parameters
    ----------
    calibrated:
        A forearm trial with ``pitch_deg`` (from :func:`apply`) and ``acc_*``.
    trial:
        Uncalibrated canonical trial for push-off detection (accelerometer).
        Defaults to ``calibrated``.
    config_path:
        Optional override for ``config.yaml``.

    Returns
    -------
    dict
        ``{"pushoffs", "strokes"}``. **strokes** columns: ``stroke_index``,
        ``t_catch``, ``t_pull`` (deepest pitch), ``t_exit``, ``pull_duration``
        (s), ``min_pitch_deg`` (deepest, signed), ``excluded`` /
        ``exclusion_reason`` (kept, flagged -- never dropped), ``flags``.
    """
    cfg = _load_config(config_path)
    pull_min = float(cfg["wrist_pull_pitch_min_deg"])
    min_sep = float(cfg["wrist_min_stroke_separation_s"])
    after_pushoff = float(cfg["exclude_after_pushoff_s"])
    before_end = float(cfg["exclude_before_end_s"])

    acc_source = trial if trial is not None else calibrated
    pushoffs = events.detect_pushoffs(acc_source, config_path=config_path)

    t = calibrated["t"].to_numpy()
    pitch = calibrated["pitch_deg"].to_numpy()
    trial_end = float(t[-1]) if t.size else 0.0
    baseline = float(np.median(pitch)) if pitch.size else 0.0
    dev = pitch - baseline
    under = dev <= -pull_min  # hand below the forward line -> pull

    raw: list[dict] = []
    for start, stop in _contiguous_runs(under):
        seg = dev[start:stop]
        k = int(np.argmin(seg))  # deepest pull
        raw.append(
            {
                "t_catch": float(t[start]),
                "t_pull": float(t[start + k]),
                "t_exit": float(t[stop - 1]),
                "min_pitch_deg": float(pitch[start + k]),
            }
        )

    # merge lobes whose pulls fall within the refractory separation (a noisy pull
    # briefly rising above threshold would otherwise split into two strokes)
    merged: list[dict] = []
    for r in raw:
        if merged and (r["t_pull"] - merged[-1]["t_pull"]) < min_sep:
            if r["min_pitch_deg"] < merged[-1]["min_pitch_deg"]:
                merged[-1] = {**r, "t_catch": merged[-1]["t_catch"]}
            else:
                merged[-1]["t_exit"] = r["t_exit"]
            continue
        merged.append(r)

    pushoff_starts = pushoffs["t_start"].to_numpy() if pushoffs.height else np.array([])

    rows: list[dict] = []
    for i, r in enumerate(merged):
        excluded = False
        reason: str | None = None
        near_pushoff = np.any(
            (pushoff_starts <= r["t_catch"]) & (r["t_catch"] < pushoff_starts + after_pushoff)
        )
        if near_pushoff:
            excluded, reason = True, REASON_PUSHOFF
        elif (trial_end - r["t_exit"]) < before_end:
            excluded, reason = True, REASON_END
        rows.append(
            {
                "stroke_index": i,
                "t_catch": r["t_catch"],
                "t_pull": r["t_pull"],
                "t_exit": r["t_exit"],
                "pull_duration": r["t_exit"] - r["t_catch"],
                "min_pitch_deg": r["min_pitch_deg"],
                "excluded": excluded,
                "exclusion_reason": reason,
                "flags": [],
            }
        )

    schema = {
        "stroke_index": pl.Int64,
        "t_catch": pl.Float64,
        "t_pull": pl.Float64,
        "t_exit": pl.Float64,
        "pull_duration": pl.Float64,
        "min_pitch_deg": pl.Float64,
        "excluded": pl.Boolean,
        "exclusion_reason": pl.Utf8,
        "flags": pl.List(pl.Utf8),
    }
    return {"pushoffs": pushoffs, "strokes": pl.DataFrame(rows, schema=schema)}


# --------------------------------------------------------------------------- #
# Metrics (one arm)
# --------------------------------------------------------------------------- #


def metrics(
    calibrated: pl.DataFrame,
    ev: dict[str, pl.DataFrame],
    *,
    side: str | None = None,
) -> pl.DataFrame:
    """One-row per-arm metric summary from the detected strokes.

    Columns: ``side``, ``stroke_count`` / ``n_valid_strokes``,
    ``stroke_rate_cpm`` (60 / mean catch-to-catch cycle over valid strokes),
    ``mean_cycle_s``, ``mean_pull_duration_s`` / ``mean_recovery_duration_s`` and
    ``pull_fraction`` (pull / (pull + recovery)), ``pitch_amplitude_deg``
    (max - min forearm pitch), ``mean_catch_pitch_deg`` / ``mean_min_pitch_deg``,
    ``flags``.
    """
    strokes = ev["strokes"]
    excluded = (
        strokes["excluded"].fill_null(False).to_numpy().astype(bool)
        if strokes.height
        else np.zeros(0, dtype=bool)
    )
    valid = ~excluded
    n_strokes = int(strokes.height)
    n_valid = int(valid.sum())

    t_catch = strokes["t_catch"].to_numpy() if strokes.height else np.array([])
    t_exit = strokes["t_exit"].to_numpy() if strokes.height else np.array([])
    pull_dur = strokes["pull_duration"].to_numpy() if strokes.height else np.array([])
    min_pitch = strokes["min_pitch_deg"].to_numpy() if strokes.height else np.array([])

    vc = t_catch[valid]
    order = np.argsort(vc)
    vc = vc[order]
    ve = t_exit[valid][order]
    vp = pull_dur[valid][order]
    vmin = min_pitch[valid][order]

    # cycle = catch-to-catch; recovery = this exit to next catch (in-length only)
    if vc.size >= 2:
        cyc = np.diff(vc)
        in_cycle = cyc[cyc < 6.0]  # a wall turn is many seconds
        mean_cycle = float(np.mean(in_cycle)) if in_cycle.size else float("nan")
        recov = vc[1:] - ve[:-1]
        recov = recov[(recov > 0) & (np.diff(vc) < 6.0)]
        mean_recovery = float(np.mean(recov)) if recov.size else float("nan")
    else:
        mean_cycle = float("nan")
        mean_recovery = float("nan")

    mean_pull = float(np.mean(vp)) if vp.size else float("nan")
    rate = round(60.0 / mean_cycle, 3) if np.isfinite(mean_cycle) and mean_cycle > 0 else None
    if np.isfinite(mean_pull) and np.isfinite(mean_recovery) and (mean_pull + mean_recovery) > 0:
        pull_fraction = round(mean_pull / (mean_pull + mean_recovery), 4)
    else:
        pull_fraction = None

    pitch = calibrated["pitch_deg"].to_numpy()
    amplitude = float(np.max(pitch) - np.min(pitch)) if pitch.size else None

    flag_union: set[str] = set()
    if strokes.height:
        for row in strokes["flags"].to_list():
            if row:
                flag_union.update(row)

    row = {
        "side": side,
        "stroke_count": n_strokes,
        "n_valid_strokes": n_valid,
        "stroke_rate_cpm": rate,
        "mean_cycle_s": round(mean_cycle, 4) if np.isfinite(mean_cycle) else None,
        "mean_pull_duration_s": round(mean_pull, 4) if np.isfinite(mean_pull) else None,
        "mean_recovery_duration_s": round(mean_recovery, 4) if np.isfinite(mean_recovery) else None,
        "pull_fraction": pull_fraction,
        "pitch_amplitude_deg": round(amplitude, 3) if amplitude is not None else None,
        "mean_min_pitch_deg": round(float(np.mean(vmin)), 3) if vmin.size else None,
        "flags": sorted(flag_union),
    }
    schema = {
        "side": pl.Utf8,
        "stroke_count": pl.Int64,
        "n_valid_strokes": pl.Int64,
        "stroke_rate_cpm": pl.Float64,
        "mean_cycle_s": pl.Float64,
        "mean_pull_duration_s": pl.Float64,
        "mean_recovery_duration_s": pl.Float64,
        "pull_fraction": pl.Float64,
        "pitch_amplitude_deg": pl.Float64,
        "mean_min_pitch_deg": pl.Float64,
        "flags": pl.List(pl.Utf8),
    }
    return pl.DataFrame([row], schema=schema)


# --------------------------------------------------------------------------- #
# Fusion: left/right symmetry (two arms)
# --------------------------------------------------------------------------- #


def symmetry(
    metrics_r: pl.DataFrame,
    metrics_l: pl.DataFrame,
    ev_r: dict[str, pl.DataFrame] | None = None,
    ev_l: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Left/right arm symmetry -- the first genuine two-sensor fusion metric.

    Each index is ``(R - L) / mean`` (0 = symmetric, sign shows the stronger
    side). When both event tables are supplied, ``mean_phase_offset_cycles`` is
    the fraction of a cycle the left pull lags the right; front crawl is
    antiphase, so ~0.5 is expected.

    Parameters
    ----------
    metrics_r, metrics_l:
        One-row outputs of :func:`metrics` for the right and left arms.
    ev_r, ev_l:
        Optional :func:`detect_events` outputs, for the phase offset.

    Returns
    -------
    polars.DataFrame
        One row: ``stroke_count_symmetry_index``, ``rate_symmetry_index``,
        ``amplitude_symmetry_index``, ``pull_duration_symmetry_index``,
        ``mean_phase_offset_cycles`` (or ``None``).
    """
    def _idx(a, b) -> float | None:
        if a is None or b is None:
            return None
        m = 0.5 * (abs(a) + abs(b))
        return round((a - b) / m, 4) if m > 0 else None

    r = metrics_r.row(0, named=True)
    lft = metrics_l.row(0, named=True)

    phase_offset: float | None = None
    if ev_r is not None and ev_l is not None:
        pr = np.sort(ev_r["strokes"].filter(~pl.col("excluded"))["t_pull"].to_numpy())
        pl_ = np.sort(ev_l["strokes"].filter(~pl.col("excluded"))["t_pull"].to_numpy())
        if pr.size >= 2 and pl_.size >= 1:
            cyc = np.median(np.diff(pr))
            # for each right pull, the fraction of a cycle to the next left pull
            offs = []
            for tp in pr:
                later = pl_[pl_ > tp]
                if later.size:
                    frac = (later[0] - tp) / cyc
                    if 0 < frac < 1.5:
                        offs.append(frac % 1.0)
            if offs:
                phase_offset = round(float(np.median(offs)), 4)

    row = {
        "stroke_count_symmetry_index": _idx(r["stroke_count"], lft["stroke_count"]),
        "rate_symmetry_index": _idx(r["stroke_rate_cpm"], lft["stroke_rate_cpm"]),
        "amplitude_symmetry_index": _idx(r["pitch_amplitude_deg"], lft["pitch_amplitude_deg"]),
        "pull_duration_symmetry_index": _idx(
            r["mean_pull_duration_s"], lft["mean_pull_duration_s"]
        ),
        "mean_phase_offset_cycles": phase_offset,
    }
    schema = {
        "stroke_count_symmetry_index": pl.Float64,
        "rate_symmetry_index": pl.Float64,
        "amplitude_symmetry_index": pl.Float64,
        "pull_duration_symmetry_index": pl.Float64,
        "mean_phase_offset_cycles": pl.Float64,
    }
    return pl.DataFrame([row], schema=schema)
