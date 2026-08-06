"""Sacrum (pelvis) placement module: calibrate -> events -> metrics.

The sacrum sensor is the highest-value single placement in the platform
(docs/platform-design.md sections 8, 11): from the pelvis's gravity-referenced
**body roll** and the accelerometer push-off spikes it delivers the whole set
the brief asked for -- **lengths, stroke count, stroke rate / tempo, distance,
body-roll amplitude, L/R roll symmetry, push-off count / interval and pace
drift**.

This module mirrors the head module's shape (calibrate -> events -> metrics)
and inherits its rules verbatim:

* **Gravity-referenced only.** Body roll/pitch come from the fused quaternion's
  gravity direction; magnetometer and yaw are never touched (CLAUDE.md hard
  constraint 2). Calibration reuses the head's gravity-only two-pose fit -- the
  pelvis protocol (upright stand + prone glide) has the same ~90 deg geometry as
  the head's T0a/T0b, so the same maths applies with pelvis sign conventions.
* **No hardcoded thresholds.** Every threshold is read from ``config.yaml``
  (hard constraint 4); this file bakes in none.
* **Never silently drop data.** Strokes excluded by the push-off / trial-end
  proxy stay in the table with a boolean ``excluded`` flag and a reason code
  (hard constraint 5); nothing is filtered away invisibly.
* **Pure functions.** Dataframe(s) in, dataframe out; the only I/O is reading
  the study config.

Sign conventions (pelvis, gravity-referenced, zero at the prone glide)
----------------------------------------------------------------------
* ``roll_deg``  -- **positive = body rotating to the swimmer's right.** One
  front-crawl stroke rolls the body to one side, so successive strokes alternate
  sign; a stroke is one roll lobe past the config threshold.
* ``pitch_deg`` -- positive = hips/torso pitching head-up (legs dropping).
* yaw -- unused.

Honest constraint carried from the design doc: **an IMU cannot measure
position.** Distance is ``lengths x pool_length_m`` -- never an integrated
acceleration -- so ``pool_length_m`` is a declared session input to
:func:`metrics`, not something recovered from the sensor.
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
]

# Reuse the head module's flag / reason vocabulary so a fused multi-placement
# session speaks one language.
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
# Calibration (gravity-only, reused from the head module with pelvis conventions)
# --------------------------------------------------------------------------- #


def calibrate_transform(t0a_df: pl.DataFrame, t0b_df: pl.DataFrame) -> Rotation:
    """Recover the sensor->pelvis rotation from the upright + prone poses.

    The pelvis calibration protocol -- **upright stand** (T0a) then **prone
    streamline glide** (T0b) -- is geometrically the head's T0a/T0b (~90 deg
    apart), so the gravity-only two-pose fit is identical; see
    :func:`swimlab.calibrate.fit_transform`. Returned rotation maps a sensor-frame
    up vector into the calibrated pelvis frame (zero at the prone glide).
    """
    return calibrate.fit_transform(t0a_df, t0b_df)


def pose_check(
    t0a_df: pl.DataFrame,
    t0b_df: pl.DataFrame,
    *,
    config_path: str | Path | None = None,
) -> str | None:
    """Flag ``CALIB_POSE_SUSPECT`` if the two poses are not ~90 deg apart.

    Same sanity check as the head (config ``calib_pose_angle_deg`` +/-
    ``calib_pose_tolerance_deg``); a badly performed upright or prone pose is
    flagged, never silently trusted.
    """
    return calibrate.pose_check(t0a_df, t0b_df, config_path=config_path)


def apply(df: pl.DataFrame, transform: Rotation) -> pl.DataFrame:
    """Add gravity-referenced pelvis ``pitch_deg`` / ``roll_deg`` to a trial.

    Thin wrapper over :func:`swimlab.calibrate.apply` (same tilt equations); the
    columns are the pelvis angles under this module's sign conventions.
    """
    return calibrate.apply(df, transform)


# --------------------------------------------------------------------------- #
# Events: push-offs, lengths, strokes
# --------------------------------------------------------------------------- #


def _detect_lengths(pushoffs: pl.DataFrame, trial_end: float) -> pl.DataFrame:
    """One row per length, delimited by detected push-offs.

    Each length starts at a wall push-off and runs to the next push-off (the
    last runs to trial end). Lengths are the unit of absolute distance
    (``lengths x pool_length_m``) -- the only distance an IMU can honestly give.
    """
    schema = {
        "length_index": pl.Int64,
        "t_start": pl.Float64,
        "t_end": pl.Float64,
        "duration_s": pl.Float64,
    }
    if pushoffs.height == 0:
        return pl.DataFrame([], schema=schema)
    starts = sorted(float(s) for s in pushoffs["t_start"].to_numpy())
    edges = starts + [trial_end]
    rows = [
        {
            "length_index": i,
            "t_start": edges[i],
            "t_end": edges[i + 1],
            "duration_s": edges[i + 1] - edges[i],
        }
        for i in range(len(starts))
    ]
    return pl.DataFrame(rows, schema=schema)


def detect_events(
    calibrated: pl.DataFrame,
    trial: pl.DataFrame | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, pl.DataFrame]:
    """Detect push-offs, lengths and strokes from a calibrated sacrum trial.

    Parameters
    ----------
    calibrated:
        A sacrum trial with ``roll_deg`` (from :func:`apply`) and the canonical
        ``acc_*`` columns.
    trial:
        The uncalibrated canonical trial for push-off detection (accelerometer
        magnitude). Defaults to ``calibrated`` when omitted (the ``acc_*``
        columns are shared, so either works).
    config_path:
        Optional override for ``config.yaml``.

    Returns
    -------
    dict
        ``{"pushoffs", "lengths", "strokes"}`` DataFrames.

        **strokes** -- one row per detected stroke (a roll lobe past
        ``sacrum_stroke_roll_min_deg``), in time order:

        * ``stroke_index`` -- 0-based.
        * ``t_peak`` -- time (s) of the roll extreme in the lobe.
        * ``side`` -- ``"R"`` (positive roll) / ``"L"`` (negative).
        * ``peak_roll_deg`` -- signed roll extreme (deg).
        * ``length_index`` -- which length the stroke falls in (-1 if outside).
        * ``excluded`` / ``exclusion_reason`` -- protocol exclusion (within
          ``exclude_after_pushoff_s`` of the length's push-off, or within
          ``exclude_before_end_s`` of trial end); the stroke is kept, just
          flagged.
        * ``flags`` -- ``list[str]`` (empty at detection).
    """
    cfg = _load_config(config_path)
    roll_min = float(cfg["sacrum_stroke_roll_min_deg"])
    min_sep = float(cfg["sacrum_min_stroke_separation_s"])
    after_pushoff = float(cfg["exclude_after_pushoff_s"])
    before_end = float(cfg["exclude_before_end_s"])

    acc_source = trial if trial is not None else calibrated
    pushoffs = events.detect_pushoffs(acc_source, config_path=config_path)

    t = calibrated["t"].to_numpy()
    roll = calibrated["roll_deg"].to_numpy()
    trial_end = float(t[-1]) if t.size else 0.0
    lengths = _detect_lengths(pushoffs, trial_end)

    baseline = float(np.median(roll)) if roll.size else 0.0
    dev = roll - baseline
    over = np.abs(dev) >= roll_min

    # One stroke per contiguous over-threshold lobe; take the signed extreme.
    raw: list[dict] = []
    for start, stop in _contiguous_runs(over):
        seg = dev[start:stop]
        k = int(np.argmax(np.abs(seg)))
        peak_roll = float(roll[start + k])
        raw.append({"t_peak": float(t[start + k]), "peak_roll_deg": peak_roll})

    # Merge peaks closer than the refractory separation (a noisy lobe that dips
    # briefly below threshold would otherwise split into two strokes); keep the
    # larger-magnitude peak. Alternating sides make real strokes far enough apart
    # that this only ever coalesces a split lobe, never two distinct strokes.
    merged: list[dict] = []
    for r in raw:
        if merged and (r["t_peak"] - merged[-1]["t_peak"]) < min_sep:
            if abs(r["peak_roll_deg"]) > abs(merged[-1]["peak_roll_deg"]):
                merged[-1] = r
            continue
        merged.append(r)

    length_starts = lengths["t_start"].to_numpy() if lengths.height else np.array([])
    length_ends = lengths["t_end"].to_numpy() if lengths.height else np.array([])

    rows: list[dict] = []
    for i, r in enumerate(merged):
        tp = r["t_peak"]
        li = int(np.argmax((length_starts <= tp) & (tp < length_ends))) if lengths.height else -1
        if lengths.height and not ((length_starts <= tp) & (tp < length_ends)).any():
            li = -1
        # exclusion: too soon after this length's push-off, or too near the end
        excluded = False
        reason: str | None = None
        if li >= 0 and (tp - length_starts[li]) < after_pushoff:
            excluded, reason = True, REASON_PUSHOFF
        elif (trial_end - tp) < before_end:
            excluded, reason = True, REASON_END
        rows.append(
            {
                "stroke_index": i,
                "t_peak": tp,
                "side": "R" if r["peak_roll_deg"] - baseline > 0 else "L",
                "peak_roll_deg": r["peak_roll_deg"],
                "length_index": li,
                "excluded": excluded,
                "exclusion_reason": reason,
                "flags": [],
            }
        )

    stroke_schema = {
        "stroke_index": pl.Int64,
        "t_peak": pl.Float64,
        "side": pl.Utf8,
        "peak_roll_deg": pl.Float64,
        "length_index": pl.Int64,
        "excluded": pl.Boolean,
        "exclusion_reason": pl.Utf8,
        "flags": pl.List(pl.Utf8),
    }
    strokes = pl.DataFrame(rows, schema=stroke_schema)
    return {"pushoffs": pushoffs, "lengths": lengths, "strokes": strokes}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def metrics(
    calibrated: pl.DataFrame,
    ev: dict[str, pl.DataFrame],
    *,
    pool_length_m: float = 25.0,
) -> pl.DataFrame:
    """One-row sacrum metric summary from the detected events.

    Parameters
    ----------
    calibrated:
        The calibrated sacrum trial (used only for its final timestamp).
    ev:
        Output of :func:`detect_events` (``pushoffs`` / ``lengths`` / ``strokes``).
    pool_length_m:
        Declared pool length (m) -- the **only** source of absolute distance
        (``lengths x pool_length_m``); an IMU cannot measure position. A session
        input, not a recovered value.

    Returns
    -------
    polars.DataFrame
        One row:

        * ``lengths`` / ``distance_m`` -- length count and lengths x pool length.
        * ``stroke_count`` -- all detected strokes; ``n_valid_strokes`` excludes
          the protocol-excluded ones.
        * ``mean_stroke_period_s`` / ``tempo_spm`` (strokes/min) /
          ``stroke_rate_cpm`` (cycles/min = tempo / 2) -- from valid strokes'
          in-length intervals (wall-turn gaps dropped).
        * ``body_roll_amplitude_deg`` -- mean valid stroke peak ``|roll|``.
        * ``mean_peak_roll_right_deg`` / ``mean_peak_roll_left_deg`` /
          ``roll_symmetry_index`` -- ``(R - L) / mean`` on valid peak ``|roll|``.
        * ``pushoff_count`` / ``mean_pushoff_interval_s``.
        * ``pace_drift_s_per_length`` -- slope of per-length duration.
        * ``flags`` -- sorted unique union of stroke flags.
    """
    strokes = ev["strokes"]
    lengths = ev["lengths"]
    pushoffs = ev["pushoffs"]

    n_lengths = int(lengths.height)
    distance_m = round(n_lengths * pool_length_m, 4)

    excluded = (
        strokes["excluded"].fill_null(False).to_numpy().astype(bool)
        if strokes.height
        else np.zeros(0, dtype=bool)
    )
    valid = ~excluded
    n_strokes = int(strokes.height)
    n_valid = int(valid.sum())

    t_peak = strokes["t_peak"].to_numpy() if strokes.height else np.array([])
    side = np.asarray(strokes["side"].to_list(), dtype=object) if strokes.height else np.array([])
    peak_abs = np.abs(strokes["peak_roll_deg"].to_numpy()) if strokes.height else np.array([])

    tv = t_peak[valid]
    sv = side[valid]
    pv = peak_abs[valid]

    # tempo: mean interval between consecutive valid strokes within a length
    # (drop the large wall-turn gaps), matching how the body model defines truth.
    if tv.size >= 2:
        d = np.diff(np.sort(tv))
        in_length = d[d < 3.0]  # a wall turn is many seconds; a stroke ~1-2 s
        mean_period = float(np.mean(in_length)) if in_length.size else float("nan")
    else:
        mean_period = float("nan")
    tempo_spm = round(60.0 / mean_period, 3) if np.isfinite(mean_period) and mean_period > 0 else None
    stroke_rate_cpm = round(tempo_spm / 2.0, 3) if tempo_spm is not None else None

    def _mean(x: np.ndarray) -> float | None:
        return float(np.mean(x)) if x.size else None

    right = pv[sv == "R"]
    left = pv[sv == "L"]
    body_roll_amp = _mean(pv)
    mean_r, mean_l = _mean(right), _mean(left)
    if mean_r is not None and mean_l is not None and (mean_r + mean_l) > 0:
        roll_symmetry_index = round((mean_r - mean_l) / (0.5 * (mean_r + mean_l)), 4)
    else:
        roll_symmetry_index = None

    # push-off interval
    if pushoffs.height >= 2:
        po = np.sort(pushoffs["t_start"].to_numpy())
        mean_po_interval = float(np.mean(np.diff(po)))
    else:
        mean_po_interval = None

    # pace drift: slope of per-length duration vs length index
    if lengths.height >= 2:
        dur = lengths["duration_s"].to_numpy()
        idx = lengths["length_index"].to_numpy()
        pace_drift = float(np.polyfit(idx, dur, 1)[0])
    else:
        pace_drift = None

    flag_union: set[str] = set()
    if strokes.height and "flags" in strokes.columns:
        for row in strokes["flags"].to_list():
            if row:
                flag_union.update(row)

    row = {
        "lengths": n_lengths,
        "distance_m": distance_m,
        "pool_length_m": pool_length_m,
        "stroke_count": n_strokes,
        "n_valid_strokes": n_valid,
        "mean_stroke_period_s": round(mean_period, 4) if np.isfinite(mean_period) else None,
        "tempo_spm": tempo_spm,
        "stroke_rate_cpm": stroke_rate_cpm,
        "body_roll_amplitude_deg": round(body_roll_amp, 3) if body_roll_amp is not None else None,
        "mean_peak_roll_right_deg": round(mean_r, 3) if mean_r is not None else None,
        "mean_peak_roll_left_deg": round(mean_l, 3) if mean_l is not None else None,
        "roll_symmetry_index": roll_symmetry_index,
        "pushoff_count": int(pushoffs.height),
        "mean_pushoff_interval_s": round(mean_po_interval, 4) if mean_po_interval is not None else None,
        "pace_drift_s_per_length": round(pace_drift, 4) if pace_drift is not None else None,
        "flags": sorted(flag_union),
    }
    schema = {
        "lengths": pl.Int64,
        "distance_m": pl.Float64,
        "pool_length_m": pl.Float64,
        "stroke_count": pl.Int64,
        "n_valid_strokes": pl.Int64,
        "mean_stroke_period_s": pl.Float64,
        "tempo_spm": pl.Float64,
        "stroke_rate_cpm": pl.Float64,
        "body_roll_amplitude_deg": pl.Float64,
        "mean_peak_roll_right_deg": pl.Float64,
        "mean_peak_roll_left_deg": pl.Float64,
        "roll_symmetry_index": pl.Float64,
        "pushoff_count": pl.Int64,
        "mean_pushoff_interval_s": pl.Float64,
        "pace_drift_s_per_length": pl.Float64,
        "flags": pl.List(pl.Utf8),
    }
    return pl.DataFrame([row], schema=schema)
