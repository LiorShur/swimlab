"""Event detection for swimlab trials: push-offs and breath windows.

This module locates the two protocol events the downstream metric layer keys
off of:

* **Push-offs** -- the wall push at the start of each length, found from a
  spike in accelerometer *magnitude*. The canonical accelerometer includes
  gravity, so a still sensor reads ~1 g; stroke surges stay well under 2 g and
  a wall push peaks around 3 g. The trigger threshold (in g) lives in
  ``config.yaml`` as ``pushoff_g_threshold`` -- never hardcoded here
  (CLAUDE.md hard constraint 4).

* **Breath windows** -- detected from the gravity-referenced *roll* angle
  (``roll_deg`` from :func:`swimlab.calibrate.apply`; magnetometer/yaw are
  never touched, hard constraint 2). A breath window is, per CLAUDE.md, the
  interval where roll crosses ``breath_roll_threshold_deg`` away from its
  baseline -- first crossing to return. Side follows the CLAUDE.md sign
  convention: **positive roll = face to the swimmer's right = ``"R"``**,
  negative = ``"L"``.

Exclusions and insufficient-cycle handling follow CLAUDE.md hard constraint 5
("never silently drop data"): rejected breaths stay in the table with a
boolean ``excluded`` flag, a reason code, and an accumulating ``flags``
column. Nothing is filtered away invisibly.

All functions are pure: dataframe(s) in, dataframe out. No globals mutated, no
I/O beyond reading the study config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.spatial.transform import Rotation

from swimlab import calibrate

__all__ = [
    "INSUFFICIENT_CYCLES",
    "REASON_PUSHOFF",
    "REASON_END",
    "detect_pushoffs",
    "detect_breath_windows",
    "apply_exclusions",
]

# Quality flag surfaced when too few valid breaths remain (CLAUDE.md exclusion
# rules: "< 20 valid cycles -> mark participant INSUFFICIENT_CYCLES").
INSUFFICIENT_CYCLES = "INSUFFICIENT_CYCLES"

# Exclusion reason codes. These strings match the synthetic ground-truth
# generator's ``exclusion_reason`` values exactly so the two can be compared
# without a translation table.
REASON_PUSHOFF = "within_4s_of_pushoff"
REASON_END = "within_2s_of_end"

# Standard gravity, m/s^2. Only used to convert the canonical accelerometer
# (m/s^2, gravity-inclusive) to g so the config threshold can stay in g.
_G = 9.80665

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_config(config_path: str | Path | None) -> dict:
    """Load the study config (``config.yaml``). All thresholds live there."""
    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    with open(path) as handle:
        return yaml.safe_load(handle)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` index pairs (half-open) of each ``True`` run."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def detect_pushoffs(
    df: pl.DataFrame,
    config_path: str | Path | None = None,
) -> pl.DataFrame:
    """Detect length-boundary push-offs from an accelerometer magnitude spike.

    A push-off is a contiguous run of samples whose accelerometer magnitude
    ``|acc|`` exceeds ``config.pushoff_g_threshold`` (expressed in g). The
    accelerometer is gravity-inclusive, so the still baseline is ~1 g and only
    the wall push (~3 g) clears a ~2 g threshold.

    Parameters
    ----------
    df:
        Canonical dataframe (needs ``t`` and ``acc_x/y/z`` in m/s^2).
    config_path:
        Optional override for ``config.yaml``.

    Returns
    -------
    polars.DataFrame
        One row per push-off, in time order, with columns:

        * ``t_start`` -- time (s) of the first sample over threshold.
        * ``t_peak``  -- time (s) of the magnitude maximum in the run.
        * ``t_end``   -- time (s) of the last sample over threshold.
        * ``peak_g``  -- peak magnitude of the run, in g.

        Empty (with the same schema) when no push-off is found.
    """
    cfg = _load_config(config_path)
    threshold_g = float(cfg["pushoff_g_threshold"])

    t = df["t"].to_numpy()
    acc = df.select(["acc_x", "acc_y", "acc_z"]).to_numpy()
    mag_g = np.linalg.norm(acc, axis=1) / _G

    over = mag_g > threshold_g
    rows = []
    for start, stop in _contiguous_runs(over):
        seg = mag_g[start:stop]
        peak_local = int(np.argmax(seg))
        rows.append(
            {
                "t_start": float(t[start]),
                "t_peak": float(t[start + peak_local]),
                "t_end": float(t[stop - 1]),
                "peak_g": float(seg[peak_local]),
            }
        )

    schema = {
        "t_start": pl.Float64,
        "t_peak": pl.Float64,
        "t_end": pl.Float64,
        "peak_g": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def detect_breath_windows(
    df: pl.DataFrame,
    transform: Rotation | None = None,
    config_path: str | Path | None = None,
) -> pl.DataFrame:
    """Detect breath windows from gravity-referenced roll.

    Per CLAUDE.md, a breath window is where roll crosses
    ``breath_roll_threshold_deg`` away from its baseline -- from the first
    crossing to the return back inside the band. Baseline is the median of
    ``roll_deg`` over the trial (robust to the brief per-breath excursions).

    The detector operates on ``roll_deg`` from :func:`swimlab.calibrate.apply`.
    If ``df`` already carries a ``roll_deg`` column it is used directly;
    otherwise ``transform`` (the sensor->skull rotation from
    :func:`swimlab.calibrate.fit_transform`) must be supplied and the roll is
    computed via ``calibrate.apply``. Only pitch/roll from the fused
    quaternion are used -- magnetometer and yaw are never referenced
    (hard constraint 2).

    Parameters
    ----------
    df:
        Canonical trial dataframe (needs ``t`` and either ``roll_deg`` or the
        ``quat_*`` columns plus a ``transform``).
    transform:
        Sensor->skull rotation; required only when ``roll_deg`` is absent.
    config_path:
        Optional override for ``config.yaml``.

    Returns
    -------
    polars.DataFrame
        One row per breath, in time order, with columns:

        * ``t_start`` -- window start (s), first threshold crossing.
        * ``t_end``   -- window end (s), return inside the band.
        * ``side``    -- ``"R"`` (positive roll, face to swimmer's right) or
          ``"L"`` (negative roll).
        * ``peak_roll_deg`` -- signed roll of largest magnitude in the window.
        * ``baseline_roll_deg`` -- the trial roll baseline used (deg).
        * ``flags`` -- ``list[str]`` of quality flags (empty at detection).

    Notes
    -----
    A pure roll-threshold detector cannot see a breath whose true roll never
    reaches the threshold (e.g. many ``FLAT`` breaths). Such breaths are
    genuinely undetectable from roll alone, not silently dropped: they simply
    produce no window. Callers comparing against ground truth should score
    recall against the *detectable* subset and report the low-roll misses,
    rather than lowering the threshold below config to chase them.
    """
    cfg = _load_config(config_path)
    threshold_deg = float(cfg["breath_roll_threshold_deg"])

    if "roll_deg" not in df.columns:
        if transform is None:
            raise ValueError(
                "detect_breath_windows needs roll: pass a df with 'roll_deg' "
                "(from calibrate.apply) or a 'transform' to compute it."
            )
        df = calibrate.apply(df, transform)

    t = df["t"].to_numpy()
    roll = df["roll_deg"].to_numpy()
    baseline = float(np.median(roll))
    dev = roll - baseline

    over = np.abs(dev) >= threshold_deg
    rows = []
    for start, stop in _contiguous_runs(over):
        seg_dev = dev[start:stop]
        peak_local = int(np.argmax(np.abs(seg_dev)))
        peak_roll = float(roll[start + peak_local])
        side = "R" if peak_roll - baseline > 0 else "L"
        rows.append(
            {
                "t_start": float(t[start]),
                "t_end": float(t[stop - 1]),
                "side": side,
                "peak_roll_deg": peak_roll,
                "baseline_roll_deg": baseline,
                "flags": [],
            }
        )

    schema = {
        "t_start": pl.Float64,
        "t_end": pl.Float64,
        "side": pl.Utf8,
        "peak_roll_deg": pl.Float64,
        "baseline_roll_deg": pl.Float64,
        "flags": pl.List(pl.Utf8),
    }
    return pl.DataFrame(rows, schema=schema)


def apply_exclusions(
    breaths: pl.DataFrame,
    pushoffs: pl.DataFrame,
    df: pl.DataFrame,
    config_path: str | Path | None = None,
) -> pl.DataFrame:
    """Mark protocol-excluded breaths and flag insufficient valid cycles.

    Nothing is deleted (CLAUDE.md hard constraint 5). Each breath keeps its
    row and gains:

    * ``excluded`` -- bool, ``True`` if the window starts within
      ``exclude_after_pushoff_s`` of a detected push-off or within
      ``exclude_before_end_s`` of trial end.
    * ``exclusion_reason`` -- ``"within_4s_of_pushoff"`` or
      ``"within_2s_of_end"`` (or ``None`` when kept). When both apply the
      push-off reason wins, matching the generator's precedence.

    The two boundaries use different edges of the breath window, matching the
    ground-truth generator exactly: the push-off rule keys on the window
    **start** (a breath *starting* during the post-push-off glide is dropped),
    while the trial-end rule keys on the window **end** (a breath whose window
    *extends into* the final ``exclude_before_end_s`` seconds is dropped).
    * ``flags`` -- the detection flags, plus ``INSUFFICIENT_CYCLES`` on every
      row when the number of non-excluded breaths falls below
      ``min_valid_cycles``.

    The exclusion window boundaries are a *time-based proxy* for the
    protocol's "first and last 5 m": a head IMU cannot measure position, so we
    reject the ``exclude_after_pushoff_s`` seconds of glide after each push-off
    and the final ``exclude_before_end_s`` seconds of the trial.

    Insufficient valid cycles is a quality flag, not an error -- the data are
    readable, just too few for primary analysis -- so it is surfaced in the
    ``flags`` column rather than raised.

    Parameters
    ----------
    breaths:
        Output of :func:`detect_breath_windows`.
    pushoffs:
        Output of :func:`detect_pushoffs` (detected push-off times). Using the
        detected -- rather than ground-truth -- times keeps the pipeline
        self-contained; detection lands within a few samples of the true wall
        push, well inside the seconds-scale exclusion window.
    df:
        The trial dataframe, used only for its final timestamp (trial end).
    config_path:
        Optional override for ``config.yaml``.

    Returns
    -------
    polars.DataFrame
        ``breaths`` with ``excluded``, ``exclusion_reason`` and an updated
        ``flags`` column appended.
    """
    cfg = _load_config(config_path)
    after_pushoff_s = float(cfg["exclude_after_pushoff_s"])
    before_end_s = float(cfg["exclude_before_end_s"])
    min_valid = int(cfg["min_valid_cycles"])

    t = df["t"].to_numpy()
    trial_end = float(t[-1]) if t.size else 0.0
    exclude_from = trial_end - before_end_s
    pushoff_starts = pushoffs["t_start"].to_numpy() if pushoffs.height else np.array([])

    starts = breaths["t_start"].to_numpy()
    ends = breaths["t_end"].to_numpy()
    excluded: list[bool] = []
    reasons: list[str | None] = []
    for ts, te in zip(starts, ends):
        near_pushoff = np.any(
            (pushoff_starts <= ts) & (ts < pushoff_starts + after_pushoff_s)
        )
        near_end = te > exclude_from
        if near_pushoff:
            excluded.append(True)
            reasons.append(REASON_PUSHOFF)
        elif near_end:
            excluded.append(True)
            reasons.append(REASON_END)
        else:
            excluded.append(False)
            reasons.append(None)

    n_valid = int(sum(not e for e in excluded))
    insufficient = n_valid < min_valid

    flags_col = breaths["flags"].to_list() if breaths.height else []
    new_flags: list[list[str]] = []
    for existing in flags_col:
        f = list(existing)
        if insufficient:
            f.append(INSUFFICIENT_CYCLES)
        new_flags.append(f)

    return breaths.with_columns(
        pl.Series("excluded", excluded, dtype=pl.Boolean),
        pl.Series("exclusion_reason", reasons, dtype=pl.Utf8),
        pl.Series("flags", new_flags, dtype=pl.List(pl.Utf8)),
    )
