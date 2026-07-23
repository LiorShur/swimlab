"""End-to-end session pipeline: a session directory -> per-breath metric table.

This module wires the already-built and independently-tested stages together
into the single entry point the study runs per participant-session::

    calibrate (pose check + transform)
        -> events (push-offs, breath windows, exclusions)
        -> metrics (per-breath table)

It adds no new signal processing of its own -- every stage is called through
the public API of :mod:`swimlab.calibrate`, :mod:`swimlab.events` and
:mod:`swimlab.metrics`. The only I/O is reading the session's canonical-schema
parquet files off disk; everything after the read is pure.

Provisional on-disk session layout
-----------------------------------
``io.py`` (Movella DOT export -> canonical dataframe) is blocked until a real
export file exists, so there is no vendor reader yet. In the interim a
"session" is a directory of **canonical-schema** parquet files -- the exact
schema :mod:`swimlab.synth` emits and every module already consumes:

===================  ============================================  ==========
file                 contents                                      required
===================  ============================================  ==========
``t0a.parquet``      T0a calibration pose (upright, gaze level)    yes
``t0b.parquet``      T0b calibration pose (face-down float)        yes
``t0c.parquet``      T0c sync-nod segment                          no (logged)
``trial.parquet``    the T7 front-crawl trial to analyse           yes
===================  ============================================  ==========

This layout is **provisional**: once ``io.py`` lands, ``run_session`` will read
a real DOT export directory instead, but the downstream contract (a calibrated
trial fed through events -> metrics) is unchanged. ``t0c`` is read only to
confirm its presence is optional; the gravity-only transform needs T0a/T0b.

Quality flags (CLAUDE.md hard constraint 5 -- never silently drop data) are
accumulated and returned alongside the table rather than raised:

* ``CALIB_POSE_SUSPECT`` -- T0a/T0b differ from the expected ~90 deg pitch
  separation by more than the config tolerance (from
  :func:`swimlab.calibrate.pose_check`). The session is still processed; the
  flag warns that the calibration transform may be untrustworthy.
* ``INSUFFICIENT_CYCLES`` -- fewer than ``config.min_valid_cycles`` non-excluded
  breaths remain (from :func:`swimlab.events.apply_exclusions`).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from swimlab import calibrate, events, metrics

__all__ = [
    "SESSION_FILES",
    "run_session",
]

# Provisional canonical-parquet session layout (see module docstring). ``t0c``
# is optional and read only to note its presence; the transform is gravity-only
# (T0a/T0b). ``io.py`` will replace this with a real DOT-export reader.
SESSION_FILES = {
    "t0a": "t0a.parquet",
    "t0b": "t0b.parquet",
    "t0c": "t0c.parquet",  # optional
    "trial": "trial.parquet",
}


def _read_parquet(path: Path) -> pl.DataFrame:
    """Read one canonical-schema parquet file, with a clear error if absent."""
    if not path.exists():
        raise FileNotFoundError(f"session file missing: {path}")
    return pl.read_parquet(path)


def run_session(
    path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Run one session directory through calibrate -> events -> metrics.

    Reads the provisional canonical-parquet session layout (see the module
    docstring), calibrates the trial against that session's own T0a/T0b poses,
    detects push-offs and breath windows, applies the protocol exclusions, and
    returns the per-breath metric table plus the accumulated session-level
    quality flags.

    Calibration is per session by construction: the transform is fit from this
    directory's T0a/T0b, so a trial is always decoded against a calibration at
    its own swimmer's prone baseline (the gravity-referenced zero). Sharing one
    calibration across swimmers with different baselines would inject a
    baseline-mismatch error and is precisely what this layout prevents.

    Parameters
    ----------
    path:
        Session directory containing ``t0a.parquet``, ``t0b.parquet``,
        ``trial.parquet`` and (optionally) ``t0c.parquet``.
    config_path:
        Optional override for ``config.yaml`` (thresholds). Passed through to
        every stage so the whole pipeline reads one config.

    Returns
    -------
    (metrics_table, flags)
        ``metrics_table`` is the :func:`swimlab.metrics.per_breath_metrics`
        table (one row per detected breath, carrying ``excluded`` /
        ``exclusion_reason`` and the per-breath ``flags`` / ``metric_flags``).
        ``flags`` is the sorted unique list of **session-level** quality flags
        (e.g. ``CALIB_POSE_SUSPECT``, ``INSUFFICIENT_CYCLES``); ``[]`` when the
        session is clean.
    """
    session = Path(path)

    t0a = _read_parquet(session / SESSION_FILES["t0a"])
    t0b = _read_parquet(session / SESSION_FILES["t0b"])
    trial = _read_parquet(session / SESSION_FILES["trial"])
    # t0c is optional -- read it when present so its schema is validated, but
    # the gravity-only transform does not need it.
    t0c_path = session / SESSION_FILES["t0c"]
    if t0c_path.exists():
        _read_parquet(t0c_path)

    session_flags: set[str] = set()

    # 1. Calibration: sanity-check the poses (flag, never raise) then fit the
    #    sensor->skull transform from this session's own T0a/T0b.
    pose_flag = calibrate.pose_check(t0a, t0b, config_path=config_path)
    if pose_flag is not None:
        session_flags.add(pose_flag)
    transform = calibrate.fit_transform(t0a, t0b)
    calibrated = calibrate.apply(trial, transform)

    # 2. Events: push-offs, breath windows, protocol exclusions.
    pushoffs = events.detect_pushoffs(trial, config_path=config_path)
    breaths = events.detect_breath_windows(calibrated, config_path=config_path)
    breaths = events.apply_exclusions(
        breaths, pushoffs, calibrated, config_path=config_path
    )

    # 3. Metrics: per-breath table on the calibrated trial.
    per_breath = metrics.per_breath_metrics(calibrated, breaths)

    # Surface any breath-level session flags (e.g. INSUFFICIENT_CYCLES, which
    # apply_exclusions stamps on every row) at the session level.
    if "flags" in per_breath.columns and per_breath.height:
        for row in per_breath["flags"].to_list():
            session_flags.update(row)

    return per_breath, sorted(session_flags)
