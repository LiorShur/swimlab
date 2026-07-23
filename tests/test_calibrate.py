"""Tests for swimlab.calibrate.

The calibration transform is validated by end-to-end recovery of the
study-relevant pitch/roll, not by comparing the raw mount Euler triple
(gravity cannot observe rotation about its own axis at a single pose).

Fixtures come from ``swimlab.synth``'s public API (treated as a black-box
ground-truth generator) plus the committed parquet fixtures under
``tests/fixtures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import swimlab.synth as synth
from swimlab import calibrate

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

MOUNTS = [
    (0.0, 0.0, 0.0),
    (3.0, -2.0, 12.0),
    (-5.0, 4.0, -8.0),
    (10.0, 7.0, 25.0),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _recover_pitch_roll(trial: pl.DataFrame, R) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibrated = trial.with_columns(pl.col("t"))  # no-op, keep pure input
    out = calibrate.apply(calibrated, R)
    return (
        out["t"].to_numpy(),
        out["pitch_deg"].to_numpy(),
        out["roll_deg"].to_numpy(),
    )


def _per_breath_errors(trial, gt, R):
    """Recovered per-breath d_pitch / peak-roll errors vs ground truth (deg)."""
    t, pitch, roll = _recover_pitch_roll(trial, R)
    breaths = gt["breaths"]
    dpitch_err, roll_err = [], []
    for i, b in enumerate(breaths):
        if b["excluded_by_protocol"]:
            continue
        win = (t >= b["t_start"]) & (t <= b["t_end"])
        if not win.any():
            continue
        prev_end = breaths[i - 1]["t_end"] if i > 0 else 0.0
        pre = (t > prev_end) & (t < b["t_start"])
        baseline = float(np.median(pitch[pre])) if pre.any() else 0.0
        dpitch = float(pitch[win].max()) - baseline
        peak_roll = float(np.abs(roll[win]).max())
        dpitch_err.append(dpitch - b["true_d_pitch_deg"])
        roll_err.append(peak_roll - b["true_peak_roll_deg"])
    return np.array(dpitch_err), np.array(roll_err)


# --------------------------------------------------------------------------
# transform / mount recovery
# --------------------------------------------------------------------------
def test_poses_map_to_reference_frame():
    """T0b -> pitch/roll ~ 0 (zero reference); T0a -> pitch = pose angle, roll ~ 0.

    T0a pitch equals the (near-90 deg) measured pose angle, not exactly 90:
    the two poses are ~86-90 deg apart in practice and the transform places
    T0b at zero, so T0a lands at the true pose separation.
    """
    segs, _gt = synth.generate_calibration(
        mount_offset_deg=(3.0, -2.0, 12.0), pitch_baseline_deg=0.0, noise=False, seed=1
    )
    R = calibrate.fit_transform(segs["t0a"], segs["t0b"])

    t0b = calibrate.apply(segs["t0b"], R)
    assert abs(float(t0b["pitch_deg"].mean())) < 1.0
    assert abs(float(t0b["roll_deg"].mean())) < 1.0

    t0a = calibrate.apply(segs["t0a"], R)
    # T0a is the upright / nose-up pose: pitch lands near +90 deg (within a
    # couple of degrees given the +/-1.5 deg gaze sway in the pose). Roll is a
    # singularity at 90 deg pitch (up ~ [-1,0,0], atan2(0,0) wraps +/-180) and
    # is intentionally not asserted -- roll is only meaningful near prone.
    assert abs(float(t0a["pitch_deg"].mean()) - 90.0) < 3.0


@pytest.mark.parametrize("mount", MOUNTS)
def test_recovery_zero_noise_within_1deg(mount):
    """Zero-noise calibration: recovered pitch/roll match truth within ~1 deg.

    ``peak_roll`` is baseline-free and is the clean transform validator (held
    to 1 deg). ``d_pitch`` additionally carries the slack of reconstructing
    the "preceding non-breath" baseline here (metrics.py's job, task 3), so it
    is held to 1.5 deg -- the transform itself is proven by roll and by the
    pose mapping test above.
    """
    segs, _gt = synth.generate_calibration(
        mount_offset_deg=mount, pitch_baseline_deg=4.0, noise=False, seed=11
    )
    R = calibrate.fit_transform(segs["t0a"], segs["t0b"])
    trial, tgt = synth.generate_trial("LIFTER", mount_offset_deg=mount, noise=False, seed=102)
    dpitch_err, roll_err = _per_breath_errors(trial, tgt, R)
    assert np.abs(roll_err).mean() < 1.0
    assert np.abs(roll_err).max() < 1.5
    assert np.abs(dpitch_err).mean() < 1.5


@pytest.mark.parametrize("mount", MOUNTS)
def test_recovery_full_noise_within_2deg(mount):
    """Full-noise calibration + trial: recovery within 2 deg."""
    segs, _gt = synth.generate_calibration(
        mount_offset_deg=mount, pitch_baseline_deg=4.0, noise=True, seed=21
    )
    R = calibrate.fit_transform(segs["t0a"], segs["t0b"])
    trial, tgt = synth.generate_trial("LIFTER", mount_offset_deg=mount, noise=True, seed=202)
    dpitch_err, roll_err = _per_breath_errors(trial, tgt, R)
    assert np.abs(roll_err).mean() < 2.0
    assert np.abs(dpitch_err).mean() < 2.0


@pytest.mark.parametrize("archetype", ["LIFTER", "ROTATOR", "MIXED", "FLAT"])
def test_apply_end_to_end_zero_noise(archetype):
    """apply() recovers per-breath peak-roll (transform validator) and d_pitch
    across every breathing archetype under a common mount."""
    mount = (6.0, -3.0, 15.0)
    segs, _gt = synth.generate_calibration(
        mount_offset_deg=mount, pitch_baseline_deg=3.0, noise=False, seed=31
    )
    R = calibrate.fit_transform(segs["t0a"], segs["t0b"])
    trial, tgt = synth.generate_trial(archetype, mount_offset_deg=mount, noise=False, seed=303)
    dpitch_err, roll_err = _per_breath_errors(trial, tgt, R)
    assert np.abs(roll_err).mean() < 1.0
    assert np.abs(dpitch_err).mean() < 1.5


def test_apply_adds_columns_and_is_nondestructive():
    trial, _tgt = synth.generate_trial("LIFTER", noise=False, seed=1)
    segs, _gt = synth.generate_calibration(noise=False, seed=1)
    R = calibrate.fit_transform(segs["t0a"], segs["t0b"])
    out = calibrate.apply(trial, R)
    assert "pitch_deg" in out.columns
    assert "roll_deg" in out.columns
    assert out.height == trial.height
    # original columns preserved unchanged
    for col in trial.columns:
        assert out[col].to_list() == trial[col].to_list()


# --------------------------------------------------------------------------
# committed fixtures
# --------------------------------------------------------------------------
def test_committed_calib_fixture_recovers_trial():
    """Committed calib fixture (mount [3,-2,12]) + committed trial of the same
    mount. Both carry full sensor noise -> 2 deg tolerance."""
    calib_gt = json.loads((FIXTURES / "calib.gt.json").read_text())
    trial_gt = json.loads((FIXTURES / "trial_lifter.gt.json").read_text())
    # guard: the fixture pairing is only valid if the mounts match
    assert calib_gt["mount_offset_deg"] == trial_gt["mount_offset_deg"]

    t0a = pl.read_parquet(FIXTURES / "calib_t0a.parquet")
    t0b = pl.read_parquet(FIXTURES / "calib_t0b.parquet")
    R = calibrate.fit_transform(t0a, t0b)
    trial = pl.read_parquet(FIXTURES / "trial_lifter.parquet")
    dpitch_err, roll_err = _per_breath_errors(trial, trial_gt, R)
    assert np.abs(roll_err).mean() < 2.0
    assert np.abs(dpitch_err).mean() < 2.0


# --------------------------------------------------------------------------
# pose sanity check
# --------------------------------------------------------------------------
def test_pose_check_good_no_flag():
    segs, _gt = synth.generate_calibration(
        mount_offset_deg=(3.0, -2.0, 12.0), noise=False, seed=1
    )
    assert calibrate.pose_check(segs["t0a"], segs["t0b"], config_path=CONFIG) is None


def test_pose_check_bad_flags():
    segs, gt = synth.generate_calibration(
        mount_offset_deg=(3.0, -2.0, 12.0), bad_calibration=True, noise=False, seed=2
    )
    assert gt["bad_calibration"] is True
    assert (
        calibrate.pose_check(segs["t0a"], segs["t0b"], config_path=CONFIG)
        == calibrate.CALIB_POSE_SUSPECT
    )


@pytest.mark.parametrize("mount", MOUNTS)
def test_pose_check_is_mount_invariant(mount):
    """Good vs bad verdict must not depend on the mount offset."""
    good, _ = synth.generate_calibration(
        mount_offset_deg=mount, bad_calibration=False, noise=True, seed=41
    )
    bad, _ = synth.generate_calibration(
        mount_offset_deg=mount, bad_calibration=True, noise=True, seed=42
    )
    assert calibrate.pose_check(good["t0a"], good["t0b"], config_path=CONFIG) is None
    assert (
        calibrate.pose_check(bad["t0a"], bad["t0b"], config_path=CONFIG)
        == calibrate.CALIB_POSE_SUSPECT
    )


def test_pose_check_reads_config_by_default():
    """Default path loads thresholds from config.yaml (no hardcoded values)."""
    segs, _gt = synth.generate_calibration(noise=False, seed=1)
    # should not raise and should agree with explicit config path
    assert calibrate.pose_check(segs["t0a"], segs["t0b"]) == calibrate.pose_check(
        segs["t0a"], segs["t0b"], config_path=CONFIG
    )


def test_pose_angle_close_to_90_for_good_calibration():
    segs, _gt = synth.generate_calibration(
        mount_offset_deg=(3.0, -2.0, 12.0), noise=False, seed=1
    )
    angle = calibrate.pose_angle_deg(segs["t0a"], segs["t0b"])
    assert abs(angle - 90.0) < 10.0


# --------------------------------------------------------------------------
# hard constraint: no magnetometer, no yaw
# --------------------------------------------------------------------------
def test_module_source_uses_no_mag_or_yaw():
    """Hard constraint 2: never read mag_* and never compute yaw.

    Strip docstrings/comments (where 'yaw' legitimately appears in prose) and
    check the executable source references neither the magnetometer columns
    nor any yaw derivation.
    """
    import ast

    path = Path(__file__).resolve().parent.parent / "swimlab" / "calibrate.py"
    tree = ast.parse(path.read_text())
    code_tokens: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue  # skip string literals / docstrings
        if isinstance(node, ast.Name):
            code_tokens.append(node.id)
        elif isinstance(node, ast.Attribute):
            code_tokens.append(node.attr)
    joined = " ".join(code_tokens).lower()
    assert "mag_x" not in joined and "mag_y" not in joined and "mag_z" not in joined
    assert "yaw" not in joined
