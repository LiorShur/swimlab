"""Tests for ``swimlab.synth`` (task 1).

The three acceptance tests the generator is *responsible* for -- 5 (bad
calibration caught), 6 (yaw independence) and 8 (determinism) from
SYNTHETIC_DATA_SPEC.md section 8 -- are checkable with the generator alone.
Tests 1-4 and 7 need the downstream modules (calibrate/events/metrics) and are
exercised in task 6. A handful of structural checks guard the canonical schema
and the archetype ground truth so a regression in the generator is caught here
rather than three modules downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from scipy.spatial.transform import Rotation

from swimlab import synth

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


# --------------------------------------------------------------------------- #
# Structural / schema checks
# --------------------------------------------------------------------------- #


def _assert_canonical(df: pl.DataFrame) -> None:
    assert df.columns == list(synth.CANONICAL_COLUMNS)
    assert all(dt == pl.Float64 for dt in df.dtypes)
    # quaternions are unit norm
    q = df.select(["quat_w", "quat_x", "quat_y", "quat_z"]).to_numpy()
    assert np.allclose((q**2).sum(axis=1), 1.0, atol=1e-9)
    # sample spacing is exactly 1/120 s
    t = df["t"].to_numpy()
    assert np.allclose(np.diff(t), 1.0 / synth.SAMPLE_RATE_HZ)


@pytest.mark.parametrize("archetype", sorted(synth.ARCHETYPES))
def test_trial_schema(archetype: str) -> None:
    df, gt = synth.generate_trial(archetype, seed=0)
    _assert_canonical(df)
    assert gt["archetype"] == archetype
    assert gt["summary"]["n_breaths"] > 0
    assert len(gt["pushoffs"]) == gt["n_lengths"]


def test_calibration_and_bobs_schema() -> None:
    segs, _ = synth.generate_calibration(seed=0)
    assert set(segs) == {"t0a", "t0b", "t0c"}
    for seg in segs.values():
        _assert_canonical(seg)
    for exhale in (True, False):
        df, gt = synth.generate_bobs(exhale=exhale, seed=0)
        _assert_canonical(df)
        assert gt["exhale"] is exhale


def test_archetype_ground_truth_separates() -> None:
    """LIFTER shows large pitch / smaller roll; ROTATOR the opposite. This is
    the injected ground truth, not a pipeline result -- if it does not separate
    here, the generator itself is broken and every downstream test is meaningless.
    """
    _, lifter = synth.generate_trial("LIFTER", noise=False, seed=3)
    _, rotator = synth.generate_trial("ROTATOR", noise=False, seed=3)
    lift_dp = np.mean([b["true_d_pitch_deg"] for b in lifter["breaths"]])
    rot_dp = np.mean([b["true_d_pitch_deg"] for b in rotator["breaths"]])
    lift_roll = np.mean([b["true_peak_roll_deg"] for b in lifter["breaths"]])
    rot_roll = np.mean([b["true_peak_roll_deg"] for b in rotator["breaths"]])
    assert lift_dp > rot_dp + 8.0
    assert rot_roll > lift_roll + 15.0


def test_asymmetric_archetype_is_asymmetric() -> None:
    """The ASYMMETRIC archetype must carry a clear left/right d_pitch split so
    ``asymmetry_index`` has something real to recover downstream.
    """
    _, gt = synth.generate_trial("ASYMMETRIC", noise=False, seed=4)
    left = np.mean([b["true_d_pitch_deg"] for b in gt["breaths"] if b["side"] == "L"])
    right = np.mean([b["true_d_pitch_deg"] for b in gt["breaths"] if b["side"] == "R"])
    assert abs(right - left) > 8.0


def test_breathing_side_alternates_bilateral_but_not_unilateral() -> None:
    """Every-3 breathing alternates sides (bilateral); every-2 stays on one
    side (unilateral). This distinction drives the asymmetry metric, so pin it.
    """
    _, tri = synth.generate_trial("ROTATOR", breathe_every_n_strokes=3, seed=0)
    sides_tri = [b["side"] for b in tri["breaths"]]
    assert "L" in sides_tri and "R" in sides_tri

    _, bi = synth.generate_trial("ROTATOR", breathe_every_n_strokes=2, seed=0)
    sides_bi = {b["side"] for b in bi["breaths"]}
    assert len(sides_bi) == 1


def test_exclusions_carry_reason_codes(config: dict) -> None:
    """Nothing is dropped silently (CLAUDE.md #5): every excluded breath has a
    reason code, and exclusions match the config thresholds.
    """
    _, gt = synth.generate_trial("LIFTER", seed=1)
    for b in gt["breaths"]:
        if b["excluded_by_protocol"]:
            assert b["exclusion_reason"] in {"within_4s_of_pushoff", "within_2s_of_end"}
        else:
            assert b["exclusion_reason"] is None
    # at least the first breath of the trial is excluded (right after push-off)
    assert gt["breaths"][0]["excluded_by_protocol"]


# --------------------------------------------------------------------------- #
# Acceptance test 8 -- determinism
# --------------------------------------------------------------------------- #


def test_determinism_trial() -> None:
    kwargs = dict(
        archetype="MIXED",
        mount_offset_deg=(3.0, -2.0, 12.0),
        mount_slip_deg_per_min=1.0,
        noise=True,
        seed=12345,
    )
    df1, gt1 = synth.generate_trial(**kwargs)
    df2, gt2 = synth.generate_trial(**kwargs)
    assert df1.equals(df2)
    assert gt1 == gt2


def test_determinism_calibration_and_bobs() -> None:
    c1, g1 = synth.generate_calibration(mount_offset_deg=(1.0, 2.0, 3.0), seed=7)
    c2, g2 = synth.generate_calibration(mount_offset_deg=(1.0, 2.0, 3.0), seed=7)
    assert all(c1[k].equals(c2[k]) for k in c1)
    assert g1 == g2

    b1, gb1 = synth.generate_bobs(exhale=True, seed=9)
    b2, gb2 = synth.generate_bobs(exhale=True, seed=9)
    assert b1.equals(b2)
    assert gb1 == gb2


def test_different_seeds_differ() -> None:
    df1, _ = synth.generate_trial("LIFTER", seed=1)
    df2, _ = synth.generate_trial("LIFTER", seed=2)
    assert not df1.equals(df2)


# --------------------------------------------------------------------------- #
# Acceptance test 6 -- yaw independence
# --------------------------------------------------------------------------- #


def _quat_gravity_dir(df: pl.DataFrame) -> np.ndarray:
    q = df.select(["quat_w", "quat_x", "quat_y", "quat_z"]).to_numpy()
    rot = Rotation.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]]))
    return rot.inv().apply(np.tile([0.0, 0.0, 1.0], (len(q), 1)))


@pytest.mark.parametrize("noise", [False, True])
def test_yaw_independence(noise: bool) -> None:
    """Doubling the injected yaw corruption must not move any gravity-referenced
    quantity. The accelerometer is computed from a yaw-free orientation, so its
    channels are byte-identical; the quaternion carries yaw but the gravity
    direction extracted from it (rotation about global vertical) is invariant to
    well under 0.1 deg. The gyroscope *does* change -- proof the yaw is live and
    the test is not passing trivially.
    """
    common = dict(
        archetype="ROTATOR", mount_offset_deg=(5.0, 3.0, 9.0), noise=noise, seed=11
    )
    df1, _ = synth.generate_trial(**common, yaw_corruption_scale=1.0)
    df2, _ = synth.generate_trial(**common, yaw_corruption_scale=2.0)

    acc1 = df1.select(["acc_x", "acc_y", "acc_z"]).to_numpy()
    acc2 = df2.select(["acc_x", "acc_y", "acc_z"]).to_numpy()
    assert np.array_equal(acc1, acc2)  # exact: gravity is yaw-free by construction

    g1, g2 = _quat_gravity_dir(df1), _quat_gravity_dir(df2)
    max_deg = np.degrees(np.linalg.norm(g1 - g2, axis=1)).max()
    assert max_deg < 0.1

    gyr1 = df1.select(["gyr_x", "gyr_y", "gyr_z"]).to_numpy()
    gyr2 = df2.select(["gyr_x", "gyr_y", "gyr_z"]).to_numpy()
    assert not np.allclose(gyr1, gyr2)


# --------------------------------------------------------------------------- #
# Acceptance test 5 -- bad calibration caught
# --------------------------------------------------------------------------- #


def _pose_angle_deg(t0a: pl.DataFrame, t0b: pl.DataFrame) -> float:
    """Angle between the mean gravity directions of the two calibration poses.

    This is the mount-invariant quantity ``calibrate.py`` will use for its T0a/T0b
    sanity check: both poses' gravity vectors are rotated by the same mount, so
    the angle between them is unchanged, and it equals the true head-pitch change
    between upright and face-down.
    """

    def mean_dir(df: pl.DataFrame) -> np.ndarray:
        a = df.select(["acc_x", "acc_y", "acc_z"]).to_numpy().mean(axis=0)
        return a / np.linalg.norm(a)

    ga, gb = mean_dir(t0a), mean_dir(t0b)
    return float(np.degrees(np.arccos(np.clip(ga @ gb, -1.0, 1.0))))


def _pose_flag(t0a: pl.DataFrame, t0b: pl.DataFrame, config: dict) -> str | None:
    """The CALIB_POSE_SUSPECT rule from CLAUDE.md, applied to the two segments."""
    angle = _pose_angle_deg(t0a, t0b)
    nominal = float(config["calib_pose_angle_deg"])
    tol = float(config["calib_pose_tolerance_deg"])
    return "CALIB_POSE_SUSPECT" if abs(angle - nominal) > tol else None


def test_good_calibration_passes_pose_check(config: dict) -> None:
    segs, gt = synth.generate_calibration(
        mount_offset_deg=(3.0, -2.0, 12.0), pitch_baseline_deg=4.0, seed=1
    )
    assert not gt["bad_calibration"]
    assert _pose_flag(segs["t0a"], segs["t0b"], config) is None


def test_bad_calibration_raises_pose_suspect(config: dict) -> None:
    """Acceptance test 5: with ``bad_calibration=True`` the T0a/T0b pitch change
    falls outside 90 +/- 10 deg, so the pose check raises CALIB_POSE_SUSPECT.
    """
    segs, gt = synth.generate_calibration(
        mount_offset_deg=(3.0, -2.0, 12.0),
        pitch_baseline_deg=4.0,
        bad_calibration=True,
        seed=2,
    )
    assert gt["bad_calibration"]
    assert _pose_flag(segs["t0a"], segs["t0b"], config) == "CALIB_POSE_SUSPECT"


def test_bad_calibration_flag_is_mount_invariant(config: dict) -> None:
    """The pose check must catch the bad pose regardless of the mount offset --
    the whole point of using the angle between gravity vectors.
    """
    for mount in [(0.0, 0.0, 0.0), (10.0, 15.0, 20.0), (-8.0, 4.0, 30.0)]:
        segs, _ = synth.generate_calibration(
            mount_offset_deg=mount, pitch_baseline_deg=2.0, bad_calibration=True, seed=5
        )
        assert _pose_flag(segs["t0a"], segs["t0b"], config) == "CALIB_POSE_SUSPECT"


# --------------------------------------------------------------------------- #
# Bobs (Q4) band-power sanity -- the fixture must actually carry the difference
# --------------------------------------------------------------------------- #


def test_bobs_exhale_has_more_high_band_power() -> None:
    """The exhale fixture injects a 10-60 Hz component during submersion; the
    breath-hold one does not. Confirm the band-power difference exists so the
    Q4 analysis code has something to detect (while noting -- per spec section 6
    -- that this is a code test, not evidence the effect is real).
    """
    df_ex, _ = synth.generate_bobs(exhale=True, noise=False, seed=1)
    df_no, _ = synth.generate_bobs(exhale=False, noise=False, seed=1)

    def high_band_power(df: pl.DataFrame) -> float:
        a = df.select(["acc_x", "acc_y", "acc_z"]).to_numpy()
        mag = np.linalg.norm(a, axis=1)
        mag = mag - mag.mean()
        freqs = np.fft.rfftfreq(mag.size, 1.0 / synth.SAMPLE_RATE_HZ)
        psd = np.abs(np.fft.rfft(mag)) ** 2
        return float(psd[(freqs >= 10.0) & (freqs <= 60.0)].sum())

    assert high_band_power(df_ex) > 5.0 * high_band_power(df_no)
