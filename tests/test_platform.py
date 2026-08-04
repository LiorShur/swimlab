"""Tests for the platform Phase 0 abstraction: the placement registry, the
unified body model :func:`swimlab.synth.generate_swim`, and
:func:`swimlab.synth.virtual_sensor`.

The load-bearing claim of Phase 0 -- *"the head re-expressed through the body
model, with no metric change"* -- is pinned here as a **byte-for-byte**
equivalence: a skull placement sampled from a body reproduces the shipped
:func:`swimlab.synth.generate_trial` output exactly, across noise/clean and
static/slipping mounts. If that ever drifts, the abstraction has changed the
head result and this test fails loudly.

The rest guards the registry, the canonical schema for a second placement
(sacrum), determinism, one-body-many-sensors consistency, and that the pelvis
roll the sacrum module will consume is real and recoverable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.spatial.transform import Rotation

from swimlab import calibrate, events, metrics, placements, synth

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


# --------------------------------------------------------------------------- #
# Placement registry
# --------------------------------------------------------------------------- #


def test_registry_is_self_consistent() -> None:
    for pid, p in placements.PLACEMENTS.items():
        assert p.id == pid
        assert p.segment  # every placement names a body segment
        assert p.side in (None, "L", "R")
        assert placements.get(pid) is p
        assert placements.segment_of(pid) == p.segment
        # L/R placements come in pairs on distinct segments
        if p.side in ("L", "R"):
            assert p.segment.endswith(("_l", "_r"))


def test_head_is_the_implemented_placement() -> None:
    impl = {p.id for p in placements.implemented()}
    assert "head" in impl  # the shipped module
    assert placements.get("head").segment == "skull"


def test_unknown_placement_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError, match="registered placements"):
        placements.get("elbow_l")


# --------------------------------------------------------------------------- #
# Canonical schema for a virtual sensor
# --------------------------------------------------------------------------- #


def _assert_canonical(df: pl.DataFrame) -> None:
    assert df.columns == list(synth.CANONICAL_COLUMNS)
    assert all(dt == pl.Float64 for dt in df.dtypes)
    q = df.select(["quat_w", "quat_x", "quat_y", "quat_z"]).to_numpy()
    assert np.allclose((q**2).sum(axis=1), 1.0, atol=1e-9)
    t = df["t"].to_numpy()
    assert np.allclose(np.diff(t), 1.0 / synth.SAMPLE_RATE_HZ)


@pytest.mark.parametrize("placement_id", ["head", "sacrum"])
def test_virtual_sensor_is_canonical(placement_id: str) -> None:
    body = synth.generate_swim("ROTATOR", seed=1, pitch_baseline_deg=4.0)
    df, info = synth.virtual_sensor(
        body, placement_id, mount_offset_deg=(3.0, -2.0, 12.0), seed=1
    )
    _assert_canonical(df)
    assert info["placement"] == placement_id
    assert info["segment"] == placements.segment_of(placement_id)


def test_virtual_sensor_accepts_placement_object_and_raw_segment() -> None:
    body = synth.generate_swim("LIFTER", seed=2, pitch_baseline_deg=4.0)
    a, _ = synth.virtual_sensor(body, placements.get("head"), seed=2)
    b, _ = synth.virtual_sensor(body, "head", seed=2)
    c, _ = synth.virtual_sensor(body, "skull", seed=2)  # raw segment name
    assert a.equals(b) and b.equals(c)


def test_body_missing_segment_raises() -> None:
    body = synth.generate_swim("FLAT", seed=3, pitch_baseline_deg=4.0)
    with pytest.raises(KeyError, match="no segment"):
        synth.virtual_sensor(body, "shank_l", seed=3)  # ankle segment not built yet


# --------------------------------------------------------------------------- #
# THE Phase-0 acceptance: head re-expressed through the body model, no change
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("archetype", sorted(synth.ARCHETYPES))
@pytest.mark.parametrize(
    "kw",
    [
        dict(mount_offset_deg=(0.0, 0.0, 0.0), noise=False),
        dict(mount_offset_deg=(3.0, -2.0, 12.0), noise=True),
        dict(mount_offset_deg=(5.0, 3.0, 9.0), mount_slip_deg_per_min=2.0, noise=True),
    ],
)
def test_head_virtual_sensor_matches_generate_trial_byte_for_byte(
    archetype: str, kw: dict
) -> None:
    """A skull placement sampled from the body IS the shipped head trial.

    Same seed, same mount, same noise switches -> identical canonical dataframe.
    This is the concrete meaning of "the head re-expressed through the unified
    model, with no metric change": the bytes -- and therefore every downstream
    metric -- are unchanged.
    """
    seed = 777
    baseline = 4.0
    df_trial, _ = synth.generate_trial(
        archetype, seed=seed, pitch_baseline_deg=baseline, **kw
    )
    body = synth.generate_swim(archetype, seed=seed, pitch_baseline_deg=baseline)
    df_vs, _ = synth.virtual_sensor(body, "head", seed=seed, **kw)
    assert df_vs.equals(df_trial)


def test_head_ground_truth_matches_generate_trial() -> None:
    """The body model carries the same head breath ground truth as the head
    generator (it is computed by the same code on the same timeline)."""
    seed = 21
    _, gt = synth.generate_trial("MIXED", seed=seed, pitch_baseline_deg=4.0)
    body = synth.generate_swim("MIXED", seed=seed, pitch_baseline_deg=4.0)
    assert body.truth["head"]["summary"] == gt["summary"]
    assert body.truth["head"]["breaths"] == gt["breaths"]


def test_head_metrics_recover_through_the_unified_model() -> None:
    """End-to-end: a clean skull sensor from the body model, calibrated against a
    matched head calibration and run through events -> metrics, recovers the
    body's head ground-truth mean d_pitch -- the head pipeline is unchanged by
    routing through the abstraction.
    """
    seed, baseline = 7, 4.0
    body = synth.generate_swim("LIFTER", seed=seed, pitch_baseline_deg=baseline)
    trial, _ = synth.virtual_sensor(
        body, "head", mount_offset_deg=(0.0, 0.0, 0.0), noise=False, seed=seed
    )
    segs, _ = synth.generate_calibration(
        mount_offset_deg=(0.0, 0.0, 0.0), pitch_baseline_deg=baseline, noise=False, seed=1
    )
    R = calibrate.fit_transform(segs["t0a"], segs["t0b"])
    calibrated = calibrate.apply(trial, R)
    pushoffs = events.detect_pushoffs(trial)
    breaths = events.detect_breath_windows(calibrated)
    breaths = events.apply_exclusions(breaths, pushoffs, calibrated)
    per_breath = metrics.per_breath_metrics(calibrated, breaths)
    summary = metrics.trial_summary(per_breath)

    recovered = summary["mean_d_pitch_breath"][0]
    truth = body.truth["head"]["summary"]["true_mean_d_pitch_deg"]
    # same tolerance band the head module meets in its own round-trip.
    assert abs(recovered - truth) < 2.0


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_generate_swim_is_deterministic() -> None:
    a = synth.generate_swim("MIXED", seed=99, pitch_baseline_deg=4.0)
    b = synth.generate_swim("MIXED", seed=99, pitch_baseline_deg=4.0)
    assert np.array_equal(a.t, b.t)
    assert a.truth == b.truth
    assert a.meta == b.meta


def test_virtual_sensor_is_deterministic() -> None:
    body = synth.generate_swim("ROTATOR", seed=4, pitch_baseline_deg=4.0)
    kw = dict(mount_offset_deg=(2.0, 1.0, 8.0), mount_slip_deg_per_min=1.0, seed=4)
    d1, _ = synth.virtual_sensor(body, "sacrum", **kw)
    d2, _ = synth.virtual_sensor(body, "sacrum", **kw)
    assert d1.equals(d2)


# --------------------------------------------------------------------------- #
# One body, many sensors
# --------------------------------------------------------------------------- #


def test_sensors_from_one_body_share_the_timeline() -> None:
    """Head and sacrum sampled from the same body see the same push-offs -- the
    whole point of a single body model (fusion needs one timeline)."""
    body = synth.generate_swim("ROTATOR", seed=8, pitch_baseline_deg=4.0)
    head, _ = synth.virtual_sensor(body, "head", noise=False, seed=8)
    sacrum, _ = synth.virtual_sensor(body, "sacrum", noise=False, seed=9)

    po_head = events.detect_pushoffs(head)["t_peak"].to_numpy()
    po_sacrum = events.detect_pushoffs(sacrum)["t_peak"].to_numpy()
    assert po_head.size == po_sacrum.size == body.truth["pushoff_count"]
    assert np.allclose(po_head, po_sacrum, atol=0.05)


def test_distinct_seeds_give_independent_sensor_noise() -> None:
    body = synth.generate_swim("LIFTER", seed=10, pitch_baseline_deg=4.0)
    a, _ = synth.virtual_sensor(body, "sacrum", noise=True, seed=10)
    b, _ = synth.virtual_sensor(body, "sacrum", noise=True, seed=11)
    ga = a.select(["gyr_x", "gyr_y", "gyr_z"]).to_numpy()
    gb = b.select(["gyr_x", "gyr_y", "gyr_z"]).to_numpy()
    assert not np.allclose(ga, gb)


# --------------------------------------------------------------------------- #
# Pelvis segment sanity: the sacrum ground truth is real and recoverable
# --------------------------------------------------------------------------- #


def _grav_referenced_roll(df: pl.DataFrame) -> np.ndarray:
    q = df.select(["quat_x", "quat_y", "quat_z", "quat_w"]).to_numpy()
    up = Rotation.from_quat(q).inv().apply(np.tile([0.0, 0.0, 1.0], (len(q), 1)))
    return np.degrees(np.arctan2(up[:, 1], up[:, 2]))


def test_pelvis_roll_is_recoverable_to_injected_amplitude() -> None:
    """A clean sacrum sensor recovers the injected body-roll amplitude. (No
    "pelvis rolls more than head" claim: a ROTATOR's head rides the body roll and
    is deliberately large -- the two are comparable, not ordered.)"""
    body = synth.generate_swim(
        "ROTATOR", seed=5, pitch_baseline_deg=4.0, body_roll_amp_deg=50.0
    )
    sacrum, _ = synth.virtual_sensor(body, "sacrum", mount_offset_deg=(0, 0, 0), noise=False, seed=5)
    roll_pelvis = _grav_referenced_roll(sacrum)
    assert np.abs(roll_pelvis).max() == pytest.approx(
        body.truth["body_roll_amp_deg"], abs=3.0
    )


def test_roll_asymmetry_ground_truth_has_correct_sign() -> None:
    """Positive ``roll_asymmetry_frac`` makes right-side roll exceed left; the
    symmetry index the sacrum module recovers must reflect that."""
    body = synth.generate_swim(
        "ROTATOR", seed=6, pitch_baseline_deg=4.0, body_roll_amp_deg=50.0,
        roll_asymmetry_frac=0.15,
    )
    assert body.truth["mean_peak_roll_right_deg"] > body.truth["mean_peak_roll_left_deg"]
    assert body.truth["roll_symmetry_index"] > 0.1

    sacrum, _ = synth.virtual_sensor(body, "sacrum", noise=False, seed=6)
    roll = _grav_referenced_roll(sacrum)
    assert roll.max() > abs(roll.min())  # right peak (positive) exceeds left
