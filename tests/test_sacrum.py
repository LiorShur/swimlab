"""Tests for :mod:`swimlab.sacrum` -- the pelvis placement module.

Every metric is scored against the unified body model's ground truth
(:func:`swimlab.synth.generate_swim`), the same synth-vs-truth contract the head
module uses. The sacrum is calibrated against a matched upright+prone pose pair
(reusing :func:`swimlab.synth.generate_calibration`, whose geometry is
placement-agnostic) at the pelvis's own prone baseline.

Covered: lengths, distance (= lengths x pool length), stroke count, tempo /
stroke rate, body-roll amplitude, L/R roll symmetry, push-off count/interval,
pace drift; plus calibration pose sanity, mount-invariance (calibration removes
the mount), and never-silently-drop exclusions.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from swimlab import sacrum, synth

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _run(
    archetype: str = "ROTATOR",
    *,
    seed: int = 42,
    mount: tuple[float, float, float] = (3.0, -2.0, 12.0),
    noise: bool = True,
    body_roll_amp_deg: float = 48.0,
    roll_asymmetry_frac: float = 0.0,
    pool_length_m: float = 25.0,
) -> tuple[dict, pl.DataFrame, dict]:
    """Generate a body, sample+calibrate a sacrum sensor, return (truth, metrics, events)."""
    body = synth.generate_swim(
        archetype,
        seed=seed,
        pitch_baseline_deg=4.0,
        body_roll_amp_deg=body_roll_amp_deg,
        roll_asymmetry_frac=roll_asymmetry_frac,
        pool_length_m=pool_length_m,
    )
    trial, _ = synth.virtual_sensor(body, "sacrum", mount_offset_deg=mount, noise=noise, seed=seed)
    segs, _ = synth.generate_calibration(
        mount_offset_deg=mount,
        pitch_baseline_deg=body.meta["pelvis_baseline_deg"],
        noise=noise,
        seed=1,
    )
    R = sacrum.calibrate_transform(segs["t0a"], segs["t0b"])
    calibrated = sacrum.apply(trial, R)
    ev = sacrum.detect_events(calibrated, trial)
    m = sacrum.metrics(calibrated, ev, pool_length_m=pool_length_m)
    return body.truth, m, ev


# --------------------------------------------------------------------------- #
# Lengths & distance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_lengths,pool", [(4, 25.0), (4, 50.0)])
def test_lengths_and_distance(n_lengths: int, pool: float) -> None:
    body = synth.generate_swim("ROTATOR", n_lengths=n_lengths, seed=3, pitch_baseline_deg=4.0, pool_length_m=pool)
    trial, _ = synth.virtual_sensor(body, "sacrum", mount_offset_deg=(2, 1, 8), noise=True, seed=3)
    segs, _ = synth.generate_calibration(mount_offset_deg=(2, 1, 8), pitch_baseline_deg=body.meta["pelvis_baseline_deg"], seed=1)
    R = sacrum.calibrate_transform(segs["t0a"], segs["t0b"])
    cal = sacrum.apply(trial, R)
    m = sacrum.metrics(cal, sacrum.detect_events(cal, trial), pool_length_m=pool)
    assert m["lengths"][0] == n_lengths
    # distance is lengths x pool length -- never integrated from acceleration
    assert m["distance_m"][0] == pytest.approx(n_lengths * pool)


# --------------------------------------------------------------------------- #
# Stroke count / tempo / rate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("archetype", ["ROTATOR", "LIFTER", "MIXED"])
@pytest.mark.parametrize("seed", [1, 42, 100])
def test_stroke_count_matches_truth(archetype: str, seed: int) -> None:
    truth, m, _ = _run(archetype, seed=seed, noise=True)
    assert m["stroke_count"][0] == pytest.approx(truth["stroke_count"], abs=2)


def test_stroke_count_is_exact_when_clean() -> None:
    truth, m, _ = _run("ROTATOR", seed=7, noise=False, mount=(0, 0, 0))
    assert m["stroke_count"][0] == truth["stroke_count"]


@pytest.mark.parametrize("archetype", ["ROTATOR", "MIXED"])
def test_tempo_and_rate(archetype: str) -> None:
    truth, m, _ = _run(archetype, seed=5, noise=True)
    assert m["tempo_spm"][0] == pytest.approx(truth["tempo_spm"], abs=1.0)
    # stroke rate (cycles/min) is exactly half the stroke tempo
    assert m["stroke_rate_cpm"][0] == pytest.approx(m["tempo_spm"][0] / 2.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Body-roll amplitude & L/R symmetry
# --------------------------------------------------------------------------- #


def test_body_roll_amplitude() -> None:
    truth, m, _ = _run("ROTATOR", seed=9, noise=True, body_roll_amp_deg=50.0)
    assert m["body_roll_amplitude_deg"][0] == pytest.approx(truth["body_roll_amp_deg"], abs=3.0)


def test_roll_symmetry_recovers_injected_asymmetry() -> None:
    truth, m, _ = _run("ROTATOR", seed=11, noise=True, body_roll_amp_deg=50.0, roll_asymmetry_frac=0.15)
    assert truth["roll_symmetry_index"] > 0.1  # ground truth really is asymmetric
    assert m["roll_symmetry_index"][0] == pytest.approx(truth["roll_symmetry_index"], abs=0.06)
    assert m["mean_peak_roll_right_deg"][0] > m["mean_peak_roll_left_deg"][0]


def test_symmetric_swimmer_has_near_zero_symmetry_index() -> None:
    _, m, _ = _run("ROTATOR", seed=13, noise=True, roll_asymmetry_frac=0.0)
    assert abs(m["roll_symmetry_index"][0]) < 0.06


# --------------------------------------------------------------------------- #
# Push-offs & pace drift
# --------------------------------------------------------------------------- #


def test_pushoff_count_and_interval() -> None:
    truth, m, _ = _run("ROTATOR", seed=15, noise=True)
    assert m["pushoff_count"][0] == truth["pushoff_count"]
    assert m["mean_pushoff_interval_s"][0] is not None
    assert m["mean_pushoff_interval_s"][0] > 0


def test_pace_drift_matches_truth_sign_and_scale() -> None:
    truth, m, _ = _run("ROTATOR", seed=17, noise=True)
    assert m["pace_drift_s_per_length"][0] == pytest.approx(
        truth["pace_drift_s_per_length"], abs=0.2
    )


# --------------------------------------------------------------------------- #
# Calibration sanity
# --------------------------------------------------------------------------- #


def test_good_pose_passes_bad_pose_flags() -> None:
    good, _ = synth.generate_calibration(mount_offset_deg=(3, -2, 12), pitch_baseline_deg=2.0, seed=1)
    assert sacrum.pose_check(good["t0a"], good["t0b"]) is None
    bad, _ = synth.generate_calibration(
        mount_offset_deg=(3, -2, 12), pitch_baseline_deg=2.0, bad_calibration=True, seed=2
    )
    assert sacrum.pose_check(bad["t0a"], bad["t0b"]) == sacrum.CALIB_POSE_SUSPECT


# --------------------------------------------------------------------------- #
# Mount invariance (calibration removes the mount)
# --------------------------------------------------------------------------- #


def test_metrics_are_mount_invariant() -> None:
    """A fixed mount offset is removed by calibration, so the recovered metrics
    must not depend on it (the head module proves the same for its metrics)."""
    out = []
    for mount in [(0.0, 0.0, 0.0), (10.0, 15.0, 20.0), (-8.0, 4.0, 30.0)]:
        _, m, _ = _run("ROTATOR", seed=21, noise=False, mount=mount)
        out.append(m)
    base = out[0]
    for m in out[1:]:
        assert m["stroke_count"][0] == base["stroke_count"][0]
        assert m["body_roll_amplitude_deg"][0] == pytest.approx(base["body_roll_amplitude_deg"][0], abs=0.5)
        assert m["roll_symmetry_index"][0] == pytest.approx(base["roll_symmetry_index"][0], abs=0.02)


# --------------------------------------------------------------------------- #
# Never silently drop data (hard constraint 5)
# --------------------------------------------------------------------------- #


def test_excluded_strokes_stay_with_reason_codes() -> None:
    _, m, ev = _run("ROTATOR", seed=23, noise=True)
    strokes = ev["strokes"]
    # some strokes fall in the post-push-off exclusion window and are flagged,
    # not removed
    assert strokes.height == m["stroke_count"][0]
    assert m["n_valid_strokes"][0] < m["stroke_count"][0]
    for row in strokes.iter_rows(named=True):
        if row["excluded"]:
            assert row["exclusion_reason"] in {sacrum.REASON_PUSHOFF, sacrum.REASON_END}
        else:
            assert row["exclusion_reason"] is None


def test_stroke_sides_alternate() -> None:
    """Front crawl rolls the body to alternating sides each stroke; the detector
    should see that alternation in the run of valid strokes."""
    _, _, ev = _run("ROTATOR", seed=25, noise=True)
    sides = [r["side"] for r in ev["strokes"].iter_rows(named=True) if not r["excluded"]]
    # count adjacent equal pairs -- should be a small fraction of an alternating run
    same = sum(1 for a, b in zip(sides, sides[1:]) if a == b)
    assert same <= 0.15 * len(sides)


# --------------------------------------------------------------------------- #
# Schema / purity
# --------------------------------------------------------------------------- #


def test_metrics_returns_single_row_with_expected_columns() -> None:
    _, m, _ = _run()
    assert m.height == 1
    for col in (
        "lengths",
        "distance_m",
        "stroke_count",
        "tempo_spm",
        "stroke_rate_cpm",
        "body_roll_amplitude_deg",
        "roll_symmetry_index",
        "pushoff_count",
        "pace_drift_s_per_length",
        "flags",
    ):
        assert col in m.columns


def test_events_do_not_mutate_input() -> None:
    body = synth.generate_swim("ROTATOR", seed=27, pitch_baseline_deg=4.0)
    trial, _ = synth.virtual_sensor(body, "sacrum", mount_offset_deg=(2, 1, 8), noise=True, seed=27)
    segs, _ = synth.generate_calibration(mount_offset_deg=(2, 1, 8), pitch_baseline_deg=body.meta["pelvis_baseline_deg"], seed=1)
    R = sacrum.calibrate_transform(segs["t0a"], segs["t0b"])
    cal = sacrum.apply(trial, R)
    cols_before = cal.columns
    sacrum.detect_events(cal, trial)
    assert cal.columns == cols_before  # pure: no in-place column additions
