"""Tests for :mod:`swimlab.wrist` -- the forearm placement module (L and R).

Each arm's metrics are scored against the body model's per-arm ground truth
(``body.truth["arms"][side]``); the L/R symmetry is scored as a two-sensor
fusion metric. The forearm is calibrated against a matched forward-horizontal +
at-side pose pair (reusing :func:`swimlab.synth.generate_calibration`, whose
gravity geometry is placement-agnostic) at the forearm's forward-horizontal zero.
"""

from __future__ import annotations

import polars as pl
import pytest

from swimlab import synth, wrist

_PLACEMENT = {"R": "wrist_r", "L": "wrist_l"}


def _run_arm(
    side: str,
    *,
    archetype: str = "ROTATOR",
    seed: int = 42,
    mount: tuple[float, float, float] = (4.0, -3.0, 11.0),
    noise: bool = True,
    arm_asymmetry_frac: float = 0.0,
    body: synth.BodyModel | None = None,
) -> tuple[dict, pl.DataFrame, dict, synth.BodyModel]:
    if body is None:
        body = synth.generate_swim(
            archetype, seed=seed, pitch_baseline_deg=4.0, arm_asymmetry_frac=arm_asymmetry_frac
        )
    trial, _ = synth.virtual_sensor(body, _PLACEMENT[side], mount_offset_deg=mount, noise=noise, seed=seed)
    segs, _ = synth.generate_calibration(
        mount_offset_deg=mount, pitch_baseline_deg=body.meta["forearm_baseline_deg"],
        noise=noise, seed=seed + 7,
    )
    R = wrist.calibrate_transform(segs["t0a"], segs["t0b"])
    cal = wrist.apply(trial, R)
    ev = wrist.detect_events(cal, trial)
    m = wrist.metrics(cal, ev, side=side)
    return body.truth["arms"][side], m, ev, body


# --------------------------------------------------------------------------- #
# Stroke count / rate / amplitude
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("side", ["R", "L"])
@pytest.mark.parametrize("seed", [1, 42, 100])
def test_stroke_count_matches_truth(side: str, seed: int) -> None:
    truth, m, _, _ = _run_arm(side, seed=seed, noise=True)
    assert m["stroke_count"][0] == pytest.approx(truth["stroke_count"], abs=2)


@pytest.mark.parametrize("side", ["R", "L"])
def test_stroke_count_exact_when_clean(side: str) -> None:
    truth, m, _, _ = _run_arm(side, seed=7, noise=False, mount=(0, 0, 0))
    assert m["stroke_count"][0] == truth["stroke_count"]


def test_stroke_rate_matches_truth() -> None:
    truth, m, _, _ = _run_arm("R", seed=5, noise=True)
    assert m["stroke_rate_cpm"][0] == pytest.approx(truth["stroke_rate_cpm"], abs=1.5)


def test_pitch_amplitude_matches_truth() -> None:
    truth, m, _, _ = _run_arm("R", seed=9, noise=True)
    assert m["pitch_amplitude_deg"][0] == pytest.approx(truth["pitch_amplitude_deg"], abs=6.0)


def test_pull_precedes_recovery_and_fraction_in_range() -> None:
    _, m, _, _ = _run_arm("R", seed=3, noise=True)
    pf = m["pull_fraction"][0]
    assert pf is not None and 0.05 < pf < 0.6  # a pull is a minority of the cycle


# --------------------------------------------------------------------------- #
# L/R symmetry (two-sensor fusion)
# --------------------------------------------------------------------------- #


def test_symmetric_swimmer_has_near_zero_symmetry() -> None:
    body = synth.generate_swim("ROTATOR", seed=11, pitch_baseline_deg=4.0, arm_asymmetry_frac=0.0)
    _, mR, evR, _ = _run_arm("R", seed=11, body=body)
    _, mL, evL, _ = _run_arm("L", seed=11, body=body)
    sym = wrist.symmetry(mR, mL, evR, evL).row(0, named=True)
    assert abs(sym["amplitude_symmetry_index"]) < 0.05
    assert abs(sym["stroke_count_symmetry_index"]) < 0.1


def test_arm_asymmetry_is_recovered() -> None:
    """A stronger right arm (larger pitch sweep) shows a positive amplitude
    symmetry index."""
    body = synth.generate_swim("ROTATOR", seed=13, pitch_baseline_deg=4.0, arm_asymmetry_frac=0.18)
    _, mR, evR, _ = _run_arm("R", seed=13, body=body)
    _, mL, evL, _ = _run_arm("L", seed=13, body=body)
    assert mR["pitch_amplitude_deg"][0] > mL["pitch_amplitude_deg"][0]
    sym = wrist.symmetry(mR, mL, evR, evL).row(0, named=True)
    assert sym["amplitude_symmetry_index"] > 0.1


def test_arms_are_antiphase() -> None:
    """Front crawl arms are half a cycle apart -- the phase offset is ~0.5."""
    body = synth.generate_swim("ROTATOR", seed=15, pitch_baseline_deg=4.0)
    _, mR, evR, _ = _run_arm("R", seed=15, body=body)
    _, mL, evL, _ = _run_arm("L", seed=15, body=body)
    sym = wrist.symmetry(mR, mL, evR, evL).row(0, named=True)
    assert sym["mean_phase_offset_cycles"] == pytest.approx(0.5, abs=0.12)


# --------------------------------------------------------------------------- #
# Calibration, exclusions, mount-invariance, schema
# --------------------------------------------------------------------------- #


def test_good_pose_passes_bad_pose_flags() -> None:
    good, _ = synth.generate_calibration(mount_offset_deg=(4, -3, 11), pitch_baseline_deg=0.0, seed=1)
    assert wrist.pose_check(good["t0a"], good["t0b"]) is None
    bad, _ = synth.generate_calibration(
        mount_offset_deg=(4, -3, 11), pitch_baseline_deg=0.0, bad_calibration=True, seed=2
    )
    assert wrist.pose_check(bad["t0a"], bad["t0b"]) == wrist.CALIB_POSE_SUSPECT


def test_excluded_strokes_stay_with_reason_codes() -> None:
    _, m, ev, _ = _run_arm("R", seed=23, noise=True)
    strokes = ev["strokes"]
    assert strokes.height == m["stroke_count"][0]
    assert m["n_valid_strokes"][0] < m["stroke_count"][0]  # some fall post-push-off
    for row in strokes.iter_rows(named=True):
        if row["excluded"]:
            assert row["exclusion_reason"] in {wrist.REASON_PUSHOFF, wrist.REASON_END}
        else:
            assert row["exclusion_reason"] is None


def test_stroke_count_is_mount_invariant() -> None:
    counts = []
    body = synth.generate_swim("ROTATOR", seed=21, pitch_baseline_deg=4.0)
    for mount in [(0.0, 0.0, 0.0), (10.0, 15.0, 20.0), (-8.0, 4.0, 30.0)]:
        _, m, _, _ = _run_arm("R", seed=21, mount=mount, noise=False, body=body)
        counts.append(m["stroke_count"][0])
    assert len(set(counts)) == 1


def test_metrics_schema_and_side() -> None:
    _, m, _, _ = _run_arm("L")
    assert m.height == 1
    assert m["side"][0] == "L"
    for col in ("stroke_count", "stroke_rate_cpm", "pull_fraction", "pitch_amplitude_deg", "flags"):
        assert col in m.columns


def test_events_do_not_mutate_input() -> None:
    body = synth.generate_swim("ROTATOR", seed=27, pitch_baseline_deg=4.0)
    trial, _ = synth.virtual_sensor(body, "wrist_r", mount_offset_deg=(2, 1, 8), noise=True, seed=27)
    segs, _ = synth.generate_calibration(mount_offset_deg=(2, 1, 8), pitch_baseline_deg=0.0, seed=1)
    R = wrist.calibrate_transform(segs["t0a"], segs["t0b"])
    cal = wrist.apply(trial, R)
    cols = cal.columns
    wrist.detect_events(cal, trial)
    assert cal.columns == cols
