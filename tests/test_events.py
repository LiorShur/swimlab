"""Tests for swimlab.events (push-off + breath-window detection, exclusions).

Fixtures come from the committed parquet trials under ``tests/fixtures/`` and
from ``swimlab.synth``'s public API (treated as a black-box ground-truth
generator). Roll is obtained through ``swimlab.calibrate``'s public API only.

The committed noisy trial fixtures all share mount ``(3, -2, 12)`` with the
committed calibration fixtures (``calib_t0a/t0b``), so a single transform fit
from those poses calibrates every noisy trial.

Matching convention (detected window <-> ground-truth breath): temporal
overlap. A detected window matches a ground-truth breath when their intervals
overlap by any positive amount.

Detectability caveat: a roll-threshold detector cannot see a breath whose true
roll never reaches +/-``breath_roll_threshold_deg``. Recall is therefore scored
against the *detectable* ground-truth subset, and the inherently-missed
low-roll breaths are reported (never hidden to inflate the number, never
"fixed" by lowering the threshold below config).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import swimlab.synth as synth
from swimlab import calibrate, events

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
ROLL_THRESHOLD_DEG = 25.0  # == config breath_roll_threshold_deg (asserted below)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _committed_transform():
    """Fit the sensor->skull transform from the committed calib fixtures."""
    t0a = pl.read_parquet(FIXTURES / "calib_t0a.parquet")
    t0b = pl.read_parquet(FIXTURES / "calib_t0b.parquet")
    return calibrate.fit_transform(t0a, t0b)


def _load_trial(name: str):
    df = pl.read_parquet(FIXTURES / f"{name}.parquet")
    gt = json.loads((FIXTURES / f"{name}.gt.json").read_text())
    return df, gt


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _match_detected_to_gt(det: pl.DataFrame, gt_breaths: list[dict]) -> dict[int, int]:
    """Map each detected-window row index -> ground-truth breath index by overlap."""
    mapping: dict[int, int] = {}
    d0 = det["t_start"].to_list()
    d1 = det["t_end"].to_list()
    for di in range(det.height):
        best_gi, best_ov = None, 0.0
        for gi, b in enumerate(gt_breaths):
            ov = _overlap(d0[di], d1[di], b["t_start"], b["t_end"])
            if ov > best_ov:
                best_gi, best_ov = gi, ov
        if best_gi is not None:
            mapping[di] = best_gi
    return mapping


# --------------------------------------------------------------------------
# config wiring: thresholds come from config.yaml, not code
# --------------------------------------------------------------------------
def test_config_threshold_matches_expectation():
    cfg = events._load_config(CONFIG)
    assert cfg["breath_roll_threshold_deg"] == ROLL_THRESHOLD_DEG
    assert cfg["pushoff_g_threshold"] == 2.0
    assert cfg["exclude_after_pushoff_s"] == 4.0
    assert cfg["exclude_before_end_s"] == 2.0
    assert cfg["min_valid_cycles"] == 20


# --------------------------------------------------------------------------
# push-offs
# --------------------------------------------------------------------------
def test_detect_pushoffs_finds_four_boundaries():
    """All four length-start push-offs found, each near a ground-truth time,
    with no extras."""
    df, gt = _load_trial("trial_rotator")
    pushoffs = events.detect_pushoffs(df)
    gt_times = sorted(p["t"] for p in gt["pushoffs"])
    assert len(gt_times) == 4  # sanity: a 4-length trial

    det_starts = sorted(pushoffs["t_start"].to_list())
    assert pushoffs.height == len(gt_times), "extra or missing push-offs"

    # Detection triggers a few samples after the true wall push; the start
    # sample is at or just after the true time and well within 0.5 s.
    for gt_t, det_t in zip(gt_times, det_starts):
        assert det_t >= gt_t - 1e-6
        assert abs(det_t - gt_t) < 0.5

    # peak is a genuine spike (~3 g) clearly above the ~2 g threshold.
    assert pushoffs["peak_g"].min() > 2.0


def test_detect_pushoffs_across_all_committed_trials():
    for name in [
        "trial_rotator",
        "trial_lifter",
        "trial_flat",
        "trial_asymmetric",
        "trial_mixed",
    ]:
        df, gt = _load_trial(name)
        pushoffs = events.detect_pushoffs(df)
        assert pushoffs.height == len(gt["pushoffs"]), name


# --------------------------------------------------------------------------
# breath recall / precision
# --------------------------------------------------------------------------
def _score_recall_precision(name: str, R):
    df, gt = _load_trial(name)
    det = events.detect_breath_windows(df, transform=R)

    gt_breaths = gt["breaths"]
    detectable = [
        gi
        for gi, b in enumerate(gt_breaths)
        if abs(b["true_peak_roll_deg"]) >= ROLL_THRESHOLD_DEG
    ]
    low_roll_missed = len(gt_breaths) - len(detectable)

    d0 = det["t_start"].to_list()
    d1 = det["t_end"].to_list()
    matched_gt: set[int] = set()
    matched_det: set[int] = set()
    for di in range(det.height):
        for gi, b in enumerate(gt_breaths):
            if _overlap(d0[di], d1[di], b["t_start"], b["t_end"]) > 0:
                matched_gt.add(gi)
                matched_det.add(di)

    n_detectable_found = sum(1 for gi in detectable if gi in matched_gt)
    recall = n_detectable_found / max(1, len(detectable))
    false_positives = det.height - len(matched_det)
    fp_rate = false_positives / max(1, det.height)
    return recall, fp_rate, false_positives, low_roll_missed, det.height


@pytest.mark.parametrize("name", ["trial_rotator", "trial_lifter"])
def test_breath_recall_and_precision_on_real_roll(name):
    """>= 95% recall on detectable breaths, <= 5% false positives, on
    archetypes with genuine roll signal."""
    R = _committed_transform()
    recall, fp_rate, fp, missed, ndet = _score_recall_precision(name, R)
    assert recall >= 0.95, f"{name}: recall {recall:.3f}"
    assert fp_rate <= 0.05, f"{name}: fp_rate {fp_rate:.3f} ({fp} of {ndet})"
    # These archetypes have roll well above threshold: nothing inherently missed.
    assert missed == 0, f"{name}: unexpected low-roll breaths {missed}"


def test_flat_low_roll_breaths_are_reported_not_hidden(capsys):
    """FLAT breaths mostly stay under the roll threshold and are inherently
    undetectable. Recall on the *detectable* subset must still hold, and the
    misses are reported rather than silently dropped or chased with a lower
    threshold."""
    R = _committed_transform()
    recall, fp_rate, fp, missed, ndet = _score_recall_precision("trial_flat", R)
    # The few FLAT breaths that do cross the threshold are still all found.
    assert recall >= 0.95
    assert fp_rate <= 0.05
    # Honesty check: FLAT genuinely has low-roll breaths a roll detector cannot see.
    assert missed > 0
    print(
        f"FLAT: {missed} low-roll breaths inherently undetectable by a "
        f"+/-{ROLL_THRESHOLD_DEG:g} deg roll detector (reported, not dropped)."
    )


def test_side_convention_positive_roll_is_right():
    """CLAUDE.md sign convention: positive roll -> face to swimmer's right -> 'R'."""
    R = _committed_transform()
    df, gt = _load_trial("trial_rotator")
    det = events.detect_breath_windows(df, transform=R)
    for row in det.iter_rows(named=True):
        base = row["baseline_roll_deg"]
        if row["peak_roll_deg"] - base > 0:
            assert row["side"] == "R"
        else:
            assert row["side"] == "L"


def test_detect_breath_windows_requires_roll_source():
    df, _ = _load_trial("trial_rotator")
    with pytest.raises(ValueError):
        events.detect_breath_windows(df)  # no roll_deg, no transform


def test_detect_breath_windows_accepts_precomputed_roll():
    R = _committed_transform()
    df, gt = _load_trial("trial_rotator")
    calibrated = calibrate.apply(df, R)
    det = events.detect_breath_windows(calibrated)  # roll_deg already present
    assert det.height == gt["summary"]["n_breaths"]


# --------------------------------------------------------------------------
# exclusions (spec test 7): exact match against ground truth
# --------------------------------------------------------------------------
def test_exclusions_match_ground_truth_exactly():
    """On a fully-detectable fixture, the breaths the pipeline marks excluded
    match the ground-truth ``excluded_by_protocol`` set exactly -- same
    breaths, same reason codes -- using *detected* push-off times."""
    R = _committed_transform()
    df, gt = _load_trial("trial_rotator")
    gt_breaths = gt["breaths"]

    det = events.detect_breath_windows(df, transform=R)
    pushoffs = events.detect_pushoffs(df)
    marked = events.apply_exclusions(det, pushoffs, df)

    mapping = _match_detected_to_gt(det, gt_breaths)
    assert len(mapping) == det.height  # every detected window maps to a gt breath
    assert len(set(mapping.values())) == det.height  # one-to-one

    excluded = marked["excluded"].to_list()
    reasons = marked["exclusion_reason"].to_list()
    for di, gi in mapping.items():
        assert excluded[di] == gt_breaths[gi]["excluded_by_protocol"], (
            f"breath gt#{gi}: excluded {excluded[di]} != "
            f"{gt_breaths[gi]['excluded_by_protocol']}"
        )
        assert reasons[di] == gt_breaths[gi]["exclusion_reason"], (
            f"breath gt#{gi}: reason {reasons[di]!r} != "
            f"{gt_breaths[gi]['exclusion_reason']!r}"
        )


def test_within_2s_of_end_reason_matches_ground_truth():
    """trial_mixed carries a genuine 'within_2s_of_end' exclusion; the
    pipeline reproduces the same reason on the same breath."""
    R = _committed_transform()
    df, gt = _load_trial("trial_mixed")
    gt_breaths = gt["breaths"]
    # This fixture is expected to contain the end-exclusion case.
    assert any(b["exclusion_reason"] == events.REASON_END for b in gt_breaths)

    det = events.detect_breath_windows(df, transform=R)
    pushoffs = events.detect_pushoffs(df)
    marked = events.apply_exclusions(det, pushoffs, df)
    mapping = _match_detected_to_gt(det, gt_breaths)

    reasons = marked["exclusion_reason"].to_list()
    for di, gi in mapping.items():
        assert reasons[di] == gt_breaths[gi]["exclusion_reason"], (
            f"mixed breath gt#{gi}: {reasons[di]!r} != "
            f"{gt_breaths[gi]['exclusion_reason']!r}"
        )
    # And specifically the end case was reproduced.
    reproduced = {reasons[di] for di in mapping}
    assert events.REASON_END in reproduced


def test_apply_exclusions_is_pure_marking_not_dropping():
    """Hard constraint 5: excluded breaths stay in the table."""
    R = _committed_transform()
    df, _ = _load_trial("trial_rotator")
    det = events.detect_breath_windows(df, transform=R)
    pushoffs = events.detect_pushoffs(df)
    marked = events.apply_exclusions(det, pushoffs, df)
    assert marked.height == det.height  # nothing removed
    assert marked["excluded"].sum() > 0  # some are excluded
    # every excluded row carries a reason; every kept row does not
    for row in marked.iter_rows(named=True):
        if row["excluded"]:
            assert row["exclusion_reason"] in (events.REASON_PUSHOFF, events.REASON_END)
        else:
            assert row["exclusion_reason"] is None


def test_apply_exclusions_unit_end_and_pushoff():
    """Direct unit test of the exclusion boundaries on a synthetic breath
    table, independent of the detector."""
    # Trial spanning 0..100 s, one push-off at t=10.
    df = pl.DataFrame({"t": np.linspace(0.0, 100.0, 12001)})
    pushoffs = pl.DataFrame(
        {"t_start": [10.0], "t_peak": [10.05], "t_end": [10.1], "peak_g": [3.0]}
    )
    breaths = pl.DataFrame(
        {
            "t_start": [11.0, 13.5, 50.0, 98.5, 99.5],
            "t_end": [11.8, 14.3, 50.8, 99.3, 100.0],
            "side": ["R", "L", "R", "L", "R"],
            "peak_roll_deg": [40.0, -40.0, 40.0, -40.0, 40.0],
            "baseline_roll_deg": [0.0] * 5,
            "flags": [[] for _ in range(5)],
        }
    )
    marked = events.apply_exclusions(breaths, pushoffs, df, config_path=CONFIG)
    reasons = marked["exclusion_reason"].to_list()
    # 11.0 within [10,14) -> pushoff; 13.5 within [10,14) -> pushoff;
    # 50.0 kept; 98.5 and 99.5 within last 2 s (>98) -> end.
    assert reasons == [
        events.REASON_PUSHOFF,
        events.REASON_PUSHOFF,
        None,
        events.REASON_END,
        events.REASON_END,
    ]


# --------------------------------------------------------------------------
# insufficient cycles
# --------------------------------------------------------------------------
def _valid_count_and_flag(archetype: str, n_lengths: int, seed: int):
    """Detect + exclude a synth trial; return (n_valid, insufficient_flagged)."""
    df, gt = synth.generate_trial(archetype, n_lengths=n_lengths, seed=seed)
    seg, _ = synth.generate_calibration(
        mount_offset_deg=tuple(gt["mount_offset_deg"]),
        pitch_baseline_deg=gt["pitch_baseline_deg"],
        seed=seed + 5000,
    )
    R = calibrate.fit_transform(seg["t0a"], seg["t0b"])
    det = events.detect_breath_windows(df, transform=R)
    marked = events.apply_exclusions(det, events.detect_pushoffs(df), df)
    n_valid = int((~marked["excluded"]).sum())
    flagged = all(
        events.INSUFFICIENT_CYCLES in f for f in marked["flags"].to_list()
    )
    # Cross-check: our valid count reproduces the generator's ground truth.
    assert n_valid == gt["summary"]["n_valid_breaths"]
    return n_valid, flagged


def test_insufficient_cycles_two_sided():
    """The flag is threshold-driven: below min_valid_cycles it fires on every
    row, at/above it does not."""
    # Clearly too few valid breaths -> flagged.
    n_short, flagged_short = _valid_count_and_flag("ROTATOR", n_lengths=1, seed=5)
    assert n_short < 20
    assert flagged_short

    # Comfortably above the bar -> not flagged.
    n_long, flagged_long = _valid_count_and_flag("ROTATOR", n_lengths=6, seed=5)
    assert n_long >= 20
    assert not flagged_long


def test_full_4length_t7_clears_config_min_valid_cycles():
    """A realistic 4 x 25 m T7 (breathing every 3) clears min_valid_cycles = 20.

    Resolution of an earlier finding: with the synthetic swimmer set to a
    *recreational* pace (~30 s / 25 m), a complete 4-length T7 yields ~24
    breaths, ~22-23 of which survive the push-off/end exclusions -- above the
    20-cycle gate, as a real recreational T7 does. (The earlier ~15 valid was
    an artefact of a fit/competitive ~21 s pace, not a threshold that was set
    too high.) A genuinely short session still trips the flag -- see
    ``test_insufficient_cycles_two_sided``."""
    R = _committed_transform()
    df, gt = _load_trial("trial_rotator")
    det = events.detect_breath_windows(df, transform=R)
    marked = events.apply_exclusions(det, events.detect_pushoffs(df), df)
    n_valid = int((~marked["excluded"]).sum())
    assert n_valid == gt["summary"]["n_valid_breaths"]
    assert n_valid >= 20
    assert not any(events.INSUFFICIENT_CYCLES in f for f in marked["flags"].to_list())


# --------------------------------------------------------------------------
# hard constraint 2: no magnetometer, no yaw in executable code
# --------------------------------------------------------------------------
def test_module_source_uses_no_mag_or_yaw():
    """Strip string literals/comments (where 'yaw' appears only in prose) and
    check the executable source references neither mag_* nor yaw."""
    path = Path(__file__).resolve().parent.parent / "swimlab" / "events.py"
    tree = ast.parse(path.read_text())
    code_tokens: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue
        if isinstance(node, ast.Name):
            code_tokens.append(node.id)
        elif isinstance(node, ast.Attribute):
            code_tokens.append(node.attr)
    joined = " ".join(code_tokens).lower()
    assert "mag_x" not in joined and "mag_y" not in joined and "mag_z" not in joined
    assert "yaw" not in joined
