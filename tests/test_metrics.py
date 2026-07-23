"""Tests for swimlab.metrics (per-breath metric table + per-trial summary).

Design (mirrors tests/test_events.py):

* ``swimlab.synth`` is a black-box ground-truth generator. We call its public
  ``generate_trial`` / ``generate_calibration`` only, never its internals.
* Calibration and event detection are used through their public APIs.
* Detected breaths are matched to ground-truth breaths by *temporal overlap*
  (any positive overlap), exactly as the events tests do.

Recovery of ``d_pitch_breath``
------------------------------
The baseline for ``d_pitch_breath`` is the median pitch over the preceding
non-breath window ``[max(prev_breath_end, t_start - L), t_start)`` with ``L``
**one stroke-cycle**, estimated data-drivenly as the dominant period of the
roll signal (see swimlab/metrics.py). One whole oscillation period avoids the
phase-dependent bias of a non-integer (e.g. 1.5-cycle) window.

The synthetic ``true_d_pitch_deg`` is **gravity-referenced** (relative to the
T0b prone pose) -- the same quantity the pipeline measures -- so with a matching
calibration the pipeline recovers it tightly and *roll-independently*: high-roll
ROTATOR breaths recover as well as low-roll ones (there is no tilt cross-talk,
because calibrate's canonical frame anchors the zero reference to T0b exactly).
Zero-noise recovery is within 0.5 deg for every archetype; full-noise committed
fixtures recover the trial mean within 2 deg.

Because the ground truth is gravity-referenced relative to the prone pose, a
trial only round-trips against a calibration at the *same* prone baseline (one
swimmer, one prone pose). The committed trial fixtures are therefore pinned to
the same ``pitch_baseline`` as the committed ``calib_t0a/t0b`` poses, and
freshly generated trials build a matching calibration via ``_from_synth``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import swimlab.synth as synth
from swimlab import calibrate, events, metrics

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _committed_transform():
    t0a = pl.read_parquet(FIXTURES / "calib_t0a.parquet")
    t0b = pl.read_parquet(FIXTURES / "calib_t0b.parquet")
    return calibrate.fit_transform(t0a, t0b)


def _pipeline(df, R):
    """Calibrate -> detect -> exclude, returning (calibrated_df, marked_breaths)."""
    cal = calibrate.apply(df, R)
    det = events.detect_breath_windows(cal, transform=R)
    pushoffs = events.detect_pushoffs(df)
    marked = events.apply_exclusions(det, pushoffs, df)
    return cal, marked


def _from_fixture(name):
    """Committed noisy fixture, calibrated with the committed poses."""
    df = pl.read_parquet(FIXTURES / f"{name}.parquet")
    gt = json.loads((FIXTURES / f"{name}.gt.json").read_text())
    R = _committed_transform()
    cal, marked = _pipeline(df, R)
    return cal, marked, gt


def _from_synth(archetype, seed, *, noise):
    """Freshly generated trial with a matching calibration fit."""
    df, gt = synth.generate_trial(archetype, noise=noise, seed=seed)
    mount = tuple(gt["mount_offset_deg"])
    baseline = gt["pitch_baseline_deg"]
    calib, _ = synth.generate_calibration(
        mount_offset_deg=mount, pitch_baseline_deg=baseline, noise=noise, seed=seed
    )
    R = calibrate.fit_transform(calib["t0a"], calib["t0b"])
    cal, marked = _pipeline(df, R)
    return cal, marked, gt


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _match_to_gt(per_breath, gt_breaths):
    """Map per-breath row index -> gt breath index by max temporal overlap."""
    mapping = {}
    d0 = per_breath["t_start"].to_list()
    d1 = per_breath["t_end"].to_list()
    for di in range(per_breath.height):
        best_gi, best_ov = None, 0.0
        for gi, b in enumerate(gt_breaths):
            ov = _overlap(d0[di], d1[di], b["t_start"], b["t_end"])
            if ov > best_ov:
                best_gi, best_ov = gi, ov
        if best_gi is not None:
            mapping[di] = best_gi
    return mapping


def _d_pitch_errors_valid(per_breath, gt):
    """Per-breath (recovered - truth) d_pitch over *valid* matched breaths."""
    mapping = _match_to_gt(per_breath, gt["breaths"])
    d = per_breath["d_pitch_breath"].to_numpy()
    excluded = per_breath["excluded"].to_numpy().astype(bool)
    errs = []
    for di, gi in mapping.items():
        if excluded[di]:
            continue
        errs.append(d[di] - gt["breaths"][gi]["true_d_pitch_deg"])
    return np.array(errs)


# --------------------------------------------------------------------------
# hard constraint 2: no magnetometer / yaw in executable code
# --------------------------------------------------------------------------
def test_module_source_uses_no_mag_or_yaw():
    """Strip string literals/comments; the executable source must reference
    neither ``mag_*`` nor ``yaw`` (hard constraint 2), same check as events."""
    path = Path(__file__).resolve().parent.parent / "swimlab" / "metrics.py"
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


# --------------------------------------------------------------------------
# purity / never-drop (hard constraint 5)
# --------------------------------------------------------------------------
def test_per_breath_preserves_every_row_and_reasons():
    """Nothing is dropped: one output row per detected breath, and every
    protocol-excluded breath keeps its exclusion reason code."""
    cal, marked, _ = _from_fixture("trial_rotator")
    pb = metrics.per_breath_metrics(cal, marked)
    assert pb.height == marked.height  # nothing removed
    assert pb["excluded"].sum() > 0  # some excluded
    for row in pb.iter_rows(named=True):
        if row["excluded"]:
            assert row["exclusion_reason"] in (events.REASON_PUSHOFF, events.REASON_END)
        else:
            assert row["exclusion_reason"] is None


def test_per_breath_does_not_mutate_input():
    cal, marked, _ = _from_fixture("trial_mixed")
    before = marked.columns
    _ = metrics.per_breath_metrics(cal, marked)
    assert marked.columns == before  # input untouched
    assert "d_pitch_breath" not in marked.columns


# --------------------------------------------------------------------------
# columns present
# --------------------------------------------------------------------------
def test_per_breath_has_all_metric_columns():
    cal, marked, _ = _from_fixture("trial_lifter")
    pb = metrics.per_breath_metrics(cal, marked)
    for col in (
        "d_pitch_breath",
        "peak_pitch_deg",
        "baseline_pitch_deg",
        "peak_roll_breath",
        "roll_pitch_ratio",
        "breath_duration",
        "metric_flags",
    ):
        assert col in pb.columns
    # breath_duration == t_end - t_start
    dur = (pb["t_end"] - pb["t_start"]).to_numpy()
    assert np.allclose(dur, pb["breath_duration"].to_numpy())
    # peak_roll_breath is a non-negative magnitude
    assert (pb["peak_roll_breath"].to_numpy() >= 0).all()


# --------------------------------------------------------------------------
# roll_pitch_ratio: present, and degenerate handled with a reason code
# --------------------------------------------------------------------------
def test_roll_pitch_ratio_defined_for_detected_breaths():
    """A detected breath clears the config roll threshold, so peak_roll_breath
    is well above zero and roll_pitch_ratio is defined for every breath."""
    cal, marked, _ = _from_fixture("trial_mixed")
    pb = metrics.per_breath_metrics(cal, marked)
    assert pb["roll_pitch_ratio"].null_count() == 0
    ratio = pb["roll_pitch_ratio"].to_numpy()
    d = pb["d_pitch_breath"].to_numpy()
    pr = pb["peak_roll_breath"].to_numpy()
    assert np.allclose(ratio, d / pr)
    # nobody was flagged degenerate
    assert all(metrics.REASON_ROLL_DEGENERATE not in f for f in pb["metric_flags"].to_list())


def test_roll_pitch_ratio_degenerate_is_reason_coded_not_dropped():
    """When peak_roll_breath is ~0 the ratio is None *and* the row is kept with
    a ROLL_NEAR_ZERO reason -- never silently dropped (hard constraint 5)."""
    t = np.arange(0.0, 2.0, 1.0 / 120.0)
    trial = pl.DataFrame(
        {
            "t": t,
            "pitch_deg": np.full_like(t, 5.0),  # any pitch
            "roll_deg": np.zeros_like(t),  # no roll at all
        }
    )
    breaths = pl.DataFrame(
        {
            "t_start": [1.0],
            "t_end": [1.3],
            "side": ["R"],
            "excluded": [False],
            "exclusion_reason": [None],
            "flags": [[]],
        }
    )
    pb = metrics.per_breath_metrics(trial, breaths)
    assert pb.height == 1  # kept
    assert pb["roll_pitch_ratio"].to_list()[0] is None
    assert metrics.REASON_ROLL_DEGENERATE in pb["metric_flags"].to_list()[0]


def test_no_baseline_samples_is_reason_coded():
    """A breath starting at the very first sample has no preceding pitch; it is
    flagged NO_BASELINE_SAMPLES and kept, not dropped."""
    t = np.arange(0.0, 2.0, 1.0 / 120.0)
    trial = pl.DataFrame(
        {"t": t, "pitch_deg": np.linspace(0, 10, t.size), "roll_deg": np.full_like(t, 40.0)}
    )
    breaths = pl.DataFrame(
        {"t_start": [0.0], "t_end": [0.3], "side": ["R"], "flags": [[]]}
    )
    pb = metrics.per_breath_metrics(trial, breaths)
    assert metrics.REASON_NO_BASELINE in pb["metric_flags"].to_list()[0]
    assert pb.height == 1


# --------------------------------------------------------------------------
# d_pitch_breath recovery
# --------------------------------------------------------------------------
def test_d_pitch_zero_noise_recovers_trial_mean_mixed():
    """MIXED's breath roll is modest, so the tilt cross-talk is small and the
    zero-noise trial-mean d_pitch matches ground truth well within 0.5 deg."""
    cal, marked, gt = _from_synth("MIXED", seed=5, noise=False)
    pb = metrics.per_breath_metrics(cal, marked)
    summ = metrics.trial_summary(pb)
    mean_mine = summ["mean_d_pitch_breath"].to_list()[0]
    err = mean_mine - gt["summary"]["true_mean_d_pitch_deg"]
    assert abs(err) < 0.5, f"MIXED zero-noise trial-mean err={err:+.3f} deg"


@pytest.mark.parametrize(
    "archetype,seed", [("MIXED", 5), ("ASYMMETRIC", 100)]
)
def test_d_pitch_zero_noise_low_bias_archetypes(archetype, seed):
    """Modest-roll archetypes recover the trial-mean d_pitch within 0.5 deg."""
    cal, marked, gt = _from_synth(archetype, seed=seed, noise=False)
    pb = metrics.per_breath_metrics(cal, marked)
    errs = _d_pitch_errors_valid(pb, gt)
    assert abs(errs.mean()) < 0.5, f"{archetype} mean d_pitch err={errs.mean():+.3f}"


def test_d_pitch_recovery_is_tight_and_roll_independent():
    """Zero-noise d_pitch recovery is tight and does NOT depend on breath roll.

    Because the ground truth is gravity-referenced (relative to the T0b prone
    pose) and calibrate anchors that zero reference to T0b exactly, there is no
    roll-dependent pitch cross-talk: high-roll ROTATOR breaths recover as well
    as low-roll ones. Pooled over archetypes and seeds, the per-breath error is
    sub-degree and essentially uncorrelated with peak roll -- the regression
    that an earlier (align_vectors) calibration frame exhibited."""
    rolls, errs = [], []
    for archetype in ("LIFTER", "ROTATOR", "MIXED", "ASYMMETRIC"):
        for seed in range(2, 12):
            cal, marked, gt = _from_synth(archetype, seed=seed, noise=False)
            pb = metrics.per_breath_metrics(cal, marked)
            mapping = _match_to_gt(pb, gt["breaths"])
            d = pb["d_pitch_breath"].to_numpy()
            pr = pb["peak_roll_breath"].to_numpy()
            excl = pb["excluded"].to_numpy().astype(bool)
            for di, gi in mapping.items():
                if excl[di]:
                    continue
                errs.append(d[di] - gt["breaths"][gi]["true_d_pitch_deg"])
                rolls.append(pr[di])

    rolls = np.array(rolls)
    errs = np.array(errs)
    # tight per-breath recovery, well within the 0.5 deg zero-noise bar
    assert np.abs(errs).mean() < 0.3, f"mean |err|={np.abs(errs).mean():.3f} deg"
    assert np.abs(errs).max() < 0.5, f"max |err|={np.abs(errs).max():.3f} deg"
    # and no roll cross-talk: high-roll breaths are no worse than low-roll ones
    hi = rolls > np.quantile(rolls, 0.75)
    lo = rolls < np.quantile(rolls, 0.25)
    assert abs(errs[hi].mean() - errs[lo].mean()) < 0.3
    assert abs(float(np.corrcoef(rolls, errs)[0, 1])) < 0.5


def test_clean_fixture_round_trips_to_ground_truth():
    """Cross-module drift guard: the committed zero-noise round-trip fixture
    (``trial_lifter_clean`` + ``calib_clean_*``) recovers the gravity-referenced
    ground-truth ``d_pitch``/``peak_roll`` to a small fraction of a degree.

    synth (ground truth) and calibrate (measurement) keep private copies of the
    canonical-frame construction for module independence; if they ever drift,
    this exact round-trip breaks -- catching it here rather than three modules
    downstream."""
    df = pl.read_parquet(FIXTURES / "trial_lifter_clean.parquet")
    gt = json.loads((FIXTURES / "trial_lifter_clean.gt.json").read_text())
    t0a = pl.read_parquet(FIXTURES / "calib_clean_t0a.parquet")
    t0b = pl.read_parquet(FIXTURES / "calib_clean_t0b.parquet")
    R = calibrate.fit_transform(t0a, t0b)
    cal, marked = _pipeline(df, R)
    pb = metrics.per_breath_metrics(cal, marked)
    mapping = _match_to_gt(pb, gt["breaths"])
    d = pb["d_pitch_breath"].to_numpy()
    pr = pb["peak_roll_breath"].to_numpy()
    excl = pb["excluded"].to_numpy().astype(bool)
    matched = 0
    for di, gi in mapping.items():
        if excl[di]:
            continue
        b = gt["breaths"][gi]
        assert abs(d[di] - b["true_d_pitch_deg"]) < 0.3, f"d_pitch drift at breath {gi}"
        assert abs(pr[di] - b["true_peak_roll_deg"]) < 0.3, f"peak_roll drift at breath {gi}"
        matched += 1
    assert matched >= 8


@pytest.mark.parametrize(
    "name", ["trial_lifter", "trial_rotator", "trial_mixed", "trial_asymmetric"]
)
def test_d_pitch_full_noise_trial_mean_within_2deg(name):
    """On the committed full-noise fixtures every detectable archetype recovers
    the trial-mean d_pitch within 2 deg."""
    cal, marked, gt = _from_fixture(name)
    pb = metrics.per_breath_metrics(cal, marked)
    summ = metrics.trial_summary(pb)
    mean_mine = summ["mean_d_pitch_breath"].to_list()[0]
    err = mean_mine - gt["summary"]["true_mean_d_pitch_deg"]
    assert abs(err) < 2.0, f"{name} full-noise trial-mean err={err:+.3f} deg"


def test_flat_under_detection_is_visible_not_hidden():
    """FLAT's sub-threshold breaths are undetectable from roll alone (events
    cannot window them). The metric layer cannot invent them; the shortfall
    surfaces as a valid-breath count far below the ground-truth count -- it is
    visible, not silently absorbed."""
    cal, marked, gt = _from_fixture("trial_flat")
    pb = metrics.per_breath_metrics(cal, marked)
    summ = metrics.trial_summary(pb)
    n_valid = summ["n_valid"].to_list()[0]
    assert n_valid < gt["summary"]["n_valid_breaths"]  # under-detection is real
    # the breaths that WERE detected still recover d_pitch sensibly
    errs = _d_pitch_errors_valid(pb, gt)
    if errs.size:
        assert np.abs(errs).max() < 2.0


# --------------------------------------------------------------------------
# asymmetry_index
# --------------------------------------------------------------------------
def test_asymmetry_index_recovers_asymmetric():
    """ASYMMETRIC injects left ~= 4-6 deg, right ~= 18-20 deg. asymmetry_index
    = (left - right)/mean is strongly negative and matches ground truth."""
    cal, marked, gt = _from_fixture("trial_asymmetric")
    pb = metrics.per_breath_metrics(cal, marked)
    summ = metrics.trial_summary(pb)
    ai = summ["asymmetry_index"].to_list()[0]

    gb = [b for b in gt["breaths"] if not b["excluded_by_protocol"]]
    gl = np.mean([b["true_d_pitch_deg"] for b in gb if b["side"] == "L"])
    gr = np.mean([b["true_d_pitch_deg"] for b in gb if b["side"] == "R"])
    gt_ai = (gl - gr) / np.mean([b["true_d_pitch_deg"] for b in gb])

    assert ai < -0.5, f"asymmetry_index should be strongly negative, got {ai:+.3f}"
    assert abs(ai - gt_ai) < 0.15, f"AI {ai:+.3f} vs gt {gt_ai:+.3f}"


def test_asymmetry_index_near_zero_for_symmetric():
    """A bilaterally-symmetric archetype (LIFTER) has near-zero asymmetry."""
    cal, marked, _ = _from_fixture("trial_lifter")
    pb = metrics.per_breath_metrics(cal, marked)
    summ = metrics.trial_summary(pb)
    ai = summ["asymmetry_index"].to_list()[0]
    assert abs(ai) < 0.2, f"symmetric asymmetry_index should be ~0, got {ai:+.3f}"


# --------------------------------------------------------------------------
# summary aggregates
# --------------------------------------------------------------------------
def test_summary_uses_only_valid_breaths():
    """mean_d_pitch_breath and pitch_variability are computed over non-excluded
    breaths only; counts add up."""
    cal, marked, _ = _from_fixture("trial_rotator")
    pb = metrics.per_breath_metrics(cal, marked)
    summ = metrics.trial_summary(pb).to_dicts()[0]

    valid = pb.filter(~pl.col("excluded"))
    expected_mean = float(valid["d_pitch_breath"].mean())
    expected_sd = float(valid["d_pitch_breath"].to_numpy().std(ddof=1))

    assert summ["n_breaths"] == pb.height
    assert summ["n_valid"] + summ["n_excluded"] == summ["n_breaths"]
    assert summ["n_valid"] == valid.height
    assert abs(summ["mean_d_pitch_breath"] - expected_mean) < 1e-9
    assert abs(summ["pitch_variability"] - expected_sd) < 1e-9


def test_summary_flags_bubble_up_insufficient_cycles():
    """A full 4x25 T7 falls below config min_valid_cycles (events sets the flag
    on every breath); the summary surfaces it once."""
    cal, marked, _ = _from_fixture("trial_lifter")
    pb = metrics.per_breath_metrics(cal, marked)
    summ = metrics.trial_summary(pb).to_dicts()[0]
    # events sets INSUFFICIENT_CYCLES because 16 valid < 20; it must appear.
    assert events.INSUFFICIENT_CYCLES in summ["flags"]


# --------------------------------------------------------------------------
# pitch_drift_100m (defined for T9; runs on any multi-length trial)
# --------------------------------------------------------------------------
def test_pitch_drift_per_length_runs_and_has_slope():
    """One row per length with a median pitch, plus an OLS slope. Meaningful
    only on the T9 drift protocol -- here we just check it computes."""
    df = pl.read_parquet(FIXTURES / "trial_lifter.parquet")
    R = _committed_transform()
    cal = calibrate.apply(df, R)
    pushoffs = events.detect_pushoffs(df)
    drift = metrics.pitch_drift_per_length(cal, pushoffs)
    assert drift.height >= 1
    assert "median_pitch_deg" in drift.columns
    slope = drift["pitch_drift_per_length"].to_list()[0]
    # 4x25 T7 has >=2 lengths, so the slope is defined (a finite float).
    assert slope is not None and np.isfinite(slope)
    # slope is constant across the rows (a per-trial scalar broadcast per length)
    assert len(set(drift["pitch_drift_per_length"].to_list())) == 1
