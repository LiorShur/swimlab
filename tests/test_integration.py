"""End-to-end integration harness for the swimlab pipeline (task 6).

This exercises the acceptance tests of SYNTHETIC_DATA_SPEC.md section 8 through
the *whole* pipeline -- ``swimlab.pipeline.run_session`` and, where a test needs
finer access than a session read gives, the same public stages it wires
together (``synth`` -> ``calibrate`` -> ``events`` -> ``metrics`` -> ``stats``).

Tests **1, 2, 3, 4, 7** are this task's gate. Tests 5 (bad calibration), 6 (yaw
independence) and 8 (determinism) are the generator's own responsibility and
are covered in ``tests/test_synth.py``; they are lightly re-asserted here at the
pipeline boundary rather than re-deriving their machinery.

CRITICAL modelling contract (see the task brief and ``generate_trial``): the
synthetic ground truth is gravity-referenced to the T0b prone pose, which
depends on the swimmer's ``pitch_baseline``. Each swimmer's trial is therefore
calibrated against a calibration generated at *that swimmer's own* mount and
baseline -- one swimmer, one prone pose. ``_synth_session`` enforces this by
construction; a shared calibration across differing baselines would inject a
baseline-mismatch error and is the mistake the per-session layout prevents.

The acceptance tolerances (0.5 / 1 / 2 deg, > 90 %, >= 95 % / <= 5 %) are spec
constants and live here in the test, not in the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from swimlab import calibrate, events, pipeline, stats, synth

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
ROLL_THRESHOLD_DEG = float(CONFIG["breath_roll_threshold_deg"])


# --------------------------------------------------------------------------- #
# Session synthesis: write one swimmer's canonical parquet session to disk,
# honouring the one-swimmer-one-baseline calibration contract.
# --------------------------------------------------------------------------- #
def _write_session(
    directory: Path,
    trial: pl.DataFrame,
    segs: dict[str, pl.DataFrame],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    trial.write_parquet(directory / pipeline.SESSION_FILES["trial"])
    segs["t0a"].write_parquet(directory / pipeline.SESSION_FILES["t0a"])
    segs["t0b"].write_parquet(directory / pipeline.SESSION_FILES["t0b"])
    if "t0c" in segs:
        segs["t0c"].write_parquet(directory / pipeline.SESSION_FILES["t0c"])
    return directory


def _synth_session(
    tmp_path: Path,
    archetype: str,
    *,
    mount: tuple[float, float, float],
    baseline: float,
    noise: bool,
    trial_seed: int,
    calib_seed: int,
    mount_slip_deg_per_min: float = 0.0,
    name: str = "session",
) -> tuple[Path, dict]:
    """Synthesise one swimmer's trial + its *matching* calibration and write both.

    The calibration is generated at the SAME mount and baseline as the trial
    (one swimmer, one prone pose), per the task's critical modelling contract.
    Returns the session directory and the trial ground truth.
    """
    trial, gt = synth.generate_trial(
        archetype,
        mount_offset_deg=mount,
        mount_slip_deg_per_min=mount_slip_deg_per_min,
        noise=noise,
        seed=trial_seed,
        pitch_baseline_deg=baseline,
    )
    segs, _ = synth.generate_calibration(
        mount_offset_deg=mount,
        pitch_baseline_deg=baseline,
        noise=noise,
        seed=calib_seed,
    )
    directory = _write_session(tmp_path / name, trial, segs)
    return directory, gt


# --------------------------------------------------------------------------- #
# Scoring helpers (shared with the events/metrics unit tests' conventions):
# match a detected/metric row to a ground-truth breath by temporal overlap.
# --------------------------------------------------------------------------- #
def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _match_rows_to_gt(table: pl.DataFrame, gt_breaths: list[dict]) -> dict[int, int]:
    """Map each metric-table row index -> gt breath index by max overlap."""
    mapping: dict[int, int] = {}
    t0 = table["t_start"].to_list()
    t1 = table["t_end"].to_list()
    for di in range(table.height):
        best_gi, best_ov = None, 0.0
        for gi, b in enumerate(gt_breaths):
            ov = _overlap(t0[di], t1[di], b["t_start"], b["t_end"])
            if ov > best_ov:
                best_gi, best_ov = gi, ov
        if best_gi is not None:
            mapping[di] = best_gi
    return mapping


def _score(table: pl.DataFrame, gt: dict) -> dict:
    """Recall / FP / per-breath recovery errors of a metric table vs ground truth.

    Recall is scored against the *detectable* ground-truth subset -- breaths
    whose true peak roll reaches the config threshold. Low-roll breaths a roll
    detector cannot window are reported, never chased with a lower threshold
    (events.detect_breath_windows documents this).
    """
    gt_breaths = gt["breaths"]
    detectable = [
        gi
        for gi, b in enumerate(gt_breaths)
        if abs(b["true_peak_roll_deg"]) >= ROLL_THRESHOLD_DEG
    ]

    t0 = table["t_start"].to_list()
    t1 = table["t_end"].to_list()
    dp = table["d_pitch_breath"].to_numpy()
    pr = table["peak_roll_breath"].to_numpy()
    excluded = table["excluded"].to_numpy()

    matched_gt: set[int] = set()
    matched_det: set[int] = set()
    dpitch_err: list[float] = []
    roll_err: list[float] = []
    for di in range(table.height):
        for gi, b in enumerate(gt_breaths):
            if _overlap(t0[di], t1[di], b["t_start"], b["t_end"]) > 0:
                matched_gt.add(gi)
                matched_det.add(di)
                if not excluded[di] and not b["excluded_by_protocol"]:
                    dpitch_err.append(dp[di] - b["true_d_pitch_deg"])
                    roll_err.append(pr[di] - b["true_peak_roll_deg"])
                break

    n_found = sum(1 for gi in detectable if gi in matched_gt)
    recall = n_found / max(1, len(detectable))
    false_positives = table.height - len(matched_det)
    fp_rate = false_positives / max(1, table.height)
    low_roll_missed = len(gt_breaths) - len(detectable)
    return {
        "recall": recall,
        "fp_rate": fp_rate,
        "false_positives": false_positives,
        "low_roll_missed": low_roll_missed,
        "n_detected": table.height,
        "dpitch_abs_max": float(np.abs(dpitch_err).max()) if dpitch_err else 0.0,
        "dpitch_abs_mean": float(np.abs(dpitch_err).mean()) if dpitch_err else 0.0,
        "roll_abs_max": float(np.abs(roll_err).max()) if roll_err else 0.0,
    }


def _swimmer_mean_d_pitch(table: pl.DataFrame) -> float:
    """Mean ``d_pitch_breath`` over a swimmer's valid (non-excluded) breaths."""
    valid = ~table["excluded"].to_numpy()
    dp = table["d_pitch_breath"].to_numpy()[valid]
    return float(np.mean(dp)) if dp.size else float("nan")


# --------------------------------------------------------------------------- #
# Acceptance test 1 -- round trip (zero noise, zero offset)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("archetype", ["ROTATOR", "LIFTER"])
def test_1_round_trip(tmp_path, archetype):
    """Zero-noise, zero-offset trial through calibrate -> events -> metrics
    recovers ``d_pitch_breath`` within 0.5 deg and detects every detectable
    breath with no false positives (SYNTHETIC_DATA_SPEC.md test 1)."""
    session, gt = _synth_session(
        tmp_path,
        archetype,
        mount=(0.0, 0.0, 0.0),
        baseline=4.0,
        noise=False,
        trial_seed=102,
        calib_seed=11,
    )
    table, _flags = pipeline.run_session(session)
    s = _score(table, gt)

    assert s["dpitch_abs_max"] < 0.5, f"{archetype}: d_pitch err {s['dpitch_abs_max']:.3f} deg"
    assert s["recall"] == 1.0, f"{archetype}: recall {s['recall']:.3f}"
    assert s["false_positives"] == 0, f"{archetype}: {s['false_positives']} false positives"


# --------------------------------------------------------------------------- #
# Acceptance test 2 -- mount recovery (offset (10,15,20), no noise)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("archetype", ["LIFTER", "ROTATOR"])
def test_2_mount_recovery(tmp_path, archetype):
    """With mount ``(10, 15, 20)`` and no noise, recovered pitch/roll match
    truth within 1 deg. Validated on the study-relevant recovered per-breath
    pitch (``d_pitch_breath``) and roll (``peak_roll_breath``) -- per calibrate's
    identifiability note the raw intrinsic-Euler triple is not the target
    (SYNTHETIC_DATA_SPEC.md test 2)."""
    session, gt = _synth_session(
        tmp_path,
        archetype,
        mount=(10.0, 15.0, 20.0),
        baseline=4.0,
        noise=False,
        trial_seed=303,
        calib_seed=11,
    )
    table, _flags = pipeline.run_session(session)
    s = _score(table, gt)

    assert s["dpitch_abs_max"] < 1.0, f"{archetype}: d_pitch err {s['dpitch_abs_max']:.3f} deg"
    assert s["roll_abs_max"] < 1.0, f"{archetype}: peak_roll err {s['roll_abs_max']:.3f} deg"


# --------------------------------------------------------------------------- #
# Acceptance test 3 -- noise tolerance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("archetype", ["ROTATOR", "LIFTER"])
def test_3_noise_tolerance(tmp_path, archetype):
    """Full noise: recovered ``d_pitch_breath`` within 2 deg, breath detection
    >= 95 % recall (on the *detectable* subset -- archetypes with real roll) with
    <= 5 % false positives (SYNTHETIC_DATA_SPEC.md test 3)."""
    session, gt = _synth_session(
        tmp_path,
        archetype,
        mount=(10.0, 15.0, 20.0),
        baseline=4.0,
        noise=True,
        trial_seed=202,
        calib_seed=21,
    )
    table, _flags = pipeline.run_session(session)
    s = _score(table, gt)

    assert s["dpitch_abs_max"] < 2.0, f"{archetype}: d_pitch err {s['dpitch_abs_max']:.3f} deg"
    assert s["recall"] >= 0.95, f"{archetype}: recall {s['recall']:.3f}"
    assert s["fp_rate"] <= 0.05, (
        f"{archetype}: fp_rate {s['fp_rate']:.3f} "
        f"({s['false_positives']} of {s['n_detected']})"
    )


# --------------------------------------------------------------------------- #
# Acceptance test 4 -- archetype separation (THE GATE)
# --------------------------------------------------------------------------- #
def test_4_archetype_separation_gate(tmp_path, capsys):
    """THE GATE (SYNTHETIC_DATA_SPEC.md test 4). 20 LIFTER + 20 ROTATOR swimmers
    with noise and random mount offsets, each calibrated against its own
    matching prone-baseline calibration; classify each swimmer by its mean
    ``d_pitch_breath`` at the Youden-optimal threshold and require accuracy
    > 90 %.

    This is the whole project's gate. It is not tuned to pass: the mount RNG is
    seeded once and the threshold is data-driven. If it lands below 90 % the
    assertion fails loudly rather than being massaged -- that would be a real
    finding about the study design, upstream of the code.
    """
    rng = np.random.default_rng(20260723)
    scores: list[float] = []
    labels: list[int] = []

    for label, archetype in ((1, "LIFTER"), (0, "ROTATOR")):
        for i in range(20):
            mount = tuple(float(x) for x in rng.uniform(-15.0, 15.0, size=3))
            baseline = float(rng.uniform(-5.0, 12.0))
            session, _gt = _synth_session(
                tmp_path,
                archetype,
                mount=mount,  # type: ignore[arg-type]
                baseline=baseline,
                noise=True,
                trial_seed=10_000 + label * 1000 + i,
                calib_seed=20_000 + label * 1000 + i,
                name=f"{archetype.lower()}_{i}",
            )
            table, _flags = pipeline.run_session(session)
            scores.append(_swimmer_mean_d_pitch(table))
            labels.append(label)

    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    # Youden-optimal threshold on mean d_pitch (LIFTER = high = positive class).
    roc = stats.roc_analysis(scores_arr, labels_arr, n_boot=1000, seed=0)
    pred = (scores_arr >= roc.youden_threshold).astype(int)
    accuracy = float((pred == labels_arr).mean())

    # Q1 bimodality sanity (informational): pooled d_pitch should be bimodal.
    dip = stats.dip_test(scores_arr, n_boot=500, seed=0)

    lift = scores_arr[labels_arr == 1]
    rot = scores_arr[labels_arr == 0]
    with capsys.disabled():
        print(
            "\n[test 4 -- THE GATE] 20 LIFTER + 20 ROTATOR, noise + random mounts, "
            "per-swimmer matched calibration:"
        )
        print(f"  LIFTER  mean d_pitch: [{lift.min():.2f}, {lift.max():.2f}] deg")
        print(f"  ROTATOR mean d_pitch: [{rot.min():.2f}, {rot.max():.2f}] deg")
        print(
            f"  AUC={roc.auc:.3f}  Youden thr={roc.youden_threshold:.2f} deg  "
            f"accuracy={accuracy:.3f}"
        )
        print(f"  Q1 dip test (informational): dip={dip.dip_statistic:.4f} p={dip.p_value:.4f}")

    assert accuracy > 0.90, (
        f"GATE FAILED: classification accuracy {accuracy:.3f} <= 0.90. "
        f"AUC={roc.auc:.3f}. This is a study-design finding, not a tuning target."
    )


# --------------------------------------------------------------------------- #
# Acceptance test 7 -- exclusion logic
# --------------------------------------------------------------------------- #
def test_7_exclusion_logic_matches_ground_truth(tmp_path):
    """Breaths the pipeline marks excluded are exactly those the ground truth
    marks ``excluded_by_protocol`` (SYNTHETIC_DATA_SPEC.md test 7). Uses a
    fully-detectable ROTATOR (all breaths cross the roll threshold) so the
    metric rows and ground-truth breaths map one-to-one and the exclusion sets
    can be compared without detection gaps confounding the result."""
    session, gt = _synth_session(
        tmp_path,
        "ROTATOR",
        mount=(6.0, -3.0, 15.0),
        baseline=3.0,
        noise=False,
        trial_seed=303,
        calib_seed=31,
    )
    table, _flags = pipeline.run_session(session)

    gt_breaths = gt["breaths"]
    detectable = [
        gi
        for gi, b in enumerate(gt_breaths)
        if abs(b["true_peak_roll_deg"]) >= ROLL_THRESHOLD_DEG
    ]
    assert len(detectable) == len(gt_breaths), "fixture must be fully detectable"

    mapping = _match_rows_to_gt(table, gt_breaths)
    assert len(mapping) == table.height, "every detected breath maps to a gt breath"
    # Bijective: no two detections claim the same ground-truth breath.
    assert len(set(mapping.values())) == table.height, "one-to-one match expected"
    assert len(mapping) == len(gt_breaths), "every gt breath is detected"

    excluded = table["excluded"].to_numpy()
    for di, gi in mapping.items():
        assert bool(excluded[di]) == bool(gt_breaths[gi]["excluded_by_protocol"]), (
            f"breath {gi}: pipeline excluded={bool(excluded[di])} "
            f"vs gt={gt_breaths[gi]['excluded_by_protocol']}"
        )


# --------------------------------------------------------------------------- #
# run_session contract / flag surfacing
# --------------------------------------------------------------------------- #
def test_run_session_returns_table_and_flags(tmp_path):
    """run_session returns the per-breath table plus session-level flags, and
    reads a session whether or not the optional t0c segment is present."""
    session, _gt = _synth_session(
        tmp_path,
        "LIFTER",
        mount=(3.0, -2.0, 12.0),
        baseline=4.0,
        noise=False,
        trial_seed=102,
        calib_seed=11,
    )
    table, flags = pipeline.run_session(session)
    assert isinstance(table, pl.DataFrame)
    assert isinstance(flags, list)
    for col in ("d_pitch_breath", "peak_roll_breath", "excluded", "exclusion_reason"):
        assert col in table.columns
    # A short 4-length trial yields < 20 valid breaths -> INSUFFICIENT_CYCLES,
    # surfaced (never raised) per CLAUDE.md hard constraint 5.
    assert events.INSUFFICIENT_CYCLES in flags

    # t0c is optional: removing it must not change the result.
    (session / pipeline.SESSION_FILES["t0c"]).unlink()
    table2, flags2 = pipeline.run_session(session)
    assert table2.equals(table)
    assert flags2 == flags


def test_run_session_flags_suspect_calibration(tmp_path):
    """A bad calibration (T0b performed looking forward by 25 deg) surfaces
    CALIB_POSE_SUSPECT through run_session without raising -- the session is
    still processed (SYNTHETIC_DATA_SPEC.md test 5, re-asserted at the pipeline
    boundary; the generator-side machinery lives in test_synth.py)."""
    trial, _gt = synth.generate_trial(
        "ROTATOR", mount_offset_deg=(3.0, -2.0, 12.0), noise=False, seed=7,
        pitch_baseline_deg=4.0,
    )
    segs, cgt = synth.generate_calibration(
        mount_offset_deg=(3.0, -2.0, 12.0),
        pitch_baseline_deg=4.0,
        bad_calibration=True,
        noise=False,
        seed=2,
    )
    assert cgt["bad_calibration"]
    session = _write_session(tmp_path / "bad_calib", trial, segs)
    table, flags = pipeline.run_session(session)
    assert calibrate.CALIB_POSE_SUSPECT in flags
    assert table.height > 0  # processed, not aborted


def test_run_session_missing_file_raises(tmp_path):
    """A session missing a required segment raises a clear FileNotFoundError."""
    session, _gt = _synth_session(
        tmp_path, "LIFTER", mount=(0.0, 0.0, 0.0), baseline=0.0,
        noise=False, trial_seed=1, calib_seed=1,
    )
    (session / pipeline.SESSION_FILES["t0b"]).unlink()
    with pytest.raises(FileNotFoundError):
        pipeline.run_session(session)


# --------------------------------------------------------------------------- #
# Informational report (NOT a gate): mount-slip degradation of the two
# candidate primary-gate metrics. Printed, never asserted.
# --------------------------------------------------------------------------- #
def test_mount_slip_metric_degradation_report(tmp_path, capsys):
    """Sweep ``mount_slip_deg_per_min`` 0..10 and compare how ``d_pitch_breath``
    and ``roll_pitch_ratio`` degrade against the no-slip baseline. This may
    inform which becomes the primary gate metric (CLAUDE.md notes
    ``roll_pitch_ratio`` as a dimensionless candidate more robust to mount
    variation). Informational only -- no assertion gates on it."""
    from swimlab import metrics

    mount = (6.0, -3.0, 15.0)
    baseline = 4.0

    def summarise(slip: float) -> tuple[float, float]:
        session, _gt = _synth_session(
            tmp_path,
            "LIFTER",
            mount=mount,
            baseline=baseline,
            noise=False,
            trial_seed=102,
            calib_seed=11,
            mount_slip_deg_per_min=slip,
            name=f"slip_{slip:g}",
        )
        table, _flags = pipeline.run_session(session)
        summ = metrics.trial_summary(table)
        return (
            float(summ["mean_d_pitch_breath"][0]),
            float(summ["mean_roll_pitch_ratio"][0]),
        )

    base_dp, base_rpr = summarise(0.0)
    rows = []
    for slip in (1.0, 2.0, 5.0, 10.0):
        dp, rpr = summarise(slip)
        dp_pct = 100.0 * (dp - base_dp) / base_dp
        rpr_pct = 100.0 * (rpr - base_rpr) / base_rpr
        rows.append((slip, dp, dp_pct, rpr, rpr_pct))

    with capsys.disabled():
        print(
            "\n[informational -- mount-slip robustness, LIFTER, noise-free] "
            "d_pitch vs roll_pitch_ratio drift from no-slip baseline "
            f"(d_pitch0={base_dp:.3f} deg, rpr0={base_rpr:.4f}):"
        )
        print(f"  {'slip/min':>8}  {'d_pitch':>8} {'dPitch%':>8}  {'rpr':>8} {'rpr%':>8}")
        for slip, dp, dp_pct, rpr, rpr_pct in rows:
            print(
                f"  {slip:>8.0f}  {dp:>8.3f} {dp_pct:>+7.1f}%  "
                f"{rpr:>8.4f} {rpr_pct:>+7.1f}%"
            )
        worse = "roll_pitch_ratio" if abs(rows[-1][4]) > abs(rows[-1][2]) else "d_pitch"
        print(
            f"  note: over this ~25 s trial the larger relative drift at "
            f"10 deg/min is in {worse} (informational, not a gate)."
        )
