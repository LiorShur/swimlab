"""Tests for swimlab.io (Movella DOT reader — DRAFT).

IMPORTANT: no real Movella DOT export exists yet, so these tests validate the
reader's *logic* (header skipping, timestamp/units conversion, gravity
reconstruction, validation checks, nod detection) against a **synthetic CSV
written in the assumed format**. They do NOT confirm the assumed format matches
a real export — that is exactly what ``io.validate_dot_export`` is for, to be
run on the first real recording. See the assumptions flagged in swimlab/io.py.
"""

from __future__ import annotations

import csv

import numpy as np
import polars as pl
import pytest

from swimlab import calibrate, events, io, metrics, synth

def _write_fake_dot_csv(df: pl.DataFrame, path, *, acc="raw", ned=False,
                        include_mag=True, drop=(), preamble=True):
    """Write a canonical dataframe as a Movella-DOT-style CSV, using the **real**
    column conventions derived from the DOT payload format (see swimlab/io.py):

    * ``acc="raw"``  -> ``Acc_X/Y/Z`` with gravity (Rate Quantities / Custom Mode 5).
    * ``acc="free"`` -> ``FreeAcc_X/Y/Z`` with gravity removed (orientation modes).
    * ``include_mag=False`` -> no ``Mag_*`` (Custom Mode 5 has no magnetometer).
    * ``ned`` flips gravity into a z-down world (writes raw ``Acc_*``).
    * ``drop`` omits columns (to exercise the validator).

    ``PacketCounter``/``SampleTimeFine`` and a metadata preamble mirror a real
    export. This proves the reader's logic in place; it is not a real recording.
    """
    t = df["t"].to_numpy()
    quat = df.select(["quat_w", "quat_x", "quat_y", "quat_z"]).to_numpy()
    acc_raw = df.select(["acc_x", "acc_y", "acc_z"]).to_numpy()
    gyr = df.select(["gyr_x", "gyr_y", "gyr_z"]).to_numpy()
    mag = df.select(["mag_x", "mag_y", "mag_z"]).to_numpy()
    grav = io._gravity_in_sensor(quat)  # sensor-frame gravity from the quaternion

    if ned:
        acc_vals, acc_cols = -grav, ("Acc_X", "Acc_Y", "Acc_Z")  # z-down world
    elif acc == "free":
        acc_vals, acc_cols = acc_raw - grav, ("FreeAcc_X", "FreeAcc_Y", "FreeAcc_Z")
    else:
        acc_vals, acc_cols = acc_raw, ("Acc_X", "Acc_Y", "Acc_Z")

    stf = np.round(t * 1_000_000.0).astype(np.int64)  # SampleTimeFine (us)
    header = ["PacketCounter", "SampleTimeFine", "Quat_W", "Quat_X", "Quat_Y", "Quat_Z",
              *acc_cols, "Gyr_X", "Gyr_Y", "Gyr_Z"]
    if include_mag:
        header += ["Mag_X", "Mag_Y", "Mag_Z"]
    header = [c for c in header if c not in drop]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if preamble:  # a few metadata rows, as real exports carry
            fh.write("// Start Time: 2026-07-23 10:00:00.000\n")
            fh.write("// Device Tag: DOT-OCCIPUT\n")
            fh.write("// Firmware Version: 1.10.0\n")
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(len(t)):
            row = {
                "PacketCounter": i, "SampleTimeFine": int(stf[i]),
                "Quat_W": quat[i, 0], "Quat_X": quat[i, 1], "Quat_Y": quat[i, 2], "Quat_Z": quat[i, 3],
                acc_cols[0]: acc_vals[i, 0], acc_cols[1]: acc_vals[i, 1], acc_cols[2]: acc_vals[i, 2],
                "Gyr_X": gyr[i, 0], "Gyr_Y": gyr[i, 1], "Gyr_Z": gyr[i, 2],
                "Mag_X": mag[i, 0], "Mag_Y": mag[i, 1], "Mag_Z": mag[i, 2],
            }
            w.writerow([row[c] for c in header])


@pytest.fixture(scope="module")
def trial():
    df, _ = synth.generate_trial("LIFTER", n_lengths=1, noise=True, seed=3,
                                 pitch_baseline_deg=5.0)
    return df


# --------------------------------------------------------------------------- #
# schema / structure
# --------------------------------------------------------------------------- #

def test_canonical_schema_matches_synth():
    """io must emit exactly the schema synth/the pipeline expect."""
    assert io.CANONICAL_COLUMNS == synth.CANONICAL_COLUMNS


def test_preamble_is_skipped(tmp_path, trial):
    p = tmp_path / "e.csv"
    _write_fake_dot_csv(trial, p, preamble=True)
    out = io.read_dot_export(p)
    assert out.columns == list(io.CANONICAL_COLUMNS)
    assert out.height == trial.height


# --------------------------------------------------------------------------- #
# timestamp
# --------------------------------------------------------------------------- #

def test_time_seconds_unwraps_and_zeroes():
    fs = 120.0
    ticks = (np.arange(600) * (1_000_000.0 / fs))
    ticks = (ticks + (io._TIME_WRAP - ticks[300])) % io._TIME_WRAP  # force a wrap mid-stream
    t = io._time_seconds(ticks)
    assert t[0] == 0.0
    assert np.all(np.diff(t) > 0)                       # monotonic after unwrap
    assert np.allclose(np.diff(t), 1.0 / fs, atol=1e-6)  # steady 120 Hz


# --------------------------------------------------------------------------- #
# round-trip: the reader recovers the canonical data
# --------------------------------------------------------------------------- #

def test_roundtrip_with_gravity(tmp_path, trial):
    p = tmp_path / "g.csv"
    _write_fake_dot_csv(trial, p, acc="raw")
    out = io.read_dot_export(p)
    assert np.allclose(out["t"].to_numpy(), trial["t"].to_numpy(), atol=1e-5)
    for c in ("quat_w", "acc_x", "acc_z", "gyr_y", "mag_z"):
        assert np.allclose(out[c].to_numpy(), trial[c].to_numpy(), atol=1e-6), c


def test_custom_mode_5_export_no_magnetometer(tmp_path, trial):
    """The recommended recording mode (Custom Mode 5) is quaternion + raw
    acceleration + gyro, with NO magnetometer. It must read cleanly, with the
    (unused) magnetometer carried as NaN."""
    p = tmp_path / "cm5.csv"
    _write_fake_dot_csv(trial, p, acc="raw", include_mag=False)
    out = io.read_dot_export(p)
    for c in ("acc_x", "acc_z", "gyr_y"):
        assert np.allclose(out[c].to_numpy(), trial[c].to_numpy(), atol=1e-6), c
    assert out["mag_x"].is_nan().all()  # mag absent -> NaN, never used downstream


def test_roundtrip_free_acceleration_reconstructs_gravity(tmp_path, trial):
    """An orientation-mode export carries FreeAcc_* (gravity removed); the reader
    reconstructs gravity-inclusive acc from the quaternion."""
    p = tmp_path / "free.csv"
    _write_fake_dot_csv(trial, p, acc="free")
    out = io.read_dot_export(p)
    for c in ("acc_x", "acc_y", "acc_z"):
        assert np.allclose(out[c].to_numpy(), trial[c].to_numpy(), atol=1e-6), c


def test_write_dot_export_custom_mode_5_roundtrips(tmp_path, trial):
    """write_dot_export (Custom Mode 5) then read_dot_export recovers the canonical
    data — the synth → DOT-CSV → reader loop, validatable with no hardware."""
    p = tmp_path / "cm5_export.csv"
    io.write_dot_export(trial, p, mode="custom5")
    # a Custom Mode 5 file validates cleanly (raw acc, quaternion, gyro; no mag)
    rep = io.validate_dot_export(p)
    assert rep.ok, str(rep)
    out = io.read_dot_export(p)
    assert out.columns == list(io.CANONICAL_COLUMNS)
    for c in ("t", "quat_w", "quat_z", "acc_x", "acc_z", "gyr_y"):
        assert np.allclose(out[c].to_numpy(), trial[c].to_numpy(), atol=1e-5), c
    assert out["mag_x"].is_nan().all()  # Custom Mode 5 has no magnetometer


def test_write_dot_export_free_acceleration_roundtrips(tmp_path, trial):
    """The Complete-Quaternion export writes FreeAcc_*; the reader reconstructs
    gravity-inclusive acc back to the original."""
    p = tmp_path / "cq_export.csv"
    io.write_dot_export(trial, p, mode="complete_quaternion")
    out = io.read_dot_export(p)
    for c in ("acc_x", "acc_y", "acc_z"):
        assert np.allclose(out[c].to_numpy(), trial[c].to_numpy(), atol=1e-5), c


def test_read_output_is_pipeline_compatible(tmp_path):
    """A read-back export runs through calibrate->events->metrics and matches the
    same pipeline on the original synth dataframe (io is a faithful front-end)."""
    base = (4.0, -3.0, 11.0)
    df, _ = synth.generate_trial("LIFTER", mount_offset_deg=base, noise=False,
                                 seed=7, pitch_baseline_deg=5.0)
    p = tmp_path / "trial.csv"
    _write_fake_dot_csv(df, p, acc="raw")
    read = io.read_dot_export(p)

    segs, _ = synth.generate_calibration(mount_offset_deg=base, pitch_baseline_deg=5.0,
                                         noise=False, seed=7)
    R = calibrate.fit_transform(segs["t0a"], segs["t0b"])

    def dpitch(frame):
        cal = calibrate.apply(frame, R)
        marked = events.apply_exclusions(events.detect_breath_windows(cal),
                                         events.detect_pushoffs(cal), cal)
        return float(metrics.trial_summary(metrics.per_breath_metrics(cal, marked))
                     ["mean_d_pitch_breath"][0])

    assert abs(dpitch(read) - dpitch(df)) < 0.05


# --------------------------------------------------------------------------- #
# validator
# --------------------------------------------------------------------------- #

def test_validate_good_export_passes(tmp_path, trial):
    p = tmp_path / "ok.csv"
    _write_fake_dot_csv(trial, p, acc="raw")
    rep = io.validate_dot_export(p)
    assert rep.ok, str(rep)
    by = {c.name: c.status for c in rep.checks}
    assert by["orientation"] == "PASS"
    assert by["sample_rate"] == "PASS"
    assert by["acc_has_gravity"] == "PASS"
    assert by["gravity_frame"] == "PASS"


def test_validate_warns_on_missing_gyro(tmp_path, trial):
    """Gyro is unused downstream (Complete Quaternion mode has none), so a missing
    gyroscope is a WARN, not a failure — the file is still usable."""
    p = tmp_path / "nogyr.csv"
    _write_fake_dot_csv(trial, p, drop=("Gyr_X", "Gyr_Y", "Gyr_Z"))
    rep = io.validate_dot_export(p)
    assert rep.ok, str(rep)  # gyro optional -> still OK
    assert any(c.name == "gyr_columns" and c.status == "WARN" for c in rep.checks)


def test_validate_fails_on_missing_acceleration(tmp_path, trial):
    """Acceleration IS required (push-off detection). Dropping it must FAIL."""
    p = tmp_path / "noacc.csv"
    _write_fake_dot_csv(trial, p, drop=("Acc_X", "Acc_Y", "Acc_Z"))
    rep = io.validate_dot_export(p)
    assert not rep.ok
    assert any(c.name == "acc_columns" and c.status == "FAIL" for c in rep.checks)


def test_validate_warns_on_free_acceleration(tmp_path, trial):
    p = tmp_path / "free.csv"
    _write_fake_dot_csv(trial, p, acc="free")
    rep = io.validate_dot_export(p)
    assert any(c.name == "acc_has_gravity" and c.status == "WARN" for c in rep.checks)


def test_validate_catches_ned_frame(tmp_path, trial):
    """A z-down (NED) world frame would silently invert the whole pipeline —
    the validator must catch it via the acc/quaternion-gravity alignment."""
    p = tmp_path / "ned.csv"
    _write_fake_dot_csv(trial, p, ned=True)
    rep = io.validate_dot_export(p)
    assert not rep.ok
    assert any(c.name == "gravity_frame" and c.status == "FAIL" for c in rep.checks)


def test_validate_rejects_non_dot_file(tmp_path):
    p = tmp_path / "random.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    rep = io.validate_dot_export(p)
    assert not rep.ok
    assert rep.checks[0].name == "header" and rep.checks[0].status == "FAIL"


# --------------------------------------------------------------------------- #
# nod-marker detection (video sync)
# --------------------------------------------------------------------------- #

def test_find_sync_nods_on_t0c():
    """T0c has three ~+/-25 deg pitch nods (~0.8, 2.0, 3.2 s). Find all three."""
    segs, _ = synth.generate_calibration(mount_offset_deg=(3.0, -2.0, 12.0),
                                         pitch_baseline_deg=4.0, noise=True, seed=1)
    nods = io.find_sync_nods(segs["t0c"])
    assert len(nods) == 3, nods
    for expected, got in zip([0.8, 2.0, 3.2], nods):
        assert abs(got - expected) < 0.15, (expected, got)


def test_no_nods_in_a_swim(tmp_path):
    """A plain swim (no deliberate nods) should not trip the nod detector as if
    it were a sync marker — head pitch during breaths is well under the amplitude
    of a deliberate nod is NOT guaranteed, so we only assert it does not find a
    tidy triple at nod amplitude on a calm segment."""
    df, _ = synth.generate_trial("ROTATOR", n_lengths=1, noise=False, seed=2,
                                 pitch_baseline_deg=4.0)
    nods = io.find_sync_nods(df, min_amplitude_deg=40.0)  # nods are ~25 deg; 40 is above swim
    assert nods == []
