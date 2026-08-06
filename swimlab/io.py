"""Movella DOT export -> canonical dataframe.  Draft, but format-derived.

STATUS
------
Drafted against Movella/Xsens DOT's data format, now **cross-checked against a
reference implementation** -- the BLE payload parser in
``jiminghe/Xsens_DOT_PC_Reader`` (``movella_dot_py/core/parser.py`` +
``models/data_structures.py``) -- plus Movella's documented CSV column names.
That pins the field layout, types, units and the free-vs-raw acceleration
distinction below. It still has **not been run on a real export file**, so
format details that only a real recording can confirm (exact Euler convention,
the free-acceleration frame, the delimiter/locale, that gyro really is deg/s)
stay flagged ``# TODO(real-file)``. When the first real recording exists, run
:func:`validate_dot_export` on it, fix anything it flags, then trust
:func:`read_dot_export`.

The pipeline contract (what this must produce)
----------------------------------------------
The canonical dataframe (CLAUDE.md): ``t`` (s from start), ``quat_w..quat_z``
(unit quaternion, **sensor->global**), ``acc_*`` (m/s^2, sensor frame,
**includes gravity**), ``gyr_*`` (deg/s, sensor frame), ``mag_*`` (logged,
never used). calibrate/events/metrics all read this schema.

Derived DOT format (from the reference parser + Movella docs)
-------------------------------------------------------------
Onboard recording is exported to CSV; columns depend on the **payload mode**
the recording used. Field encodings in the BLE payload (little-endian):

* ``SampleTimeFine`` -- uint32, microseconds (1 MHz), starts at power-on, wraps
  at 2**32 (~1.2 h).  ``PacketCounter`` precedes it.
* ``Quat_W/X/Y/Z``   -- 4x float32, order **w, x, y, z**, sensor->earth.
* ``Euler_X/Y/Z``    -- 3x float32 = roll, pitch, yaw (deg).
* ``Acc_X/Y/Z``      -- 3x float32, m/s^2, **raw acceleration WITH gravity**.
* ``FreeAcc_X/Y/Z``  -- 3x float32, m/s^2, **free acceleration, gravity REMOVED**.
* ``Gyr_X/Y/Z``      -- 3x float32, deg/s.
* ``Mag_X/Y/Z``      -- 3x int16 / 4096 -> ~Gauss (arbitrary units, **not uT**).
* ``Status``         -- uint16 bitfield (clipping flags); ClipCount Acc/Gyr uint8.

**Which acceleration you get depends on the mode** (this is the load-bearing
fact for us): every orientation mode -- Complete/Extended Quaternion, all Euler
modes, Custom Modes 1-2 -- outputs **FreeAcc only** (gravity removed). Raw
``Acc`` (with gravity, which the pipeline needs) appears **only** in
``RATE_QUANTITIES(_WITH_MAG)`` and **``CUSTOM_MODE_5``**.

*Recommended recording mode:* **Custom Mode 5** (payload id 26) =
``Timestamp + Quaternion + Acceleration(raw) + Angular velocity``. That is the
canonical schema minus the (unused) magnetometer, at **120 Hz** (120 Hz is
*recording-only*; BLE streaming caps at 60 Hz) -- no gravity reconstruction
needed. Failing that, any FreeAcc export is handled by reconstructing gravity
from the quaternion (see :func:`_resolve_acceleration`), at the cost of relying
on the free-acceleration frame assumption.

Remaining real-file confirmations (each a ``# TODO(real-file)``)
----------------------------------------------------------------
A. Header strings / delimiter (locale may use ``;``); which columns the chosen
   mode actually emits.
B. Free-acceleration **frame** (this reader assumes the sensor/local frame when
   reconstructing) -- moot if you record Custom Mode 5 (raw Acc).
C. World frame ENU / z-up (gravity along -Z). NED / z-down would invert the
   pipeline; the validator checks acc-vs-quaternion-gravity alignment.
D. Euler rotation order (only used if quaternion columns are absent).
E. Gyro really deg/s (not rad/s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.transform import Rotation

# Canonical schema -- must match swimlab.synth.CANONICAL_COLUMNS (a drift test in
# tests/test_io.py asserts they are identical).
CANONICAL_COLUMNS: tuple[str, ...] = (
    "t",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "mag_x", "mag_y", "mag_z",
)

_G = 9.80665                 # m/s^2
_SAMPLE_RATE_HZ = 120.0      # F: assumed logging rate  # TODO(real-file)
_TIME_TICK_HZ = 1_000_000.0  # D: SampleTimeFine is 1 MHz ticks (us)  # TODO(real-file)
_TIME_WRAP = 2 ** 32         # D: SampleTimeFine wraps at 2**32  # TODO(real-file)

# A: assumed vendor column names -> canonical names.  # TODO(real-file): confirm
# exact header strings (case/underscores may differ between app/firmware versions).
_TIME_COL = "SampleTimeFine"
_QUAT_MAP = {"quat_w": "Quat_W", "quat_x": "Quat_X", "quat_y": "Quat_Y", "quat_z": "Quat_Z"}
# Raw acceleration WITH gravity (Rate Quantities / Custom Mode 5). Preferred.
_ACC_MAP = {"acc_x": "Acc_X", "acc_y": "Acc_Y", "acc_z": "Acc_Z"}
# Free acceleration, gravity REMOVED (every orientation mode). Gravity is
# reconstructed from the quaternion when only these are present.
_FREEACC_MAP = {"acc_x": "FreeAcc_X", "acc_y": "FreeAcc_Y", "acc_z": "FreeAcc_Z"}
_GYR_MAP = {"gyr_x": "Gyr_X", "gyr_y": "Gyr_Y", "gyr_z": "Gyr_Z"}
_MAG_MAP = {"mag_x": "Mag_X", "mag_y": "Mag_Y", "mag_z": "Mag_Z"}

#: The recording mode that yields the canonical schema directly (quaternion +
#: raw acceleration + gyro), minus the unused magnetometer. See module docstring.
RECOMMENDED_RECORDING_MODE = "Custom Mode 5 (Quaternion + Acceleration + Angular velocity), 120 Hz"
# Euler fallback (some modes export Euler, not quaternion).  # TODO(real-file):
# confirm the Euler rotation order/units before trusting the conversion.
_EULER_COLS = ("Euler_X", "Euler_Y", "Euler_Z")

_HEADER_MARKERS = (_TIME_COL, "PacketCounter")  # a data header row contains one of these


# --------------------------------------------------------------------------- #
# Header / preamble handling
# --------------------------------------------------------------------------- #

def _locate_header(path: Path) -> int:
    """Return the 0-based line index of the CSV header row.

    Movella DOT exports begin with a metadata preamble (device id, firmware,
    ordinal counter, ...) before the column header. We skip until the first line
    that looks like the data header (contains a known marker column).
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for i, line in enumerate(fh):
            if any(m in line for m in _HEADER_MARKERS):
                return i
    raise ValueError(
        f"{path}: no data header found (looked for {_HEADER_MARKERS}). "
        "Is this a Movella DOT CSV export? Run validate_dot_export() for detail."
    )


def _read_raw(path: Path) -> pl.DataFrame:
    """Read the tabular part of a DOT CSV (skipping the preamble)."""
    skip = _locate_header(path)
    # TODO(real-file): confirm the delimiter is ',' (some locales export ';').
    return pl.read_csv(path, skip_rows=skip, infer_schema_length=2000, truncate_ragged_lines=True)


# --------------------------------------------------------------------------- #
# Timestamp
# --------------------------------------------------------------------------- #

def _time_seconds(sample_time_fine: np.ndarray) -> np.ndarray:
    """SampleTimeFine (us, 1 MHz, wraps at 2**32) -> seconds from trial start."""
    ticks = np.asarray(sample_time_fine, dtype=np.float64)
    # unwrap counter resets: whenever it steps backwards, add a full period
    diffs = np.diff(ticks)
    wraps = np.cumsum(np.where(diffs < 0, _TIME_WRAP, 0.0))
    ticks = ticks + np.concatenate([[0.0], wraps])
    seconds = ticks / _TIME_TICK_HZ
    return seconds - seconds[0]


# --------------------------------------------------------------------------- #
# Acceleration (gravity handling)
# --------------------------------------------------------------------------- #

def _gravity_in_sensor(quat_wxyz: np.ndarray) -> np.ndarray:
    """Specific-force-at-rest (gravity) vector in the sensor frame, per sample.

    For a sensor->global quaternion in a z-up world, the accelerometer at rest
    reads +up in the sensor frame: ``R^{-1} @ [0,0,g]``. Used to reconstruct
    gravity-inclusive acceleration from a *free-acceleration* export.
    """
    # scipy wants [x,y,z,w]
    rot = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    up = np.tile([0.0, 0.0, _G], (quat_wxyz.shape[0], 1))
    return rot.inv().apply(up)


def _resolve_acceleration(
    acc: np.ndarray, quat_wxyz: np.ndarray, mode: str
) -> tuple[np.ndarray, str]:
    """Return (gravity-inclusive acc, note). ``mode`` in {'auto','with_gravity',
    'free_sensor'}.

    B: the pipeline needs gravity in ``acc``. ``auto`` inspects the resting
    magnitude: ~g means gravity is present (use as-is); ~0 means free
    acceleration (reconstruct by adding the quaternion-derived gravity).
    ``# TODO(real-file)``: confirm the export mode and, if free, the frame the
    vendor expresses free acceleration in (this assumes *sensor* frame).
    """
    mag = np.linalg.norm(acc, axis=1)
    resting = float(np.median(mag))
    if mode == "auto":
        mode = "with_gravity" if resting > 0.5 * _G else "free_sensor"
    if mode == "with_gravity":
        return acc, f"acc used as-is (median |acc|={resting:.2f} m/s^2 ~ g)"
    if mode == "free_sensor":
        recon = acc + _gravity_in_sensor(quat_wxyz)
        return recon, (
            f"acc reconstructed from free-acceleration + quaternion gravity "
            f"(median |acc_in|={resting:.2f}); ASSUMES free-accel in sensor frame "
            f"# TODO(real-file)"
        )
    raise ValueError(f"unknown acceleration mode {mode!r}")


# --------------------------------------------------------------------------- #
# Public: read
# --------------------------------------------------------------------------- #

def read_dot_export(
    path: str | Path, *, acceleration: str = "auto"
) -> pl.DataFrame:
    """Parse a Movella DOT CSV export into the canonical dataframe. **DRAFT.**

    Parameters
    ----------
    path:
        A DOT CSV export file.
    acceleration:
        ``'auto'`` (default) detects free vs gravity-inclusive acceleration;
        ``'with_gravity'`` / ``'free_sensor'`` force the interpretation (see
        :func:`_resolve_acceleration`).

    Returns
    -------
    polars.DataFrame with the canonical schema (:data:`CANONICAL_COLUMNS`).

    Notes
    -----
    Drafted against the documented format; run :func:`validate_dot_export` on a
    real file first. Raises ``ValueError`` with actionable guidance when a
    required column/mode assumption is not met, rather than emitting silently
    wrong data.
    """
    path = Path(path)
    raw = _read_raw(path)
    cols = set(raw.columns)

    if _TIME_COL not in cols:
        raise ValueError(f"{path}: missing '{_TIME_COL}' column.  # TODO(real-file): map the real time column")
    t = _time_seconds(raw[_TIME_COL].to_numpy())

    # orientation: quaternion preferred, Euler fallback
    if all(v in cols for v in _QUAT_MAP.values()):
        quat = np.column_stack([raw[_QUAT_MAP[k]].to_numpy() for k in
                                ("quat_w", "quat_x", "quat_y", "quat_z")])
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = quat / np.where(norms == 0, 1.0, norms)  # renormalise defensively
    elif all(c in cols for c in _EULER_COLS):
        # C/A fallback: convert Euler -> quaternion.  # TODO(real-file): confirm
        # the Euler order ('xyz' intrinsic degrees assumed) and that it is
        # sensor->global before trusting this branch.
        eul = np.column_stack([raw[c].to_numpy() for c in _EULER_COLS])
        xyzw = Rotation.from_euler("xyz", eul, degrees=True).as_quat()
        quat = xyzw[:, [3, 0, 1, 2]]
    else:
        raise ValueError(
            f"{path}: no orientation columns found (need {list(_QUAT_MAP.values())} "
            f"or {list(_EULER_COLS)}). Re-export in a quaternion logging mode."
        )

    # Acceleration: prefer raw Acc_* (with gravity, e.g. Custom Mode 5); fall back
    # to FreeAcc_* (gravity removed -- every orientation mode) and reconstruct.
    has_raw = all(v in cols for v in _ACC_MAP.values())
    has_free = all(v in cols for v in _FREEACC_MAP.values())
    if has_raw:
        acc = np.column_stack([raw[_ACC_MAP[k]].to_numpy() for k in ("acc_x", "acc_y", "acc_z")])
        acc, _note = _resolve_acceleration(acc, quat, acceleration)
    elif has_free:
        free = np.column_stack([raw[_FREEACC_MAP[k]].to_numpy() for k in ("acc_x", "acc_y", "acc_z")])
        acc = free + _gravity_in_sensor(quat)  # reconstruct gravity-inclusive acc
        _note = "gravity reconstructed from FreeAcc_* + quaternion (frame assumed sensor)"
    else:
        raise ValueError(
            f"{path}: no acceleration columns. Need raw {list(_ACC_MAP.values())} "
            f"(preferred -- record in {RECOMMENDED_RECORDING_MODE}) or free "
            f"{list(_FREEACC_MAP.values())}. Re-export in a mode that includes acceleration."
        )

    if not all(v in cols for v in _GYR_MAP.values()):
        raise ValueError(f"{path}: missing gyroscope columns {list(_GYR_MAP.values())}.")
    gyr = np.column_stack([raw[_GYR_MAP[k]].to_numpy() for k in ("gyr_x", "gyr_y", "gyr_z")])
    # TODO(real-file): confirm gyro is deg/s. If rad/s, multiply by 180/pi here.

    # E: magnetometer optional; normalised units, never used downstream.
    if all(v in cols for v in _MAG_MAP.values()):
        mag = np.column_stack([raw[_MAG_MAP[k]].to_numpy() for k in ("mag_x", "mag_y", "mag_z")])
    else:
        mag = np.full((len(t), 3), np.nan)  # mode without mag; carried as NaN

    out = pl.DataFrame({
        "t": t.astype(np.float64),
        "quat_w": quat[:, 0], "quat_x": quat[:, 1], "quat_y": quat[:, 2], "quat_z": quat[:, 3],
        "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
        "gyr_x": gyr[:, 0], "gyr_y": gyr[:, 1], "gyr_z": gyr[:, 2],
        "mag_x": mag[:, 0], "mag_y": mag[:, 1], "mag_z": mag[:, 2],
    }).select(CANONICAL_COLUMNS)
    return out


# --------------------------------------------------------------------------- #
# Public: validate
# --------------------------------------------------------------------------- #

@dataclass
class Check:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str


@dataclass
class ExportReport:
    """Result of :func:`validate_dot_export`. ``ok`` is False if any FAIL."""
    path: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def ok(self) -> bool:
        return all(c.status != "FAIL" for c in self.checks)

    def __str__(self) -> str:
        lines = [f"DOT export validation — {self.path}  [{'OK' if self.ok else 'FAIL'}]"]
        for c in self.checks:
            lines.append(f"  [{c.status:4}] {c.name}: {c.detail}")
        return "\n".join(lines)


def validate_dot_export(path: str | Path) -> ExportReport:
    """Check a real DOT export against every assumption this module makes.

    Run this on the first real recording. Each check is PASS / WARN / FAIL:
    header located, required columns present, sample rate ~120 Hz, timestamp
    monotonic, quaternion unit-norm, acceleration carries gravity, gravity
    points the expected way (ENU / z-up). A FAIL means ``read_dot_export`` output
    cannot be trusted until this module's flagged assumptions are corrected.
    """
    path = Path(path)
    rep = ExportReport(path=str(path))

    try:
        hdr = _locate_header(path)
        rep.add("header", "PASS", f"data header at line {hdr}")
    except ValueError as e:
        rep.add("header", "FAIL", str(e))
        return rep

    raw = _read_raw(path)
    cols = set(raw.columns)

    # columns
    if _TIME_COL in cols:
        rep.add("time_column", "PASS", _TIME_COL)
    else:
        rep.add("time_column", "FAIL", f"'{_TIME_COL}' not found; columns={sorted(cols)}")
    has_quat = all(v in cols for v in _QUAT_MAP.values())
    has_eul = all(c in cols for c in _EULER_COLS)
    rep.add("orientation", "PASS" if has_quat else ("WARN" if has_eul else "FAIL"),
            "quaternion" if has_quat else ("Euler only — will be converted (confirm order)"
            if has_eul else "no orientation columns; re-export in quaternion mode"))
    has_raw_acc = all(v in cols for v in _ACC_MAP.values())
    has_free_acc = all(v in cols for v in _FREEACC_MAP.values())
    rep.add("acc_columns",
            "PASS" if has_raw_acc else ("WARN" if has_free_acc else "FAIL"),
            "Acc_X/Y/Z (raw, with gravity)" if has_raw_acc
            else ("FreeAcc_X/Y/Z only (gravity removed) — will be reconstructed; "
                  f"prefer recording {RECOMMENDED_RECORDING_MODE}" if has_free_acc
                  else "no acceleration columns (need Acc_* or FreeAcc_*)"))
    rep.add("gyr_columns", "PASS" if all(v in cols for v in _GYR_MAP.values()) else "FAIL",
            "Gyr_X/Y/Z present" if all(v in cols for v in _GYR_MAP.values()) else "missing")
    rep.add("mag_columns", "PASS" if all(v in cols for v in _MAG_MAP.values()) else "WARN",
            "Mag_X/Y/Z present" if all(v in cols for v in _MAG_MAP.values())
            else "no magnetometer (ok — never used)")

    if _TIME_COL not in cols:
        return rep

    # sample rate + monotonic
    t = _time_seconds(raw[_TIME_COL].to_numpy())
    if len(t) > 1:
        dt = np.diff(t)
        rate = 1.0 / float(np.median(dt))
        rep.add("sample_rate", "PASS" if abs(rate - _SAMPLE_RATE_HZ) < 5 else "WARN",
                f"median {rate:.1f} Hz (expected {_SAMPLE_RATE_HZ:.0f})")
        rep.add("monotonic_time", "PASS" if np.all(dt > 0) else "FAIL",
                "strictly increasing" if np.all(dt > 0) else
                f"{int(np.sum(dt <= 0))} non-increasing steps (timestamp unwrap wrong?)")

    # quaternion norm
    if has_quat:
        q = np.column_stack([raw[_QUAT_MAP[k]].to_numpy() for k in
                             ("quat_w", "quat_x", "quat_y", "quat_z")])
        norm = np.linalg.norm(q, axis=1)
        rep.add("quat_unit_norm", "PASS" if np.allclose(norm, 1.0, atol=1e-2) else "WARN",
                f"mean |q|={float(np.mean(norm)):.4f} (expected 1.0)")

    # acceleration: gravity present? and pointing the ENU way?
    if has_raw_acc:
        acc = np.column_stack([raw[_ACC_MAP[k]].to_numpy() for k in ("acc_x", "acc_y", "acc_z")])
        resting = float(np.median(np.linalg.norm(acc, axis=1)))
        if resting > 0.5 * _G:
            rep.add("acc_has_gravity", "PASS", f"median |acc|={resting:.2f} ~ g (gravity present)")
            if has_quat:
                # C: check the quaternion-implied gravity matches the measured one.
                # If they anti-correlate, the world frame is z-down (NED) not z-up.
                g_pred = _gravity_in_sensor(q)
                dots = np.sum(acc * g_pred, axis=1) / (
                    np.linalg.norm(acc, axis=1) * np.linalg.norm(g_pred, axis=1) + 1e-9)
                md = float(np.median(dots))
                rep.add("gravity_frame", "PASS" if md > 0.7 else "FAIL",
                        f"acc vs quaternion-gravity alignment={md:+.2f} "
                        f"({'z-up/ENU as assumed' if md > 0.7 else 'MISMATCH — likely NED/z-down; pipeline would invert'})")
        else:
            rep.add("acc_has_gravity", "WARN",
                    f"Acc_* median |acc|={resting:.2f} << g — labelled raw but looks free; "
                    "confirm the export mode")
    elif has_free_acc:
        free = np.column_stack([raw[_FREEACC_MAP[k]].to_numpy() for k in ("acc_x", "acc_y", "acc_z")])
        resting = float(np.median(np.linalg.norm(free, axis=1)))
        rep.add("acc_has_gravity", "WARN",
                f"FreeAcc only (median |free|={resting:.2f} << g) — read_dot_export "
                f"reconstructs gravity from the quaternion; prefer recording "
                f"{RECOMMENDED_RECORDING_MODE} so raw Acc (with gravity) is logged directly")

    # NaN in required channels
    acc_cols = list(_ACC_MAP.values()) if has_raw_acc else (
        list(_FREEACC_MAP.values()) if has_free_acc else [])
    req = [_TIME_COL] + acc_cols + list(_GYR_MAP.values())
    if has_quat:
        req += list(_QUAT_MAP.values())
    nan_cols = [c for c in req if c in cols and raw[c].is_null().any()]
    rep.add("no_missing_values", "PASS" if not nan_cols else "FAIL",
            "no nulls in required channels" if not nan_cols else f"nulls in {nan_cols}")

    return rep


# --------------------------------------------------------------------------- #
# Public: nod-marker detection (video sync, TASKS.md task 7)
# --------------------------------------------------------------------------- #

def _pitch_from_quat(df: pl.DataFrame) -> np.ndarray:
    """Gravity-referenced pitch (deg) from the fused quaternion, per sample."""
    q = df.select(["quat_w", "quat_x", "quat_y", "quat_z"]).to_numpy()
    up = Rotation.from_quat(q[:, [1, 2, 3, 0]]).inv().apply(np.tile([0, 0, 1.0], (len(q), 1)))
    return np.degrees(np.arctan2(-up[:, 0], np.hypot(up[:, 1], up[:, 2])))


def find_sync_nods(
    df: pl.DataFrame, *, min_amplitude_deg: float = 15.0, min_separation_s: float = 0.25
) -> list[float]:
    """Detect the T0c sync-nod marker times (s) for aligning IMU to video.

    The T0c calibration segment is three deliberate ~+/-25 deg pitch nods. This
    finds the prominent pitch extrema (nods) so the first one can anchor the
    video clock. Operates on the fused quaternion, so it needs no calibration.
    Returns the nod peak times, sorted.
    """
    t = df["t"].to_numpy()
    pitch = _pitch_from_quat(df)
    pitch = pitch - np.median(pitch)  # relative to the resting pose
    fs = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else _SAMPLE_RATE_HZ

    # prominent |pitch| peaks separated by min_separation
    mag = np.abs(pitch)
    gap = max(int(min_separation_s * fs), 1)
    peaks: list[int] = []
    i = 1
    while i < len(mag) - 1:
        if mag[i] >= min_amplitude_deg and mag[i] >= mag[i - 1] and mag[i] > mag[i + 1]:
            if not peaks or (i - peaks[-1]) >= gap:
                peaks.append(i)
            elif mag[i] > mag[peaks[-1]]:
                peaks[-1] = i
        i += 1
    return [round(float(t[p]), 4) for p in peaks]
