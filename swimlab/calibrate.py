"""Calibration: recover the sensor-to-skull transform from the two static
T0 poses and express head orientation as gravity-referenced pitch/roll.

Coordinate conventions (see CLAUDE.md)
--------------------------------------
Angles live in the calibrated, z-up-at-prone skull frame. At the T0b
face-down float reference the calibrated sensor z-axis points up and the
gravity ("up") vector reads ``[0, 0, +1]``.

* **pitch** -- rotation about the medio-lateral axis, degrees.
  **Positive = nose up (head lift).** Zero is the T0b prone pose.
* **roll** -- rotation about the longitudinal (nose-to-occiput) axis,
  degrees. **Positive = face rotating to the swimmer's right.**
* **yaw** -- never computed. Gravity cannot observe rotation about the
  gravity axis and pool-hall magnetics are unusable (hard constraint 2).

Method
------
The transform is recovered from **gravity only**. For each static pose we
take the yaw-independent "up" direction in the sensor frame (from the fused
quaternion, which is robust to linear acceleration) and average it over the
pose. The calibrated frame is then built directly from those two vectors:

    z-axis = T0b (prone) up          -- the exact zero pitch/roll reference
    x-axis = sagittal component of T0a up, negated (nose-up -> +90 deg pitch)
    y-axis = z x x                   -- right-handed

The prone pose defines ``+z`` *exactly*; it is the zero reference and must
not be compromised. The upright pose is a pure pitch from prone, so its up
vector lies in the sagittal (x-z) plane and its component orthogonal to z
fixes the azimuth (the x axis). A least-squares two-vector fit (Wahba) is
deliberately avoided here: T0a and T0b are ~86 deg apart (T0b sits at the
swimmer's habitual head pitch, not a right angle from upright), so an
equal-weight fit would split that ~4 deg mismatch and rotate the zero
reference off T0b -- which, decoded through the tilt equations, injects a
small roll-dependent error into pitch at large roll. Anchoring z to T0b
removes that cross-talk, so a perfect (noise-free) trial round-trips to the
gravity-referenced ground truth exactly.

The one quantity gravity genuinely *cannot* observe is rotation about the
gravity axis at a single pose -- but that null direction differs between the
two poses, so combining them removes the ambiguity for the study-relevant
pitch/roll. (The raw intrinsic-xyz mount Euler triple is not compared
directly; pitch/roll on a trial are, which is what the study measures.)

Pitch/roll are then read from the calibrated up vector with the standard
tilt equations, which -- unlike a naive gravity-axis construction -- do not
cross-talk pitch into roll::

    pitch = atan2(-g_x, hypot(g_y, g_z))
    roll  = atan2(g_y, g_z)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.spatial.transform import Rotation

__all__ = [
    "CALIB_POSE_SUSPECT",
    "canonical_frame",
    "fit_transform",
    "apply",
    "pose_check",
    "pose_angle_deg",
]

CALIB_POSE_SUSPECT = "CALIB_POSE_SUSPECT"

# Global "up" unit vector. The canonical quaternion is sensor->global with a
# gravity-down / z-up global frame, so the up direction in the sensor frame is
# ``R_quat.inv() @ [0, 0, 1]``.
_UP_GLOBAL = np.array([0.0, 0.0, 1.0])

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _up_sensor(df: pl.DataFrame) -> np.ndarray:
    """Per-sample gravity ("up") unit vectors in the sensor frame.

    Derived from the fused quaternion (``quat_*``, sensor->global): the up
    direction is yaw-independent and, unlike the raw accelerometer, is not
    contaminated by linear acceleration during a swim. Returns an
    ``(n, 3)`` array of unit vectors.
    """
    quat_xyzw = df.select(["quat_x", "quat_y", "quat_z", "quat_w"]).to_numpy()
    up = Rotation.from_quat(quat_xyzw).inv().apply(_UP_GLOBAL)
    return up / np.linalg.norm(up, axis=1, keepdims=True)


def _mean_up_sensor(df: pl.DataFrame) -> np.ndarray:
    """Mean sensor-frame up unit vector over a (near-static) pose segment."""
    mean = _up_sensor(df).mean(axis=0)
    return mean / np.linalg.norm(mean)


def canonical_frame(up_upright: np.ndarray, up_prone: np.ndarray) -> Rotation:
    """Build the calibrated-skull frame from two static-pose up vectors.

    ``up_prone`` (T0b) defines the calibrated ``+z`` axis exactly -- the zero
    pitch/roll reference. ``up_upright`` (T0a), a pure pitch from prone, lies in
    the sagittal plane; its component orthogonal to z fixes the ``x`` axis, with
    nose-up mapping toward ``-x`` (i.e. positive pitch). ``y = z x x`` completes
    a right-handed frame. Returned rotation maps a sensor-frame up vector into
    the calibrated skull frame.

    This is the single source of truth for the gravity-referenced frame; the
    synthetic generator computes its ground truth with the identical definition
    (guarded by a round-trip consistency test) so a clean trial recovers the
    ground-truth pitch/roll exactly.
    """
    z = up_prone / np.linalg.norm(up_prone)
    x_dir = up_upright - np.dot(up_upright, z) * z
    x = -x_dir / np.linalg.norm(x_dir)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.vstack([x, y, z]))


def fit_transform(t0a_df: pl.DataFrame, t0b_df: pl.DataFrame) -> Rotation:
    """Recover the sensor-to-calibrated-skull rotation from the two poses.

    Uses gravity only, via :func:`canonical_frame` on the mean sensor-frame up
    vectors of T0a (upright) and T0b (face-down prone). The prone pose anchors
    the zero reference exactly; see the module docstring for why this is
    preferred to an equal-weight two-vector Wahba fit.

    Parameters
    ----------
    t0a_df, t0b_df:
        Canonical dataframes for the T0a (upright, gaze horizontal) and T0b
        (face-down float) static calibration poses.

    Returns
    -------
    scipy.spatial.transform.Rotation
        ``R`` such that ``R.apply(up_sensor)`` gives the up vector in the
        calibrated skull frame.
    """
    return canonical_frame(_mean_up_sensor(t0a_df), _mean_up_sensor(t0b_df))


def apply(df: pl.DataFrame, R: Rotation) -> pl.DataFrame:
    """Add gravity-referenced ``pitch_deg`` and ``roll_deg`` to a trial frame.

    Parameters
    ----------
    df:
        Canonical dataframe (must contain ``quat_*``).
    R:
        Sensor-to-skull rotation from :func:`fit_transform`.

    Returns
    -------
    polars.DataFrame
        Copy of ``df`` with two columns appended:

        * ``pitch_deg`` -- head pitch, degrees, positive = nose up.
        * ``roll_deg``  -- head roll, degrees, positive = face to swimmer's
          right.

        Both are zero at the T0b prone reference pose.
    """
    up_skull = R.apply(_up_sensor(df))
    g_x, g_y, g_z = up_skull[:, 0], up_skull[:, 1], up_skull[:, 2]
    pitch = np.degrees(np.arctan2(-g_x, np.hypot(g_y, g_z)))
    roll = np.degrees(np.arctan2(g_y, g_z))
    return df.with_columns(
        pl.Series("pitch_deg", pitch),
        pl.Series("roll_deg", roll),
    )


def pose_angle_deg(t0a_df: pl.DataFrame, t0b_df: pl.DataFrame) -> float:
    """Angle (degrees) between the T0a and T0b mean gravity directions.

    Mount-invariant: both pose up vectors are rotated by the same fixed mount,
    so the angle between them is preserved and equals the true head-pitch
    change between upright and prone (~90 deg for a valid calibration).
    """
    ua = _mean_up_sensor(t0a_df)
    ub = _mean_up_sensor(t0b_df)
    return float(np.degrees(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0))))


def _load_pose_thresholds(config_path: str | Path | None) -> tuple[float, float]:
    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    return float(cfg["calib_pose_angle_deg"]), float(cfg["calib_pose_tolerance_deg"])


def pose_check(
    t0a_df: pl.DataFrame,
    t0b_df: pl.DataFrame,
    *,
    expected_angle_deg: float | None = None,
    tolerance_deg: float | None = None,
    config_path: str | Path | None = None,
) -> str | None:
    """Sanity-check the two calibration poses; flag rather than raise.

    A valid calibration has T0a (upright) and T0b (prone) differing in head
    pitch by ``calib_pose_angle_deg`` +/- ``calib_pose_tolerance_deg`` (from
    ``config.yaml``, per hard constraint 4 -- no hardcoded thresholds). If the
    measured :func:`pose_angle_deg` falls outside that band one pose was
    performed incorrectly.

    Returns
    -------
    str | None
        ``CALIB_POSE_SUSPECT`` when the pose angle is out of band, else
        ``None``. The caller accumulates the returned flag into the trial's
        ``flags`` column (CLAUDE.md: quality flags accumulate, never raise on
        merely-suspect data).
    """
    if expected_angle_deg is None or tolerance_deg is None:
        cfg_angle, cfg_tol = _load_pose_thresholds(config_path)
        expected_angle_deg = cfg_angle if expected_angle_deg is None else expected_angle_deg
        tolerance_deg = cfg_tol if tolerance_deg is None else tolerance_deg

    measured = pose_angle_deg(t0a_df, t0b_df)
    if abs(measured - expected_angle_deg) > tolerance_deg:
        return CALIB_POSE_SUSPECT
    return None
