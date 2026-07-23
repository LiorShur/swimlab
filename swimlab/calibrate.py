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
quaternion, which is robust to linear acceleration), average it over the
pose, and solve Wahba's two-vector problem for the rotation ``R``
(sensor -> skull) that maps:

    T0b up  ->  [0, 0, 1]   (prone zero reference, z-up)
    T0a up  ->  [-1, 0, 0]  (upright, +90 deg pitch, nose-up direction)

Two non-parallel gravity vectors ~90 deg apart fully constrain a 3-DOF
rotation, so ``R`` is identifiable up to the residual pose noise. The one
quantity gravity genuinely *cannot* observe is rotation about the gravity
axis at a single pose -- but that null direction differs between the two
poses, so combining them removes the ambiguity for the study-relevant
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

# Skull-frame targets for the two calibration poses.
_T0B_TARGET = np.array([0.0, 0.0, 1.0])   # prone: up is +z (zero reference)
_T0A_TARGET = np.array([-1.0, 0.0, 0.0])  # upright: +90 deg pitch (nose up)

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


def fit_transform(t0a_df: pl.DataFrame, t0b_df: pl.DataFrame) -> Rotation:
    """Recover the sensor-to-calibrated-skull rotation from the two poses.

    Uses gravity only. Solves the two-vector Wahba problem aligning the mean
    sensor-frame up vectors of T0a (upright) and T0b (face-down prone) to
    their skull-frame targets ``[-1, 0, 0]`` and ``[0, 0, 1]`` respectively.

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
    src = np.vstack([_mean_up_sensor(t0a_df), _mean_up_sensor(t0b_df)])
    tgt = np.vstack([_T0A_TARGET, _T0B_TARGET])
    rotation, _rmsd = Rotation.align_vectors(tgt, src)
    return rotation


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
