"""Synthetic head-IMU trace generator with ground truth (swimlab task 1).

Why this module exists
----------------------
The Movella DOT has not shipped, and real pool data will *never* carry ground
truth -- you will only ever know what the sensor and a coach report, never a
swimmer's true head pitch. This generator injects a known motion, derives
physically consistent sensor channels from it, and emits the ground truth
alongside. It is the only dataset in the project where "correct" is defined,
and it is what every downstream module (`calibrate`, `events`, `metrics`) is
tested against.

Generation approach (see SYNTHETIC_DATA_SPEC.md section 2)
---------------------------------------------------------
Orientation is generated *first*, then the accelerometer, gyroscope and
quaternion are derived from that single rotation trajectory. Synthesising
accel and gyro independently would produce physically inconsistent data that
lets broken pipeline code pass its tests.

Coordinate conventions (see CLAUDE.md)
--------------------------------------
The *calibrated* skull frame used to define the injected angles is:

    x = swim-forward (top-of-head direction),  y = swimmer's left,  z = up.

* ``pitch`` -- rotation about the medio-lateral (y) axis. Positive = nose up
  (head lift). Zero is the T0b face-down float pose.
* ``roll``  -- rotation about the longitudinal (x) axis. Positive = face
  rotating to the swimmer's right.
* ``yaw``   -- heading about the *global vertical* axis. Deliberately
  corrupted (magnetic disturbance). Never used downstream; gravity-referenced
  pitch/roll are analytically independent of it.

Because yaw is a rotation about global vertical, the gravity vector in the
sensor frame -- and therefore any gravity-referenced pitch/roll -- does not
depend on it. This module computes the accelerometer gravity component from a
yaw-free orientation so that independence is exact, while the gyroscope and
quaternion carry the yaw corruption (as a real fusion output would).

Deliberate limitations (state these so nobody mistakes synthetic performance
for validated performance, per spec section 9)
----------------------------------------------------------------------------
* Real head motion is not sinusoidal. Real swimmers vary stroke period, pause,
  sight the wall and adjust goggles mid-length. This model does none of that.
* The generator encodes the pipeline author's assumptions about what lifting
  and rotating look like. Those assumptions are the hypothesis under test --
  the generator cannot validate them.
* Passing every acceptance test here means the pipeline is internally correct.
  It says nothing about whether head pitch is a useful diagnostic gate. Only
  the pool answers that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------------- #
# Fixed physical / study constants
# --------------------------------------------------------------------------- #

SAMPLE_RATE_HZ: float = 120.0  # onboard DOT logging rate; fixed (spec section 1)
G: float = 9.81  # m/s^2, gravitational acceleration magnitude

#: Canonical dataframe columns, in order (see CLAUDE.md).
CANONICAL_COLUMNS: tuple[str, ...] = (
    "t",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "mag_x",
    "mag_y",
    "mag_z",
)

#: The frame in which the injected ``pitch``/``roll`` are expressed is the
#: *calibrated skull frame*: at the T0b prone reference the sensor z-axis points
#: up (an occiput mount faces the surface when the swimmer is face down), so the
#: reference orientation is the identity and gravity reads ``[0, 0, +g]``. Pitch
#: (about y) and roll (about x) then both tilt gravity and are recovered exactly
#: by the standard tilt equations, independent of the heading (yaw about global
#: vertical). The 90 deg relationship between the upright T0a pose and the prone
#: T0b pose that the calibration sanity check verifies is represented as a 90 deg
#: pitch offset applied to the T0a segment (see :func:`generate_calibration`).
_UPRIGHT_PITCH_DEG = 90.0

# --------------------------------------------------------------------------- #
# Archetypes (spec section 3). Ranges are sampled per generated swimmer.
# --------------------------------------------------------------------------- #

#: Per-archetype parameter ranges. ``bump`` is the breath pitch bump amplitude
#: in degrees; ``roll`` is the breath-side roll amplitude in degrees. Values are
#: (low, high) drawn uniformly per swimmer. ``LR`` archetypes carry explicit
#: per-side values to exercise ``asymmetry_index``.
ARCHETYPES: dict[str, dict[str, Any]] = {
    # correct technique: rotates to breathe, little pitch change
    "ROTATOR": {"bump": (2.0, 6.0), "roll": (65.0, 80.0)},
    # lifts the head instead of rotating
    "LIFTER": {"bump": (15.0, 28.0), "roll": (35.0, 50.0)},
    # inconsistent -- the hard case. High cycle-to-cycle SD (see _CYCLE_JITTER).
    "MIXED": {"bump": (8.0, 14.0), "roll": (45.0, 65.0), "cycle_sd_frac": 0.45},
    # barely rolls; lifts by default
    "FLAT": {"bump": (10.0, 18.0), "roll": (20.0, 30.0)},
    # left/right asymmetry -- tests asymmetry_index
    "ASYMMETRIC": {
        "bump_left": 4.0,
        "bump_right": 20.0,
        "roll_left": 70.0,
        "roll_right": 40.0,
    },
}

#: Default per-cycle jitter as a fraction of the mean bump (spec: SD ~= 15%).
_CYCLE_JITTER: float = 0.15

# Timing model (seconds). Not study thresholds -- these shape the synthetic
# swimmer and never enter the analysis pipeline.
_GLIDE_MIN_S = 1.5
_GLIDE_MAX_S = 3.0
# Nominal wall-to-wall duration of one 25 m length. Set for a *recreational*
# swimmer (~30 s / 25 m, i.e. ~1:00 / 100 m) -- the study population. At the
# default 1.4 s stroke period this is ~21 strokes/length, so a 4-length T7
# breathing every 3 yields ~24 breaths / ~22 valid after exclusions, clearing
# the min_valid_cycles = 20 gate as a real recreational T7 does. (An earlier
# 21 s value was a fit/competitive pace and under-produced breaths, tripping
# INSUFFICIENT_CYCLES on every valid T7.)
_TARGET_LENGTH_S = 30.0
_PUSHOFF_DUR_S = 0.20
_PUSHOFF_G_LOW = 2.0
_PUSHOFF_G_HIGH = 4.0
_ROLL_LOBE_HALFWIDTH_FRAC = 0.62  # raised-cosine half-width as a fraction of period
_BREATH_RAMP_S = 0.15  # raised-cosine ramp in/out of the breath roll enhancement

# Movella DOT noise figures (spec section 5).
_GYRO_NOISE_DENSITY = 0.007  # deg/s/sqrt(Hz)
_GYRO_BIAS_DEG_PER_H = 10.0  # deg/h
_ACC_NOISE_DENSITY = 120e-6  # g/sqrt(Hz)
_ACC_BIAS_MG = 0.03  # mg
_QUAT_ERROR_DEG = 0.5  # injected fusion inconsistency (1-sigma slow orientation error)


# --------------------------------------------------------------------------- #
# Configuration (thresholds live in config.yaml, never in code -- CLAUDE.md #4)
# --------------------------------------------------------------------------- #


def _load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load study thresholds from ``config.yaml``.

    Only the values that define the breath window and the protocol exclusions
    are read here, so that the ground truth this module emits stays consistent
    with the thresholds the pipeline will use downstream.
    """
    if path is None:
        # repo_root/config.yaml, with this file at repo_root/swimlab/synth.py
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Deterministic, channel-independent RNG streams
# --------------------------------------------------------------------------- #

#: Named RNG streams. Each channel draws from its own generator so that, for a
#: fixed seed, the accelerometer noise is byte-identical regardless of what the
#: yaw channel does -- which is what makes the yaw-independence property exact
#: even with noise on (spec acceptance test 6).
_STREAMS = (
    "swimmer",  # archetype parameter sampling
    "timing",  # stroke/glide jitter
    "cycle",  # per-cycle bump/roll jitter
    "yaw",  # heading random walk + jumps
    "acc_noise",
    "gyro_noise",
    "gyro_bias",
    "quat_error",
    "mag",
)


def _rng_streams(seed: int | None) -> dict[str, np.random.Generator]:
    """Return an independent :class:`numpy.random.Generator` per named channel.

    Uses :class:`numpy.random.SeedSequence` spawning so every stream is
    reproducible and statistically independent for a given ``seed``.
    """
    root = np.random.SeedSequence(seed)
    children = root.spawn(len(_STREAMS))
    return {name: np.random.default_rng(child) for name, child in zip(_STREAMS, children)}


# --------------------------------------------------------------------------- #
# Small maths helpers
# --------------------------------------------------------------------------- #


def _raised_cosine(t: np.ndarray, center: float, halfwidth: float) -> np.ndarray:
    """A raised-cosine lobe: 1 at ``center``, smoothly to 0 at +/- ``halfwidth``.

    Zero outside the support. Using a raised cosine (never a step) avoids the
    discontinuities that would create artificial gyro spikes the event detector
    could latch onto (spec section 3).
    """
    x = (t - center) / halfwidth
    out = np.zeros_like(t)
    inside = np.abs(x) < 1.0
    out[inside] = 0.5 * (1.0 + np.cos(np.pi * x[inside]))
    return out


def _tilt_pitch_roll(grav_sensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gravity-referenced pitch and roll (degrees) from a sensor-frame gravity
    (specific-force-at-rest) vector, using the standard accelerometer tilt
    equations for a z-up frame. Provided so tests and downstream code share one
    definition. ``grav_sensor`` may be a single (3,) vector or an (N, 3) array.
    """
    g = np.atleast_2d(grav_sensor)
    ux, uy, uz = g[:, 0], g[:, 1], g[:, 2]
    pitch = np.degrees(np.arctan2(-ux, np.hypot(uy, uz)))
    roll = np.degrees(np.arctan2(uy, uz))
    if np.ndim(grav_sensor) == 1:
        return float(pitch[0]), float(roll[0])
    return pitch, roll


def _canonical_frame(up_upright: np.ndarray, up_prone: np.ndarray) -> Rotation:
    """Calibrated-skull frame from two static-pose "up" vectors.

    Identical construction to :func:`swimlab.calibrate.canonical_frame` -- kept
    in lock-step by the round-trip consistency test in the metrics suite -- so
    that the ground truth this module emits is exactly what a correctly
    calibrated pipeline measures. ``up_prone`` (T0b) defines +z exactly (the
    zero reference); ``up_upright`` (T0a) fixes the azimuth (nose-up -> -x).
    """
    z = up_prone / np.linalg.norm(up_prone)
    x_dir = up_upright - np.dot(up_upright, z) * z
    x = -x_dir / np.linalg.norm(x_dir)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.vstack([x, y, z]))


def _gravity_referenced_angles(
    pitch_deg: np.ndarray, roll_deg: np.ndarray, pitch_baseline_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Injected intrinsic pitch/roll -> gravity-referenced pitch/roll (degrees).

    The study measures *gravity-referenced* angles relative to the T0b prone
    pose (CLAUDE.md: "gravity-referenced pitch and roll only"), not the intrinsic
    Euler angles used to synthesise the motion. The two diverge by a
    roll-dependent term at large roll, so the emitted ground truth must be
    expressed the way the pipeline measures it -- otherwise no gravity-referenced
    pipeline could ever recover it (acceptance test 1). This decodes the clean
    skull-frame gravity vector through the canonical calibrated frame whose prone
    reference sits at ``pitch_baseline_deg`` (T0b) and whose upright pose (T0a)
    is a 90 deg pitch, exactly as :mod:`swimlab.calibrate` reconstructs it.
    """
    p = np.radians(np.asarray(pitch_deg, dtype=np.float64))
    r = np.radians(np.asarray(roll_deg, dtype=np.float64))
    # Exact clean gravity "up" in the skull frame for R = Ry(pitch) . Rx(roll).
    u_skull = np.column_stack(
        [-np.sin(p), np.cos(p) * np.sin(r), np.cos(p) * np.cos(r)]
    )
    base = np.radians(pitch_baseline_deg)
    up_prone = np.array([-np.sin(base), 0.0, np.cos(base)])  # T0b: pitch=baseline, roll=0
    up_upright = np.array([-1.0, 0.0, 0.0])  # T0a: pitch=90, roll=0
    calibrated = _canonical_frame(up_upright, up_prone).apply(u_skull)
    return _tilt_pitch_roll(calibrated)


# --------------------------------------------------------------------------- #
# Archetype sampling
# --------------------------------------------------------------------------- #


def _sample_swimmer(archetype: str, rng: np.random.Generator) -> dict[str, Any]:
    """Draw a concrete swimmer's parameters from an archetype's ranges."""
    if archetype not in ARCHETYPES:
        raise ValueError(
            f"unknown archetype {archetype!r}; expected one of {sorted(ARCHETYPES)}"
        )
    spec = ARCHETYPES[archetype]
    pitch_baseline = float(rng.uniform(-5.0, 12.0))  # habitual look-forward, spec 3

    if archetype == "ASYMMETRIC":
        return {
            "archetype": archetype,
            "pitch_baseline": pitch_baseline,
            "bump_left": spec["bump_left"],
            "bump_right": spec["bump_right"],
            "roll_left": spec["roll_left"],
            "roll_right": spec["roll_right"],
            "cycle_sd_frac": _CYCLE_JITTER,
        }

    bump = float(rng.uniform(*spec["bump"]))
    roll = float(rng.uniform(*spec["roll"]))
    return {
        "archetype": archetype,
        "pitch_baseline": pitch_baseline,
        "bump_left": bump,
        "bump_right": bump,
        "roll_left": roll,
        "roll_right": roll,
        "cycle_sd_frac": float(spec.get("cycle_sd_frac", _CYCLE_JITTER)),
    }


# --------------------------------------------------------------------------- #
# Timeline: strokes, breaths, push-offs
# --------------------------------------------------------------------------- #


def _build_timeline(
    n_lengths: int,
    stroke_period_s: float,
    breathe_every_n_strokes: int,
    swimmer: dict[str, Any],
    rng: np.random.Generator,
    cyc_rng: np.random.Generator,
) -> dict[str, Any]:
    """Lay out push-offs, glides, strokes and breaths across the whole trial.

    Returns a dict with ``pushoffs`` (length-start times), ``strokes`` (list of
    per-stroke dicts with ``t``, ``side``), ``breaths`` (per-breath dicts with
    injected amplitudes) and ``duration`` (total trial seconds). Stroke indexing
    is *global* across lengths so "breathe every n strokes" is continuous, and
    the breathing side alternates for odd n (bilateral) or is fixed for even n
    (unilateral) -- exactly the distinction that drives ``asymmetry_index``.
    """
    pushoffs: list[float] = []
    strokes: list[dict[str, Any]] = []
    breaths: list[dict[str, Any]] = []

    t = 0.0
    global_stroke = 0
    breath_count = 0

    for _ in range(n_lengths):
        pushoffs.append(t)
        glide = float(rng.uniform(_GLIDE_MIN_S, _GLIDE_MAX_S))
        first_stroke_t = t + _PUSHOFF_DUR_S + glide

        # number of strokes to roughly fill the nominal length duration
        stroke_span = max(_TARGET_LENGTH_S - _PUSHOFF_DUR_S - glide, 4 * stroke_period_s)
        n_strokes = max(int(round(stroke_span / stroke_period_s)), 4)

        cur = first_stroke_t
        for _k in range(n_strokes):
            jitter = float(rng.normal(0.0, 0.03 * stroke_period_s))
            t_k = cur + jitter
            # A single notion of side, by global stroke parity, drives both the
            # body-roll lobe sign and the breathing side. With breathe_every=3
            # this makes breaths land on strokes 0,3,6,... -> parity R,L,R,L
            # (bilateral, alternating). With breathe_every=2 they land on
            # 0,2,4,... -> always the same parity (unilateral). Exactly the
            # spec's distinction, and it guarantees a breath enhances -- never
            # cancels -- its own roll lobe.
            side = "R" if (global_stroke % 2 == 0) else "L"
            is_breath = (global_stroke % breathe_every_n_strokes) == 0

            stroke = {"t": t_k, "side": side, "index": global_stroke}
            strokes.append(stroke)

            if is_breath:
                b_side = side
                bump_mean = swimmer[f"bump_{'left' if b_side == 'L' else 'right'}"]
                roll_mean = swimmer[f"roll_{'left' if b_side == 'L' else 'right'}"]
                jitter_frac = float(cyc_rng.normal(0.0, swimmer["cycle_sd_frac"]))
                bump = max(bump_mean * (1.0 + jitter_frac), 0.0)
                roll_amp = max(roll_mean * (1.0 + 0.10 * float(cyc_rng.normal())), 5.0)
                breaths.append(
                    {
                        "index": breath_count,
                        "side": b_side,
                        "center": t_k,
                        "bump": bump,
                        "roll_amp": roll_amp,
                        "pushoff_t": pushoffs[-1],
                    }
                )
                breath_count += 1

            cur = t_k + stroke_period_s
            global_stroke += 1

        # brief pause at the wall before the next push-off
        t = cur + float(rng.uniform(0.3, 0.8))

    duration = t
    return {
        "pushoffs": pushoffs,
        "strokes": strokes,
        "breaths": breaths,
        "duration": duration,
        "stroke_period_s": stroke_period_s,
    }


# --------------------------------------------------------------------------- #
# Continuous angle signals
# --------------------------------------------------------------------------- #


def _angle_signals(
    t: np.ndarray,
    timeline: dict[str, Any],
    swimmer: dict[str, Any],
    stroke_period_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``pitch(t)`` and ``roll(t)`` in the calibrated skull frame (degrees).

    Roll is a sum of alternating raised-cosine lobes (one per stroke), enhanced
    on breathing strokes. Pitch is a baseline plus a small stroke-rate
    oscillation plus a raised-cosine bump on each breath. Signals are zero-roll
    and baseline-pitch during glides because no strokes fall there.
    """
    half = _ROLL_LOBE_HALFWIDTH_FRAC * stroke_period_s
    baseline_roll_amp = 18.0  # deg, always-present body roll between breaths

    roll = np.zeros_like(t)
    for stroke in timeline["strokes"]:
        sign = 1.0 if stroke["side"] == "R" else -1.0
        roll += sign * baseline_roll_amp * _raised_cosine(t, stroke["t"], half)

    # Breath roll enhancement: ramp the breathing-side lobe up to its amplitude.
    for breath in timeline["breaths"]:
        sign = 1.0 if breath["side"] == "R" else -1.0
        extra = max(breath["roll_amp"] - baseline_roll_amp, 0.0)
        roll += sign * extra * _raised_cosine(t, breath["center"], half)

    # Pitch: baseline + small oscillation at stroke frequency + breath bumps.
    pitch = np.full_like(t, swimmer["pitch_baseline"])
    pitch += 2.5 * np.sin(2.0 * np.pi * t / (2.0 * stroke_period_s))
    for breath in timeline["breaths"]:
        pitch += breath["bump"] * _raised_cosine(t, breath["center"], half)

    return pitch, roll


def _yaw_signal(
    t: np.ndarray, rng: np.random.Generator, corruption_scale: float
) -> np.ndarray:
    """A deliberately corrupted heading signal (degrees).

    A slow plausible drift plus a random walk of several deg/s plus occasional
    10-30 deg jumps, imitating magnetic disturbance in a pool hall. Any pipeline
    code that accidentally depends on yaw fails loudly -- that is the point.
    Scaled by ``corruption_scale`` so acceptance test 6 can double it.
    """
    n = t.size
    dt = 1.0 / SAMPLE_RATE_HZ
    # random walk: several deg/s -> per-sample step sd
    step_sd = 4.0 * np.sqrt(dt)  # ~4 deg/s
    walk = np.cumsum(rng.normal(0.0, step_sd, size=n))
    slow = 8.0 * np.sin(2.0 * np.pi * t / 30.0)  # slow plausible heading sway
    jumps = np.zeros(n)
    n_jumps = max(int(t[-1] / 12.0), 1)
    idx = rng.integers(0, n, size=n_jumps)
    mag = rng.uniform(10.0, 30.0, size=n_jumps) * rng.choice([-1.0, 1.0], size=n_jumps)
    for i, m in zip(idx, mag):
        jumps[i:] += m
    return corruption_scale * (slow + walk + jumps)


# --------------------------------------------------------------------------- #
# Orientation trajectory and sensor derivation
# --------------------------------------------------------------------------- #


def _mount_rotation(
    mount_offset_deg: tuple[float, float, float],
    mount_slip_deg_per_min: float,
    t: np.ndarray,
) -> Rotation:
    """Sensor-to-skull mount rotation, optionally drifting over the trial.

    ``mount_offset_deg`` are intrinsic xyz Euler angles of the fixed mount. When
    ``mount_slip_deg_per_min > 0`` the mount rotates progressively about a fixed
    axis -- the highest-likelihood risk in the register, and what lets the study
    compare how badly slip corrupts ``d_pitch_breath`` versus ``roll_pitch_ratio``.
    Returns a single Rotation when there is no slip, else an (N,) Rotation.
    """
    base = Rotation.from_euler("xyz", mount_offset_deg, degrees=True)
    if mount_slip_deg_per_min == 0.0:
        return base
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    slip_deg = mount_slip_deg_per_min * (t / 60.0)
    slip = Rotation.from_rotvec(np.radians(slip_deg)[:, None] * axis[None, :])
    return slip * base


def _orientation(
    pitch_deg: np.ndarray,
    roll_deg: np.ndarray,
    yaw_deg: np.ndarray,
    mount: Rotation,
    base: Rotation | None = None,
) -> tuple[Rotation, Rotation]:
    """Return (R_sensor_full, R_sensor_noyaw), both sensor->global.

    ``R_sensor_full`` carries the yaw corruption (used for gyro and quaternion);
    ``R_sensor_noyaw`` omits it so the accelerometer gravity component is exactly
    yaw-independent. Composition (spec section 2):

        R_skull->global = Rz_global(yaw) . base . Ry(pitch) . Rx(roll)
        R_sensor->global = R_skull->global . R_mount^-1

    ``base`` is the reference pose; it defaults to the identity, i.e. the
    calibrated z-up-at-prone frame in which pitch and roll are expressed. The
    only caller that overrides it is the upright T0a calibration segment, which
    supplies a 90 deg pitch offset.
    """
    if base is None:
        base = Rotation.identity()
    r_pr = Rotation.from_euler(
        "y", np.asarray(pitch_deg)[:, None], degrees=True
    ) * Rotation.from_euler("x", np.asarray(roll_deg)[:, None], degrees=True)
    r_body = base * r_pr  # skull->global, no heading
    r_skull_full = (
        Rotation.from_euler("z", np.asarray(yaw_deg)[:, None], degrees=True) * r_body
    )
    mount_inv = mount.inv()
    return r_skull_full * mount_inv, r_body * mount_inv


def _linear_accel_sensor(
    t: np.ndarray,
    timeline: dict[str, Any],
    stroke_period_s: float,
    mount: Rotation,
) -> np.ndarray:
    """Head linear acceleration expressed in the sensor frame (m/s^2).

    A small forward oscillation at stroke frequency plus a push-off transient of
    2-4 g over ~200 ms at each length boundary. Defined along the body-forward
    axis (skull +x) so it is, like gravity, independent of the global heading.
    """
    forward_skull = np.array([1.0, 0.0, 0.0])
    # skull->sensor is the mount rotation itself (sensor->skull is its inverse)
    if mount.single:
        forward_sensor = np.tile(mount.apply(forward_skull), (t.size, 1))
    else:
        forward_sensor = mount.apply(np.tile(forward_skull, (t.size, 1)))

    mag = 0.8 * np.sin(2.0 * np.pi * t / stroke_period_s)  # gentle stroke surge
    for p_t in timeline["pushoffs"]:
        peak_g = _PUSHOFF_G_LOW  # deterministic mid-range; magnitude jitter not needed
        pulse = _raised_cosine(t, p_t + _PUSHOFF_DUR_S / 2.0, _PUSHOFF_DUR_S / 2.0)
        mag = mag + peak_g * G * pulse * 1.5  # ~2-4 g at the peak
    return forward_sensor * mag[:, None]


def _derive_gyro(r_full: Rotation, dt: float) -> np.ndarray:
    """Body-frame angular velocity (deg/s) by differentiating the trajectory."""
    q = r_full.as_quat()  # (N, 4)
    n = q.shape[0]
    gyro = np.zeros((n, 3))
    rel = r_full[:-1].inv() * r_full[1:]  # incremental body-frame rotation
    gyro[:-1] = np.degrees(rel.as_rotvec()) / dt
    gyro[-1] = gyro[-2]
    return gyro


def _accel(r_noyaw: Rotation, lin_sensor: np.ndarray) -> np.ndarray:
    """Accelerometer specific force (m/s^2): gravity (yaw-free) + linear accel."""
    up_g = np.tile([0.0, 0.0, G], (lin_sensor.shape[0], 1))
    grav_sensor = r_noyaw.inv().apply(up_g)
    return grav_sensor + lin_sensor


def _emit_quaternion(
    r_full: Rotation, rng: np.random.Generator, inject_error: bool
) -> np.ndarray:
    """Fused sensor->global quaternion as (N,4) [w,x,y,z].

    A small, slowly varying orientation error is injected so the fused output is
    not perfectly consistent with the raw channels -- real DOT fusion never is.
    """
    n = len(r_full)
    if inject_error:
        walk = np.cumsum(rng.normal(0.0, 0.05, size=(n, 3)), axis=0)
        walk *= np.radians(_QUAT_ERROR_DEG) / max(np.std(walk) + 1e-9, 1e-9)
        r_err = Rotation.from_rotvec(walk)
        r_out = r_full * r_err
    else:
        r_out = r_full
    q_xyzw = r_out.as_quat()
    # canonicalise sign (w >= 0) for stable, deterministic output
    flip = q_xyzw[:, 3] < 0
    q_xyzw[flip] *= -1.0
    return np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]])


def _magnetometer(
    r_full: Rotation, rng: np.random.Generator, n: int, add_noise: bool
) -> np.ndarray:
    """Plausible but corrupted magnetometer (uT). Logged, never used (CLAUDE.md #2)."""
    field_global = np.array([22.0, 0.0, -42.0])  # a plausible indoor field
    mag = r_full.inv().apply(np.tile(field_global, (n, 1)))
    # pool-hall disturbance: slow drift + spikes; makes the channel unusable
    drift = np.cumsum(rng.normal(0.0, 0.4, size=(n, 3)), axis=0)
    mag = mag + drift
    if add_noise:
        mag = mag + rng.normal(0.0, 1.5, size=(n, 3))
    return mag


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #


def _add_sensor_noise(
    acc: np.ndarray,
    gyro: np.ndarray,
    streams: dict[str, np.random.Generator],
    gyro_bias_walk: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Add Movella DOT noise/bias to accelerometer and gyroscope (spec section 5)."""
    n = acc.shape[0]
    bw = SAMPLE_RATE_HZ / 2.0  # noise bandwidth (Nyquist)

    acc_sd = _ACC_NOISE_DENSITY * G * np.sqrt(bw)
    acc_bias = _ACC_BIAS_MG * 1e-3 * G
    acc = acc + streams["acc_noise"].normal(0.0, acc_sd, size=(n, 3))
    acc = acc + acc_bias * streams["acc_noise"].standard_normal(3)[None, :]

    gyro_sd = _GYRO_NOISE_DENSITY * np.sqrt(bw)
    gyro = gyro + streams["gyro_noise"].normal(0.0, gyro_sd, size=(n, 3))

    bias0 = (_GYRO_BIAS_DEG_PER_H / 3600.0) * streams["gyro_bias"].standard_normal(3)
    if gyro_bias_walk:
        dt = 1.0 / SAMPLE_RATE_HZ
        walk_sd = (_GYRO_BIAS_DEG_PER_H / 3600.0) * 0.5 * np.sqrt(dt)
        bias = bias0[None, :] + np.cumsum(
            streams["gyro_bias"].normal(0.0, walk_sd, size=(n, 3)), axis=0
        )
    else:
        bias = bias0[None, :]
    gyro = gyro + bias
    return acc, gyro


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def _breath_ground_truth(
    t: np.ndarray,
    pitch_clean: np.ndarray,
    roll_clean: np.ndarray,
    timeline: dict[str, Any],
    swimmer: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-breath ground truth plus a trial summary.

    ``true_d_pitch_deg`` and ``true_peak_roll_deg`` are computed from the *clean*
    (noise-free) angle signals using the same definitions the metrics module
    will use, so the round-trip test can compare like with like. Each breath is
    marked ``excluded_by_protocol`` if its window starts within
    ``exclude_after_pushoff_s`` of a push-off or within ``exclude_before_end_s``
    of trial end -- nothing is dropped silently (CLAUDE.md #5); every exclusion
    carries a reason code.
    """
    roll_thr = float(cfg["breath_roll_threshold_deg"])
    after_pushoff = float(cfg["exclude_after_pushoff_s"])
    before_end = float(cfg["exclude_before_end_s"])
    trial_end = float(t[-1])
    stroke_period = timeline["stroke_period_s"]

    records: list[dict[str, Any]] = []
    for breath in timeline["breaths"]:
        center = breath["center"]
        # window: contiguous span around the breath where |roll| >= threshold
        ci = int(round(center * SAMPLE_RATE_HZ))
        ci = min(max(ci, 0), t.size - 1)
        over = np.abs(roll_clean) >= roll_thr
        if not over[ci]:
            # find nearest sample over threshold within half a stroke
            span = int(round(0.5 * stroke_period * SAMPLE_RATE_HZ))
            lo, hi = max(ci - span, 0), min(ci + span, t.size - 1)
            local = np.where(over[lo:hi + 1])[0]
            if local.size:
                ci = lo + local[np.argmin(np.abs(local + lo - ci))]
            else:
                ci = ci  # degenerate (tiny roll); window falls back to a point
        # walk out to the crossings
        i0 = ci
        while i0 > 0 and over[i0 - 1]:
            i0 -= 1
        i1 = ci
        while i1 < t.size - 1 and over[i1 + 1]:
            i1 += 1
        t_start, t_end = float(t[i0]), float(t[i1])

        win = slice(i0, i1 + 1)
        peak_roll = float(np.max(np.abs(roll_clean[win]))) if i1 >= i0 else 0.0
        # d_pitch: peak pitch in window minus median pitch over the preceding
        # (non-breath) cycle span -- the metrics definition applied to clean data
        pre_lo = max(int(round((t_start - 2.0 * stroke_period) * SAMPLE_RATE_HZ)), 0)
        pre_hi = max(i0, pre_lo + 1)
        pre_median = float(np.median(pitch_clean[pre_lo:pre_hi]))
        peak_pitch = float(np.max(pitch_clean[win])) if i1 >= i0 else float(pitch_clean[ci])
        d_pitch = peak_pitch - pre_median

        excluded = False
        reason: str | None = None
        if (t_start - breath["pushoff_t"]) < after_pushoff:
            excluded, reason = True, "within_4s_of_pushoff"
        elif (trial_end - t_end) < before_end:
            excluded, reason = True, "within_2s_of_end"

        records.append(
            {
                "index": breath["index"],
                "side": breath["side"],
                "t_start": round(t_start, 4),
                "t_end": round(t_end, 4),
                "true_d_pitch_deg": round(d_pitch, 4),
                "true_peak_roll_deg": round(peak_roll, 4),
                "excluded_by_protocol": excluded,
                "exclusion_reason": reason,
            }
        )

    valid = [r for r in records if not r["excluded_by_protocol"]]
    mean_d = (
        float(np.mean([r["true_d_pitch_deg"] for r in valid])) if valid else 0.0
    )
    summary = {
        "true_mean_d_pitch_deg": round(mean_d, 4),
        "n_breaths": len(records),
        "n_valid_breaths": len(valid),
    }
    return records, summary


# --------------------------------------------------------------------------- #
# Assembly helper
# --------------------------------------------------------------------------- #


def _make_dataframe(
    t: np.ndarray,
    quat: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    mag: np.ndarray,
) -> pl.DataFrame:
    """Assemble the canonical dataframe (schema and column order per CLAUDE.md)."""
    return pl.DataFrame(
        {
            "t": t.astype(np.float64),
            "quat_w": quat[:, 0],
            "quat_x": quat[:, 1],
            "quat_y": quat[:, 2],
            "quat_z": quat[:, 3],
            "acc_x": acc[:, 0],
            "acc_y": acc[:, 1],
            "acc_z": acc[:, 2],
            "gyr_x": gyro[:, 0],
            "gyr_y": gyro[:, 1],
            "gyr_z": gyro[:, 2],
            "mag_x": mag[:, 0],
            "mag_y": mag[:, 1],
            "mag_z": mag[:, 2],
        }
    ).select(CANONICAL_COLUMNS)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def generate_trial(
    archetype: str,
    n_lengths: int = 4,
    stroke_period_s: float = 1.4,
    breathe_every_n_strokes: int = 3,
    mount_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mount_slip_deg_per_min: float = 0.0,
    noise: bool = True,
    gyro_bias_walk: bool = True,
    seed: int | None = None,
    yaw_corruption_scale: float = 1.0,
    pitch_baseline_deg: float | None = None,
    config_path: str | Path | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Generate one synthetic front-crawl trial (T7) with ground truth.

    Parameters
    ----------
    archetype:
        One of ``ROTATOR``, ``LIFTER``, ``MIXED``, ``FLAT``, ``ASYMMETRIC``.
    n_lengths:
        Number of 25 m lengths (default 4, i.e. the T7 protocol).
    stroke_period_s:
        Seconds per stroke (one arm pull). One cycle is two strokes.
    breathe_every_n_strokes:
        Breathe every n strokes. Odd n breathes bilaterally (alternating sides);
        even n breathes unilaterally (same side every cycle).
    mount_offset_deg:
        Intrinsic xyz Euler angles of the fixed sensor-to-skull mount. This is
        the transform ``calibrate.py`` must recover.
    mount_slip_deg_per_min:
        If > 0, the mount rotates progressively over the trial (models slip).
    noise:
        Add Movella DOT sensor noise/bias when True.
    gyro_bias_walk:
        Add a slow random-walk gyro bias when True.
    seed:
        Seed for fully reproducible output (acceptance test 8).
    yaw_corruption_scale:
        Multiplies the injected yaw corruption. A test hook for acceptance test
        6 (yaw independence); the accelerometer is unaffected by construction.
    pitch_baseline_deg:
        Override the swimmer's habitual head pitch (degrees). When ``None`` it
        is drawn from ``seed`` per archetype. Because the ground truth is now
        gravity-referenced relative to the T0b prone pose, a trial and the
        calibration it is decoded against must share the same baseline (one
        swimmer, one prone pose); pin both here to pair them.
    config_path:
        Path to ``config.yaml`` (defaults to the repo root copy).

    Returns
    -------
    (dataframe, ground_truth)
        ``dataframe`` is the canonical schema; ``ground_truth`` is the dict
        documented in SYNTHETIC_DATA_SPEC.md section 7.
    """
    cfg = _load_config(config_path)
    streams = _rng_streams(seed)
    swimmer = _sample_swimmer(archetype, streams["swimmer"])
    if pitch_baseline_deg is not None:
        swimmer["pitch_baseline"] = float(pitch_baseline_deg)

    timeline = _build_timeline(
        n_lengths,
        stroke_period_s,
        breathe_every_n_strokes,
        swimmer,
        streams["timing"],
        streams["cycle"],
    )

    dt = 1.0 / SAMPLE_RATE_HZ
    n = int(round(timeline["duration"] * SAMPLE_RATE_HZ)) + 1
    t = np.arange(n) * dt

    pitch, roll = _angle_signals(t, timeline, swimmer, stroke_period_s)
    yaw = _yaw_signal(t, streams["yaw"], yaw_corruption_scale)

    mount = _mount_rotation(mount_offset_deg, mount_slip_deg_per_min, t)
    r_full, r_noyaw = _orientation(pitch, roll, yaw, mount)

    lin_sensor = _linear_accel_sensor(t, timeline, stroke_period_s, mount)
    acc = _accel(r_noyaw, lin_sensor)
    gyro = _derive_gyro(r_full, dt)
    if noise:
        acc, gyro = _add_sensor_noise(acc, gyro, streams, gyro_bias_walk)

    quat = _emit_quaternion(r_full, streams["quat_error"], inject_error=noise)
    mag = _magnetometer(r_full, streams["mag"], n, add_noise=noise)

    df = _make_dataframe(t, quat, acc, gyro, mag)

    # Ground truth is expressed in gravity-referenced angles (relative to the
    # T0b prone pose) -- the quantity the pipeline measures -- not the intrinsic
    # Euler angles used to drive the orientation. These diverge at large roll.
    pitch_grav, roll_grav = _gravity_referenced_angles(
        pitch, roll, swimmer["pitch_baseline"]
    )
    breaths, summary = _breath_ground_truth(
        t, pitch_grav, roll_grav, timeline, swimmer, cfg
    )
    ground_truth: dict[str, Any] = {
        "archetype": archetype,
        "seed": seed,
        "n_lengths": n_lengths,
        "mount_offset_deg": list(mount_offset_deg),
        "mount_slip_deg_per_min": mount_slip_deg_per_min,
        "pitch_baseline_deg": round(swimmer["pitch_baseline"], 4),
        "stroke_period_s": stroke_period_s,
        "breathe_every_n_strokes": breathe_every_n_strokes,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "pushoffs": [{"t": round(p, 4)} for p in timeline["pushoffs"]],
        "breaths": breaths,
        "summary": summary,
    }
    return df, ground_truth


def generate_calibration(
    mount_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pitch_baseline_deg: float = 0.0,
    bad_calibration: bool = False,
    noise: bool = True,
    seed: int | None = None,
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    """Generate the T0a / T0b / T0c calibration segments.

    * **T0a** -- upright, gaze horizontal, 10 s, near-static with +/-1.5 deg sway.
    * **T0b** -- face-down float, 10 s, near-static; pitch = ``pitch_baseline_deg``.
    * **T0c** -- three ~+/-25 deg pitch nods (~0.4 s each) as a sync marker.

    The same ``mount_offset_deg`` is applied to every segment -- recovering it is
    exactly what ``calibrate.py`` is tested on. With ``bad_calibration=True`` the
    T0b pose is performed looking forward by 25 deg, so T0a and T0b differ in
    pitch by ~65 deg instead of 90 deg and ``calibrate.py`` must raise
    ``CALIB_POSE_SUSPECT`` (acceptance test 5).

    Returns ``({"t0a", "t0b", "t0c"}, ground_truth)``.
    """
    streams = _rng_streams(seed)
    mount = Rotation.from_euler("xyz", mount_offset_deg, degrees=True)
    dt = 1.0 / SAMPLE_RATE_HZ

    def _segment(
        make_pitch_roll_yaw,
        dur_s: float,
        rng: np.random.Generator,
        t0: float,
    ) -> pl.DataFrame:
        n = int(round(dur_s * SAMPLE_RATE_HZ)) + 1
        t = t0 + np.arange(n) * dt
        pitch, roll, yaw = make_pitch_roll_yaw(t, n, rng)
        r_full, r_noyaw = _orientation(pitch, roll, yaw, mount)
        lin = np.zeros((n, 3))
        acc = _accel(r_noyaw, lin)
        gyro = _derive_gyro(r_full, dt)
        if noise:
            acc, gyro = _add_sensor_noise(acc, gyro, streams, gyro_bias_walk=False)
        quat = _emit_quaternion(r_full, streams["quat_error"], inject_error=noise)
        mag = _magnetometer(r_full, streams["mag"], n, add_noise=noise)
        return _make_dataframe(t, quat, acc, gyro, mag)

    sway = 1.5  # deg physiological sway

    def _upright(t, n, rng):
        # upright, gaze horizontal: a 90 deg pitch from the prone reference
        pitch = _UPRIGHT_PITCH_DEG + rng.normal(0.0, sway, size=n)
        roll = rng.normal(0.0, sway, size=n)
        yaw = np.zeros(n)
        return pitch, roll, yaw

    def _facedown(t, n, rng):
        # T0b defines the prone reference (pitch = baseline). A bad pose is
        # performed "looking forward by 25 deg" -- the nose lifted 25 deg toward
        # upright -- which drops the T0a/T0b angle from ~90 deg to ~65 deg,
        # outside the 90 +/- 10 deg sanity band, so calibrate.py must raise
        # CALIB_POSE_SUSPECT.
        offset = 25.0 if bad_calibration else 0.0
        pitch = pitch_baseline_deg + offset + rng.normal(0.0, sway, size=n)
        roll = rng.normal(0.0, sway, size=n)
        yaw = np.zeros(n)
        return pitch, roll, yaw

    def _nods(t, n, rng):
        # three raised-cosine pitch nods of +/-25 deg, ~0.4 s each, about prone
        pitch = np.zeros(n)
        for k in range(3):
            center = 0.8 + k * 1.2
            pitch += 25.0 * _raised_cosine(t - t[0], center, 0.2)
        pitch += rng.normal(0.0, 0.5, size=n)
        roll = rng.normal(0.0, sway, size=n)
        yaw = np.zeros(n)
        return pitch, roll, yaw

    t0a = _segment(_upright, 10.0, streams["timing"], 0.0)
    t0b = _segment(_facedown, 10.0, streams["cycle"], 0.0)
    t0c = _segment(_nods, 4.0, streams["yaw"], 0.0)

    ground_truth = {
        "mount_offset_deg": list(mount_offset_deg),
        "pitch_baseline_deg": pitch_baseline_deg,
        "bad_calibration": bad_calibration,
        "expected_pose_angle_deg": 90.0 - (25.0 if bad_calibration else 0.0),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "seed": seed,
    }
    return {"t0a": t0a, "t0b": t0b, "t0c": t0c}, ground_truth


def generate_bobs(
    exhale: bool,
    n_bobs: int = 10,
    noise: bool = True,
    seed: int | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Generate the Q4 vertical-bobbing trial (T1/T2).

    Ten ~1.2 s bobbing cycles, pitch near vertical, with large vertical
    acceleration excursions. When ``exhale=True`` a low-amplitude 10-60 Hz
    broadband component is added during the submerged phase (standing in for
    exhale-induced vibration); when ``exhale=False`` there is no such component,
    the submerged phase is slightly longer, and bob-to-bob timing varies more.

    Honesty note (spec section 6): 120 Hz logging gives a 60 Hz Nyquist limit,
    well below where real bubble acoustics live. This fixture tests only whether
    the analysis code can find a band-power difference *if one exists*. It is not
    evidence that the difference exists in reality; Q4 is expected to fail on
    real data and this generator must not be cited as support for it.
    """
    streams = _rng_streams(seed)
    dt = 1.0 / SAMPLE_RATE_HZ

    period = 1.2
    submerged_windows: list[tuple[float, float]] = []
    centers: list[float] = []
    t_cursor = 0.5
    for _ in range(n_bobs):
        jitter = streams["timing"].normal(0.0, 0.05 if exhale else 0.12)
        c = t_cursor + jitter
        centers.append(c)
        sub_len = 0.45 if exhale else 0.6
        submerged_windows.append((c - sub_len / 2.0, c + sub_len / 2.0))
        t_cursor = c + period

    duration = t_cursor
    n = int(round(duration * SAMPLE_RATE_HZ)) + 1
    t = np.arange(n) * dt

    # pitch near vertical (head-down): oscillate around 80 deg with the bob
    pitch = 80.0 + 8.0 * np.sin(2.0 * np.pi * t / period)
    roll = 3.0 * np.sin(2.0 * np.pi * t / period + 0.5)
    yaw = _yaw_signal(t, streams["yaw"], 1.0)

    mount = Rotation.from_euler("xyz", (0.0, 0.0, 0.0), degrees=True)
    r_full, r_noyaw = _orientation(pitch, roll, yaw, mount, base=Rotation.identity())

    # large vertical acceleration excursions along body-forward (head axis)
    vert = np.zeros(n)
    for c in centers:
        vert += 1.5 * G * np.sin(2.0 * np.pi * (t - c) / period) * _raised_cosine(
            t, c, period / 2.0
        )
    lin_sensor = mount.apply(np.array([1.0, 0.0, 0.0]))[None, :] * vert[:, None]
    lin_sensor = np.broadcast_to(lin_sensor, (n, 3)).copy()

    acc = _accel(r_noyaw, lin_sensor)

    if exhale:
        # 10-60 Hz broadband component during submerged phases (band-limited)
        raw = streams["acc_noise"].normal(0.0, 1.0, size=n)
        freqs = np.fft.rfftfreq(n, dt)
        spec = np.fft.rfft(raw)
        spec[(freqs < 10.0) | (freqs > 60.0)] = 0.0
        band = np.fft.irfft(spec, n=n)
        band /= max(np.std(band), 1e-9)
        env = np.zeros(n)
        for lo, hi in submerged_windows:
            env += ((t >= lo) & (t <= hi)).astype(float)
        acc = acc + 0.15 * G * (band * env)[:, None]

    gyro = _derive_gyro(r_full, dt)
    if noise:
        acc, gyro = _add_sensor_noise(acc, gyro, streams, gyro_bias_walk=False)
    quat = _emit_quaternion(r_full, streams["quat_error"], inject_error=noise)
    mag = _magnetometer(r_full, streams["mag"], n, add_noise=noise)

    df = _make_dataframe(t, quat, acc, gyro, mag)
    ground_truth = {
        "trial": "bobs",
        "exhale": exhale,
        "n_bobs": n_bobs,
        "period_s": period,
        "submerged_windows": [[round(a, 4), round(b, 4)] for a, b in submerged_windows],
        "exhale_band_hz": [10.0, 60.0] if exhale else None,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "seed": seed,
    }
    return df, ground_truth


# --------------------------------------------------------------------------- #
# Fixture generation
# --------------------------------------------------------------------------- #


def write_fixtures(out_dir: str | Path | None = None) -> list[Path]:
    """Generate the standard fixture set under ``tests/fixtures/``.

    Each trial is written as a parquet dataframe plus a ``*.gt.json`` ground
    truth file. Parquet (not CSV) is used deliberately: ``.gitignore`` excludes
    ``*.csv`` and ``.claudeignore`` excludes ``tests/fixtures/*.csv`` to keep
    large tables out of context. Returns the list of files written.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _dump(name: str, df: pl.DataFrame, gt: dict[str, Any]) -> None:
        pq = out_dir / f"{name}.parquet"
        gj = out_dir / f"{name}.gt.json"
        df.write_parquet(pq)
        gj.write_text(json.dumps(gt, indent=2, sort_keys=True), encoding="utf-8")
        written.extend([pq, gj])

    # All committed trials share one prone baseline with the committed
    # calibration poses (BASELINE below). The ground truth is gravity-referenced
    # relative to T0b, so a trial only round-trips against a calibration at the
    # same baseline -- one swimmer, one prone pose. Pinning the baseline lets the
    # single committed calibration decode every trial fixture.
    BASELINE = 4.0
    NOISY_MOUNT = (3.0, -2.0, 12.0)

    # One noisy trial per archetype, with deterministic seeds.
    for i, arch in enumerate(sorted(ARCHETYPES)):
        df, gt = generate_trial(
            arch, mount_offset_deg=NOISY_MOUNT, noise=True, seed=100 + i,
            pitch_baseline_deg=BASELINE,
        )
        _dump(f"trial_{arch.lower()}", df, gt)

    # A zero-noise, zero-offset round-trip fixture (for spec test 1 downstream),
    # paired with a matching zero-noise, zero-offset calibration.
    df, gt = generate_trial(
        "LIFTER", mount_offset_deg=(0.0, 0.0, 0.0), noise=False, seed=7,
        pitch_baseline_deg=BASELINE,
    )
    _dump("trial_lifter_clean", df, gt)
    clean_segs, _ = generate_calibration(
        mount_offset_deg=(0.0, 0.0, 0.0), pitch_baseline_deg=BASELINE,
        noise=False, seed=1,
    )
    for seg_name, seg_df in clean_segs.items():
        pq = out_dir / f"calib_clean_{seg_name}.parquet"
        seg_df.write_parquet(pq)
        written.append(pq)

    # Calibration: good and bad (mounted, matching the archetype trials).
    segs, gt = generate_calibration(
        mount_offset_deg=NOISY_MOUNT, pitch_baseline_deg=BASELINE, seed=1
    )
    for seg_name, seg_df in segs.items():
        pq = out_dir / f"calib_{seg_name}.parquet"
        seg_df.write_parquet(pq)
        written.append(pq)
    (out_dir / "calib.gt.json").write_text(
        json.dumps(gt, indent=2, sort_keys=True), encoding="utf-8"
    )
    written.append(out_dir / "calib.gt.json")

    segs_bad, gt_bad = generate_calibration(
        mount_offset_deg=(3.0, -2.0, 12.0),
        pitch_baseline_deg=4.0,
        bad_calibration=True,
        seed=2,
    )
    for seg_name, seg_df in segs_bad.items():
        pq = out_dir / f"calib_bad_{seg_name}.parquet"
        seg_df.write_parquet(pq)
        written.append(pq)
    (out_dir / "calib_bad.gt.json").write_text(
        json.dumps(gt_bad, indent=2, sort_keys=True), encoding="utf-8"
    )
    written.append(out_dir / "calib_bad.gt.json")

    # Bobs: exhale and breath-hold.
    for tag, ex in (("exhale", True), ("hold", False)):
        df, gt = generate_bobs(exhale=ex, seed=50 + int(ex))
        _dump(f"bobs_{tag}", df, gt)

    return written


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    files = write_fixtures()
    print(f"wrote {len(files)} fixture files to tests/fixtures/")
