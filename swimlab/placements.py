"""Placement registry -- the platform's core abstraction (see
``docs/platform-design.md`` in swimlab-viewer, section 2).

The same physical Movella DOT can be worn on the head, the sacrum, either
wrist, either upper arm, or either ankle. **Interpretation depends on where the
sensor was and how it was calibrated, not on the sensor itself.** So the
platform is organised around *placements*, not sensors: each recording declares
its placement, and the pipeline routes to that placement's calibration protocol
and analysis module.

This module is intentionally dependency-free -- it imports nothing from the rest
of ``swimlab`` -- so both the synthetic body model (:mod:`swimlab.synth`) and the
per-placement analysis modules can import it without a cycle. A placement names
the body **segment** it measures; the unified synthetic swimmer
(:func:`swimlab.synth.generate_swim`) produces one kinematic trajectory per
segment, and :func:`swimlab.synth.virtual_sensor` samples the segment a placement
sits on. "Same sensor, placed differently" is simply choosing a different
registry entry for that recording.

Nothing here hardcodes a study threshold (CLAUDE.md hard constraint 4); the
fields are structural (which segment, which poses, which sign conventions),
never tuned numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CalibrationPose",
    "Placement",
    "PLACEMENTS",
    "get",
    "implemented",
    "segment_of",
]


@dataclass(frozen=True)
class CalibrationPose:
    """One static (or slow) calibration pose in a placement's protocol.

    Two poses ~90 deg apart give a gravity-only sensor->segment transform, the
    same construction the head module already uses (T0a upright + T0b prone). The
    ``expected_pair_angle_deg`` on the :class:`Placement` says how far the two
    poses should differ so a badly performed pose can be flagged rather than
    silently trusted (CLAUDE.md calibration sanity check).
    """

    id: str
    description: str


@dataclass(frozen=True)
class Placement:
    """One body location a sensor can be worn, and how to interpret it.

    Attributes
    ----------
    id:
        Stable placement key, e.g. ``"head"``, ``"sacrum"``, ``"wrist_l"``.
    segment:
        The rigid body segment this placement measures (``"skull"``,
        ``"pelvis"``, ``"forearm_l"``, ...). Keys into the segment trajectories
        that :func:`swimlab.synth.generate_swim` produces.
    side:
        ``"L"``, ``"R"`` or ``None`` (midline placements like head/sacrum).
    calibration:
        The ordered pose protocol used to build the sensor->segment transform.
    expected_pair_angle_deg:
        Nominal angle (deg) between the first two calibration poses' gravity
        directions -- the pose-sanity target (like the head's ~90 deg between
        upright and prone). ``None`` when the placement's poses are not a
        fixed-angle pair.
    frame:
        Human-readable anatomical sign conventions for the calibrated angles
        (the per-placement analogue of the head's pitch/roll/yaw notes).
    metrics:
        The metric keys this placement's module is expected to produce. Advisory
        documentation of the placement's catalogue, not an enforced schema.
    module:
        Dotted name of the analysis module for this placement, or ``None`` while
        it is still on the roadmap. Referenced by name (not imported) so the
        registry stays dependency-free.
    implemented:
        ``True`` once the placement's module + tests exist. Lets the app show
        which placements are live versus planned without importing them.
    """

    id: str
    segment: str
    calibration: tuple[CalibrationPose, ...]
    frame: str
    side: str | None = None
    expected_pair_angle_deg: float | None = 90.0
    metrics: tuple[str, ...] = ()
    module: str | None = None
    implemented: bool = False
    notes: str = ""


# --------------------------------------------------------------------------- #
# Shared calibration poses
# --------------------------------------------------------------------------- #

_UPRIGHT = CalibrationPose("upright", "Upright stand, gaze horizontal, still ~5 s")
_PRONE = CalibrationPose("prone", "Face-down streamline float, still ~5 s")
_ARM_OVERHEAD = CalibrationPose(
    "arm_overhead", "Arm extended overhead in streamline, still ~5 s"
)
_ARM_SIDE = CalibrationPose("arm_side", "Arm relaxed at the side, still ~5 s")
_ARM_FORWARD = CalibrationPose(
    "arm_forward", "Arm forward-horizontal, palm down, still ~5 s"
)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

#: Every placement the platform targets. ``implemented`` marks which have a
#: working module + tests today; the rest are the phased roadmap
#: (docs/platform-design.md section 11). Segment names are shared with
#: :func:`swimlab.synth.generate_swim`.
PLACEMENTS: dict[str, Placement] = {
    "head": Placement(
        id="head",
        segment="skull",
        side=None,
        calibration=(_UPRIGHT, _PRONE),
        expected_pair_angle_deg=90.0,
        frame=(
            "Calibrated skull frame, gravity-referenced, zero at T0b prone. "
            "pitch: + = nose up (head lift). roll: + = face to swimmer's right. "
            "yaw: unused."
        ),
        metrics=(
            "d_pitch_breath",
            "peak_roll_breath",
            "roll_pitch_ratio",
            "breath_duration",
            "asymmetry_index",
            "pitch_variability",
            "pitch_drift_100m",
        ),
        module="swimlab.calibrate+events+metrics",
        implemented=True,
        notes="The shipped module. Breath technique (lift vs rotate).",
    ),
    "sacrum": Placement(
        id="sacrum",
        segment="pelvis",
        side=None,
        calibration=(_UPRIGHT, _PRONE),
        expected_pair_angle_deg=90.0,
        frame=(
            "Calibrated pelvis frame, gravity-referenced, zero at prone glide. "
            "roll: + = body rotating to swimmer's right. pitch: + = hips/torso "
            "pitching head-up (dropping legs). yaw: unused."
        ),
        metrics=(
            "lengths",
            "stroke_count",
            "stroke_rate",
            "tempo",
            "distance_m",
            "body_roll_amplitude",
            "roll_symmetry_index",
            "pushoff_count",
            "pushoff_interval",
            "pace_drift",
        ),
        module="swimlab.sacrum",
        implemented=False,
        notes="Highest-value single sensor: time/lengths/stroke-count/rate/distance.",
    ),
    "wrist_l": Placement(
        id="wrist_l",
        segment="forearm_l",
        side="L",
        calibration=(_ARM_FORWARD, _ARM_SIDE),
        expected_pair_angle_deg=90.0,
        frame=(
            "Calibrated left-forearm frame, gravity-referenced, zero at arm "
            "extended forward-horizontal. pitch: + = hand above the forward line "
            "(recovery over water), - = hand below (catch/pull under water); roll "
            "tracks pronation/supination. yaw: unused. (Calibration poses are "
            "forward-horizontal + arm-at-side, ~90 deg apart -- overhead vs side "
            "would be ~180 deg and cannot span a frame.)"
        ),
        metrics=(
            "stroke_count",
            "catch_time",
            "pull_time",
            "recovery_time",
            "entry_exit_timing",
        ),
        module="swimlab.wrist",
        implemented=False,
        notes="Per-arm stroke phases; pairs with wrist_r for L/R symmetry.",
    ),
    "wrist_r": Placement(
        id="wrist_r",
        segment="forearm_r",
        side="R",
        calibration=(_ARM_FORWARD, _ARM_SIDE),
        expected_pair_angle_deg=90.0,
        frame=(
            "Calibrated right-forearm frame, gravity-referenced, zero at arm "
            "extended forward-horizontal. pitch: + = hand above the forward line "
            "(recovery over water), - = hand below (catch/pull under water); roll "
            "tracks pronation/supination. yaw: unused."
        ),
        metrics=(
            "stroke_count",
            "catch_time",
            "pull_time",
            "recovery_time",
            "entry_exit_timing",
        ),
        module="swimlab.wrist",
        implemented=False,
        notes="Per-arm stroke phases; pairs with wrist_l for L/R symmetry.",
    ),
    "upper_arm_l": Placement(
        id="upper_arm_l",
        segment="upper_arm_l",
        side="L",
        calibration=(_ARM_SIDE, _ARM_FORWARD),
        expected_pair_angle_deg=90.0,
        frame="Calibrated left-upper-arm frame; shoulder rotation, elbow-high catch proxy.",
        metrics=("shoulder_rotation", "elbow_high_proxy"),
        module="swimlab.upper_arm",
        implemented=False,
    ),
    "upper_arm_r": Placement(
        id="upper_arm_r",
        segment="upper_arm_r",
        side="R",
        calibration=(_ARM_SIDE, _ARM_FORWARD),
        expected_pair_angle_deg=90.0,
        frame="Calibrated right-upper-arm frame; shoulder rotation, elbow-high catch proxy.",
        metrics=("shoulder_rotation", "elbow_high_proxy"),
        module="swimlab.upper_arm",
        implemented=False,
    ),
    "ankle_l": Placement(
        id="ankle_l",
        segment="shank_l",
        side="L",
        calibration=(_UPRIGHT, _PRONE),
        expected_pair_angle_deg=90.0,
        frame="Calibrated left-shank frame; kick amplitude/rate from pitch oscillation.",
        metrics=("kick_count", "kick_rate", "kick_amplitude", "kick_symmetry_index"),
        module="swimlab.ankle",
        implemented=False,
    ),
    "ankle_r": Placement(
        id="ankle_r",
        segment="shank_r",
        side="R",
        calibration=(_UPRIGHT, _PRONE),
        expected_pair_angle_deg=90.0,
        frame="Calibrated right-shank frame; kick amplitude/rate from pitch oscillation.",
        metrics=("kick_count", "kick_rate", "kick_amplitude", "kick_symmetry_index"),
        module="swimlab.ankle",
        implemented=False,
    ),
}


def get(placement_id: str) -> Placement:
    """Look up a placement by id, with a clear error listing the valid ids."""
    try:
        return PLACEMENTS[placement_id]
    except KeyError:
        raise KeyError(
            f"unknown placement {placement_id!r}; "
            f"registered placements are {sorted(PLACEMENTS)}"
        ) from None


def implemented() -> list[Placement]:
    """The placements with a working module + tests today (app 'live' list)."""
    return [p for p in PLACEMENTS.values() if p.implemented]


def segment_of(placement_id: str) -> str:
    """The body segment a placement measures (its key into a BodyModel)."""
    return get(placement_id).segment
