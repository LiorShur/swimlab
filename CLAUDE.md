# swimlab

Analysis pipeline for a head-mounted IMU pilot study on recreational swimmers.
Sensor: Movella DOT, occipital mount, 120 Hz onboard logging.

**Scope:** offline analysis only. No app, no cloud, no real-time. Read a session
directory, produce a metric table and per-participant plots.

## What this study asks

| ID | Question | Pass criterion |
|----|----------|----------------|
| Q1 | Does head pitch change during the breath separate "lifters" from "rotators"? | Hartigan dip test p < 0.05 |
| Q2 | Does IMU classification agree with a blinded coach rating? | Cohen κ ≥ 0.70, accuracy ≥ 85% |
| Q3 | Is the measurement repeatable across sessions? | ICC(2,1) ≥ 0.75 |
| Q4 | Can the accelerometer detect breath-holding? | AUC ≥ 0.75 (expected: no) |

## Hard constraints

1. **Never commit anything under `data/`.** Participant video and sensor data. Check `.gitignore` before any commit.
2. **Magnetometer and yaw are unusable.** Pool halls have severe magnetic disturbance. Use gravity-referenced pitch and roll only. Log magnetometer, never compute from it.
3. **All angles are relative to the T0 calibration transform.** There is no meaningful "absolute" head angle in this study.
4. **No hardcoded thresholds in `metrics.py`.** Every threshold comes from `config.yaml`. They are study outputs, not inputs.
5. **Never silently drop data.** Rejected cycles are returned with a reason code, never filtered away invisibly. A pipeline that hides bad data will make every result look good.

## Coordinate conventions

After calibration, angles are expressed in a skull-fixed frame:

- **pitch** — rotation about the medio-lateral axis. **Positive = nose up (head lift).** Zero is defined by the T0b face-down float pose.
- **roll** — rotation about the longitudinal (nose-to-occiput) axis. **Positive = face rotating to the swimmer's right.**
- **yaw** — not used. Do not compute, do not report.

**Calibration sanity check:** T0a (upright, gaze horizontal) and T0b (face-down float) should differ in pitch by 90° ± 10°. Any larger deviation means one pose was performed incorrectly — emit a `CALIB_POSE_SUSPECT` quality flag and do not silently proceed.

## Canonical dataframe

Every reader returns this schema. Nothing downstream touches vendor formats.

| column | type | unit |
|--------|------|------|
| `t` | float64 | seconds from trial start |
| `quat_w`, `quat_x`, `quat_y`, `quat_z` | float64 | unit quaternion, sensor→global |
| `acc_x`, `acc_y`, `acc_z` | float64 | m/s², sensor frame, includes gravity |
| `gyr_x`, `gyr_y`, `gyr_z` | float64 | °/s, sensor frame |
| `mag_x`, `mag_y`, `mag_z` | float64 | µT — logged, never used |

## Metric definitions

Computed on T7 (4 × 25 m front crawl, breathing every 3).

| metric | definition | unit |
|--------|-----------|------|
| `d_pitch_breath` | peak pitch in breath window − median pitch over the preceding non-breath cycles | ° |
| `peak_roll_breath` | max abs roll in breath window | ° |
| `roll_pitch_ratio` | `d_pitch_breath / peak_roll_breath` | — |
| `breath_duration` | breath window duration | s |
| `asymmetry_index` | (left − right) / mean, on `d_pitch_breath` | — |
| `pitch_variability` | SD of `d_pitch_breath` across cycles | ° |
| `pitch_drift_100m` | slope of per-length median pitch across T9 | °/length |

`roll_pitch_ratio` was proposed as a candidate to replace `d_pitch_breath` as the primary gate — dimensionless, and hypothesised to be more robust to mount variation. **Synthetic-pipeline analysis does not support this** (to be confirmed on real swimmers): after calibration *both* metrics are exactly invariant to a fixed mount offset — calibration removes it — so neither is "more robust" there; and under progressive mount *slip* `roll_pitch_ratio` degrades **more** than `d_pitch_breath` (≈2× the relative drift at 10°/min across LIFTER/ROTATOR/MIXED), because the ratio compounds pitch and roll errors. It is also undefined for low-roll swimmers. On the mount-robustness criterion `d_pitch_breath` is therefore the better primary gate. Both are still computed (the study confirms on real data); the slip/variation comparison is pinned in `tests/test_integration.py`.

**Breath window:** roll crosses ±25° from baseline, first crossing to return. This detector must be validated frame-by-frame against video on the first two participants before batch processing.

## Exclusion rules

- Discard cycles within **4 s after a detected push-off** and within **2 s of trial end**. This is a time-based proxy for the protocol's "first and last 5 m" — a head IMU alone cannot measure position. Document this approximation in any writeup.
- Push-off detection: acceleration magnitude spike above `config.pushoff_g_threshold`.
- Require **≥ 20 valid cycles** per participant for T7. Below that, mark the participant `INSUFFICIENT_CYCLES` and exclude from primary analysis.

## Module map

```
swimlab/
  io.py          DOT export → canonical dataframe.  BLOCKED: needs a real export file.
  synth.py       Synthetic trace generator with ground truth.  BUILD FIRST.
  calibrate.py   T0a/T0b → sensor-to-skull transform + pose sanity check
  events.py      Push-off detection, breath window detection
  metrics.py     Metric table from a calibrated trial
  stats.py       Dip test, ICC(2,1), SEM, MDC95, Bland-Altman, ROC, Cohen κ
  report.py      Per-participant plots and summary PDF
```

## Conventions

- Python 3.11+, `polars` for dataframes, `scipy` / `statsmodels` for stats, `numpy-quaternion` or `scipy.spatial.transform.Rotation` for orientation maths.
- Metric functions are **pure**: dataframe in, dataframe out, no I/O, no globals.
- Every metric function has a unit test against a `synth.py` fixture with known ground truth. Recovered value must match truth within a stated tolerance.
- Type hints throughout. Docstrings state units and sign conventions explicitly.
- Quality flags accumulate in a `flags` column, never as exceptions, unless the data is genuinely unreadable.

## Working notes

- `io.py` is blocked until a real Movella DOT export exists. Do not guess the column schema. Everything else is built and tested against `synth.py`.
- Build order: `synth` → `calibrate` → `events` → `metrics` → `stats` → `report` → `io`.
- These modules share almost no state beyond the canonical schema. Use a fresh session per module.
