# `synth.py` — Synthetic Head-IMU Trace Generator

**Why this is task one.** The sensor has not shipped. Real data will never have ground
truth — you will never know a swimmer's *true* head pitch, only what the sensor and a
coach say. A generator with known ground truth lets you build and validate the entire
pipeline before hardware arrives, and gives you the only dataset where "correct" is
defined. If the pipeline cannot recover a 20° pitch lift that you yourself injected,
it will not find one in a pool.

---

## 1. Interface

```python
def generate_trial(
    archetype: str,              # "LIFTER" | "ROTATOR" | "MIXED" | "FLAT" | "ASYMMETRIC"
    n_lengths: int = 4,          # 25 m lengths
    stroke_period_s: float = 1.4,
    breathe_every_n_strokes: int = 3,
    mount_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mount_slip_deg_per_min: float = 0.0,
    noise: bool = True,
    gyro_bias_walk: bool = True,
    seed: int | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Returns (canonical_dataframe, ground_truth)."""
```

Plus `generate_calibration()` producing T0a / T0b / T0c segments, and
`generate_bobs(exhale: bool)` for the Q4 T1/T2 trials.

**Sample rate: 120 Hz.** Fixed.

---

## 2. Generation approach

Generate the **orientation trajectory first**, then derive sensor readings from it.
Do not synthesise accelerometer and gyroscope signals directly — they will not be
physically consistent with each other, and the pipeline will pass tests it should fail.

**Step 1 — build Euler angle time series** `pitch(t)`, `roll(t)`, `yaw(t)` in the
true skull frame.

**Step 2 — convert to a rotation trajectory** `R_skull→global(t)`.

**Step 3 — apply the mount offset**: `R_sensor→global(t) = R_skull→global(t) · R_mount⁻¹`,
where `R_mount` is the fixed (or slowly drifting) sensor-to-skull rotation. This is the
transform `calibrate.py` must recover.

**Step 4 — derive gyroscope** by numerical differentiation of the rotation trajectory,
expressed in the sensor frame.

**Step 5 — derive accelerometer** as gravity rotated into the sensor frame, plus linear
acceleration of the head (small: a slow forward velocity with a superimposed oscillation
at stroke frequency, plus push-off transients).

**Step 6 — add noise and bias** (Section 5).

**Step 7 — emit quaternions** from the noisy trajectory to imitate the DOT's onboard
fusion output. Deliberately inject a small orientation error so that the fused output is
not perfectly consistent with the raw channels — real fusion output never is.

---

## 3. The motion model

### Stroke and breathing timing

One **cycle** = 2 strokes (left arm + right arm). With `breathe_every_n_strokes = 3`,
a breath occurs every 1.5 cycles, **alternating sides**. Get this right — it is the
difference between a bilateral and a unilateral pattern, and it drives `asymmetry_index`.

With `breathe_every_n_strokes = 2`, breaths occur every cycle on the **same side**.

### Roll

Baseline body roll, present on every cycle:

```
roll_base(t) = A_roll · sin(2π t / T_cycle)
```

with `A_roll ≈ 40°` (configurable). During a breath window the amplitude increases on
the breathing side to `A_breath ≈ 70°`, ramping smoothly in and out over ~150 ms. Use a
raised-cosine window, not a step — a discontinuity will create artificial gyro spikes
that the event detector will latch onto.

### Pitch — the signal under test

```
pitch(t) = pitch_baseline + pitch_oscillation(t) + pitch_breath_bump(t)
```

- `pitch_baseline`: constant per swimmer, default 0°, range −5° to +12° (a swimmer who
  habitually looks forward even when not breathing).
- `pitch_oscillation`: ±2–3° at stroke frequency.
- `pitch_breath_bump`: a raised-cosine bump coincident with the breath window, peak
  amplitude determined by archetype.

### Archetypes

| archetype | breath pitch bump | roll amplitude | notes |
|-----------|------------------|----------------|-------|
| `ROTATOR` | 2–6° | 65–80° | correct technique |
| `LIFTER` | 15–28° | 35–50° | lifts instead of rotating |
| `MIXED` | 8–14°, high cycle-to-cycle SD | 45–65° | inconsistent — the hard case |
| `FLAT` | 10–18° | 20–30° | barely rolls at all; lifts by default |
| `ASYMMETRIC` | 4° left, 20° right | 70° left, 40° right | tests `asymmetry_index` |

Randomise within each range per generated swimmer, and add per-cycle jitter (SD ≈ 15% of
the mean bump) so no two cycles are identical.

`MIXED` is the archetype that matters most. If the classifier separates LIFTER from
ROTATOR but assigns MIXED arbitrarily, Q1's bimodality result is an artefact of an
easy sample.

### Push-offs

At each length boundary: an acceleration transient of 2–4 g over ~200 ms, followed by a
glide phase of 1.5–3 s with near-zero roll and pitch close to baseline, then stroke
resumption. `events.py` must find these.

### Yaw

Generate a plausible yaw signal, then **corrupt it** — add a random walk of several
degrees per second plus occasional 10–30° jumps, imitating magnetic disturbance. Any
pipeline code that accidentally depends on yaw will fail loudly. That is the point.

---

## 4. Calibration segments

`generate_calibration()` produces:

- **T0a** — upright, gaze horizontal, 10 s, near-static with physiological sway (±1.5°)
- **T0b** — face-down float, 10 s, near-static, pitch = `pitch_baseline` by definition
- **T0c** — three head nods, ±25° pitch, ~0.4 s each, for the sync marker

The same `mount_offset_deg` must be applied to calibration and trial segments. Recovering
it is precisely what `calibrate.py` is being tested on.

Also provide a `bad_calibration=True` mode where T0b is performed looking forward by 25°.
`calibrate.py` must raise `CALIB_POSE_SUSPECT` on this input.

---

## 5. Noise model

Use published Movella DOT figures:

| source | value |
|--------|-------|
| gyroscope noise density | 0.007 °/s/√Hz |
| gyroscope bias | 10 °/h |
| accelerometer noise density | 120 µg/√Hz |
| accelerometer bias | 0.03 mg |

Add a slow random-walk gyro bias when `gyro_bias_walk=True`.

**Mount slip:** when `mount_slip_deg_per_min > 0`, rotate `R_mount` progressively over
the trial. This models the highest-likelihood risk in the register. The pipeline should
be tested for how badly a 5°/min slip corrupts `d_pitch_breath` versus `roll_pitch_ratio`
— that comparison alone may decide which metric becomes the primary gate.

---

## 6. Bobs (Q4)

`generate_bobs(exhale: bool)` produces 10 vertical bobbing cycles.

- **Common:** ~1.2 s period, pitch near vertical, large vertical acceleration excursions.
- **`exhale=True`:** add a low-amplitude broadband component in the 10–60 Hz band during
  the submerged phase, standing in for exhale-induced vibration.
- **`exhale=False`:** no such component; slightly longer submerged phase and greater
  bob-to-bob timing variability.

**Be honest about what this tests.** 120 Hz logging gives a 60 Hz Nyquist limit, well
below where bubble acoustics live. This fixture tests only whether the *analysis code*
can find a band-power difference if one exists. It is not evidence that the difference
exists in reality. Q4 is expected to fail on real data, and this generator must not be
cited as support for it.

---

## 7. Ground truth output

```json
{
  "archetype": "LIFTER",
  "seed": 42,
  "mount_offset_deg": [3.2, -1.8, 12.0],
  "mount_slip_deg_per_min": 0.0,
  "pitch_baseline_deg": 4.1,
  "stroke_period_s": 1.4,
  "pushoffs": [{"t": 0.0}, {"t": 21.3}, {"t": 43.1}, {"t": 64.8}],
  "breaths": [
    {"index": 0, "side": "R", "t_start": 3.42, "t_end": 4.06,
     "true_d_pitch_deg": 21.4, "true_peak_roll_deg": 44.2,
     "excluded_by_protocol": true, "exclusion_reason": "within_4s_of_pushoff"}
  ],
  "summary": {"true_mean_d_pitch_deg": 20.8, "n_breaths": 46, "n_valid_breaths": 38}
}
```

Ground truth must mark which breaths the exclusion rules should remove, so
`events.py` can be tested on exclusion logic independently of detection accuracy.

---

## 8. Acceptance tests

The generator is done when all of these pass:

1. **Round trip.** Feed a zero-noise, zero-offset trial through `calibrate` → `events` →
   `metrics`. Recovered `d_pitch_breath` matches ground truth within **0.5°**, and all
   breaths are detected with no false positives.
2. **Mount recovery.** With `mount_offset_deg = (10, 15, 20)` and no noise,
   `calibrate.py` recovers the transform within **1°** on each axis.
3. **Noise tolerance.** With full noise, recovered `d_pitch_breath` matches truth within
   **2°**, and breath detection achieves ≥ 95% recall with ≤ 5% false positives.
4. **Archetype separation.** Generate 20 LIFTER and 20 ROTATOR swimmers with noise and
   random mount offsets; the pipeline's classification accuracy exceeds **90%**. If it
   cannot separate synthetic archetypes it has no chance on real swimmers.
5. **Bad calibration caught.** `bad_calibration=True` raises `CALIB_POSE_SUSPECT`.
6. **Yaw independence.** Doubling the injected yaw corruption changes no output metric by
   more than **0.1°**.
7. **Exclusion logic.** Breaths flagged `excluded_by_protocol` in ground truth are exactly
   those the pipeline excludes.
8. **Determinism.** Same seed produces byte-identical output.

Test 4 is the one that matters. Tests 1–3 verify the maths; test 4 verifies the study
design is capable of answering Q1 and Q2 *in principle*. If test 4 fails, the problem
is upstream of the code.

---

## 9. Deliberate limitations

State these in the module docstring so nobody later mistakes synthetic performance for
validated performance:

- Real head motion is not sinusoidal. Real swimmers vary stroke period, pause, sight the
  wall, and adjust goggles mid-length.
- The generator encodes the pipeline author's assumptions about what lifting and rotating
  look like. Those assumptions are the hypothesis under test — the generator cannot
  validate them.
- Passing every test here means the pipeline is internally correct. It says nothing about
  whether head pitch is a useful diagnostic gate. Only the pool answers that.
