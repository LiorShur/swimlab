# swimlab — Claude Code task sequence

Seven tasks. **One fresh Claude Code session per task.** They share almost no state
beyond the canonical dataframe schema in `CLAUDE.md`, so carrying context between them
buys nothing and costs everything — this is the accumulated-context failure mode, not a
message-length one.

Start each session by reading `CLAUDE.md` only. Reference `SYNTHETIC_DATA_SPEC.md` in
task 1 only.

---

## Task 0 — Scaffold (15 min, do it yourself)

Don't hand this to Claude Code; it's faster to type.

```
swimlab/
  CLAUDE.md
  SYNTHETIC_DATA_SPEC.md
  TASKS.md
  config.yaml
  pyproject.toml
  .gitignore
  .claudeignore
  data/            # empty, gitignored
  swimlab/__init__.py
  tests/fixtures/
  notebooks/
```

**`.gitignore`**
```
data/
notebooks/*.ipynb
*.mp4
*.csv
.venv/
__pycache__/
_keys/
```

**`.claudeignore`** — large CSVs and notebook JSON will eat context instantly.
```
data/
notebooks/
tests/fixtures/*.csv
*.mp4
```

**`config.yaml`** — every threshold lives here, none in code.
```yaml
sample_rate_hz: 120
breath_roll_threshold_deg: 25.0
pushoff_g_threshold: 2.0
exclude_after_pushoff_s: 4.0
exclude_before_end_s: 2.0
min_valid_cycles: 20
calib_pose_angle_deg: 90.0
calib_pose_tolerance_deg: 10.0
```

---

## Task 1 — `synth.py`

**Read:** `CLAUDE.md`, `SYNTHETIC_DATA_SPEC.md`
**Deliver:** generator + ground-truth emitter + fixture files in `tests/fixtures/`
**Done when:** acceptance tests 5, 6 and 8 from the spec pass (the others need
downstream modules and are checked in task 6).

The highest-value task in the sequence. Do not rush it. Generate orientation first,
derive sensor readings from it — synthesising accel and gyro independently produces
physically inconsistent data that will let broken code pass.

---

## Task 2 — `calibrate.py`

**Deliver:** `fit_transform(t0a_df, t0b_df) -> Rotation`, `apply(df, R) -> df` with
`pitch_deg` and `roll_deg` columns added, and the T0a/T0b 90° pose sanity check.

**Done when:**
- Mount offset recovered within 1° on synthetic zero-noise data (spec test 2)
- Within 2° with full noise
- `bad_calibration=True` fixture raises `CALIB_POSE_SUSPECT`
- No code path reads `mag_*` or computes yaw

---

## Task 3 — `events.py`

**Deliver:** `detect_pushoffs(df)`, `detect_breath_windows(df)`, `apply_exclusions(...)`,
each returning a table of events with start/end times, side (L/R), and flags.

**Done when:**
- ≥ 95% breath recall, ≤ 5% false positives on the full-noise fixture
- Push-off detection finds all four boundaries in a 4-length fixture
- Excluded breaths match ground truth `excluded_by_protocol` exactly (spec test 7)
- `INSUFFICIENT_CYCLES` raised below `min_valid_cycles`

The trap here: a detector tuned on synthetic data will be too clean. Leave the
thresholds in `config.yaml` and expect to retune on the first real participant.

---

## Task 4 — `metrics.py`

**Deliver:** the metric table from `CLAUDE.md`, one row per breath plus a per-trial
summary. Pure functions only.

**Done when:**
- `d_pitch_breath` within 0.5° of truth (zero noise), 2° (full noise)
- `roll_pitch_ratio` computed alongside, never instead
- `asymmetry_index` correctly recovers the `ASYMMETRIC` archetype
- Every rejected cycle carries a reason code — nothing dropped silently

---

## Task 5 — `stats.py`

**Deliver:** Hartigan dip test, Cohen κ with confusion matrix, ROC with bootstrapped CI
on the Youden-optimal threshold, ICC(2,1) with CI, SEM, MDC₉₅ = 1.96 × √2 × SEM,
Bland–Altman with bias and limits of agreement.

**Done when:** each function is verified against a published worked example or a
reference implementation — not against synthetic data. These are standard statistics;
correctness is checkable independently, and a subtly wrong ICC would invalidate Q3
without any visible symptom.

---

## Task 6 — Integration

**Deliver:** `run_session(path) -> metrics table + flags`, and a script that runs all
eight acceptance tests from the spec end to end.

**Done when:** spec tests 1, 2, 3, 4 and 7 pass.

**Test 4 is the gate for the whole project.** 20 synthetic LIFTERs and 20 ROTATORs with
noise and random mount offsets, classification accuracy > 90%. If this fails, stop and
fix the design before the pool sessions — a pipeline that can't separate archetypes you
built yourself will not separate real swimmers.

Also worth running here, though not an acceptance test: sweep `mount_slip_deg_per_min`
from 0 to 10 and compare how `d_pitch_breath` and `roll_pitch_ratio` degrade. That
comparison may pick the primary gate metric before you collect a single real swim.

---

## Task 7 — `io.py` — **BLOCKED**

Do not start until a real Movella DOT export exists on disk. Do not guess the column
schema.

**Deliver:** `read_dot_export(path) -> canonical dataframe`, unit conversion, timestamp
normalisation to trial start, nod-marker detection for video sync.

**Done when:** a real export round-trips into the canonical schema and produces a metric
table, and the nod marker is found within 1 sample of manual inspection.

---

## Task 8 — `report.py` (after first real data)

Per-participant plots: pitch and roll traces with breath windows shaded, `d_pitch_breath`
per cycle with the fitted threshold, a Bland–Altman for retest participants, and a
one-page summary. Deliberately last — plot design should follow from looking at real
traces, not precede it.

---

## What is deliberately not here

No app, no database, no API, no cloud, no dashboard, no real-time processing, no wrist or
ankle sensor support. Every one of those is downstream of knowing whether head pitch is a
real gate. Adding any of them now is the most likely way this project stalls.
