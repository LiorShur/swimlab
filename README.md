# swimlab

Offline analysis pipeline for a head-mounted IMU pilot study on recreational
swimmers. Sensor: Movella DOT, occipital mount, 120 Hz onboard logging.

**Scope:** offline analysis only. No app, no cloud, no real-time. Read a
session directory, produce a metric table and per-participant plots.

## Study questions

| ID | Question | Pass criterion |
|----|----------|----------------|
| Q1 | Does head pitch during the breath separate "lifters" from "rotators"? | Hartigan dip test p < 0.05 |
| Q2 | Does IMU classification agree with a blinded coach rating? | Cohen κ ≥ 0.70, accuracy ≥ 85% |
| Q3 | Is the measurement repeatable across sessions? | ICC(2,1) ≥ 0.75 |
| Q4 | Can the accelerometer detect breath-holding? | AUC ≥ 0.75 (expected: no) |

## Module map

```
swimlab/
  io.py          DOT export -> canonical dataframe.  BLOCKED: needs a real export file.
  synth.py       Synthetic trace generator with ground truth.  BUILD FIRST.
  calibrate.py   T0a/T0b -> sensor-to-skull transform + pose sanity check
  events.py      Push-off detection, breath window detection
  metrics.py     Metric table from a calibrated trial
  stats.py       Dip test, ICC(2,1), SEM, MDC95, Bland-Altman, ROC, Cohen kappa
  report.py      Per-participant plots and summary PDF
```

See `CLAUDE.md` for conventions and hard constraints, `SYNTHETIC_DATA_SPEC.md`
for the generator spec, and `TASKS.md` for the build sequence.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Hard constraints (see CLAUDE.md)

1. Never commit anything under `data/`.
2. Magnetometer and yaw are unusable — logged, never used.
3. All angles are relative to the T0 calibration transform.
4. No hardcoded thresholds in `metrics.py` — every threshold comes from `config.yaml`.
5. Never silently drop data — rejected cycles carry a reason code.
