# BehaviorLens

**Do synthetic users predict real behavior—or merely sound plausible?**

An independent evaluation project for probabilistic behavioral predictions.

## Status

Initial evaluation kernel, not a completed benchmark. No real-data results or LLM runs yet. The included CSV is handcrafted test data and must not be cited as empirical evidence.

Implemented: strict probability/label validation, duplicate checks, temporal metadata checks, Brier score, clipped log loss, calibration bins, ECE, descriptive subgroup metrics, paired household bootstrap and JSON/HTML reporting with an input fingerprint.

Planned: verified Complete Journey ingestion, train/validation/test construction, fitted baselines, LLM adapters, experiment lineage, real experiments, presentation and video.

## Run without dependencies

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m behaviorlens examples/fixture.csv --evidence fixture --output outputs/fixture
```

Input CSV: `household_id,cutoff,history_end,label_end,y,probability,baseline_probability,segment`.
Each row represents the same outcome and horizon for model and baseline. Dates use ISO format. `history_end < cutoff < label_end` is required. Only one row per household/cutoff is permitted; evaluate each task/model/run separately. Baseline probabilities must be fitted without test labels.

The interval estimates the observation-weighted Brier difference, model minus baseline; negative favors the model. Whole households are resampled, keeping paired predictions together. This does not account for shared temporal shocks. Small subgroups are descriptive only. Metadata checks do not prove absence of upstream leakage.

`--evidence observed` is an operator declaration, not an automatic verification of data provenance. Raw data, credentials and generated outputs are excluded from Git.

This is an independent project.
