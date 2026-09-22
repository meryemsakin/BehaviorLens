# BehaviorLens

Testing behavioral predictions against held-out retailer transaction records.

## First baseline experiment

**Status: real-data baselines completed; OpenAI engineering pilot completed. Full LLM/persona benchmark pending.**

Task: purchase in `FLUID MILK PRODUCTS` within 28 dataset days. Category chosen using only records before day 365. Test: 6,939 observations from 2,398 recently active households.

| Model | Brier ↓ | Average precision ↑ | Δ Brier vs prevalence | 95% household-cluster interval |
|---|---:|---:|---:|---|
| prevalence | 0.2432 | 0.5854 | +0.0000 | [+0.0000, +0.0000] |
| logistic | 0.1709 | 0.8715 | -0.0724 | [-0.0781, -0.0667] |
| gradient_boosting | 0.1701 | 0.8709 | -0.0731 | [-0.0789, -0.0674] |

Historical behavioral features improve prediction over a fixed training-prevalence forecast in this task. The two fitted models have similar point estimates; their difference has not been tested. **No conclusion about persona value is available yet.**

This is one category at one retailer, not a test of general population representativeness. The source describes a representation of transactions; demographic codes are not decoded into age or income. Intervals resample households, not shared temporal shocks.

## Live OpenAI pilot

A five-household pilot completed 39 API calls over 13 prediction points using GPT-4.1 mini. Estimated token cost: **$0.00565**. This verifies the pipeline, not a research hypothesis. [Pilot report](reports/openai_pilot_v1/report.html) · [Protocol and cost guard](docs/OPENAI_PILOT.md).

## Reproduce

```sh
uv sync --extra research --frozen
uv run python scripts/download_journey.py
uv run python -m behaviorlens.benchmark --output outputs/my_run
```

Use a fresh output directory for each run. Each run saves the protocol, source and code hashes, metrics, matched predictions, feature snapshots and an HTML report with calibration and a baseline failure explorer. Raw and household-level data stay local.

```sh
uv run python -m behaviorlens.prepare_llm --run outputs/my_run --output outputs/my_llm_plan
```

This prepares label-free requests for two LLM conditions from identical source features; it does not call a model. Use the exact same selected subset for baseline comparison.

## Validate the kernel

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m behaviorlens examples/fixture.csv --evidence fixture --output outputs/fixture
```

The fixture is handcrafted smoke-test data, not research evidence. The evaluation CLI accepts ISO dates or integer dataset days. Metadata checks alone do not prove feature provenance.

## Evidence and scope

- [Aggregate results and provenance](reports/baseline_v1/results.json)
- [Dataset card](docs/DATASET_CARD.md)
- [Fixed experiment protocol](configs/complete_journey_v1.json)
***REMOVED***

Next: expand matched LLM conditions beyond the engineering pilot; evaluate failures and uncertainty; prepare research note, video and presentation. This is an independent project.
