# Expansion protocol v1

The expansion uses 200 households absent from the engineering pilot. Selection extends the same fixed SHA-256 ranking to 205 households and excludes the five pilot households. All eligible snapshots of the selected households are retained, giving 566 matched observations. Selection is not based on labels or pilot success.

- Same GPT-4.1 mini snapshot, temperature, prompts, output schemas and 512-token ceiling as the pilot.
- Primary contrast: persona minus structured-history Brier score.
- Secondary contrast: persona minus gradient-boosting Brier score. Logistic contrast is exploratory.
- Paired household bootstrap: 5,000 resamples, seed 42. Percentile 95% intervals are marginal, with no multiple-comparison adjustment.
- A 0.01 Brier difference is a discussion margin, not a validated business decision threshold.
- Complete the fixed sample or explicitly report an incomplete run. No outcome-based stopping or prompt tuning.
- Baseline aggregate test results and pilot outcomes were previously inspected. This is an exploratory extension, not external preregistration.
- No model stability conclusion from one run at temperature zero. No causal explanation from inspecting generated text alone.

## Budget and failure handling

The 0.50 USD application cap includes the earlier pilot's usage-derived cost. Six workers reserve conservative maximum cost under a shared lock before transmission; successful responses settle against usage. Unknown charges retain reservations. The first request or parsing failure stops new work, and already running calls drain. There are no automatic retries. An incomplete run cannot be silently scored on its successful subset.

The budget tracks this authorized expansion and its pilot, not unrelated account usage or independent runs. It is an application guard using verified standard prices, not an account billing limit.

## Reporting

All model rows share the exact same prediction IDs and labels. Scoring verifies request, source, and baseline file hashes. Confidence intervals refer to direct paired differences, not overlapping per-model intervals. Per-window and per-segment tables are descriptive. Failure examples are selected after evaluation by signed loss difference in both directions.

The repository report contains aggregate metrics and protocol details. The local report also includes feature snapshots, generated personas, call IDs and per-observation loss contributions; raw household-level artifacts remain excluded from Git.
