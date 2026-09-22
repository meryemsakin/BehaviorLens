# BehaviorLens — research note

Persona summaries increased prediction error relative to structured history; gradient boosting had a lower Brier score.

## Question

Does replacing structured behavioral history with an LLM-generated persona improve 28-day category purchase predictions?

## Design

The Complete Journey official package provides a representation of transactions for 2,500 frequent-shopper households at one retailer. Nonpositive quantities were excluded. The category FLUID MILK PRODUCTS was chosen using only distinct household buyers in GROCERY before dataset day 365. Day indices are relative, not inferred calendar dates.

Training cutoffs: 365, 393, 421, 449, 477. Validation: 533, 561. Test: 617, 645, 673. Labels use [cutoff, cutoff + 28); historical features use only the prior 84 days. Eligibility uses past activity only. No future activity requirement is imposed.

Fixed supervised baselines use eight recency, frequency and retailer-sales features. The LLM receives those same historical feature values. The persona summary is generated from exactly that source. Neither LLM condition receives future labels. Prompts and model snapshot were unchanged after the engineering pilot.

The 200 expansion households were selected by deterministic hash ranking, excluding all five pilot households. Their 566 eligible test points are matched across all five conditions. The pilot and aggregate baseline outcomes had already been inspected; this is exploratory work, not external preregistration.

## Results

| Condition | Brier | Log loss | Average precision |
|---|---:|---:|---:|
| Gradient boosting | 0.1578 | 0.4764 | 0.8865 |
| Logistic regression | 0.1581 | 0.4770 | 0.8869 |
| LLM · structured history | 0.1756 | 0.6442 | 0.8419 |
| LLM · persona | 0.1965 | 0.7760 | 0.7927 |
| Prevalence baseline | 0.2462 | 0.6855 | 0.5618 |

Primary difference, persona minus structured: **+0.0210; 95% CI [+0.0097, +0.0326]**.

Secondary difference, persona minus gradient boosting: **+0.0387; 95% CI [+0.0248, +0.0531]**.

Negative differences favor persona. Percentile intervals use 5,000 paired household bootstrap draws. They are marginal intervals without multiplicity correction and do not account for shared temporal shocks. The 0.01 Brier discussion margin was set before expansion LLM outcomes; it is not a validated commercial decision threshold.

## Where the signal went

**Facts survived.** 89% of source feature values appear verbatim in generated personas (conservative: words such as “no visits” count as missing). Mean persona-minus-structured loss was +0.0254 where every value was retained and +0.0106 otherwise, so the loss is not explained by dropped numbers.

**Discrimination did not.** Using the report's ten fixed bins, resolution fell from 0.0820 (structured) to 0.0668 (persona), and reliability error rose from 0.0112 to 0.0168. Gradient boosting: 0.0928 and 0.0054.

| Milk trips, prior 84 days | n | Observed | GBDT mean p | Structured mean p | Persona mean p | Persona − structured Δ loss |
|---|---:|---:|---:|---:|---:|---:|
| none | 123 | 0.19 | 0.17 | 0.02 | 0.04 | -0.0067 |
| 1–2 | 148 | 0.38 | 0.41 | 0.41 | 0.38 | +0.0261 |
| 3–5 | 118 | 0.64 | 0.68 | 0.63 | 0.58 | +0.0171 |
| 6+ | 177 | 0.93 | 0.93 | 0.79 | 0.74 | +0.0384 |

Persona forecasts were most conservative for the most frequent category buyers. Both LLM conditions assigned very low probabilities to households with no recent category purchase, while gradient boosting tracked their observed rate. These groups were defined after viewing outcomes and are descriptive; they suggest hypotheses, not a mechanism.

## Failure analysis

The local report shows the three largest positive and three largest negative persona-minus-structured loss differences. Each example includes exact source features, generated persona, predictions, outcome, source hash and call identifiers. These are post-evaluation examples, not an unbiased estimate of failure prevalence. Numerical omissions or interpretive changes may suggest follow-up hypotheses; inspecting a summary does not establish a causal mechanism.

## Limits and next experiment

One category, one retailer, one model snapshot and a single draw per condition. Missing purchase records do not establish absence of purchases elsewhere. Demographic fields are partly generic codes and are not interpreted as actual age or income. No general synthetic-population fidelity or causal claim is made. Persona length and summarization are part of the treatment; this design does not isolate all representation effects.

The next independent replication should freeze new categories and time windows, include repeated model calls and inspect summary fidelity with a prespecified rubric. Any prompt changes after viewing these outcomes require a new exploratory version and a fresh evaluation set.

## Reproducibility

Run: `expansion_v1`. Model: `gpt-4.1-mini-2025-04-14`. Plan SHA-256: `71df9f32656a82b10c02357cb99159ba80f5f038b80a6b44b76f4414877cdd14`.

1698 API calls; 0 failed calls. Usage-derived run cost $0.24824; cumulative with pilot $0.25390.

All figures, tables and reported differences derive from results.json. Raw requests/responses, historical snapshots and labels remain local. See the repository reproduction instructions and fixed protocol. Independent project.
