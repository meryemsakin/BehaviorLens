# BehaviorLens v2: pre-registered protocol

Frozen on 2026-09-22, before any v2 language-model output existed. The git commit that adds this file is the timestamp. The prompts, sampling, and analysis code in that commit are the ones that will run. Any deviation will be listed in the results under **Deviations**.

## Motivation

Version 1 found that an LLM persona written from eight aggregate features predicted fluid-milk purchases worse than the same features given as structured text: Brier +0.021, 95% CI [+0.010, +0.033]. Two things limit that result:

1. A real synthetic-user pipeline builds personas from much richer data.
2. The result covers one category and one older model.

Version 2 tests whether the result holds under realistic conditions, in two tasks, with current models.

## Studies

**Study A: category purchase.** Will a household buy each of five categories at this retailer in the next 28 days?

- Categories are chosen by a fixed rule on data before day 365: ranks 3, 6, 12, 24 and 48 by distinct buyers. Rank 1, fluid milk, was the v1 target. The resulting categories are BAKED BREAD/BUNS/ROLLS, BEEF, CONDIMENTS/SAUCES, MILK BY-PRODUCTS and YOGURT.
- Cutoffs: train 365–477, validation 533 and 561, test 617, 645 and 673.
- Test sample: 500 households (1,456 household-cutoff rows, 7,280 labels), sampled by hash. The 205 households used in v1 are excluded.
- Validation sample: 250 households (490 rows).

**Study B: coupon-campaign response.** Will a household that received a campaign redeem at least one of its coupons?

- Each household–campaign pair is one row. The prediction day is the campaign start.
- Split by campaign start day:
  - Train: before day 500.
  - Validation: days 500–574, excluding campaigns that run into the test period. Campaign 15 is excluded for this reason.
  - Test: day 575 onward (campaigns 17–25).
- Test covers all 2,512 pairs. Validation uses a hash sample of 400 households (539 pairs). Households inactive in the prior 84 days are excluded (35 pairs).

## Conditions

All conditions see only records dated before the prediction day.

| Condition | Input |
|---|---|
| Prevalence | Training base rate |
| Logistic, gradient boosting | Engineered household features. Study A adds category history; Study B adds campaign overlap and past coupon use. Trained on the train split only, with fixed hyperparameters (`v2_analysis.GBDT`). |
| LLM · structured | Item-level shopping record: demographic codes, trips, spend, private-label, promotion and coupon shares, department mix, the top 40 categories with trips and recency, and the target categories always shown. Study B also gets the prior coupon-campaign record and a campaign description that is the same for every household. |
| LLM · persona | A 150–200 word persona generated from exactly that record. The forecast then sees only the persona, plus the same campaign description in Study B. |
| Calibrated LLM | Platt scaling of each LLM condition, fit on the validation sample, separately per category. |
| Hybrid | Logistic stack of the gradient-boosting and LLM logits, fit on validation. |

- **Models.** Main: `gpt-5.4-mini-2026-03-17`. Robustness: `gpt-5.5-2026-04-23` on a hash subset of 40 Study A households and 100 Study B households.
- **Model settings.** Reasoning effort `none`, strict JSON schema, one draw per request, OpenAI Batch API.
- **Prompts.** `src/behaviorlens/prompts.py`. Both LLM conditions share the task wording and the calibration instruction; only the evidence differs.

## Hypotheses and contrasts

Brier score is the primary metric. Each contrast is model minus reference; negative favours the model. Intervals come from a 5,000-draw household-cluster percentile bootstrap with seed 42.

| ID | Contrast | Interval |
|---|---|---|
| **H1** (co-primary, both studies) | persona − structured | 97.5% (Bonferroni over the two studies) |
| H2 | hybrid − gradient boosting: does the LLM add information to a strong tabular model? | 95% |
| H3 | calibrated persona − calibrated structured: is any persona gap only miscalibration? | 95% |
| E1, E2 | Exploratory: structured − gradient boosting; persona hybrid − gradient boosting | 95% |

- A contrast is called supported when its interval excludes zero. A difference under 0.005 Brier is described as practically small, whatever its interval.
- **Population fidelity (descriptive).**
  - Study A: mean predicted versus observed rate per category and cutoff.
  - Study B: the same per campaign, with Spearman rank correlation across the nine test campaigns. This asks whether simulated customers can rank campaigns the way real customers did.
- **Secondary metrics.** Log loss, AUC, average precision, and a reliability–resolution decomposition.
- **Robustness.** The gpt-5.5 conditions are compared with gpt-5.4-mini on identical rows.

## Failures and stopping

- Failed or unparsable requests are not retried.
- A row missing any LLM condition is removed from every condition, and the count is reported. If more than 2% of rows fail, this is reported as a major limitation.
- The full samples are run. There is no early stopping and no prompt changes.

## Budget

- Hard application cap: `BEHAVIORLENS_V2_CAP_USD`. A batch is refused if its token-exact upper bound plus everything already committed exceeds the cap.
- Expected cost: about $12. Worst-case bound: about $20.
- Prices are OpenAI Batch-tier, checked on 2026-09-22.

## What was seen before freezing

- All v1 results.
- v2 validation-set metrics of the supervised baselines. These led to two changes, both made to improve validation size and baseline stability:
  - The Study B validation boundary moved from day 531 to day 500, because the earlier split had 37 positives.
  - Stronger gradient-boosting regularization.
- Two single test API calls to confirm parameter support.
- An end-to-end dry run with shuffled labels and synthetic LLM outputs.

No v2 test-set outcome was inspected for any condition.

## Fingerprints

| Artifact | SHA-256 |
|---|---|
| inputs.json (label-free prompts source) | `93220f7e1120f8887df75c2b29731d060baf4ffe65d9bebbc1aa38dedac434d2` |
| labels.json | `d9eaa7b666c756c78370dde377b5bbec2be0e2ee77e7cafbd5c15421b028ff40` |
| frames.pkl | `fa892a4620daef511debe72ef002af32af58361a090734784e59618da3a769a3` |

## Known limitations

- One retailer, 2009–2011-era anonymised data.
- The LLM may have seen this public dataset during training.
- Demographic codes are anonymised bands.
- Study A absence of purchase means no record at this retailer.
- TypeA coupon assignment depends on past purchases in ways the data does not reveal.
- One draw per condition, so model variance is not measured.
