# BehaviorLens — evidence card

**Question:** Does a persona summary preserve predictive signal?

**Headline:** The persona kept the facts but lost the signal.

**Finding:** Persona summaries increased prediction error relative to structured history; gradient boosting had a lower Brier score.

**Evidence:** 200 new households, 566 matched held-out predictions; one retailer and category; unchanged model and prompts; five pilot households excluded.

**Primary contrast:** Persona minus structured Brier +0.0210; 95% CI [+0.0097, +0.0326].

**Secondary contrast:** Persona minus GBDT Brier +0.0387; 95% CI [+0.0248, +0.0531].

**Where the signal went (descriptive):** 89% of source values kept verbatim; resolution 0.0820 → 0.0668 (structured → persona).

**Not claimed:** General population fidelity, causal mechanism, cross-model stability or commercial uplift.

**Inspect:** report.html → differences → local failure explorer → source hash and raw call records.

**Trace:** run `expansion_v1`; plan `71df9f32656a82b10c02357cb99159ba80f5f038b80a6b44b76f4414877cdd14`.
