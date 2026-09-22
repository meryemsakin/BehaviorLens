# Complete Journey: source and scope

Official source: https://www.dunnhumby.com/source-files/

Acquired 2026-09-22 using the Complete Journey archive linked in that page's content. The separate “Let's Get Sort-of-Real” dataset was not used. The package contains a 2023 user guide describing a representation of transactions for frequent shoppers at one retailer. We evaluate the distributed transaction records; we do not certify that they are unmodified raw observations.

The guide describes classroom and academic research use and carries an “All rights reserved” notice. No broad redistribution license was found in the archive. Raw files and household-level derived artifacts remain excluded from Git. This card does not assert permission for commercial use or redistribution. Aggregate research results and code are stored separately.

## Interpretation

- DAY is a relative dataset day. No real calendar epoch is inferred.
- Positive quantity defines a purchase record. Nonpositive quantity rows are excluded and counted in the audit. Negative or zero sales with positive quantity are retained; sales are retailer receipts, not inferred consumer willingness to pay.
- Missing product joins are fatal instead of silently dropping transactions.
- Demographics include generic classification fields. They are not decoded as actual income, gender or age. This initial baseline uses only behavioral features.
- Eligibility uses a purchase in the 84 days strictly before the cutoff. It does not depend on future activity. Households with no future records remain negative for the retailer-specific outcome.
- Basket counts deduplicate basket IDs within the relevant household/window/category; line-item rows are retained when summing retailer sales.
- No data outside the observed retailer is available. Absence of a purchase record is not evidence of no purchase elsewhere.

## Fixed first experiment

The JSON protocol is saved before data ingestion. Category selection uses distinct household buyers within GROCERY before day 365 only. Hyperparameters are fixed, not selected on test performance. Train windows start on days 365–477, validation on 533 and 561, test on 617, 645 and 673. Every label window is 28 days, half-open: [cutoff, cutoff + 28).

The two model baselines use the same eight pre-cutoff behavioral features. Scaling is fitted only on training observations. This is prediction for recently active, potentially previously seen households, not a cold-start household experiment.

The manifest includes exact source, protocol and code hashes. Protocol files are reproducibility records, not external preregistration. An analyst could change them; any follow-up version after viewing results must be identified as such.
