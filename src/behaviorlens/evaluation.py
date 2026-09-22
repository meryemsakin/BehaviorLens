"""Strict binary prediction evaluation; no model or network dependency."""
import math
import random
from collections import defaultdict
from datetime import date


def validate(rows):
    if not rows:
        raise ValueError("No observations")
    seen = set()
    for r in rows:
        key = (r['household_id'], r['cutoff'])
        if not r['household_id'] or key in seen:
            raise ValueError("Empty household or duplicate household/cutoff")
        seen.add(key)
        if r['y'] not in ('0', '1', 0, 1):
            raise ValueError("Labels must be binary")
        for field in ('probability', 'baseline_probability'):
            p = float(r[field])
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f"Invalid {field}")
        history, cutoff, end = map(date.fromisoformat,
                                   (r['history_end'], r['cutoff'], r['label_end']))
        if not history < cutoff < end:
            raise ValueError("Require history_end < cutoff < label_end")


def metrics(rows, field='probability', bins=10):
    if not rows or bins < 1:
        raise ValueError("Nonempty rows and positive bins required")
    pairs = [(int(r['y']), float(r[field])) for r in rows]
    n = len(pairs)
    buckets = [[] for _ in range(bins)]
    for y, p in pairs:
        buckets[min(int(p * bins), bins - 1)].append((y, p))
    curve = []
    for i, bucket in enumerate(buckets):
        if bucket:
            curve.append(dict(bin=i, n=len(bucket), predicted=sum(p for _, p in bucket)/len(bucket),
                              observed=sum(y for y, _ in bucket)/len(bucket)))
    return dict(n=n, households=len({r['household_id'] for r in rows}),
                brier=sum((p-y)**2 for y, p in pairs)/n,
                log_loss=-sum(y*math.log(max(p, 1e-15)) + (1-y)*math.log(max(1-p, 1e-15))
                              for y, p in pairs)/n,
                ece=sum(b['n']*abs(b['predicted']-b['observed']) for b in curve)/n,
                prevalence=sum(y for y, _ in pairs)/n,
                mean_probability=sum(p for _, p in pairs)/n,
                calibration=curve)


def paired_bootstrap(rows, repeats=2000, seed=42):
    """Resample households, retaining all their rows and both predictions.

    Estimand: observation-weighted Brier difference (model minus baseline).
    Negative differences favor the model. Does not account for shared time shocks.
    """
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    groups = defaultdict(list)
    for r in rows:
        y = int(r['y'])
        groups[r['household_id']].append((float(r['probability'])-y)**2 -
                                          (float(r['baseline_probability'])-y)**2)
    if len(groups) < 2:
        return dict(ci95=None, reason='Fewer than two households')
    aggregates = [(sum(v), len(v)) for _, v in sorted(groups.items())]
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        selected = rng.choices(aggregates, k=len(aggregates))
        samples.append(sum(s for s, _ in selected)/sum(n for _, n in selected))
    samples.sort()
    return dict(ci95=[samples[int(.025*(repeats-1))], samples[int(.975*(repeats-1))]],
                seed=seed, repeats=repeats, unit='household', method='percentile')


def evaluate(rows, repeats=2000, seed=42):
    validate(rows)
    model, baseline = metrics(rows), metrics(rows, 'baseline_probability')
    groups = defaultdict(list)
    for r in rows:
        groups[r.get('segment') or 'unknown'].append(r)
    return dict(model=model, baseline=baseline,
                brier_difference=model['brier']-baseline['brier'],
                uncertainty=paired_bootstrap(rows, repeats, seed),
                subgroups={k: dict(metrics=metrics(v), descriptive_only=True)
                           for k, v in sorted(groups.items())},
                limitations=['Timestamp checks use supplied metadata; they do not prove feature provenance.',
                             'Intervals do not account for shared temporal shocks.',
                             'Subgroups are descriptive; no multiple-comparison or small-sample claims.',
                             'One binary outcome does not establish population representativeness.'])
