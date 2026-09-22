"""Frozen v2 analysis plan, written before any v2 LLM output existed. See docs/PREREGISTRATION_V2.md."""
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from .batch import results
from .mechanism import murphy

SEED, REPEATS, EPS = 42, 5000, 1e-3
GBDT = dict(max_iter=100, max_leaf_nodes=7, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED)
CONTRASTS = [  # (name, model, reference, confidence); model minus reference Brier, negative favours model
    ('H1 persona vs structured LLM', 'persona', 'structured', .975),
    ('H2 hybrid vs gradient boosting', 'hybrid', 'gradient_boosting', .95),
    ('H3 calibrated persona vs calibrated structured', 'persona_cal', 'structured_cal', .95),
    ('E1 structured LLM vs gradient boosting', 'structured', 'gradient_boosting', .95),
    ('E2 persona hybrid vs gradient boosting', 'hybrid_persona', 'gradient_boosting', .95),
]


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1-EPS)
    return np.log(p/(1-p))


def fit_logistic(columns, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=2000).fit(np.column_stack(columns), y)


def cluster_ci(households, loss_diff, confidence, repeats=REPEATS, seed=SEED):
    """Household-cluster percentile bootstrap of a mean per-label loss difference."""
    groups = defaultdict(lambda: [0., 0])
    for h, d in zip(households, loss_diff):
        groups[h][0] += d
        groups[h][1] += 1
    sums = np.array([v[0] for v in groups.values()])
    counts = np.array([v[1] for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(repeats, len(sums)))
    stats = sums[draws].sum(1) / counts[draws].sum(1)
    tail = (1-confidence)/2
    return dict(estimate=float(sums.sum()/counts.sum()), ci=[float(np.quantile(stats, tail)), float(np.quantile(stats, 1-tail))],
                confidence=confidence, households=len(sums), labels=int(counts.sum()), repeats=repeats, seed=seed)


def metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y, p = np.asarray(y), np.asarray(p, float)
    pc = np.clip(p, 1e-15, 1-1e-15)
    m = murphy(list(zip(y.tolist(), p.tolist())))
    return dict(n=len(y), prevalence=float(y.mean()), mean_prediction=float(p.mean()), brier=float(((p-y)**2).mean()),
                log_loss=float(-(y*np.log(pc)+(1-y)*np.log(1-pc)).mean()), auc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)), **m)


def baselines(frames, study, categories):
    """Train-only fits; returns id -> {condition: probability} per label key for validation and test frames."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {split: defaultdict(dict) for split in ('validation', 'test')}
    keys = categories if study == 'A' else [None]
    for key in keys:
        get_x = (lambda r: r['x'][key]) if key else (lambda r: r['x'])
        get_y = (lambda r: r['y'][key]) if key else (lambda r: r['y'])
        train = frames[f'{study}_train']
        names = sorted(get_x(train[0]))
        X = [[get_x(r)[n] for n in names] for r in train]
        y = [get_y(r) for r in train]
        models = dict(logistic=make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                      gradient_boosting=HistGradientBoostingClassifier(**GBDT))
        for m in models.values():
            m.fit(X, y)
        rate = sum(y)/len(y)
        for split in out:
            rows = frames[f'{study}_{split}']
            Xs = [[get_x(r)[n] for n in names] for r in rows]
            preds = {n: m.predict_proba(Xs)[:, 1] for n, m in models.items()}
            for i, r in enumerate(rows):
                out[split][r['id']][key or 'probability'] = dict(prevalence=rate, **{n: float(v[i]) for n, v in preds.items()})
    return out


def llm_predictions(stage1, stage2):
    """custom-id results -> id -> {'structured': {key: p}, 'persona': {key: p}}; failures listed separately."""
    s1, s2 = results(stage1), results(stage2) if Path(stage2).exists() else {}
    preds, failures = defaultdict(dict), defaultdict(list)
    for cid, value in list(s1.items()) + list(s2.items()):
        rid, kind = cid.rsplit('|', 1)
        if isinstance(value, str) and value.startswith('error:'):
            failures[kind].append(rid)
        elif kind in ('structured', 'persona'):
            preds[rid][kind] = value
    return preds, {k: sorted(v) for k, v in failures.items()}


def long_table(ids, labels, base, llm, keys):
    """One record per (row, label key) with every condition's probability; rows missing any LLM value are dropped."""
    table, dropped = [], []
    for rid in ids:
        p = llm.get(rid, {})
        if 'structured' not in p or 'persona' not in p:
            dropped.append(rid)
            continue
        for key in keys:
            y = labels[rid][key] if isinstance(labels[rid], dict) else labels[rid]
            bkey = key if isinstance(labels[rid], dict) else 'probability'
            table.append(dict(id=rid, household=rid.split(':')[1], key=key, y=int(y),
                              structured=float(p['structured'][key]), persona=float(p['persona'][key]),
                              **base[rid][bkey]))
    return table, dropped


def add_fitted(test, validation):
    """Platt calibration and stacked hybrids, fit on validation only (per label key), applied to test."""
    for key in sorted({r['key'] for r in test}):
        v = [r for r in validation if r['key'] == key]
        t = [r for r in test if r['key'] == key]
        yv = [r['y'] for r in v]
        for name, cols in [('structured_cal', ['structured']), ('persona_cal', ['persona']),
                           ('hybrid', ['gradient_boosting', 'structured']), ('hybrid_persona', ['gradient_boosting', 'persona'])]:
            model = fit_logistic([logit([r[c] for r in v]) for c in cols], yv)
            fitted = model.predict_proba(np.column_stack([logit([r[c] for r in t]) for c in cols]))[:, 1]
            for r, p in zip(t, fitted):
                r[name] = float(p)
    return test


CONDITIONS = ['prevalence', 'logistic', 'gradient_boosting', 'structured', 'persona', 'structured_cal', 'persona_cal',
              'hybrid', 'hybrid_persona']


def summarize(table):
    y = [r['y'] for r in table]
    out = dict(conditions={c: metrics(y, [r[c] for r in table]) for c in CONDITIONS if c in table[0]}, contrasts={})
    for name, model, ref, conf in CONTRASTS:
        if model in table[0] and ref in table[0]:
            diff = [(r[model]-r['y'])**2 - (r[ref]-r['y'])**2 for r in table]
            out['contrasts'][name] = dict(model=model, reference=ref, **cluster_ci([r['household'] for r in table], diff, conf))
    return out


def fidelity(table, group):
    """Population-level fidelity: per-group mean prediction vs observed rate, MAE and Spearman rank correlation."""
    from scipy.stats import spearmanr
    groups = defaultdict(list)
    for r in table:
        groups[group(r)].append(r)
    observed = {g: float(np.mean([r['y'] for r in rs])) for g, rs in groups.items()}
    out = dict(groups={g: dict(n=len(rs), observed=observed[g]) for g, rs in groups.items()}, conditions={})
    for c in CONDITIONS:
        if c not in table[0]:
            continue
        pred = {g: float(np.mean([r[c] for r in rs])) for g, rs in groups.items()}
        for g in groups:
            out['groups'][g][c] = pred[g]
        keys = sorted(groups)
        rho = spearmanr([pred[g] for g in keys], [observed[g] for g in keys]).statistic if len(keys) > 2 else None
        out['conditions'][c] = dict(mae=float(np.mean([abs(pred[g]-observed[g]) for g in keys])),
                                    spearman=None if rho is None or math.isnan(rho) else float(rho))
    return out


def analyze(out, main, robust):
    out = Path(out)
    inputs = json.loads((out/'inputs.json').read_text())
    labels = json.loads((out/'labels.json').read_text())
    with (out/'frames.pkl').open('rb') as f:
        frames = pickle.load(f)
    cats = inputs['categories']
    llm, failures = llm_predictions(out/f'stage1_{main}', out/f'stage2_{main}')
    report = dict(models=dict(main=main, robustness=robust), failures={k: len(v) for k, v in failures.items()}, studies={})
    for study, keys in (('A', cats), ('B', ['probability'])):
        base = baselines(frames, study, cats)
        ids = lambda s: [r['id'] for r in inputs['data'][f'{study}_{s}']]
        val, drop_v = long_table(ids('validation'), labels[f'{study}_validation'], base['validation'], llm, keys)
        test, drop_t = long_table(ids('test'), labels[f'{study}_test'], base['test'], llm, keys)
        add_fitted(test, val)
        s = summarize(test)
        s.update(dropped=dict(validation=len(drop_v), test=len(drop_t)), rows=len({r['id'] for r in test}))
        if study == 'A':
            s['by_category'] = {k: summarize([r for r in test if r['key'] == k]) for k in keys}
            s['fidelity'] = fidelity(test, lambda r: f"{r['key']} @ day {r['id'].split(':')[2]}")
        else:
            campaign = {r['id']: r['campaign'] for r in inputs['data']['B_test']}
            s['fidelity'] = fidelity(test, lambda r: campaign[r['id']])
        report['studies'][study] = s
        with (out/f'table_{study}.pkl').open('wb') as f:
            pickle.dump(test, f)
    if robust and (out/f'stage1_{robust}').exists():
        rllm, rfail = llm_predictions(out/f'stage1_{robust}', out/f'stage2_{robust}')
        report['robustness'] = dict(failures={k: len(v) for k, v in rfail.items()})
        for study, keys in (('A', cats), ('B', ['probability'])):
            base = baselines(frames, study, cats)['test']
            rid = inputs['robust'][f'{study}_test']
            main_t, _ = long_table(rid, labels[f'{study}_test'], base, llm, keys)
            rob_t, _ = long_table(rid, labels[f'{study}_test'], base, rllm, keys)
            common = {r['id'] for r in main_t} & {r['id'] for r in rob_t}
            main_t = [r for r in main_t if r['id'] in common]
            rob_t = [r for r in rob_t if r['id'] in common]
            report['robustness'][study] = dict(
                rows=len(common), main=summarize(main_t), robust=summarize(rob_t),
                robust_minus_main_structured=cluster_ci([r['household'] for r in rob_t],
                    [(a['structured']-a['y'])**2-(b['structured']-b['y'])**2 for a, b in zip(rob_t, main_t)], .95),
                robust_minus_main_persona=cluster_ci([r['household'] for r in rob_t],
                    [(a['persona']-a['y'])**2-(b['persona']-b['y'])**2 for a, b in zip(rob_t, main_t)], .95))
    (out/'results_v2.json').write_text(json.dumps(report, indent=2))
    return report
