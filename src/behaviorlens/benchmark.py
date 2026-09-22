"""Run fixed baseline protocol; persist one source of truth for reports."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .journey import FEATURES, fingerprint, load, make_rows, select_category, validate_protocol
from .evaluation import evaluate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(raw, protocol_path, output):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score
    p = json.loads(Path(protocol_path).read_text())
    validate_protocol(p)
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists. Use a new run directory to preserve evidence.')
    output.mkdir(parents=True)
    # Persist exact protocol before loading outcomes or selecting the category.
    write_json(output/'protocol.json', p)
    households, demographics, audit = load(raw)
    if max(p['test_cutoffs'])+p['horizon_days']-1 > audit['day_max']:
        raise ValueError('Test label horizon exceeds dataset coverage')
    category = select_category(households, p['category_selection_before_day'])
    write_json(output/'selection.json', dict(category=category, selection_before=p['category_selection_before_day']))
    splits = {s: make_rows(households, demographics, p[s+'_cutoffs'], category,
                           p['horizon_days'], p['history_days']) for s in ('train', 'validation', 'test')}
    audit['splits'] = {s: dict(n=len(rs), households=len({r['household_id'] for r in rs}),
                              prevalence=sum(r['y'] for r in rs)/len(rs)) for s, rs in splits.items()}
    write_json(output/'data_audit.json', audit)
    train = splits['train']
    x = [[r['features'][f] for f in FEATURES] for r in train]
    y = [r['y'] for r in train]
    prevalence = sum(y)/len(y)
    models = dict(logistic=make_pipeline(StandardScaler(), LogisticRegression(**p['models']['logistic'], random_state=p['seed'])),
                  gradient_boosting=HistGradientBoostingClassifier(**p['models']['gradient_boosting'],
                                                                 early_stopping=False, random_state=p['seed']))
    for model in models.values():
        model.fit(x, y)
    results = {}
    for split in ('validation', 'test'):
        rs = splits[split]
        matrix = [[r['features'][f] for f in FEATURES] for r in rs]
        results[split] = {}
        for name, probabilities in [('prevalence', [prevalence]*len(rs))] + [
                (name, model.predict_proba(matrix)[:, 1].tolist()) for name, model in models.items()]:
            predictions = [{**{k: v for k, v in r.items() if k != 'features'},
                            'probability': prob, 'baseline_probability': prevalence}
                           for r, prob in zip(rs, probabilities)]
            result = evaluate(predictions, p['bootstrap_repeats'], p['seed'])
            result['average_precision'] = float(average_precision_score([r['y'] for r in rs], probabilities))
            results[split][name] = result
            with (output/f'{split}_{name}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
        with (output/f'{split}_snapshots.jsonl').open('w') as f:
            for r in rs:
                # Outcomes are separate from simulator inputs, never part of a snapshot.
                snapshot = {k: r[k] for k in ['household_id', 'cutoff', 'history_end', 'features', 'segment']}
                payload = json.dumps(snapshot, sort_keys=True)
                f.write(json.dumps(dict(snapshot=snapshot, sha256=hashlib.sha256(payload.encode()).hexdigest()))+'\n')
    write_json(output/'metrics.json', results)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), category=category,
                    evidence='retailer transaction representation from official Complete Journey package',
                    protocol_sha256=fingerprint(protocol_path), source=audit['source_fingerprints'],
                    code_sha256={f.name:fingerprint(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
                    git_commit=revision, python=platform.python_version(), sklearn=sklearn.__version__,
                    llm_status='not run', model_selection='fixed parameters; no test tuning',
                    artifacts={f.name:fingerprint(f) for f in sorted(output.iterdir()) if f.is_file()})
    write_json(output/'manifest.json', manifest)
    from .reporting import render
    render(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--protocol', type=Path, default=Path('configs/complete_journey_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.raw, args.protocol, args.output))


if __name__ == '__main__':
    main()
