"""v2 orchestration: prepare (outcome-blind) -> stage1 batch -> stage2 batch -> score.

Labels are written to a separate file at prepare time and are never read by request construction."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
from .batch import Ledger, submit, wait, results, env
from .prompts import requests_for, persona_request
from .studies import select_categories, study_a, study_b, sample, attach_history

MAIN, ROBUST = 'gpt-5.4-mini-2026-03-17', 'gpt-5.5-2026-04-23'
SAMPLES = dict(A_test=('A', 'test', 500, 'v2-A-test'), A_validation=('A', 'validation', 250, 'v2-A-validation'),
               B_test=('B', 'test', None, None), B_validation=('B', 'validation', 400, 'v2-B-validation'))
ROBUST_SAMPLES = dict(A_test=40, B_test=100)  # first households of the main test samples by the same hash order


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def v1_households():
    ids = set()
    for plan in ('outputs/expansion_plan_v1', 'outputs/llm_pilot_plan_v1'):
        path = Path(plan)/'requests.jsonl'
        if path.exists():
            ids |= {json.loads(x)['prediction_id'].split(':')[0] for x in path.read_text().splitlines()}
    return ids


def prepare(raw, out):
    from .household import Journey
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    journey = Journey(raw)
    categories, top = select_categories(journey)
    A = study_a(journey, categories)
    B, excluded = study_b(journey)
    data, labels, frame = {}, {}, {}
    exclude = v1_households()
    for name, (study, split, n, salt) in SAMPLES.items():
        rows = (A if study == 'A' else B)[split]
        rows = rows if n is None else sample(rows, n, salt, exclude if name == 'A_test' else ())
        attach_history(journey, rows, categories if study == 'A' else ())
        data[name] = [{k: v for k, v in r.items() if k not in ('y', 'x')} for r in rows]
        labels[name] = {r['id']: r['y'] for r in rows}
    for split in ('train', 'validation', 'test'):  # full frames for supervised baselines
        frame['A_'+split] = [dict(id=r['id'], household=r['household'], cutoff=r['cutoff'], x=r['x'], y=r['y']) for r in A[split]]
        frame['B_'+split] = [dict(id=r['id'], household=r['household'], campaign=r['campaign'], cutoff=r['cutoff'],
                                  x=r['x'], y=r['y']) for r in B[split]]
    robust = {}
    for name, n in ROBUST_SAMPLES.items():
        study, split, _, salt = SAMPLES[name]
        hh = sorted({r['household'] for r in data[name]}, key=lambda h: hashlib.sha256(f'v2-robust:{h}'.encode()).hexdigest())[:n]
        robust[name] = sorted(r['id'] for r in data[name] if r['household'] in set(hh))
    (out/'inputs.json').write_text(json.dumps(dict(categories=categories, data=data, robust=robust), indent=1))
    (out/'labels.json').write_text(json.dumps(labels))
    with (out/'frames.pkl').open('wb') as f:
        pickle.dump(frame, f)
    plan = dict(categories=categories, category_buyers_before_365=top, b_excluded=excluded,
                v1_households_excluded_from_A_test=len(exclude), models=dict(main=MAIN, robustness=ROBUST),
                rows={k: len(v) for k, v in data.items()}, households={k: len({r['household'] for r in v}) for k, v in data.items()},
                robust_rows={k: len(v) for k, v in robust.items()},
                inputs_sha256=sha(out/'inputs.json'), labels_sha256=sha(out/'labels.json'), frames_sha256=sha(out/'frames.pkl'))
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    return plan


def stage1(out):
    out = Path(out)
    inputs = json.loads((out/'inputs.json').read_text())
    reqs = {MAIN: {}, ROBUST: {}}
    for name, rows in inputs['data'].items():
        study = name[0]
        for r in rows:
            cats = inputs['categories'] if study == 'A' else None
            reqs[MAIN].update(requests_for(r, study, MAIN, cats))
            if r['id'] in set(inputs['robust'].get(name, [])):
                reqs[ROBUST].update(requests_for(r, study, ROBUST, cats))
    return reqs


def stage2(out, stage1_dirs):
    out = Path(out)
    inputs = json.loads((out/'inputs.json').read_text())
    reqs = {}
    for model, folder in stage1_dirs.items():
        personas = results(folder)
        reqs[model] = {}
        for name, rows in inputs['data'].items():
            study = name[0]
            for r in rows:
                persona = personas.get(r['id']+'|persona_gen')
                if isinstance(persona, str) and not persona.startswith('error:'):
                    reqs[model][r['id']+'|persona'] = persona_request(r, study, model, persona,
                                                                      inputs['categories'] if study == 'A' else None)
    return reqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'estimate', 'stage1', 'stage2', 'wait'])
    p.add_argument('--out', default='outputs/v2')
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--stage', help='stage folder for wait')
    a = p.parse_args()
    out = Path(a.out)
    if a.command == 'prepare':
        print(json.dumps(prepare(a.raw, out), indent=2))
        return
    if a.command == 'wait':
        cap = float(env('BEHAVIORLENS_V2_CAP_USD'))
        print(wait(a.stage, Ledger(out/'ledger.jsonl', cap)))
        return
    if a.command == 'estimate':
        from .batch import bound
        for model, rq in stage1(out).items():
            b1 = sum(bound(x) for x in rq.values())
            print(model, len(rq), 'stage1 bound $%.2f' % b1, '| stage2 approx bound $%.2f' % (b1*0.45))
        return
    cap = float(env('BEHAVIORLENS_V2_CAP_USD'))
    ledger = Ledger(out/'ledger.jsonl', cap)
    if a.command == 'stage1':
        for model, rq in stage1(out).items():
            print(submit(rq, out/f'stage1_{model}', ledger))
    else:
        dirs = {m: out/f'stage1_{m}' for m in (MAIN, ROBUST)}
        for model, rq in stage2(out, dirs).items():
            print(submit(rq, out/f'stage2_{model}', ledger))


if __name__ == '__main__':
    main()
