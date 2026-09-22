"""Small, sequential OpenAI pilot. No retries; durable pre-call budget reservations."""
import argparse
import csv
import hashlib
import json
import math
import os
import time
import urllib.request
from pathlib import Path
from .evaluation import evaluate

MODEL = 'gpt-4.1-mini-2025-04-14'
INPUT_PRICE = .40 / 1_000_000
OUTPUT_PRICE = 1.60 / 1_000_000
MAX_OUTPUT = 512


def read_env():
    values = {}
    if Path('.env').exists():
        for line in Path('.env').read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1)
                values[k.strip()] = v.strip().strip('\"').strip("'")
    key = os.environ.get('OPENAI_API_KEY') or values.get('OPENAI_API_KEY')
    cap = float(values.get('BEHAVIORLENS_MAX_COST_USD', '.50'))
    if not key or not math.isfinite(cap) or not 0 < cap <= .50:
        raise ValueError('Configure a key and a finite pilot cap between 0 and 0.50 USD')
    return key, cap


def payload(prompt, persona=False):
    field, kind = ('persona', 'string') if persona else ('probability', 'number')
    schema = {'type':'object', 'properties':{field:{'type':kind}}, 'required':[field], 'additionalProperties':False}
    return dict(model=MODEL, input=prompt, temperature=0, max_output_tokens=MAX_OUTPUT,
                store=False, text={'format':{'type':'json_schema','name':'result','strict':True,'schema':schema}})


def reserve_bound(body):
    # UTF-8 bytes upper-bound text tokens; extra allowance covers framing/schema.
    size = len(json.dumps(body).encode())
    if size > 12000:
        raise ValueError('Pilot input too long')
    return (size+2048)*INPUT_PRICE + MAX_OUTPUT*OUTPUT_PRICE


def parse_response(raw, persona=False):
    if raw.get('status') != 'completed':
        raise ValueError('Response did not complete')
    text = ''.join(c.get('text','') for item in raw.get('output',[]) for c in item.get('content',[])
                   if c.get('type') == 'output_text')
    result = json.loads(text)
    if persona:
        if not isinstance(result.get('persona'),str) or not result['persona'].strip():
            raise ValueError('Missing persona')
        return result['persona']
    p = result.get('probability')
    if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Invalid probability')
    return p


def run(plan, output):
    key, cap = read_env()
    plan, output = Path(plan), Path(output)
    requests = [json.loads(x) for x in (plan/'requests.jsonl').read_text().splitlines()]
    if len(requests)>15:
        raise ValueError('Pilot allows at most 15 prediction points')
    output.mkdir(parents=True, exist_ok=False)
    reserved = 0.
    actual = 0.
    calls = 0
    records = []
    protocol = dict(model=MODEL, cap_usd=cap, max_output_tokens=MAX_OUTPUT, temperature=0,
                    price_per_million={'input':.4,'output':1.6},
                    pricing_source='https://developers.openai.com/api/docs/models/gpt-4.1-mini',
                    pricing_checked='2026-09-22', purpose='engineering pilot, not confirmatory evidence',
                    requests_sha256=hashlib.sha256((plan/'requests.jsonl').read_bytes()).hexdigest(),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    planned_points=len(requests), retries=0)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))

    def call(prediction_id, condition, prompt, persona=False):
        nonlocal reserved, actual, calls
        body = payload(prompt,persona)
        amount = reserve_bound(body)
        if reserved+amount > cap:
            raise ValueError('Pilot budget exhausted before request')
        reserved += amount
        calls += 1
        stem = output/f'call_{calls:03d}'
        stem.with_suffix('.request.json').write_text(json.dumps(body,indent=2))
        event = dict(call=calls,prediction_id=prediction_id,condition=condition,
                     reserved_usd=amount,cumulative_reserved_usd=reserved,status='reserved')
        with (output/'ledger.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n'); f.flush(); os.fsync(f.fileno())
        started=time.monotonic()
        req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),
                                   headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=45) as response:
                raw=json.load(response)
            stem.with_suffix('.response.json').write_text(json.dumps(raw,indent=2))
            usage=raw['usage']
            cost=usage['input_tokens']*INPUT_PRICE+usage['output_tokens']*OUTPUT_PRICE
            actual += cost
            event.update(status='received',latency_seconds=time.monotonic()-started,usage=usage,cost_usd=cost)
            with (output/'ledger.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            if cost > amount:
                raise ValueError('Cost exceeded conservative reservation; stopped')
            return parse_response(raw,persona)
        except Exception as error:
            # Exception bodies/headers are not logged: never expose credentials.
            event.update(status='failed',error_type=type(error).__name__,http_status=getattr(error,'code',None))
            with (output/'ledger.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            raise RuntimeError('Pilot stopped; inspect redacted ledger. No automatic retry.') from None

    status='completed'
    try:
        for i,r in enumerate(requests):
            pid=r['prediction_id']
            history=call(pid,'structured',r['structured_prompt'])
            persona=call(pid,'summary',r['persona_generation_prompt'],True)
            probability=call(pid,'persona',r['persona_prediction_prefix']+'\nPersona evidence:\n'+persona)
            record=dict(prediction_id=pid,source_sha256=r['source_sha256'],structured=history,persona=probability)
            records.append(record)
            with (output/'predictions.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(f'Completed {i+1}/{len(requests)} points; estimated usage cost ${actual:.5f}',flush=True)
    except (RuntimeError,ValueError) as error:
        status='stopped'
        print(str(error),flush=True)
    summary=dict(status=status,planned_points=len(requests),completed_points=len(records),calls=calls,
                 usage_cost_usd=actual,reserved_upper_estimate_usd=reserved,budget_usd=cap,
                 note='Usage-derived estimate at verified prices, not account billing balance. Failed/uncertain calls retain reservations.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def score(run_path, output):
    output, run_path=Path(output),Path(run_path)
    summary=json.loads((output/'summary.json').read_text())
    if summary['status']!='completed':
        raise ValueError('Incomplete pilot: do not silently score only successful points')
    preds=[json.loads(x) for x in (output/'predictions.jsonl').read_text().splitlines()]
    ids={r['prediction_id'] for r in preds}
    indexed={r['prediction_id']:r for r in preds}
    results={}
    for name in ['prevalence','logistic','gradient_boosting','structured','persona']:
        source=name if name in ['prevalence','logistic','gradient_boosting'] else 'gradient_boosting'
        with (run_path/f'test_{source}.csv').open() as f:
            rows=[r for r in csv.DictReader(f) if r['household_id']+':'+r['cutoff'] in ids]
        if len(rows)!=len(ids):raise ValueError('Pilot and labels do not match')
        if name in ['structured','persona']:
            for r in rows:r['probability']=indexed[r['household_id']+':'+r['cutoff']][name]
        results[name]=evaluate(rows,repeats=5000)
    (output/'metrics.json').write_text(json.dumps(dict(status='engineering pilot only',results=results),indent=2))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--baseline-run',required=True)
    a=p.parse_args()
    summary=run(a.plan,a.output)
    if summary['status']=='completed':score(a.baseline_run,a.output)
    print(json.dumps(summary))
