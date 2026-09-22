"""Budget-guarded expansion of the fixed pilot, with durable concurrent reservations."""
import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .openai_pilot import MODEL, INPUT_PRICE, OUTPUT_PRICE, read_env, payload, reserve_bound, parse_response


class Budget:
    def __init__(self, cap, prior, path):
        if not 0 <= prior < cap <= .5:
            raise ValueError('Invalid budget')
        self.cap, self.spent, self.pending = cap, prior, {}
        self.path=Path(path)
        self.lock=threading.Lock()
        self.count=0

    def log(self, event):
        with self.path.open('a') as f:
            f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())

    def reserve(self, amount, pid, condition):
        with self.lock:
            if amount <= 0 or self.spent+sum(self.pending.values())+amount > self.cap:
                raise ValueError('Budget limit reached')
            self.count+=1
            key=self.count
            self.pending[key]=amount
            self.log(dict(event='reserved',call=key,amount=amount,prediction_id=pid,condition=condition))
            return key

    def settle(self, key, cost, metadata):
        with self.lock:
            bound=self.pending[key]
            if not 0 <= cost <= bound:
                raise ValueError('Usage outside reserved bound')
            self.log(dict(event='settled',call=key,cost=cost,**metadata))
            self.spent+=cost
            del self.pending[key]


def prepare_expansion(baseline, pilot_plan, output):
    from .prepare_llm import prepare
    output=Path(output)
    prepare(baseline,output,205)
    old={json.loads(x)['prediction_id'].split(':')[0] for x in (Path(pilot_plan)/'requests.jsonl').read_text().splitlines()}
    path=output/'requests.jsonl'
    requests=[json.loads(x) for x in path.read_text().splitlines()]
    requests=[r for r in requests if r['prediction_id'].split(':')[0] not in old]
    ids={r['prediction_id'].split(':')[0] for r in requests}
    if len(ids)!=200:raise ValueError('Expected exactly 200 nonpilot households')
    path.write_text(''.join(json.dumps(r)+'\n' for r in requests))
    p=json.loads((output/'plan.json').read_text())
    p.update(sample_households=len(ids),prediction_points=len(requests),expected_calls=3*len(requests),
             pilot_households_excluded=len(old),purpose='exploratory expansion; pilot and aggregate baseline outcomes already inspected',
             primary_comparison='persona minus structured Brier',secondary_comparison='persona minus gradient boosting Brier',
             interval='paired household percentile bootstrap, 5000 replicates, seed 42; marginal intervals, no familywise correction',
             practical_margin_brier=.01,stop_rule='complete fixed sample or report incomplete; no outcome-based early stopping',
             prompt_changes_from_pilot=False,stability='not evaluated; one draw at temperature 0')
    (output/'plan.json').write_text(json.dumps(p,indent=2))
    return p


def run(plan, output, pilot, workers=6):
    api_key,cap=read_env()
    plan,output,pilot=Path(plan),Path(output),Path(pilot)
    prior=json.loads((pilot/'summary.json').read_text())
    if prior['status']!='completed':raise ValueError('Resolve prior uncertain cost first')
    requests=[json.loads(x) for x in (plan/'requests.jsonl').read_text().splitlines()]
    if len(requests)>600 or not 1<=workers<=6:raise ValueError('Expansion size or concurrency too large')
    output.mkdir(parents=True,exist_ok=False)
    budget=Budget(cap,prior['usage_cost_usd'],output/'ledger.jsonl')
    stop=threading.Event()
    file_lock=threading.Lock()
    (output/'protocol.json').write_text(json.dumps(dict(model=MODEL,workers=workers,cap_including_pilot=cap,
         prior_cost=budget.spent,plan=json.loads((plan/'plan.json').read_text()),
         requests_sha256=hashlib.sha256((plan/'requests.jsonl').read_bytes()).hexdigest(),
         code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         prices={'input_per_million':.4,'output_per_million':1.6},retries=0),indent=2))

    def call(pid,condition,prompt,persona=False):
        if stop.is_set():raise RuntimeError('Run stopped')
        body=payload(prompt,persona)
        call_id=budget.reserve(reserve_bound(body),pid,condition)
        stem=output/f'call_{call_id:04d}'
        stem.with_suffix('.request.json').write_text(json.dumps(body,indent=2))
        start=time.monotonic()
        try:
            request=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),
                headers={'Authorization':'Bearer '+api_key,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=45) as response:raw=json.load(response)
            stem.with_suffix('.response.json').write_text(json.dumps(raw,indent=2))
            usage=raw['usage']; cost=usage['input_tokens']*INPUT_PRICE+usage['output_tokens']*OUTPUT_PRICE
            budget.settle(call_id,cost,dict(usage=usage,latency_seconds=time.monotonic()-start,prediction_id=pid,condition=condition))
            result=parse_response(raw,persona)
            return result,call_id
        except Exception as e:
            stop.set()
            with budget.lock:budget.log(dict(event='failed',call=call_id,prediction_id=pid,condition=condition,
                                           error_type=type(e).__name__,http_status=getattr(e,'code',None)))
            raise RuntimeError('Request failed; no automatic retry') from None

    def point(r):
        if stop.is_set():return None
        pid=r['prediction_id']
        try:
            structured,a=call(pid,'structured',r['structured_prompt'])
            persona,b=call(pid,'summary',r['persona_generation_prompt'],True)
            prob,c=call(pid,'persona',r['persona_prediction_prefix']+'\nPersona evidence:\n'+persona)
            record=dict(prediction_id=pid,source_sha256=r['source_sha256'],structured=structured,persona=prob,
                        calls=dict(structured=a,summary=b,persona=c))
            with file_lock:
                with (output/'predictions.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            return record
        except Exception:
            stop.set()
            return None
    done=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(point,r) for r in requests]
        for f in as_completed(futures):
            result=f.result()
            if result is not None:
                done.append(result)
                if len(done)%20==0:print(f'{len(done)}/{len(requests)} points; total incl. pilot ${budget.spent:.4f}',flush=True)
    summary=dict(status='completed' if len(done)==len(requests) else 'incomplete',planned_points=len(requests),
                 completed_points=len(done),calls=budget.count,usage_cost_usd=budget.spent-prior['usage_cost_usd'],
                 cumulative_cost_usd=budget.spent,pending_reserved_usd=sum(budget.pending.values()),cap_usd=cap,
                 note='Usage-based standard-price estimate; not account balance. Uncertain requests keep reservations.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--pilot',default='outputs/openai_pilot_v1');a=p.parse_args()
    run(a.plan,a.output,a.pilot)
