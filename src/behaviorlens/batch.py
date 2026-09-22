"""OpenAI Batch API runner with a hard pre-submission cost bound. No automatic retries; every file is kept."""
import json
import math
import time
import urllib.request
import uuid
from pathlib import Path
from .openai_pilot import parse_response

# Batch-tier USD per 1M tokens (input, output); verified at developers.openai.com/api/docs/pricing on 2026-09-22.
PRICES = {'gpt-5.4-mini-2026-03-17': (.375, 2.25), 'gpt-5.5-2026-04-23': (2.5, 15.0)}
API = 'https://api.openai.com/v1'


def tokens(text):
    import tiktoken
    return len(tiktoken.get_encoding('o200k_base').encode(text))


def bound(body):
    """Upper cost bound: exact prompt tokens plus schema and framing allowance, and the full output ceiling."""
    p_in, p_out = PRICES[body['model']]
    n = tokens(body['instructions']) + tokens(body['input']) + tokens(json.dumps(body['text'])) + 64
    return (n*p_in + body['max_output_tokens']*p_out) / 1e6


def env(name, required=True):
    values = {}
    if Path('.env').exists():
        for line in Path('.env').read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    if required and not values.get(name):
        raise ValueError(f'Set {name} in .env')
    return values.get(name)


def call(method, path, data=None, content_type='application/json', raw=False):
    request = urllib.request.Request(API+path, data=data, method=method,
                                     headers={'Authorization': 'Bearer '+env('OPENAI_API_KEY'), 'Content-Type': content_type})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    return payload if raw else json.loads(payload)


def upload(path):
    boundary = uuid.uuid4().hex
    data = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{Path(path).name}"\r\n'
            'Content-Type: application/jsonl\r\n\r\n').encode() + Path(path).read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    return call('POST', '/files', data, 'multipart/form-data; boundary='+boundary)['id']


class Ledger:
    """Cumulative spend across all v2 batches: settled usage plus the bound of anything still in flight."""

    def __init__(self, path, cap):
        self.path, self.cap = Path(path), cap
        if not (math.isfinite(cap) and 0 < cap <= 25):
            raise ValueError('Cap must be between 0 and 25 USD')

    def entries(self):
        return [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []

    def committed(self):
        state = {}
        for e in self.entries():
            state[e['batch']] = e
        return sum(e['settled'] if e['event'] == 'settled' else e['bound'] for e in state.values())

    def log(self, event):
        with self.path.open('a') as f:
            f.write(json.dumps(event)+'\n')


def submit(requests, stage_dir, ledger, dry_run=False):
    """requests: {custom_id: body}; one model per batch. Refuses if the bound would exceed the cap."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=False)
    models = {b['model'] for b in requests.values()}
    if len(models) != 1:
        raise ValueError('One model per batch')
    lines = [json.dumps(dict(custom_id=k, method='POST', url='/v1/responses', body=b)) for k, b in requests.items()]
    (stage_dir/'input.jsonl').write_text('\n'.join(lines)+'\n')
    total = sum(bound(b) for b in requests.values())
    info = dict(model=models.pop(), requests=len(requests), bound_usd=total, committed_before=ledger.committed(), cap=ledger.cap)
    if info['committed_before'] + total > ledger.cap:
        (stage_dir/'refused.json').write_text(json.dumps(info, indent=2))
        raise ValueError(f"Bound ${total:.2f} + committed ${info['committed_before']:.2f} exceeds cap ${ledger.cap:.2f}")
    if dry_run:
        (stage_dir/'dry_run.json').write_text(json.dumps(info, indent=2))
        return info
    file_id = upload(stage_dir/'input.jsonl')
    batch = call('POST', '/batches', json.dumps(dict(input_file_id=file_id, endpoint='/v1/responses',
                                                     completion_window='24h')).encode())
    info.update(batch=batch['id'], input_file=file_id, submitted=time.time())
    (stage_dir/'batch.json').write_text(json.dumps(info, indent=2))
    ledger.log(dict(event='submitted', batch=batch['id'], stage=str(stage_dir), bound=total))
    return info


def wait(stage_dir, ledger, poll=60):
    stage_dir = Path(stage_dir)
    info = json.loads((stage_dir/'batch.json').read_text())
    while True:
        batch = call('GET', '/batches/'+info['batch'])
        (stage_dir/'status.json').write_text(json.dumps(batch, indent=2))
        if batch['status'] in ('completed', 'failed', 'expired', 'cancelled'):
            break
        time.sleep(poll)
    for key, name in (('output_file_id', 'output.jsonl'), ('error_file_id', 'errors.jsonl')):
        if batch.get(key):
            (stage_dir/name).write_bytes(call('GET', f'/files/{batch[key]}/content', raw=True))
    cost = 0.
    p_in, p_out = PRICES[info['model']]
    if (stage_dir/'output.jsonl').exists():
        for line in (stage_dir/'output.jsonl').read_text().splitlines():
            body = json.loads(line)['response'].get('body') or {}
            usage = body.get('usage') or {}
            cost += (usage.get('input_tokens', 0)*p_in + usage.get('output_tokens', 0)*p_out) / 1e6
    ledger.log(dict(event='settled', batch=info['batch'], stage=str(stage_dir), status=batch['status'],
                    counts=batch.get('request_counts'), settled=cost, bound=info['bound_usd']))
    return batch['status'], cost


def results(stage_dir, persona=False):
    """custom_id -> parsed value, or an error string. Parsing failures are kept, never dropped silently."""
    out = {}
    for name in ('output.jsonl', 'errors.jsonl'):
        path = Path(stage_dir)/name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            response = r.get('response') or {}
            if response.get('status_code') != 200:
                out[r['custom_id']] = f"error:{response.get('status_code')}"
                continue
            try:
                kind = r['custom_id'].rsplit('|', 1)[1]
                out[r['custom_id']] = parse(response['body'], kind == 'persona_gen')
            except Exception as e:
                out[r['custom_id']] = f'error:{type(e).__name__}'
    return out


def parse(raw, persona):
    if persona:
        return parse_response(raw, True)
    if raw.get('status') != 'completed':
        raise ValueError('Response did not complete')
    text = ''.join(c.get('text', '') for item in raw.get('output', []) for c in item.get('content', []) or []
                   if c.get('type') == 'output_text')
    value = json.loads(text)
    probs = value['probabilities'] if 'probabilities' in value else {'probability': value['probability']}
    for p in probs.values():
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid probability')
    return probs
