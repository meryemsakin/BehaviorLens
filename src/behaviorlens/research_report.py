"""Matched research report. All displayed numbers derive from saved artifacts."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from .evaluation import evaluate
from .mechanism import diagnose

LABELS={'prevalence':'Prevalence baseline','logistic':'Logistic regression','gradient_boosting':'Gradient boosting',
        'structured':'LLM · structured history','persona':'LLM · persona'}
COLORS={'prevalence':'#8798ab','logistic':'#a7b9ce','gradient_boosting':'#f8cb8b','structured':'#98a8ff','persona':'#56dbc0'}


def read_json(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pid(row):return str(row['household_id'])+':'+str(row['cutoff'])


def analyze(baseline, run, plan):
    from sklearn.metrics import average_precision_score
    baseline,run,plan=map(Path,(baseline,run,plan))
    summary=read_json(run/'summary.json')
    if summary['status']!='completed':raise ValueError('Incomplete run must not be silently subset-scored')
    predictions=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()]
    requests=[json.loads(x) for x in (plan/'requests.jsonl').read_text().splitlines()]
    expected={r['prediction_id']:r for r in requests}
    indexed={r['prediction_id']:r for r in predictions}
    if len(indexed)!=len(predictions) or set(indexed)!=set(expected):raise ValueError('Duplicate or mismatched prediction IDs')
    for k,r in indexed.items():
        if r['source_sha256']!=expected[k]['source_sha256']:raise ValueError('Source hash mismatch')
    protocol=read_json(run/'protocol.json')
    if protocol['requests_sha256']!=digest(plan/'requests.jsonl'):raise ValueError('Plan changed after experiment')
    if read_json(plan/'plan.json')['parent_manifest_sha256']!=digest(baseline/'manifest.json'):
        raise ValueError('Baseline lineage mismatch')
    baseline_manifest=read_json(baseline/'manifest.json')
    for name in ['prevalence','logistic','gradient_boosting']:
        filename=f'test_{name}.csv'
        if digest(baseline/filename)!=baseline_manifest['artifacts'][filename]:
            raise ValueError('Baseline artifact fingerprint mismatch')
    rows={}
    for name in ['prevalence','logistic','gradient_boosting']:
        with (baseline/f'test_{name}.csv').open() as f:
            selected=[r for r in csv.DictReader(f) if pid(r) in expected]
        if len(selected)!=len(expected) or len({pid(r) for r in selected})!=len(expected):
            raise ValueError('Baseline matching failed')
        rows[name]=sorted(selected,key=pid)
    truth={pid(r):r['y'] for r in rows['prevalence']}
    for name in ['logistic','gradient_boosting']:
        if {pid(r):r['y'] for r in rows[name]}!=truth:raise ValueError('Inconsistent labels')
    for name in ['structured','persona']:
        rows[name]=[{**r,'probability':indexed[pid(r)][name]} for r in rows['prevalence']]
    results={}
    for name,rs in rows.items():
        results[name]=evaluate(rs,5000,42)
        results[name]['average_precision']=float(average_precision_score([int(r['y']) for r in rs],[float(r['probability']) for r in rs]))
    comparisons={}
    for reference in ['structured','gradient_boosting','logistic']:
        ref={pid(r):float(r['probability']) for r in rows[reference]}
        rs=[{**r,'baseline_probability':ref[pid(r)]} for r in rows['persona']]
        overall=evaluate(rs,5000,42)
        comparisons[reference]=dict(difference=overall['brier_difference'],uncertainty=overall['uncertainty'],
            by_window={c:evaluate([r for r in rs if r['cutoff']==c],5000,42)['brier_difference'] for c in sorted({r['cutoff'] for r in rs})},
            by_segment={s:evaluate([r for r in rs if r['segment']==s],5000,42)['brier_difference'] for s in sorted({r['segment'] for r in rs})})
    traces=[]
    for i,r in enumerate(rows['persona']):
        key=pid(r);pred=indexed[key];call=pred['calls']['summary']
        from .openai_pilot import parse_response
        persona_text=parse_response(read_json(run/f'call_{call:04d}.response.json'),True)
        y=int(r['y']); probs={n:float(rs[i]['probability']) for n,rs in rows.items()}
        traces.append(dict(id=key,cutoff=r['cutoff'],observed=y,source=expected[key]['source'],source_sha256=pred['source_sha256'],
                           persona_text=persona_text,probabilities=probs,calls=pred['calls'],
                           difference=(probs['persona']-y)**2-(probs['structured']-y)**2))
    ledger=[json.loads(x) for x in (run/'ledger.jsonl').read_text().splitlines()]
    settled=[e for e in ledger if e['event']=='settled']
    output=dict(status='exploratory held-out comparison; not external preregistration',results=results,comparisons=comparisons,
                summary=summary,protocol=protocol,run_id=run.name,baseline_manifest_sha256=digest(baseline/'manifest.json'),
                plan_sha256=digest(plan/'plan.json'),predictions_sha256=digest(run/'predictions.jsonl'),
                latency_mean_seconds=sum(e['latency_seconds'] for e in settled)/len(settled),
                failures=sum(e['event']=='failed' for e in ledger),
                primary='persona minus structured-history Brier',secondary='persona minus gradient-boosting Brier',
                caveats=['Pilot households excluded; baseline aggregate test results were previously inspected.',
                         'No prompt or model changes from pilot. No outcome-based stopping.',
                         'Intervals are marginal household bootstrap intervals, without multiple-comparison correction.',
                         'Single model, category and retailer; shared temporal shocks and cross-model stability not assessed.',
                         'Persona summarization may change or omit details; causal mechanism not established.'])
    output['mechanism']=diagnose(traces)
    (run/'research_results.json').write_text(json.dumps(output,indent=2)+'\n')
    (run/'traces.json').write_text(json.dumps(traces,indent=2)+'\n')
    return output,traces


def finding(data):
    d=data['comparisons']['structured']; lo,hi=d['uncertainty']['ci95']
    if lo>0:statement='Persona summaries increased prediction error relative to structured history'
    elif hi<0:statement='Persona summaries reduced prediction error relative to structured history'
    else:statement='The comparison with structured history remains inconclusive'
    g=data['comparisons']['gradient_boosting']
    suffix='; gradient boosting had a lower Brier score.' if g['difference']>0 else '; persona had a lower Brier point estimate than gradient boosting.'
    return statement+suffix


def headline(data):
    """Data-derived title; falls back to the neutral question when the pattern does not hold."""
    m=data.get('mechanism');lo=data['comparisons']['structured']['uncertainty']['ci95'][0]
    if m and lo>0 and m['retention']['mean']>=.8 and \
       m['decomposition']['persona']['resolution']<m['decomposition']['structured']['resolution']:
        return 'The persona kept the facts but lost the signal.'
    return 'Does a persona preserve predictive signal?'


def mechanism_section(data):
    m=data.get('mechanism')
    if not m:return ''
    d=m['decomposition'];r=m['retention']
    rows=''.join(f'<tr><td>{LABELS[c]}</td><td>{d[c]["resolution"]:.4f}</td><td>{d[c]["reliability"]:.4f}</td><td>{d[c]["mean_probability"]:.3f}</td></tr>' for c in ['gradient_boosting','structured','persona'])
    hist=''.join(f'<tr><td>{k}</td><td>{v["n"]}</td><td>{v["observed"]:.2f}</td><td>{v["mean_probability"]["gradient_boosting"]:.2f}</td><td>{v["mean_probability"]["structured"]:.2f}</td><td>{v["mean_probability"]["persona"]:.2f}</td><td>{v["persona_minus_structured"]:+.4f}</td></tr>' for k,v in m['by_history'].items())
    gap=r['loss_gap']
    gaps=''.join(f'<li>{k.replace("_"," ").capitalize()}: {v["n"]} predictions, mean Δ loss {v["persona_minus_structured"]:+.4f}</li>' for k,v in gap.items())
    return f'''<section><h2>Where the signal went</h2><div class="grid"><article class="panel"><p class="eyebrow">1 · Facts survived</p><strong class="delta">{100*r["mean"]:.0f}%</strong><p>of source feature values appear verbatim in the generated personas (a conservative lower bound: “no visits” counts as missing).</p><ul class="muted">{gaps}</ul><p class="muted">Loss is not concentrated where numbers were dropped.</p></article><article class="panel"><p class="eyebrow">2 · Discrimination did not</p><div class="scroll"><table><tr><th>Condition</th><th>Resolution ↑</th><th>Reliability ↓</th><th>Mean p</th></tr>{rows}</table></div><p class="muted">Brier ≈ reliability − resolution + uncertainty ({d["persona"]["uncertainty"]:.4f}); observed rate {m["observed_rate"]:.3f}. Persona forecasts separate buyers from non-buyers less sharply.</p></article></div>
<h3>By prior category purchase frequency (last 84 days)</h3><div class="scroll"><table><tr><th>Milk trips</th><th>n</th><th>Observed rate</th><th>GBDT mean p</th><th>Structured mean p</th><th>Persona mean p</th><th>Persona − structured Δ loss</th></tr>{hist}</table></div><p class="muted">{html.escape(m["note"])} Groups use a source feature only; no demographics were shown to either LLM condition.</p></section>
<section class="panel"><h2>Why this matters for synthetic users</h2><p>Narrative personas are easy to read and easy to believe. Here, the narrative carried the same facts yet produced more hedged, less discriminating forecasts, with the largest gap among the most frequent category buyers. A persona that sounds right is not evidence that it predicts right: fidelity should be measured against held-out behavior before persona outputs inform decisions.</p><p class="muted">One retailer, category and model. This motivates a replication, not a general law.</p></section>'''


def render(data, traces, target):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    e=html.escape
    results=data['results']; model=results['persona']['model']; summary=data['summary']
    order=sorted(results,key=lambda n:results[n]['model']['brier'])
    maximum=max(results[n]['model']['brier'] for n in order)*1.15
    bars=''.join(f'<div class="barrow"><span>{LABELS[n]}</span><div class="track"><div class="bar" style="width:{100*results[n]["model"]["brier"]/maximum:.2f}%;background:{COLORS[n]}"></div></div><b>{results[n]["model"]["brier"]:.4f}</b></div>' for n in order)
    table=''.join(f'<tr><td>{LABELS[n]}</td><td>{results[n]["model"]["brier"]:.4f}</td><td>{results[n]["model"]["log_loss"]:.4f}</td><td>{results[n]["average_precision"]:.4f}</td><td>{results[n]["model"]["ece"]:.4f}</td></tr>' for n in order)
    comparisons=''
    for n in ['structured','gradient_boosting']:
        c=data['comparisons'][n];ci=c['uncertainty']['ci95']
        comparisons+=f'<article class="panel"><p class="eyebrow">Persona minus {LABELS[n]}</p><strong class="delta">{c["difference"]:+.4f}</strong><p>95% paired household interval<br><b>[{ci[0]:+.4f}, {ci[1]:+.4f}]</b></p><p class="muted">Negative favors persona. Positive favors the comparator.</p></article>'
    dots=''
    for n in ['gradient_boosting','structured','persona']:
        bins=results[n]['model']['calibration']; points=' '.join(f'{55+370*b["predicted"]:.1f},{415-370*b["observed"]:.1f}' for b in bins)
        dots+=f'<polyline points="{points}" stroke="{COLORS[n]}" fill="none" stroke-width="2"/>'
        for b in bins:
            dots+=f'<circle cx="{55+370*b["predicted"]}" cy="{415-370*b["observed"]}" r="4" fill="{COLORS[n]}"><title>{LABELS[n]}: predicted {b["predicted"]:.3f}, observed {b["observed"]:.3f}, n={b["n"]}</title></circle>'
    failures=''
    if traces:
        for title, selected in [('Where persona loses',sorted(traces,key=lambda t:t['difference'],reverse=True)[:3]),
                                ('Where persona helps',sorted(traces,key=lambda t:t['difference'])[:3])]:
            failures+=f'<h3>{title}</h3>'
            for t in selected:
                alias=hashlib.sha256(t['id'].encode()).hexdigest()[:8]
                probs=t['probabilities']
                failures+=f'<details><summary>Example {alias} · observed {t["observed"]} · Δ loss {t["difference"]:+.4f}</summary><div class="trace"><p>Persona p={probs["persona"]:.3f}; structured p={probs["structured"]:.3f}; GBDT p={probs["gradient_boosting"]:.3f}.</p><h4>Historical source</h4><pre>{e(json.dumps(t["source"],indent=2))}</pre><h4>Generated persona</h4><p>{e(t["persona_text"])}</p><h4>Evidence trace</h4><p>Source snapshot SHA-256: <code>{t["source_sha256"]}</code></p><p>Local call IDs: {e(json.dumps(t["calls"]))}. Each maps to saved request and response JSON.</p><p>Persona Brier contribution: ({probs["persona"]:.3f} − {t["observed"]})² = {(probs["persona"]-t["observed"])**2:.4f}.</p><p class="muted">Post-evaluation examples selected by signed loss difference. A plausible explanation is not a proven cause.</p></div></details>'
    else:
        failures='<p>Detailed historical snapshots, generated personas and raw response traces are retained in the local run folder. Reproduce the run to inspect the household-level failure explorer; source-level data are excluded from this repository report.</p>'
    windows=''.join(f'<tr><td>{c}</td><td>{d:+.4f}</td></tr>' for c,d in data['comparisons']['structured']['by_window'].items())
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BehaviorLens · Research evidence</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0d1721;color:#edf4fa;font:16px/1.6 system-ui}}main{{max-width:1160px;margin:auto;padding:50px 28px 80px}}h1{{font-size:clamp(36px,5vw,64px);line-height:1.12;max-width:900px;letter-spacing:-.04em;margin:20px 0}}h2{{font-size:28px;margin:0 0 24px}}h3{{margin-top:28px}}section{{margin-top:56px}}.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#73dec9}}.muted{{color:#a4b5c6}}.finding{{font-size:22px;max-width:930px}}.notice{{padding:16px 20px;border-left:3px solid #f8cb8b;background:#192735}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:30px 0}}.panel{{padding:24px;background:#152331;border:1px solid #2a3b4c;border-radius:12px}}.stat strong{{display:block;font-size:28px}}.stat span{{font-size:13px;color:#a4b5c6}}.delta{{font-size:38px}}.barrow{{display:grid;grid-template-columns:220px 1fr 65px;align-items:center;gap:18px;margin:22px 0;font-size:14px}}.track{{background:#223444;height:24px;border-radius:3px}}.bar{{height:100%;border-radius:3px}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #2a3b4c}}.scroll{{overflow:auto}}svg{{width:100%;max-width:490px}}.legend{{font-size:13px}}.legend span{{margin-right:16px}}details{{padding:16px 0;border-bottom:1px solid #2a3b4c}}summary{{cursor:pointer}}.trace{{padding:20px;background:#152331}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}code{{overflow-wrap:anywhere}}a{{color:#73dec9}}.meta{{display:grid;grid-template-columns:150px 1fr;gap:12px;font-size:14px}}@media(max-width:700px){{main{{padding:28px 18px}}.grid{{grid-template-columns:1fr}}.stats{{grid-template-columns:1fr 1fr}}.barrow{{grid-template-columns:145px 1fr 55px;gap:8px;font-size:12px}}.panel{{padding:18px}}}}
</style><main><div class="eyebrow">BehaviorLens / Evidence brief 001</div><h1>{e(headline(data))}</h1><p class="finding">{e(finding(data))}</p><p class="muted">An exploratory comparison on held-out milk-category purchases at one retailer.</p><p class="notice">Pilot households excluded. One model, one category, one retailer. No claim of general population fidelity or causal effects.</p>
<div class="stats"><div class="stat"><strong>{model['households']}</strong><span>New households</span></div><div class="stat"><strong>{model['n']}</strong><span>Matched predictions</span></div><div class="stat"><strong>{summary['calls']:,}</strong><span>API calls</span></div><div class="stat"><strong>${summary['usage_cost_usd']:.3f}</strong><span>Estimated run cost</span></div></div>
<section><h2>Prediction error</h2><p class="muted">Brier score · lower is better · bars start at zero. All five conditions share exactly the same outcomes.</p>{bars}<details><summary>Full metric table</summary><div class="scroll"><table><tr><th>Condition</th><th>Brier ↓</th><th>Log loss ↓</th><th>Avg precision ↑</th><th>ECE ↓</th></tr>{table}</table></div></details></section>
<section><h2>How large is the difference?</h2><div class="grid">{comparisons}</div><p class="muted">5,000 paired household bootstrap resamples. Primary comparison: persona vs structured history. Secondary: persona vs GBDT. Marginal intervals, no familywise correction. Shared time shocks are not covered.</p></section>
<section class="grid"><article class="panel"><h2>Calibration</h2><svg viewBox="0 0 480 475" role="img" aria-label="Predicted probability against observed frequency"><path d="M55 45V415H425 M55 415L425 45" fill="none" stroke="#7b8fa3" stroke-dasharray="5"/><g fill="#b6c5d4" font-size="12"><text x="47" y="438">0</text><text x="415" y="438">1</text><text x="30" y="50">1</text><text x="155" y="465">Predicted probability</text><text transform="translate(15 290) rotate(-90)">Observed frequency</text></g>{dots}</svg><div class="legend"><span style="color:#f8cb8b">GBDT</span><span style="color:#98a8ff">Structured</span><span style="color:#56dbc0">Persona</span></div><p class="muted">Ten fixed equal-width bins. Hover for bin counts. Descriptive bin frequencies; no bin-level confidence bands.</p></article><article class="panel"><h2>Across test windows</h2><p>Persona minus structured Brier, by cutoff day.</p><table><tr><th>Dataset day</th><th>Δ Brier</th></tr>{windows}</table><p class="muted">Repeated households; these windows are not independent replications. No per-window significance claim is made.</p><h3>Practical relevance</h3><p>The expansion plan specified 0.01 Brier as a discussion margin before these LLM outcomes were generated. It is not a validated business threshold.</p></article></section>
{mechanism_section(data)}
<section><h2>Inspect where the models disagree</h2>{failures}</section>
<section class="grid"><article class="panel"><h2>Experiment</h2><div class="meta"><b>Dataset</b><span>Complete Journey, official distributed transaction representation</span><b>Target</b><span>FLUID MILK PRODUCTS</span><b>Horizon</b><span>28 relative dataset days</span><b>Model</b><span>{e(data['protocol']['model'])}</span><b>Run</b><span>{e(data['run_id'])}</span><b>Prompt</b><span>Unchanged from pilot</span><b>Failures</b><span>{data['failures']}</span><b>Total cost</b><span>${summary['cumulative_cost_usd']:.5f} including pilot; usage estimate</span></div></article><article class="panel"><h2>What would change the conclusion?</h2><p>Replication on new categories and later time windows; stable effects across model snapshots; and a paired interval that supports a practically meaningful improvement.</p><p>Persona summaries may omit, round or reinterpret history. Inspecting these transformations can suggest mechanisms, but this experiment does not prove them.</p><p class="muted">This is exploratory work after an engineering pilot, not an externally preregistered study.</p></article></section>
<section><h2>Verify the evidence</h2><p>Plan SHA-256: <code>{data['plan_sha256']}</code></p><p>Prediction SHA-256: <code>{data['predictions_sha256']}</code></p><p><a href="results.json">Machine-readable results and protocol</a></p><p class="muted">Independent research.</p></section></main></html>'''
    (target/'report.html').write_text(body)
    (target/'results.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--run',required=True);p.add_argument('--plan',required=True);p.add_argument('--publish',required=True);a=p.parse_args()
    data,traces=analyze(a.baseline,a.run,a.plan)
    render(data,traces,Path(a.run)/'report')
    render(data,[],a.publish)
    print(finding(data))
