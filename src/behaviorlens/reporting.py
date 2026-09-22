"""Static report derived exclusively from saved run artifacts."""
import csv
import html
import json
from pathlib import Path


def render(run):
    run = Path(run)
    metrics = json.loads((run/'metrics.json').read_text())['test']
    audit = json.loads((run/'data_audit.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    esc = html.escape
    rows, points = [], []
    colors = ['#9aa8b9', '#65dac4', '#f8c77e']
    for (name, result), color in zip(metrics.items(), colors):
        m = result['model']
        ci = result['uncertainty']['ci95']
        rows.append(f'<tr><td>{esc(name)}</td><td>{m["brier"]:.4f}</td><td>{m["log_loss"]:.4f}</td>'
                    f'<td>{result["average_precision"]:.4f}</td><td>{result["brier_difference"]:+.4f}</td>'
                    f'<td>[{ci[0]:+.4f}, {ci[1]:+.4f}]</td></tr>')
        for b in m['calibration']:
            points.append(f'<circle cx="{50+400*b["predicted"]}" cy="{450-400*b["observed"]}" r="5" fill="{color}">'
                          f'<title>{esc(name)}: predicted {b["predicted"]:.3f}, observed {b["observed"]:.3f}, n={b["n"]}</title></circle>')
    with (run/'test_gradient_boosting.csv').open() as f:
        predictions = list(csv.DictReader(f))
    failures = sorted(predictions, key=lambda r: (float(r['probability'])-int(r['y']))**2, reverse=True)[:10]
    snapshots = {}
    with (run/'test_snapshots.jsonl').open() as f:
        for line in f:
            item = json.loads(line)
            s = item['snapshot']
            snapshots[(str(s['household_id']), str(s['cutoff']))] = item
    traces = []
    for r in failures:
        y, p = int(r['y']), float(r['probability'])
        item = snapshots[(r['household_id'], r['cutoff'])]
        traces.append(f'<details><summary>Household {esc(r["household_id"])} · day {r["cutoff"]} · p={p:.3f} · observed={y} · loss={(p-y)**2:.4f}</summary>'
                      f'<p>Largest squared errors, selected after evaluation. Explanations are not causal findings.</p>'
                      f'<pre>{esc(json.dumps(item, indent=2))}</pre>'
                      f'<p>Label window: [{r["cutoff"]}, {r["label_end"]}). Brier contribution: ({p:.4f} − {y})² = {(p-y)**2:.4f}.</p></details>')
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BehaviorLens · Baseline evidence</title><style>
*{{box-sizing:border-box}}body{{background:#101923;color:#e5edf4;font:16px/1.6 system-ui;margin:0}}main{{max-width:1120px;margin:auto;padding:48px 24px}}h1{{font-size:56px;margin:0}}h2{{margin-top:44px}}.muted{{color:#a5b6c6}}.badge{{color:#65dac4;letter-spacing:.12em;font-size:12px}}.notice{{border-left:3px solid #f8c77e;padding:16px;background:#1a2835}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}td,th{{padding:14px 10px;border-bottom:1px solid #304253;text-align:left}}.scroll{{overflow:auto}}svg{{width:100%;max-width:500px;background:#192633}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#14212e;padding:20px;font-size:12px}}details{{border-bottom:1px solid #304253;padding:16px 0}}summary{{cursor:pointer}}a{{color:#65dac4}}
</style><main><p class="badge">BEHAVIORLENS / EXPERIMENT 001</p><h1>Predict. Compare. Inspect.</h1>
<p class="muted">Held-out purchase prediction · {esc(manifest['category'])} · 28 dataset days</p>
<p class="notice">Baseline experiment completed. LLM and persona comparisons have not run. This report cannot answer whether personas add predictive value yet.</p>
<p>{audit['households']:,} households in retained data · {audit['transaction_rows']:,} source transaction rows · {audit['splits']['test']['n']:,} test observations</p>
<h2>How predictable is the task?</h2><div class="scroll"><table><thead><tr><th>Model</th><th>Brier ↓</th><th>Log loss ↓</th><th>Avg precision ↑</th><th>Δ vs prevalence</th><th>95% cluster interval</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">Intervals compare each model with training prevalence; they do not test logistic vs gradient boosting. Fixed model settings, no test tuning. Average precision is the stepwise precision-recall summary, not trapezoidal PR-AUC.</p>
<h2>Calibration</h2><p class="muted">Predicted probability (x) vs observed purchase rate (y). Gray: prevalence · green: logistic · amber: gradient boosting. Hover for bin size.</p>
<svg viewBox="0 0 500 500" role="img" aria-label="Calibration plot"><path d="M50 50V450H450 M50 450L450 50" fill="none" stroke="#718294" stroke-dasharray="4"/><g fill="#b8c6d3" font-size="13"><text x="40" y="475">0</text><text x="440" y="475">1</text><text x="25" y="55">1</text></g>{''.join(points)}</svg>
<h2>Failure explorer</h2><p>Ten largest gradient-boosting errors. Open a row to inspect its exact historical feature snapshot and loss contribution.</p>{''.join(traces)}
<h2>Scope and limitations</h2><p>One category, one retailer, returning recently active households. Absence of a recorded purchase does not mean absence of purchases elsewhere. Demographics are partly coded and missing; codes are not decoded into age or income. Source documentation describes a representation of transactions. Cluster intervals do not cover shared time shocks. No persona fidelity or causal claims.</p>
<h2>Run provenance</h2><p>Protocol SHA-256: <code>{manifest['protocol_sha256']}</code></p><p><a href="metrics.json">All metrics and subgroup diagnostics</a> · <a href="data_audit.json">Data audit</a> · <a href="manifest.json">Manifest</a> · <a href="protocol.json">Frozen protocol</a></p></main></html>'''
    (run/'report.html').write_text(body)
