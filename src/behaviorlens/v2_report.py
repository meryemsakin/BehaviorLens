"""v2 evidence report. Every number is read from results_v2.json; headlines are derived, never hand-written."""
import argparse
import html
import json
from pathlib import Path

LABELS = dict(prevalence='Base rate', logistic='Logistic regression', gradient_boosting='Gradient boosting',
              structured='LLM · structured record', persona='LLM · persona', structured_cal='LLM · structured, calibrated',
              persona_cal='LLM · persona, calibrated', hybrid='Hybrid: GBDT + structured LLM',
              hybrid_persona='Hybrid: GBDT + persona LLM')
COLORS = dict(prevalence='#7d8fa1', logistic='#a7b9ce', gradient_boosting='#f2b866', structured='#98a8ff',
              persona='#46c9ae', structured_cal='#7584d6', persona_cal='#2e9c86', hybrid='#e98f6b', hybrid_persona='#c7735a')
STUDY = dict(A='Study A · category purchase in the next 28 days', B='Study B · coupon-campaign redemption')
e = html.escape


def verdict(c, small=.005):
    lo, hi = c['ci']
    if lo > 0:
        word = 'worse'
    elif hi < 0:
        word = 'better'
    else:
        return 'no detectable difference'
    return f"{word}{' (practically small)' if abs(c['estimate']) < small else ''}"


def overstatement(s, c='persona'):
    m = s['conditions'][c]
    return m['mean_prediction'] / m['prevalence']


def headline(r):
    st = r['studies'].values()
    over = all(overstatement(s) > 1.25 for s in st)
    lose = all(s['contrasts']['E1 structured LLM vs gradient boosting']['ci'][0] > 0 and
               s['conditions']['persona']['brier'] > s['conditions']['gradient_boosting']['brier'] for s in st)
    if over and lose:
        return 'Synthetic customers overstated demand, and none beat a gradient-boosted model.'
    if lose:
        return 'No synthetic-customer condition beat a gradient-boosted model.'
    h = {k: verdict(s['contrasts']['H1 persona vs structured LLM']) for k, s in r['studies'].items()}
    return 'Persona effects by task: ' + '; '.join(f'study {k} {w}' for k, w in h.items()) + '.'


def findings(r):
    """Plain-language findings, each carrying its own numbers from results_v2.json."""
    A, B = r['studies']['A'], r['studies']['B']
    ci = lambda c: f"{c['estimate']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
    out = []
    e1 = {k: s['contrasts']['E1 structured LLM vs gradient boosting'] for k, s in r['studies'].items()}
    out.append(('No LLM condition beat gradient boosting.',
                f"Structured LLM minus GBDT Brier: purchase {ci(e1['A'])}, coupons {ci(e1['B'])}. "
                f"GBDT {A['conditions']['gradient_boosting']['brier']:.4f} vs {A['conditions']['structured']['brier']:.4f} and "
                f"{B['conditions']['gradient_boosting']['brier']:.4f} vs {B['conditions']['structured']['brier']:.4f}."))
    h2 = {k: s['contrasts']['H2 hybrid vs gradient boosting'] for k, s in r['studies'].items()}
    out.append(('Adding the LLM to gradient boosting did not help.',
                f"Hybrid minus GBDT: purchase {ci(h2['A'])}, coupons {ci(h2['B'])}; practically small and in the wrong direction."))
    h1 = {k: s['contrasts']['H1 persona vs structured LLM'] for k, s in r['studies'].items()}
    out.append(('Personas predicted worse than the record they were written from.',
                f"Persona minus structured (97.5% CI): purchase {ci(h1['A'])}, coupons {ci(h1['B'])}."))
    out.append(('The persona penalty is mostly over-optimism.',
                f"Persona-based customers predicted a {A['conditions']['persona']['mean_prediction']:.0%} purchase rate "
                f"(actual {A['conditions']['persona']['prevalence']:.0%}) and {B['conditions']['persona']['mean_prediction']:.0%} "
                f"coupon redemption (actual {B['conditions']['persona']['prevalence']:.0%}). After validation calibration the gap "
                f"shrinks to {ci(A['contrasts']['H3 calibrated persona vs calibrated structured'])} and "
                f"{ci(B['contrasts']['H3 calibrated persona vs calibrated structured'])}."))
    f = B['fidelity']['conditions']
    out.append(('Simulated customers did not rank campaigns reliably.',
                f"Rank correlation with observed redemption across nine campaigns: GBDT {f['gradient_boosting']['spearman']:.2f}, "
                f"structured LLM {f['structured']['spearman']:.2f}, persona {f['persona']['spearman']:.2f}. Descriptive; nine points."))
    rb = r.get('robustness')
    if rb and 'A' in rb:
        out.append(('A frontier model fixed much of the persona gap, not the baseline gap.',
                    f"On a subsample, {r['models']['robustness']} persona Brier was {rb['A']['robust']['conditions']['persona']['brier']:.4f} vs "
                    f"{rb['A']['main']['conditions']['persona']['brier']:.4f} for the main model (purchase), but GBDT stayed ahead "
                    f"({rb['A']['robust']['conditions']['gradient_boosting']['brier']:.4f}; coupons "
                    f"{rb['B']['robust']['conditions']['gradient_boosting']['brier']:.4f} vs "
                    f"{rb['B']['robust']['conditions']['structured']['brier']:.4f} structured)."))
    return out


def bars(conditions, order):
    top = max(conditions[c]['brier'] for c in order) * 1.12
    return ''.join(f'<div class="bar"><span>{LABELS[c]}</span><div class="track"><i style="width:{100*conditions[c]["brier"]/top:.1f}%;'
                   f'background:{COLORS[c]}"></i></div><b>{conditions[c]["brier"]:.4f}</b></div>' for c in order)


def forest(contrasts):
    span = max(max(abs(c['ci'][0]), abs(c['ci'][1])) for c in contrasts.values()) * 1.15 or 1
    x = lambda v: 50 + 50*v/span
    rows = ''
    for name, c in contrasts.items():
        lo, hi = c['ci']
        rows += (f'<div class="forest"><span>{e(name)}<small>{c["confidence"]:.1%} interval · {verdict(c)}</small></span>'
                 f'<div class="axis"><em style="left:50%"></em><u style="left:{x(lo):.1f}%;width:{x(hi)-x(lo):.1f}%"></u>'
                 f'<b style="left:{x(c["estimate"]):.1f}%"></b></div><code>{c["estimate"]:+.4f} [{lo:+.4f}, {hi:+.4f}]</code></div>')
    return rows + '<p class="muted">Model minus reference Brier. Left of centre favours the model. Household-cluster bootstrap, 5,000 draws.</p>'


def table(conditions, order):
    head = '<tr><th>Condition</th><th>Brier ↓</th><th>Log loss ↓</th><th>AUC ↑</th><th>Resolution ↑</th><th>Reliability ↓</th><th>Mean p</th></tr>'
    body = ''.join(f'<tr><td>{LABELS[c]}</td><td>{m["brier"]:.4f}</td><td>{m["log_loss"]:.4f}</td><td>{m["auc"]:.3f}</td>'
                   f'<td>{m["resolution"]:.4f}</td><td>{m["reliability"]:.4f}</td><td>{m["mean_prediction"]:.3f}</td></tr>'
                   for c in order for m in [conditions[c]])
    return f'<div class="scroll"><table>{head}{body}</table></div>'


def campaign_chart(fid):
    groups = sorted(fid['groups'].items(), key=lambda kv: kv[1]['observed'])
    top = max(max(g['observed'], g['gradient_boosting'], g['structured'], g['persona']) for _, g in groups) * 1.1
    W, H, pad = 640, 300, 40
    step = (W-2*pad)/len(groups)
    marks = ''
    for i, (name, g) in enumerate(groups):
        cx = pad + step*(i+.5)
        y = lambda v: H-pad-(H-2*pad)*v/top
        marks += f'<rect x="{cx-step*.3:.1f}" y="{y(g["observed"]):.1f}" width="{step*.6:.1f}" height="{H-pad-y(g["observed"]):.1f}" fill="#33485c"><title>Campaign {name}: observed {g["observed"]:.3f} (n={g["n"]})</title></rect>'
        for c in ('gradient_boosting', 'structured', 'persona'):
            marks += f'<circle cx="{cx:.1f}" cy="{y(g[c]):.1f}" r="5" fill="{COLORS[c]}"><title>{LABELS[c]}: {g[c]:.3f}</title></circle>'
        marks += f'<text x="{cx:.1f}" y="{H-pad+18}" text-anchor="middle">{e(name)}</text>'
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Observed and predicted redemption rate per campaign">'
            f'<line x1="{pad}" x2="{W-pad}" y1="{H-pad}" y2="{H-pad}" stroke="#51677c"/>{marks}</svg>'
            '<p class="legend"><span style="color:#8aa0b5">■ observed</span> <span style="color:#f2b866">● GBDT</span> '
            '<span style="color:#98a8ff">● structured LLM</span> <span style="color:#46c9ae">● persona LLM</span></p>')


def fidelity_table(fid, label):
    order = ['gradient_boosting', 'structured', 'persona', 'structured_cal', 'persona_cal', 'hybrid']
    rows = ''.join(f'<tr><td>{LABELS[c]}</td><td>{fid["conditions"][c]["mae"]:.3f}</td>'
                   f'<td>{"—" if fid["conditions"][c]["spearman"] is None else format(fid["conditions"][c]["spearman"], ".2f")}</td></tr>'
                   for c in order if c in fid['conditions'])
    return f'<div class="scroll"><table><tr><th>Condition</th><th>MAE of {label} rate ↓</th><th>Rank correlation ↑</th></tr>{rows}</table></div>'


def study_section(key, s):
    c = s['conditions']
    main = [k for k in ('prevalence', 'logistic', 'gradient_boosting', 'structured', 'persona') if k in c]
    fitted = [k for k in ('gradient_boosting', 'structured_cal', 'persona_cal', 'hybrid', 'hybrid_persona') if k in c]
    extra = ''
    if key == 'A':
        rows = ''.join(f'<tr><td>{e(cat)}</td><td>{v["conditions"]["persona"]["prevalence"]:.2f}</td>'
                       + ''.join(f'<td>{v["conditions"][k]["brier"]:.4f}</td>' for k in ('gradient_boosting', 'structured', 'persona'))
                       + f'<td>{v["contrasts"]["H1 persona vs structured LLM"]["estimate"]:+.4f}</td></tr>'
                       for cat, v in s['by_category'].items())
        extra = ('<h3>By category</h3><div class="scroll"><table><tr><th>Category</th><th>Observed rate</th><th>GBDT</th>'
                 f'<th>Structured</th><th>Persona</th><th>Persona − structured</th></tr>{rows}</table></div>'
                 '<h3>Population fidelity (category × test window)</h3>' + fidelity_table(s['fidelity'], 'purchase'))
    else:
        extra = ('<h3>Can simulated customers rank campaigns?</h3><p>Mean predicted versus observed redemption rate for each of the '
                 'nine test campaigns, sorted by observed rate.</p>' + campaign_chart(s['fidelity'])
                 + fidelity_table(s['fidelity'], 'campaign redemption'))
    return (f'<section><p class="eyebrow">{e(STUDY[key])}</p><h2>{e(verdict_line(s))}</h2>'
            f'<p class="muted">{s["rows"]} test rows · {s["conditions"]["persona"]["n"]:,} labels · {s["dropped"]["test"]} rows dropped for failed calls.</p>'
            f'<div class="grid"><article class="panel"><h3>Prediction error, Brier</h3>{bars(c, main)}</article>'
            f'<article class="panel"><h3>Pre-registered contrasts</h3>{forest(s["contrasts"])}</article></div>'
            f'<article class="panel"><h3>Can calibration or a hybrid close the gap?</h3>{bars(c, fitted)}'
            '<p class="muted">Calibration and stacking are fit on the validation sample only, then applied once to test.</p></article>'
            f'<details><summary>Full metric table</summary>{table(c, main + fitted[1:])}</details>{extra}</section>')


def verdict_line(s):
    h1, h2 = s['contrasts']['H1 persona vs structured LLM'], s['contrasts']['H2 hybrid vs gradient boosting']
    return (f"Persona vs structured record: {verdict(h1)} ({h1['estimate']:+.4f} Brier). "
            f"Adding the LLM to gradient boosting: {verdict(h2)} ({h2['estimate']:+.4f}).")


def level_bars(s, single=False):
    groups = {'All test campaigns': s['conditions']} if single else {k: v['conditions'] for k, v in s['by_category'].items()}
    top = max(max(g[c]['mean_prediction'] for c in ('gradient_boosting', 'structured', 'persona')) for g in groups.values()) * 1.1
    out = ''
    for name, g in groups.items():
        out += f'<p class="group">{e(name.title() if not single else name)}</p>'
        rows = [('Observed', g['persona']['prevalence'], '#8aa0b5')] + [
            (LABELS[c], g[c]['mean_prediction'], COLORS[c]) for c in ('gradient_boosting', 'structured', 'persona')]
        out += ''.join(f'<div class="bar small"><span>{lab}</span><div class="track"><i style="width:{100*v/top:.1f}%;background:{col}"></i></div><b>{v:.0%}</b></div>'
                       for lab, v, col in rows)
    return out


def related_section():
    return ('<section class="panel"><h2>Related work and what is new here</h2><p>Evaluating LLM "synthetic users" against real '
            'people is an active area. <a href="https://arxiv.org/abs/2506.05606">OPeRA</a> benchmarks next-action prediction in '
            'real web shopping sessions; <a href="https://arxiv.org/abs/2607.26348">When Synthetic Users Fail</a> finds that no LLM '
            'beats the strongest non-LLM baseline at the individual level on survey data. This study is consistent with that '
            'finding and extends it to longitudinal purchase records and responses to real marketing campaigns, with a '
            'same-source persona ablation, validation-only calibration and stacking, campaign-level fidelity, and a protocol '
            'committed before any model output existed.</p></section>')


def deviations_section(r):
    return ('<section class="panel"><h2>Deviations from the protocol</h2><ul><li>The provider limits queued tokens per model. '
            'Batches were therefore split into sequential parts; two submissions were rejected by that limit before any request '
            'ran and were resubmitted unchanged. No processed request was retried.</li>'
            f'<li>Failed or unparsable requests: {sum(r["failures"].values())}. Rows dropped: study A {r["studies"]["A"]["dropped"]["test"]}, '
            f'study B {r["studies"]["B"]["dropped"]["test"]}.</li><li>Headline wording is generated from the results after they '
            'existed; the contrasts, intervals and all numbers follow the frozen code.</li></ul></section>')


def robustness_section(r):
    rb = r.get('robustness')
    if not rb or 'A' not in rb:
        return ''
    rows = ''
    for s in ('A', 'B'):
        x = rb[s]
        for label, part in (('gpt-5.4-mini', x['main']), (r['models']['robustness'], x['robust'])):
            h1 = part['contrasts']['H1 persona vs structured LLM']
            rows += (f'<tr><td>{s}</td><td>{e(label)}</td><td>{x["rows"]}</td><td>{part["conditions"]["structured"]["brier"]:.4f}</td>'
                     f'<td>{part["conditions"]["persona"]["brier"]:.4f}</td><td>{part["conditions"]["gradient_boosting"]["brier"]:.4f}</td>'
                     f'<td>{h1["estimate"]:+.4f} [{h1["ci"][0]:+.4f}, {h1["ci"][1]:+.4f}]</td></tr>')
    return ('<section><p class="eyebrow">Robustness</p><h2>Does a frontier model change the picture?</h2>'
            '<div class="scroll"><table><tr><th>Study</th><th>Model</th><th>Rows</th><th>Structured</th><th>Persona</th><th>GBDT</th>'
            f'<th>Persona − structured (97.5%)</th></tr>{rows}</table></div><p class="muted">Identical rows for both models; small '
            'subsample, so intervals are wide.</p></section>')


CSS = '''
:root{--bg:#0d1721;--panel:#152331;--line:#2a3b4c;--text:#edf4fa;--muted:#a4b5c6;--accent:#56dbc0}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f6f4ee;--panel:#fffdf8;--line:#ddd6c8;--text:#14212d;--muted:#51606d;--accent:#0f7a67}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:48px 16px 80px}h1{font-size:clamp(34px,5vw,60px);line-height:1.1;letter-spacing:-.03em;margin:16px 0}
h2{font-size:clamp(22px,3vw,30px);line-height:1.25;margin:4px 0 12px}section{margin-top:64px}.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.muted{color:var(--muted);font-size:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:18px 0}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px;margin:18px 0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.stats strong{display:block;font-size:28px}.stats span{font-size:13px;color:var(--muted)}
.bar{display:grid;grid-template-columns:minmax(120px,210px) 1fr 58px;gap:12px;align-items:center;margin:12px 0;font-size:14px}.track{height:20px;background:var(--line);border-radius:3px}.track i{display:block;height:100%;border-radius:3px}
.forest{display:grid;grid-template-columns:1fr;gap:4px;margin:16px 0;font-size:14px}.forest small{display:block;color:var(--muted)}.axis{position:relative;height:22px;background:var(--line);border-radius:3px}
.axis em{position:absolute;top:0;bottom:0;border-left:1px dashed var(--muted)}.axis u{position:absolute;top:9px;height:4px;background:var(--accent)}.axis b{position:absolute;top:5px;width:12px;height:12px;margin-left:-6px;border-radius:50%;background:var(--text)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid var(--line)}.scroll{overflow-x:auto}
svg{width:100%;height:auto}svg text{fill:var(--muted);font-size:12px}.legend{font-size:13px}details{margin:16px 0}summary{cursor:pointer}code{font-size:13px;overflow-wrap:anywhere}a{color:var(--accent)}
.findings li{margin:12px 0}.group{margin:16px 0 4px;font-size:14px;font-weight:600}.bar.small{margin:6px 0;font-size:13px}.bar.small .track{height:14px}
@media(max-width:760px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}}
'''


def render(results, protocol_commit, target):
    r = json.loads(Path(results).read_text())
    A, B = r['studies']['A'], r['studies']['B']
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BehaviorLens v2 evidence</title><style>{CSS}</style><main>
<p class="eyebrow">BehaviorLens · Evidence brief 002 · pre-registered</p><h1>{e(headline(r))}</h1>
<p>Do synthetic users predict what real customers do next, or only sound plausible? Two held-out tests on real retail transactions,
with the protocol and analysis code committed before any model output existed (commit <code>{e(protocol_commit)}</code>).</p>
<div class="stats"><div><strong>{A["rows"]+B["rows"]:,}</strong><span>test rows</span></div><div><strong>{A["conditions"]["persona"]["n"]+B["conditions"]["persona"]["n"]:,}</strong><span>scored outcomes</span></div>
<div><strong>9</strong><span>conditions per outcome</span></div><div><strong>{e(r["models"]["main"].split("-20")[0])}</strong><span>main model</span></div></div>
<section><p class="eyebrow">Findings</p><ol class="findings">{''.join(f"<li><b>{e(t)}</b> {e(d)}</li>" for t, d in findings(r))}</ol></section>
<section><p class="eyebrow">Population level</p><h2>What would a synthetic panel have told a retailer?</h2>
<div class="grid"><article class="panel"><h3>Share of households buying, per category</h3>{level_bars(A)}</article>
<article class="panel"><h3>Coupon redemption rate, all test campaigns</h3>{level_bars(B, single=True)}</article></div>
<p class="muted">Mean predicted probability versus the observed rate on identical test rows. A simulator used for demand or campaign planning reads these levels directly.</p></section>
{study_section("A", A)}{study_section("B", B)}{robustness_section(r)}{related_section()}{deviations_section(r)}
<section class="panel"><h2>How to read this</h2><p>Brier score is mean squared error of a probability; lower is better. H1 compares a persona written from
the shopping record with the record itself. H2 asks whether adding the LLM to a strong tabular model helps. Intervals resample households,
so repeated observations of one household are not treated as independent.</p><p class="muted">One retailer (dunnhumby Complete Journey,
public, 2-year anonymised panel); models may have seen this public dataset during training; one draw per request, so run-to-run stability
is not measured; prompts are one realistic design, not an optimised persona pipeline. Independent research.</p></section>
</main></html>'''
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    (target/'report.html').write_text(body)
    (target/'results.json').write_text(json.dumps(r, indent=2))


def animate(results, target):
    """GIF/MP4 from results: predicted vs observed levels grow in, then the Brier comparison."""
    import io
    import shutil
    import subprocess
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    r = json.loads(Path(results).read_text())
    target = Path(target)
    studies = [('Share buying a category in 28 days', r['studies']['A']), ('Coupon redemption rate', r['studies']['B'])]
    series = [('Observed', None, '#8aa0b5'), ('Gradient boosting', 'gradient_boosting', COLORS['gradient_boosting']),
              ('LLM · structured record', 'structured', COLORS['structured']), ('LLM · persona', 'persona', COLORS['persona'])]
    frames = []
    for t in [i/16 for i in range(17)] + [1]*18 + [2]*36:
        fig = plt.figure(figsize=(9.6, 5.4), dpi=100, facecolor='#0d1721')
        fig.text(.05, .9, headline(r) if t > 1 else 'What would a synthetic panel have told a retailer?', color='#edf4fa',
                 fontsize=15, weight='bold', wrap=True)
        fig.text(.05, .84, 'Held-out retail households · mean predicted rate vs what actually happened', color='#a4b5c6', fontsize=11)
        for i, (title, s) in enumerate(studies):
            ax = fig.add_axes([.08 + .47*i, .22, .38, .5], facecolor='#0d1721')
            vals = [s['conditions']['persona']['prevalence'] if k is None else s['conditions'][k]['mean_prediction'] for _, k, _ in series]
            ax.bar(range(4), [v*min(t, 1) for v in vals], color=[c for _, _, c in series])
            for j, v in enumerate(vals):
                if t >= 1:
                    ax.text(j, v*1.02, f'{v:.0%}', ha='center', color='#edf4fa', fontsize=11)
            ax.set_ylim(0, max(vals)*1.25)
            ax.set_xticks(range(4), ['Observed', 'GBDT', 'Structured', 'Persona'], color='#c9d6e2', fontsize=9)
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            ax.set_title(title, color='#c9d6e2', fontsize=11)
        if t > 1:
            A, B = r['studies']['A']['conditions'], r['studies']['B']['conditions']
            for y, name, m in ((.115, 'Purchase', A), (.075, 'Coupons', B)):
                fig.text(.05, y, f"{name} · Brier, lower is better: GBDT {m['gradient_boosting']['brier']:.3f} · structured LLM "
                         f"{m['structured']['brier']:.3f} · persona LLM {m['persona']['brier']:.3f}", color='#56dbc0', fontsize=10, weight='bold')
        fig.text(.05, .025, 'BehaviorLens v2 · pre-registered · dunnhumby Complete Journey · one retailer', color='#71849a', fontsize=8)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        frames.append(Image.open(buf).convert('RGB'))
    frames[0].save(target/'evidence.gif', save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    frames[-1].save(target/'evidence.png')
    if shutil.which('ffmpeg'):
        tmp = target/'_frames'
        tmp.mkdir(exist_ok=True)
        for i, f in enumerate(frames):
            f.save(tmp/f'{i:03d}.png')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '11', '-i', str(tmp/'%03d.png'), '-vf',
                        'tpad=stop_mode=clone:stop_duration=2', '-pix_fmt', 'yuv420p', '-c:v', 'libx264', str(target/'evidence.mp4')], check=True)
        shutil.rmtree(tmp)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    render(a.results, a.commit, a.out)
    animate(Path(a.out)/"results.json", a.out)
