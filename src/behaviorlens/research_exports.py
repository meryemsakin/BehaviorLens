"""Publication assets from aggregate evidence; never hardcode experimental scores."""
import argparse
import json
from pathlib import Path
from .research_report import LABELS, COLORS, finding, headline


def export(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=json.loads(Path(results).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    metrics=d['results'];order=sorted(metrics,key=lambda n:metrics[n]['model']['brier'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
    vals=[metrics[n]['model']['brier'] for n in order]
    axes[0].barh([LABELS[n] for n in order],vals,color=[COLORS[n] for n in order],edgecolor='#334155')
    axes[0].invert_yaxis();axes[0].set_xlim(0,max(vals)*1.25);axes[0].set_xlabel('Brier score (lower is better)');axes[0].set_title('Same held-out observations')
    for i,v in enumerate(vals):axes[0].text(v+.003,i,f'{v:.4f}',va='center',fontsize=10)
    for i,n in enumerate(['structured','gradient_boosting']):
        c=d['comparisons'][n];v=c['difference'];lo,hi=c['uncertainty']['ci95']
        axes[1].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#0f766e',capsize=5)
    axes[1].set_yticks([0,1],['Persona minus structured','Persona minus GBDT']);axes[1].set_ylim(-.6,1.6)
    axes[1].axvline(0,color='#64748b',linestyle='--');axes[1].set_xlabel('Difference in Brier (negative favors persona)');axes[1].set_title('95% paired household bootstrap intervals')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'BehaviorLens | {metrics["persona"]["model"]["households"]} households, {metrics["persona"]["model"]["n"]} predictions',fontsize=15)
    fig.savefig(out/'evidence.png',dpi=180);fig.savefig(out/'evidence.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    ax.plot([0,1],[0,1],'--',color='#64748b',label='Perfect calibration')
    for n in ['gradient_boosting','structured','persona']:
        bs=metrics[n]['model']['calibration'];ax.plot([b['predicted'] for b in bs],[b['observed'] for b in bs],'o-',label=LABELS[n],color={'gradient_boosting':'#b66a13','structured':'#545bb8','persona':'#087e6c'}[n])
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed purchase frequency',title='Calibration | fixed 10 equal-width bins')
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.savefig(out/'calibration.png',dpi=180);plt.close(fig)
    primary=d['comparisons']['structured'];secondary=d['comparisons']['gradient_boosting']
    def contrast(c):return f"{c['difference']:+.4f}; 95% CI [{c['uncertainty']['ci95'][0]:+.4f}, {c['uncertainty']['ci95'][1]:+.4f}]"
    lines=['# BehaviorLens — research note','',finding(d),'',
'## Question','', 'Does replacing structured behavioral history with an LLM-generated persona improve 28-day category purchase predictions?', '',
'## Design','', 'The Complete Journey official package provides a representation of transactions for 2,500 frequent-shopper households at one retailer. Nonpositive quantities were excluded. The category FLUID MILK PRODUCTS was chosen using only distinct household buyers in GROCERY before dataset day 365. Day indices are relative, not inferred calendar dates.', '',
'Training cutoffs: 365, 393, 421, 449, 477. Validation: 533, 561. Test: 617, 645, 673. Labels use [cutoff, cutoff + 28); historical features use only the prior 84 days. Eligibility uses past activity only. No future activity requirement is imposed.', '',
'Fixed supervised baselines use eight recency, frequency and retailer-sales features. The LLM receives those same historical feature values. The persona summary is generated from exactly that source. Neither LLM condition receives future labels. Prompts and model snapshot were unchanged after the engineering pilot.', '',
'The 200 expansion households were selected by deterministic hash ranking, excluding all five pilot households. Their 566 eligible test points are matched across all five conditions. The pilot and aggregate baseline outcomes had already been inspected; this is exploratory work, not external preregistration.', '',
'## Results','', '| Condition | Brier | Log loss | Average precision |', '|---|---:|---:|---:|']
    for n in order:lines.append(f"| {LABELS[n]} | {metrics[n]['model']['brier']:.4f} | {metrics[n]['model']['log_loss']:.4f} | {metrics[n]['average_precision']:.4f} |")
    lines+=['',f"Primary difference, persona minus structured: **{contrast(primary)}**.",'',f"Secondary difference, persona minus gradient boosting: **{contrast(secondary)}**.",'',
'Negative differences favor persona. Percentile intervals use 5,000 paired household bootstrap draws. They are marginal intervals without multiplicity correction and do not account for shared temporal shocks. The 0.01 Brier discussion margin was set before expansion LLM outcomes; it is not a validated commercial decision threshold.', '',
*mechanism_lines(d),
'## Failure analysis','', 'The local report shows the three largest positive and three largest negative persona-minus-structured loss differences. Each example includes exact source features, generated persona, predictions, outcome, source hash and call identifiers. These are post-evaluation examples, not an unbiased estimate of failure prevalence. Numerical omissions or interpretive changes may suggest follow-up hypotheses; inspecting a summary does not establish a causal mechanism.', '',
'## Limits and next experiment','', 'One category, one retailer, one model snapshot and a single draw per condition. Missing purchase records do not establish absence of purchases elsewhere. Demographic fields are partly generic codes and are not interpreted as actual age or income. No general synthetic-population fidelity or causal claim is made. Persona length and summarization are part of the treatment; this design does not isolate all representation effects.', '',
'The next independent replication should freeze new categories and time windows, include repeated model calls and inspect summary fidelity with a prespecified rubric. Any prompt changes after viewing these outcomes require a new exploratory version and a fresh evaluation set.', '',
'## Reproducibility','',f"Run: `{d['run_id']}`. Model: `{d['protocol']['model']}`. Plan SHA-256: `{d['plan_sha256']}`.",'',f"{d['summary']['calls']} API calls; {d['failures']} failed calls. Usage-derived run cost ${d['summary']['usage_cost_usd']:.5f}; cumulative with pilot ${d['summary']['cumulative_cost_usd']:.5f}.",'',
'All figures, tables and reported differences derive from results.json. Raw requests/responses, historical snapshots and labels remain local. See the repository reproduction instructions and fixed protocol. Independent project.','']
    (out/'RESEARCH_NOTE.md').write_text('\n'.join(lines))
    (out/'EVIDENCE_CARD.md').write_text(f'''# BehaviorLens — evidence card

**Question:** Does a persona summary preserve predictive signal?

**Headline:** {headline(d)}

**Finding:** {finding(d)}

**Evidence:** 200 new households, 566 matched held-out predictions; one retailer and category; unchanged model and prompts; five pilot households excluded.

**Primary contrast:** Persona minus structured Brier {contrast(primary)}.

**Secondary contrast:** Persona minus GBDT Brier {contrast(secondary)}.

{mechanism_card(d)}

**Not claimed:** General population fidelity, causal mechanism, cross-model stability or commercial uplift.

**Inspect:** report.html → differences → local failure explorer → source hash and raw call records.

**Trace:** run `{d['run_id']}`; plan `{d['plan_sha256']}`.
''')
def mechanism_lines(d):
    m=d.get('mechanism')
    if not m:return []
    dc=m['decomposition'];h=m['by_history']
    rows=[f"| {k} | {v['n']} | {v['observed']:.2f} | {v['mean_probability']['gradient_boosting']:.2f} | {v['mean_probability']['structured']:.2f} | {v['mean_probability']['persona']:.2f} | {v['persona_minus_structured']:+.4f} |" for k,v in h.items()]
    gap=m['retention']['loss_gap']
    return ['## Where the signal went','',
f"**Facts survived.** {100*m['retention']['mean']:.0f}% of source feature values appear verbatim in generated personas (conservative: words such as “no visits” count as missing). Mean persona-minus-structured loss was {gap['all_values_retained']['persona_minus_structured']:+.4f} where every value was retained and {gap['some_values_not_verbatim']['persona_minus_structured']:+.4f} otherwise, so the loss is not explained by dropped numbers.",'',
f"**Discrimination did not.** Using the report's ten fixed bins, resolution fell from {dc['structured']['resolution']:.4f} (structured) to {dc['persona']['resolution']:.4f} (persona), and reliability error rose from {dc['structured']['reliability']:.4f} to {dc['persona']['reliability']:.4f}. Gradient boosting: {dc['gradient_boosting']['resolution']:.4f} and {dc['gradient_boosting']['reliability']:.4f}.",'',
'| Milk trips, prior 84 days | n | Observed | GBDT mean p | Structured mean p | Persona mean p | Persona − structured Δ loss |','|---|---:|---:|---:|---:|---:|---:|',*rows,'',
'Persona forecasts were most conservative for the most frequent category buyers. Both LLM conditions assigned very low probabilities to households with no recent category purchase, while gradient boosting tracked their observed rate. These groups were defined after viewing outcomes and are descriptive; they suggest hypotheses, not a mechanism.','']


def mechanism_card(d):
    m=d.get('mechanism')
    if not m:return ''
    dc=m['decomposition']
    return f"**Where the signal went (descriptive):** {100*m['retention']['mean']:.0f}% of source values kept verbatim; resolution {dc['structured']['resolution']:.4f} → {dc['persona']['resolution']:.4f} (structured → persona)."


def animate(d, output):
    """Short GIF/MP4 built from the same results; bars grow, then the primary contrast appears."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import io, shutil, subprocess
    out=Path(output);metrics=d['results'];order=sorted(metrics,key=lambda n:metrics[n]['model']['brier'])
    vals=[metrics[n]['model']['brier'] for n in order];c=d['comparisons']['structured'];lo,hi=c['uncertainty']['ci95']
    n=metrics['persona']['model']
    frames=[]
    steps=[i/18 for i in range(19)]+[1]*14+[2]*30
    for t in steps:
        fig=plt.figure(figsize=(9.6,5.4),dpi=100,facecolor='#0d1721');ax=fig.add_axes([.27,.2,.63,.5],facecolor='#0d1721')
        fig.text(.05,.9,headline(d) if t>1 else 'Does a persona preserve predictive signal?',color='#edf4fa',fontsize=20,weight='bold')
        fig.text(.05,.83,f"{n['households']} held-out households · {n['n']} matched predictions · lower Brier is better",color='#a4b5c6',fontsize=11)
        ease=min(t,1)
        ax.barh([LABELS[k] for k in order],[v*ease for v in vals],color=[COLORS[k] for k in order])
        ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.25)
        for i,v in enumerate(vals):
            if t>=1:ax.text(v+.004,i,f'{v:.4f}',va='center',color='#edf4fa',fontsize=10)
        ax.tick_params(colors='#c9d6e2',labelsize=11);ax.spines[:].set_visible(False);ax.set_xticks([])
        if t>1:
            fig.text(.05,.08,f"Persona − structured history: {c['difference']:+.4f} Brier  (95% household CI {lo:+.4f} to {hi:+.4f})",color='#56dbc0',fontsize=12,weight='bold')
        fig.text(.05,.025,'BehaviorLens · Complete Journey · one retailer, one category, one model · exploratory',color='#71849a',fontsize=8)
        buf=io.BytesIO();fig.savefig(buf,format='png',facecolor=fig.get_facecolor());plt.close(fig)
        frames.append(Image.open(buf).convert('RGB'))
    frames[0].save(out/'evidence.gif',save_all=True,append_images=frames[1:],duration=80,loop=0,optimize=True)
    if shutil.which('ffmpeg'):
        tmp=out/'_frames';tmp.mkdir(exist_ok=True)
        for i,f in enumerate(frames):f.save(tmp/f'{i:03d}.png')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','12.5','-i',str(tmp/'%03d.png'),'-vf','tpad=stop_mode=clone:stop_duration=2','-pix_fmt','yuv420p','-c:v','libx264',str(out/'evidence.mp4')],check=True)
        shutil.rmtree(tmp)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args();export(a.results,a.output);animate(json.loads(Path(a.results).read_text()),a.output)
