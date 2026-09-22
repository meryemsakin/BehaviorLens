"""Post-hoc diagnostics of where persona loss arises. Aggregate outputs only; descriptive, not causal."""
import re
from collections import defaultdict

INTENSITY=[('none',0,0),('1–2',1,2),('3–5',3,5),('6+',6,10**9)]
CONDITIONS=['gradient_boosting','structured','persona']


def murphy(pairs, bins=10):
    """Brier = reliability - resolution + uncertainty, using the report's fixed equal-width bins.
    The identity is approximate because within-bin forecast variance is ignored."""
    n=len(pairs);base=sum(y for y,_ in pairs)/n;groups=defaultdict(list)
    for y,p in pairs:groups[min(int(p*bins),bins-1)].append((y,p))
    rel=res=0
    for g in groups.values():
        m=len(g);pm=sum(p for _,p in g)/m;om=sum(y for y,_ in g)/m
        rel+=m*(pm-om)**2;res+=m*(om-base)**2
    return dict(reliability=rel/n,resolution=res/n,uncertainty=base*(1-base))


def retained(source, text):
    """Which source feature values appear verbatim as numbers in generated text.
    Words such as 'no visits' count as not retained, so this is a conservative lower bound."""
    found=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',text.replace(',',''))]
    return {k:any(abs(float(v)-x)<1e-9 for x in found) for k,v in source['features'].items()}


def diagnose(traces):
    if not traces:raise ValueError('No traces')
    out=dict(decomposition={},by_history={},retention={},
             note='Post-evaluation descriptive diagnostics. Groups were defined after viewing outcomes; no significance claim.')
    for c in CONDITIONS:
        pairs=[(t['observed'],t['probabilities'][c]) for t in traces]
        out['decomposition'][c]=dict(murphy(pairs),mean_probability=sum(p for _,p in pairs)/len(pairs),
                                     distinct_values=len({round(p,4) for _,p in pairs}))
    out['observed_rate']=sum(t['observed'] for t in traces)/len(traces)
    for label,lo,hi in INTENSITY:
        g=[t for t in traces if lo<=t['source']['features']['category_visits_84']<=hi]
        if not g:continue
        out['by_history'][label]=dict(n=len(g),observed=sum(t['observed'] for t in g)/len(g),
            mean_probability={c:sum(t['probabilities'][c] for t in g)/len(g) for c in CONDITIONS},
            persona_minus_structured=sum(t['difference'] for t in g)/len(g))
    kept=[retained(t['source'],t['persona_text']) for t in traces]
    out['retention']['by_feature']={k:sum(r[k] for r in kept)/len(kept) for k in kept[0]}
    out['retention']['mean']=sum(sum(r.values())/len(r) for r in kept)/len(kept)
    full=[t for t,r in zip(traces,kept) if all(r.values())];part=[t for t,r in zip(traces,kept) if not all(r.values())]
    out['retention']['loss_gap']={name:dict(n=len(g),persona_minus_structured=sum(t['difference'] for t in g)/len(g))
                                  for name,g in [('all_values_retained',full),('some_values_not_verbatim',part)] if g}
    return out
