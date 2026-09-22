"""v2 study datasets. Study A: next-28-day purchase in five categories. Study B: coupon-campaign redemption.

Rows hold label-free inputs (`text`, `x`) and outcomes (`y`) separately; prompts are built from inputs only."""
import hashlib
from collections import Counter, defaultdict
from .household import history_text, tabular, category_history

HISTORY, HORIZON = 84, 28
A_CUTOFFS = dict(train=[365, 393, 421, 449, 477], validation=[533, 561], test=[617, 645, 673])
A_RANKS = [3, 6, 12, 24, 48]  # doubling ranks; rank 1 (fluid milk) was the v1 target
B_VALIDATION_START, B_TEST_START = 500, 575


def rank_hash(key, salt):
    return hashlib.sha256(f'{salt}:{key}'.encode()).hexdigest()


def select_categories(journey, before=365, ranks=A_RANKS):
    """Commodities ranked by distinct buyers before `before` (no outcome data), picked at fixed ranks."""
    buyers = defaultdict(set)
    for h, events in journey.events.items():
        for e in events:
            if e[0] >= before:
                break
            commodity = journey.products[e[2]][1]
            if commodity and not commodity.startswith(('NO COMMODITY', 'COUPON')):
                buyers[commodity].add(h)
    order = sorted(buyers, key=lambda c: (-len(buyers[c]), c))
    return [order[r-1] for r in ranks], {c: len(buyers[c]) for c in order[:max(ranks)]}


def study_a(journey, categories):
    rows = defaultdict(list)
    for split, cutoffs in A_CUTOFFS.items():
        for cutoff in cutoffs:
            for h in sorted(journey.events, key=int):
                base = tabular(journey, h, cutoff, HISTORY)
                if base is None:
                    continue
                future = {journey.products[e[2]][1] for e in journey.window(h, cutoff, cutoff+HORIZON)}
                x = {c: {**base, **{f'cat_{k}': v for k, v in category_history(journey, h, cutoff, c, HISTORY).items()}}
                     for c in categories}
                rows[split].append(dict(id=f'A:{h}:{cutoff}', household=h, cutoff=cutoff, x=x,
                                        y={c: int(c in future) for c in categories}))
    return dict(rows)


def campaign_features(journey, campaign, cutoff, h):
    """Engineered household-campaign features for supervised baselines (LLMs get the raw record and offer instead)."""
    info = journey.campaigns[campaign]
    products = journey.coupon_products[campaign]
    covered = {journey.products[p][1] for p in products if p in journey.products}
    prior = {c for hh, c in journey.received if hh == h and journey.campaigns[c]['start'] < cutoff}
    reds = [c for day, c in journey.redemptions.get(h, []) if day < cutoff]
    hist = category_history(journey, h, cutoff, covered, HISTORY)
    exact = {e[1] for e in journey.window(h, cutoff-HISTORY, cutoff) if e[2] in products}
    return dict(duration=info['end']-info['start']+1, eligible_products=len(products), covered_categories=len(covered),
                trips_with_eligible_product=len(exact), trips_with_covered_category=hist['trips'],
                spend_covered_category=hist['spend'], prior_campaigns=len(prior),
                prior_campaigns_redeemed=len(prior & set(reds)), prior_coupon_redemptions=len(reds),
                **{f'type_{t}': int(info['type'] == t) for t in ('TypeA', 'TypeB', 'TypeC')})


def offer_text(journey, campaign):
    """Household-independent campaign description shown identically to every LLM condition."""
    info = journey.campaigns[campaign]
    products = journey.coupon_products[campaign]
    counts = Counter(journey.products[p][1] for p in products if p in journey.products)
    selection = ('TypeA campaign: the household received 16 coupons chosen from the pool below based on its past '
                 'purchases; which 16 is not known.' if info['type'] == 'TypeA' else
                 f"{info['type']} campaign: the household received every coupon in it.")
    return (f"Starts on the prediction day and runs for {info['end']-info['start']+1} days. {selection}\n"
            f"Coupons are redeemable on {len(products)} products. Covered categories (eligible products), largest first: "
            + '; '.join(f'{c} ({n})' for c, n in counts.most_common(15)) + ('; ...' if len(counts) > 15 else ''))


def coupon_history(journey, h, cutoff):
    prior = {c for hh, c in journey.received if hh == h and journey.campaigns[c]['start'] < cutoff}
    reds = [c for day, c in journey.redemptions.get(h, []) if day < cutoff]
    return (f'- Coupon campaigns before the prediction day: received {len(prior)}; redeemed at least one coupon in '
            f'{len(prior & set(reds))} of them; {len(reds)} coupons redeemed in total.')


def b_split(info):
    if info['start'] < B_VALIDATION_START:
        return 'train'
    if info['start'] < B_TEST_START:
        # A validation campaign still running when the test period starts would overlap test outcomes.
        return 'validation' if info['end'] < B_TEST_START + HORIZON else None
    return 'test'


def study_b(journey):
    redeemed = {(h, c) for h, reds in journey.redemptions.items() for _, c in reds}
    rows, excluded = defaultdict(list), Counter()
    for h, c in sorted(journey.received, key=lambda r: (int(r[1]), int(r[0]))):
        info = journey.campaigns[c]
        split = b_split(info)
        if split is None:
            excluded['campaign_overlaps_test'] += 1
            continue
        base = tabular(journey, h, info['start'], HISTORY)
        if base is None:
            excluded['inactive_prior_84_days'] += 1
            continue
        x = {**base, **campaign_features(journey, c, info['start'], h)}
        rows[split].append(dict(id=f'B:{h}:{c}', household=h, campaign=c, cutoff=info['start'], x=x,
                                y=int((h, c) in redeemed)))
    return dict(rows), dict(excluded)


def sample(rows, households, salt, exclude=()):
    """Deterministic, outcome-blind household sample; keeps every row of each selected household."""
    ids = sorted({r['household'] for r in rows} - set(exclude), key=lambda h: rank_hash(h, salt))
    chosen = set(ids[:households])
    return [r for r in rows if r['household'] in chosen]


def attach_history(journey, rows, focus=()):
    for r in rows:
        r['history'] = history_text(journey, r['household'], r['cutoff'], HISTORY, focus)
        if r['history'] is None:
            raise ValueError('Sampled row without history')
        if 'campaign' in r:
            r['history'] += '\n' + coupon_history(journey, r['household'], r['cutoff'])
            r['offer'] = offer_text(journey, r['campaign'])
    return rows
