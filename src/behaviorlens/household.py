"""Item-level household histories for v2 studies. Every view is computed from records strictly before a cutoff."""
import csv
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

DEMOGRAPHICS = [('classification_1', 'age_band'), ('classification_3', 'income_level'), ('HOMEOWNER_DESC', 'homeowner'),
                ('classification_5', 'household_composition'), ('classification_4', 'household_size'),
                ('KID_CATEGORY_DESC', 'children'), ('classification_2', 'marital_code')]


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        yield from csv.DictReader(f)


def code(value):
    """Ordinal index embedded in anonymised codes such as 'Age Group4' or 'Level5'; None when absent."""
    digits = ''.join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


class Journey:
    """Positive-quantity item events per household, sorted by day, plus coupon and campaign tables."""

    def __init__(self, raw):
        raw = Path(raw)
        self.products = {r['PRODUCT_ID']: (r['DEPARTMENT'].strip(), r['COMMODITY_DESC'].strip(), r['BRAND'].strip())
                         for r in rows(raw/'product.csv')}
        self.demographics = {r['household_key']: r for r in rows(raw/'hh_demographic.csv')}
        events = defaultdict(list)
        for r in rows(raw/'transaction_data.csv'):
            if float(r['QUANTITY']) <= 0:
                continue
            sales = float(r['SALES_VALUE'])
            if not math.isfinite(sales):
                raise ValueError('Non-finite sales')
            events[r['household_key']].append((int(r['DAY']), r['BASKET_ID'], r['PRODUCT_ID'], sales,
                                               float(r['RETAIL_DISC']) < 0, float(r['COUPON_DISC']) < 0))
        self.events = {h: sorted(v) for h, v in events.items()}
        self.days = {h: [e[0] for e in v] for h, v in self.events.items()}
        self.campaigns = {r['CAMPAIGN']: dict(type=r['DESCRIPTION'], start=int(r['START_DAY']), end=int(r['END_DAY']))
                          for r in rows(raw/'campaign_desc.csv')}
        self.received = [(r['household_key'], r['CAMPAIGN']) for r in rows(raw/'campaign_table.csv')]
        self.coupon_products = defaultdict(set)
        for r in rows(raw/'coupon.csv'):
            self.coupon_products[r['CAMPAIGN']].add(r['PRODUCT_ID'])
        self.redemptions = defaultdict(list)
        for r in rows(raw/'coupon_redempt.csv'):
            self.redemptions[r['household_key']].append((int(r['DAY']), r['CAMPAIGN']))

    def window(self, h, start, end):
        """Events with start <= day < end."""
        d = self.days.get(h, [])
        return self.events.get(h, [])[bisect_left(d, start):bisect_left(d, end)]

    def profile(self, h):
        d = self.demographics.get(h)
        return {name: d[col].strip() for col, name in DEMOGRAPHICS} if d else None


def summarize(journey, h, cutoff, history=84, top=40):
    """Label-free structured view of the prior `history` days; None if the household was inactive."""
    past = journey.window(h, cutoff-history, cutoff)
    if not past:
        return None
    recent = [e for e in past if e[0] >= cutoff-28]
    trips = lambda es: len({e[1] for e in es})
    dept, cat_trips, cat_last, private = Counter(), defaultdict(set), {}, 0
    for day, basket, product, sales, _, _ in past:
        department, commodity, brand = journey.products[product]
        dept[department] += sales
        cat_trips[commodity].add(basket)
        cat_last[commodity] = max(cat_last.get(commodity, 0), day)
        private += brand == 'Private'
    spend = sum(e[3] for e in past)
    categories = sorted(cat_trips, key=lambda c: (-len(cat_trips[c]), c))
    return dict(
        trips_28=trips(recent), trips_84=trips(past), spend_28=round(sum(e[3] for e in recent), 2), spend_84=round(spend, 2),
        days_since_last_trip=cutoff-past[-1][0], items_84=len(past),
        private_label_share=round(private/len(past), 3), promo_item_share=round(sum(e[4] for e in past)/len(past), 3),
        coupon_items_84=sum(e[5] for e in past), distinct_categories_84=len(cat_trips),
        department_spend_share={k: round(v/spend, 3) for k, v in dept.most_common(8) if spend > 0},
        categories={c: dict(trips=len(cat_trips[c]), days_since_last=cutoff-cat_last[c]) for c in categories[:top]},
        categories_listed=min(top, len(categories)))


def history_text(journey, h, cutoff, history=84, focus=()):
    """Deterministic plain-text rendering shared by every LLM condition.
    `focus` categories are always reported, including when absent, so no condition depends on the top-40 cut."""
    s = summarize(journey, h, cutoff, history)
    if s is None:
        return None
    p = journey.profile(h)
    lines = ['Household profile (anonymised retailer codes; higher numbers are higher bands): ' +
             (', '.join(f'{k}={v}' for k, v in p.items()) if p else 'not available')]
    lines.append(f'Shopping in the {history} days before the prediction day:')
    lines.append(f"- Trips: {s['trips_28']} in last 28 days, {s['trips_84']} in last 84 days; last trip {s['days_since_last_trip']} days ago.")
    lines.append(f"- Spend: ${s['spend_28']:.2f} in last 28 days, ${s['spend_84']:.2f} in last 84 days; {s['items_84']} items.")
    lines.append(f"- Private-label share of items {s['private_label_share']:.0%}; items bought on retailer discount "
                 f"{s['promo_item_share']:.0%}; items with a manufacturer coupon: {s['coupon_items_84']}.")
    lines.append('- Spend share by department: ' + ', '.join(f'{k} {v:.0%}' for k, v in s['department_spend_share'].items()))
    lines.append(f"- Categories bought ({s['categories_listed']} of {s['distinct_categories_84']} shown, most frequent first; "
                 'trips containing the category, days since last):')
    lines.append('  ' + '; '.join(f"{c} {v['trips']} ({v['days_since_last']}d)" for c, v in s['categories'].items()))
    if focus:
        shown = []
        for c in focus:
            v = category_history(journey, h, cutoff, c, history)
            shown.append(f"{c} {v['trips']} ({v['days_since_last']}d)" if v['trips'] else f'{c} not bought in {history} days')
        lines.append('- Categories of interest (trips, days since last): ' + '; '.join(shown))
    return '\n'.join(lines)


def category_history(journey, h, cutoff, commodities, history=84):
    """Trips, spend and recency for one commodity or a set of commodities in the prior window."""
    wanted = {commodities} if isinstance(commodities, str) else set(commodities)
    past = [e for e in journey.window(h, cutoff-history, cutoff) if journey.products[e[2]][1] in wanted]
    recent = [e for e in past if e[0] >= cutoff-28]
    return dict(trips=len({e[1] for e in past}), trips_28=len({e[1] for e in recent}),
                spend=round(sum(e[3] for e in past), 2),
                days_since_last=cutoff-max(e[0] for e in past) if past else history+1)


def tabular(journey, h, cutoff, history=84):
    """Numeric household features for supervised baselines; category- or campaign-specific ones are added by studies."""
    s = summarize(journey, h, cutoff, history)
    if s is None:
        return None
    p = journey.profile(h)
    demo = {f'demo_{name}': (code(p[name]) if p and code(p[name]) is not None else -1)
            for name in ('age_band', 'income_level', 'household_size')}
    return dict(trips_28=s['trips_28'], trips_84=s['trips_84'], spend_28=s['spend_28'], spend_84=s['spend_84'],
                days_since_last_trip=s['days_since_last_trip'], private_label_share=s['private_label_share'],
                promo_item_share=s['promo_item_share'], coupon_items_84=s['coupon_items_84'],
                distinct_categories_84=s['distinct_categories_84'], demographics_available=int(p is not None),
                homeowner=int(bool(p) and p['homeowner'] == 'Homeowner'), **demo)
