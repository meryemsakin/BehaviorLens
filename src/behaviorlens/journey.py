"""Complete Journey ingestion. Feature building only consumes pre-cutoff records."""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

FEATURES = ['visits_28', 'visits_84', 'category_visits_28', 'category_visits_84',
            'category_recency', 'retailer_sales_28', 'retailer_sales_84', 'category_sales_84']


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        yield from csv.DictReader(f)


def load(raw):
    raw = Path(raw)
    products = {}
    for r in read_csv(raw/'product.csv'):
        if r['PRODUCT_ID'] in products:
            raise ValueError('Duplicate product ID')
        products[r['PRODUCT_ID']] = (r['DEPARTMENT'].strip(), r['COMMODITY_DESC'].strip())
    demographics = {r['household_key']: r for r in read_csv(raw/'hh_demographic.csv')}
    households = defaultdict(list)
    skipped = Counter()
    rows = 0
    min_day, max_day = 10**9, -1
    for r in read_csv(raw/'transaction_data.csv'):
        rows += 1
        day = int(r['DAY'])
        min_day, max_day = min(min_day, day), max(max_day, day)
        if r['PRODUCT_ID'] not in products:
            raise ValueError('Transaction product absent from product table')
        if float(r['QUANTITY']) <= 0:
            skipped['nonpositive_quantity'] += 1
            continue
        sales = float(r['SALES_VALUE'])
        if not __import__('math').isfinite(sales):
            raise ValueError('Non-finite sales')
        department, category = products[r['PRODUCT_ID']]
        households[r['household_key']].append((day, r['BASKET_ID'], category, sales, department))
    for events in households.values():
        events.sort()
    audit = dict(transaction_rows=rows, retained_rows=sum(map(len, households.values())),
                 exclusions=dict(skipped), households=len(households), day_min=min_day, day_max=max_day,
                 products=len(products), demographic_households=len(demographics),
                 missing_demographics=sum(h not in demographics for h in households),
                 source_fingerprints={n: fingerprint(raw/n) for n in
                                      ['transaction_data.csv', 'product.csv', 'hh_demographic.csv']})
    return households, demographics, audit


def select_category(households, before):
    buyers = defaultdict(set)
    for h, events in households.items():
        for day, _, category, _, department in events:
            if day < before and department == 'GROCERY' and category:
                buyers[category].add(h)
    if not buyers:
        raise ValueError('No eligible categories in training prefix')
    return sorted(buyers, key=lambda c: (-len(buyers[c]), c))[0]


def features(events, cutoff, category, history=84):
    past = [r for r in events if cutoff-history <= r[0] < cutoff]
    if not past:
        return None
    recent = [r for r in past if r[0] >= cutoff-28]
    cat = [r for r in past if r[2] == category]
    cat_recent = [r for r in recent if r[2] == category]
    visits = lambda rs: len({r[1] for r in rs})
    return [visits(recent), visits(past), visits(cat_recent), visits(cat),
            cutoff-max(r[0] for r in cat) if cat else history+1,
            sum(r[3] for r in recent), sum(r[3] for r in past), sum(r[3] for r in cat)]


def make_rows(households, demographics, cutoffs, category, horizon=28, history=84):
    rows = []
    for cutoff in cutoffs:
        for h in sorted(households, key=int):
            events = households[h]
            x = features(events, cutoff, category, history)
            if x is None:
                continue
            y = int(any(cutoff <= r[0] < cutoff+horizon and r[2] == category for r in events))
            snapshot = dict(zip(FEATURES, x))
            rows.append(dict(household_id=h, cutoff=cutoff, history_end=cutoff-1,
                             label_end=cutoff+horizon, y=y, features=snapshot,
                             segment='demographics_available' if h in demographics else 'demographics_missing'))
    return rows


def validate_protocol(p):
    if p['horizon_days'] <= 0 or p['history_days'] < 28:
        raise ValueError('Invalid horizon or history')
    windows = []
    for name in ('train_cutoffs', 'validation_cutoffs', 'test_cutoffs'):
        days = p[name]
        if not days or days != sorted(set(days)):
            raise ValueError('Cutoffs must be nonempty, unique and sorted')
        windows += days
    if windows != sorted(set(windows)):
        raise ValueError('Split windows must be chronological and unique')
    if any(a+p['horizon_days'] > b for a, b in zip(windows, windows[1:])):
        raise ValueError('Label windows overlap')
    if p['category_selection_before_day'] > windows[0]:
        raise ValueError('Category selection sees beyond training prefix')
