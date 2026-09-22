import argparse
import csv
import hashlib
import html
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate matched binary behavioral predictions')
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/report'))
    parser.add_argument('--evidence', choices=['fixture', 'observed'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    try:
        with args.predictions.open(newline='') as f:
            rows = list(csv.DictReader(f))
        result = evaluate(rows, args.bootstrap, args.seed)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    result['manifest'] = dict(evidence=args.evidence,
                              sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                              created_at=datetime.now(timezone.utc).isoformat(),
                              python=platform.python_version(), version='0.1.0',
                              seed=args.seed, bootstrap=args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, allow_nan=False)
    (args.output/'report.json').write_text(payload)
    label = ('ILLUSTRATIVE FIXTURE — NOT RESEARCH EVIDENCE' if args.evidence == 'fixture'
             else 'OBSERVED DATA — provenance supplied by operator')
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>BehaviorLens · Evidence report</title><style>
body{{font:16px system-ui;background:#101820;color:#eaf1f4;max-width:960px;margin:60px auto;padding:24px}}
h1{{font-size:48px}} .label{{color:#ffcb77}} pre{{white-space:pre-wrap;background:#192630;padding:24px}}
</style><h1>BehaviorLens</h1><p class="label">{label}</p>
<p>Brier difference vs baseline: <strong>{result['brier_difference']:.4f}</strong> (negative favors model)</p>
<p>Paired household bootstrap 95% interval: {html.escape(str(result['uncertainty']['ci95']))}</p>
<p>Timestamp metadata validated. Full feature provenance has not been audited.</p>
<h2>Reproducible evidence</h2><pre>{html.escape(payload)}</pre></html>'''
    (args.output/'report.html').write_text(body)
    print(args.output/'report.html')


if __name__ == '__main__':
    main()
