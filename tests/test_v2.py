import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from behaviorlens.v2_analysis import cluster_ci, long_table, add_fitted, fidelity
from behaviorlens.batch import Ledger, bound, parse
from behaviorlens.prompts import requests_for, persona_request
from behaviorlens.studies import b_split


class ClusterBootstrapTests(unittest.TestCase):
    def test_constant_difference_has_degenerate_interval(self):
        r = cluster_ci(['a', 'a', 'b', 'c'], [.1, .1, .1, .1], .95, repeats=200)
        self.assertAlmostEqual(r['estimate'], .1)
        self.assertAlmostEqual(r['ci'][0], .1)
        self.assertAlmostEqual(r['ci'][1], .1)
        self.assertEqual(r['households'], 3)

    def test_resamples_households_not_labels(self):
        # One household carries all the signal; clustering keeps its labels together, so the interval reaches 0.
        r = cluster_ci(['a']*50 + [str(i) for i in range(10)], [1.]*50 + [0.]*10, .95, repeats=500)
        self.assertEqual(r['ci'][0], 0.)


class TableTests(unittest.TestCase):
    def test_rows_missing_a_condition_are_dropped_everywhere(self):
        labels = {'A:1:617': {'c': 1}, 'A:2:617': {'c': 0}}
        base = {k: {'c': dict(prevalence=.5, logistic=.5, gradient_boosting=.5)} for k in labels}
        llm = {'A:1:617': {'structured': {'c': .7}, 'persona': {'c': .6}}, 'A:2:617': {'structured': {'c': .2}}}
        table, dropped = long_table(list(labels), labels, base, llm, ['c'])
        self.assertEqual(dropped, ['A:2:617'])
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]['household'], '1')

    def test_calibration_fits_on_validation_only(self):
        rng = np.random.default_rng(0)
        def rows(n):
            out = []
            for i in range(n):
                y = int(rng.random() < .5)
                out.append(dict(id=f'A:{i}:1', household=str(i), key='c', y=y, gradient_boosting=.5,
                                structured=.9 if y else .6, persona=.5))
            return out
        test = add_fitted(rows(200), rows(200))
        self.assertTrue(all('hybrid' in r and 'structured_cal' in r for r in test))
        low = np.mean([r['structured_cal'] for r in test if r['structured'] == .6])
        self.assertLess(low, .2)  # recalibration pulls the over-confident 0.6 towards its observed rate

    def test_fidelity_perfect_ranking(self):
        t = [dict(y=y, key=k, gradient_boosting=p) for k, y, p in
             [('a', 1, .9), ('a', 1, .9), ('b', 0, .1), ('b', 1, .5), ('c', 0, .05), ('c', 0, .05)]]
        f = fidelity(t, lambda r: r['key'])
        self.assertAlmostEqual(f['conditions']['gradient_boosting']['spearman'], 1.0)


class BatchTests(unittest.TestCase):
    def test_ledger_counts_inflight_bounds(self):
        with tempfile.TemporaryDirectory() as t:
            ledger = Ledger(Path(t)/'l.jsonl', 10)
            ledger.log(dict(event='submitted', batch='x', bound=4.))
            ledger.log(dict(event='submitted', batch='y', bound=3.))
            ledger.log(dict(event='settled', batch='x', settled=1.5, bound=4.))
            self.assertAlmostEqual(ledger.committed(), 4.5)
            with self.assertRaises(ValueError):
                Ledger(Path(t)/'m.jsonl', 100)

    def test_parse_rejects_out_of_range(self):
        ok = {'status': 'completed', 'output': [{'content': [{'type': 'output_text', 'text': json.dumps({'probabilities': {'a': .2}})}]}]}
        self.assertEqual(parse(ok, False), {'a': .2})
        bad = {'status': 'completed', 'output': [{'content': [{'type': 'output_text', 'text': json.dumps({'probability': 1.2})}]}]}
        with self.assertRaises(ValueError):
            parse(bad, False)

    def test_persona_prompt_never_sees_record(self):
        row = dict(id='B:1:18', history='SECRET RECORD', offer='Offer text')
        body = persona_request(row, 'B', 'gpt-5.4-mini-2026-03-17', 'A thrifty shopper.')
        self.assertNotIn('SECRET RECORD', body['input'])
        self.assertIn('Offer text', body['input'])
        stage1 = requests_for(row, 'B', 'gpt-5.4-mini-2026-03-17')
        self.assertIn('SECRET RECORD', stage1['B:1:18|structured']['input'])
        self.assertNotIn('Offer text', stage1['B:1:18|persona_gen']['input'])
        self.assertGreater(bound(stage1['B:1:18|structured']), 0)


class SplitTests(unittest.TestCase):
    def test_validation_campaign_overlapping_test_is_excluded(self):
        self.assertIsNone(b_split(dict(start=547, end=708)))
        self.assertEqual(b_split(dict(start=504, end=551)), 'validation')
        self.assertEqual(b_split(dict(start=587, end=642)), 'test')
        self.assertEqual(b_split(dict(start=412, end=460)), 'train')
