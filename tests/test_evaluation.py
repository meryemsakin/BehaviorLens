import unittest
from behaviorlens.evaluation import evaluate, metrics, validate, paired_bootstrap


def row(h='a', y=1, p=.8, baseline=.5):
    return dict(household_id=h, y=y, probability=p, baseline_probability=baseline,
                cutoff='2024-04-01', history_end='2024-03-31', label_end='2024-04-29')


class EvaluationTests(unittest.TestCase):
    def test_known_brier(self):
        self.assertAlmostEqual(metrics([row(), row('b', 0, .2)])['brier'], .04)

    def test_boundary_calibration(self):
        self.assertEqual(metrics([row(p=1), row('b', 0, 0)])['ece'], 0)

    def test_invalid_inputs(self):
        for bad in [float('nan'), float('inf'), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                validate([row(p=bad)])

    def test_temporal_leakage(self):
        r = row()
        r['history_end'] = r['cutoff']
        with self.assertRaises(ValueError):
            validate([r])

    def test_duplicates(self):
        with self.assertRaises(ValueError):
            evaluate([row(), row()])

    def test_paired_zero_difference(self):
        result = evaluate([row(p=.5), row('b', 0, .5)], repeats=100)
        self.assertEqual(result['uncertainty']['ci95'], [0, 0])

    def test_cluster_resampling(self):
        # One household's repeated rows must travel together: with two clusters,
        # only these three bootstrap means are possible.
        rows = [row(p=1, baseline=0), row(p=0, baseline=1), row('b', p=1, baseline=0)]
        result = paired_bootstrap(rows, 1000)
        self.assertEqual(result['ci95'], [-1, 0])


if __name__ == '__main__':
    unittest.main()
