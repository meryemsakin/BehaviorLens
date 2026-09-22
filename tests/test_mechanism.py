import unittest
from behaviorlens.mechanism import murphy, retained, diagnose


def trace(y, persona, structured, visits, text='visits 2'):
    return dict(observed=y, source=dict(features=dict(category_visits_84=visits, visits_84=2)), persona_text=text,
                probabilities=dict(persona=persona, structured=structured, gradient_boosting=.5),
                difference=(persona-y)**2-(structured-y)**2)


class MechanismTests(unittest.TestCase):
    def test_constant_forecast_has_no_resolution(self):
        m = murphy([(1, .5), (0, .5)])
        self.assertEqual(m['resolution'], 0)
        self.assertEqual(m['reliability'], 0)
        self.assertEqual(m['uncertainty'], .25)

    def test_perfect_forecast_decomposes_brier(self):
        m = murphy([(1, 1.0), (0, 0.0)])
        self.assertAlmostEqual(m['reliability'] - m['resolution'] + m['uncertainty'], 0)

    def test_retention_is_verbatim(self):
        r = retained(dict(features=dict(a=2, b=7.09, c=0)), 'Made 2 visits, spent 7.09, no category purchases.')
        self.assertEqual(r, dict(a=True, b=True, c=False))

    def test_history_groups_and_retention_split(self):
        d = diagnose([trace(1, .6, .8, 7, '7 milk trips, 2 visits'), trace(0, .1, .1, 0, 'no visits')])
        self.assertEqual(d['by_history']['6+']['n'], 1)
        self.assertAlmostEqual(d['by_history']['6+']['persona_minus_structured'], .16-.04)
        self.assertEqual(d['retention']['loss_gap']['all_values_retained']['n'], 1)
        with self.assertRaises(ValueError):
            diagnose([])
