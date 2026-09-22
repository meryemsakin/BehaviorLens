import unittest
from behaviorlens.journey import features, make_rows, select_category, validate_protocol


class JourneyTests(unittest.TestCase):
    def test_future_invariance(self):
        past = [(9, 'b', 'milk', 2., 'GROCERY')]
        future = [(10, 'c', 'milk', 999., 'GROCERY')]
        self.assertEqual(features(past, 10, 'milk'), features(past+future, 10, 'milk'))
        rows = make_rows({'1': past+future}, {}, [10], 'milk')
        self.assertEqual(rows[0]['y'], 1)
        self.assertEqual(rows[0]['features']['category_sales_84'], 2.)

    def test_label_right_boundary(self):
        events = [(9, 'b', 'bread', 2., 'GROCERY'), (38, 'c', 'milk', 1., 'GROCERY')]
        self.assertEqual(make_rows({'1': events}, {}, [10], 'milk')[0]['y'], 0)

    def test_selection_ignores_future(self):
        events = {'1': [(1,'a','milk',1.,'GROCERY'), (20,'b','bread',1.,'GROCERY')]}
        self.assertEqual(select_category(events, 10), 'milk')

    def test_cold_household_not_selected_from_future(self):
        self.assertEqual(make_rows({'1': [(10,'a','milk',1.,'GROCERY')]}, {}, [10], 'milk'), [])

    def test_overlapping_windows_rejected(self):
        p = dict(horizon_days=28, history_days=84, category_selection_before_day=10,
                 train_cutoffs=[20], validation_cutoffs=[30], test_cutoffs=[60])
        with self.assertRaises(ValueError):
            validate_protocol(p)
