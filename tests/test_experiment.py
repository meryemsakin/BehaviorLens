import tempfile
import unittest
from pathlib import Path
from behaviorlens.experiment import Budget

class BudgetTests(unittest.TestCase):
    def test_inflight_and_prior_count(self):
        with tempfile.TemporaryDirectory() as t:
            b=Budget(.5,.1,Path(t)/'ledger')
            k=b.reserve(.3,'p','s')
            with self.assertRaises(ValueError):b.reserve(.2,'q','s')
            b.settle(k,.01,{})
            b.reserve(.3,'q','s')
            self.assertAlmostEqual(b.spent,.11)
    def test_uncertain_reservation_not_reclaimed(self):
        with tempfile.TemporaryDirectory() as t:
            b=Budget(.5,.01,Path(t)/'ledger')
            b.reserve(.48,'p','s')
            with self.assertRaises(ValueError):b.reserve(.02,'q','s')
    def test_invalid_settlement(self):
        with tempfile.TemporaryDirectory() as t:
            b=Budget(.5,0,Path(t)/'ledger');k=b.reserve(.1,'p','s')
            with self.assertRaises(ValueError):b.settle(k,.2,{})
            self.assertEqual(b.pending[k],.1)
