import unittest
from behaviorlens.openai_pilot import payload,reserve_bound,parse_response

class PilotTests(unittest.TestCase):
    def test_cost_reservation(self):
        self.assertGreater(reserve_bound(payload('test')),512*1.6/1000000)
        with self.assertRaises(ValueError):reserve_bound(payload('x'*13000))
    def test_strict_probability(self):
        for v in ['true','-0.1','1.1','NaN']:
            r={'status':'completed','output':[{'content':[{'type':'output_text','text':'{"probability":'+v+'}'}]}]}
            with self.assertRaises(ValueError):parse_response(r)
    def test_incomplete_response(self):
        with self.assertRaises(ValueError):parse_response({'status':'incomplete'})
    def test_payload_privacy(self):
        self.assertFalse(payload('test')['store'])
        self.assertTrue(payload('test')['text']['format']['strict'])
