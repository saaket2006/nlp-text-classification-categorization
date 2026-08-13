import unittest
import numpy as np
from core.router import TieredRouter
from core.uncertainty import UncertaintyEngine

class MockTier1:
    def __init__(self, idx=0, probs=None, conf=0.9):
        self.idx = idx
        self.probs = probs if probs is not None else np.array([0.9, 0.1])
        self.conf = conf
    def predict(self, text):
        return self.idx, self.probs, self.conf

class MockTier2:
    def __init__(self, labels=None, conf=0.95):
        self.labels = labels if labels is not None else ["pos"]
        self.conf = conf
    def predict(self, text, categories, t1_label=None, sample_idx=None):
        return {"labels": self.labels, "confidence": self.conf, "vote_agreement": 1.0}, "raw"

class TestRouter(unittest.TestCase):
    def test_router_standard_mode(self):
        t1 = MockTier1(idx=0, conf=0.95)
        t2 = MockTier2()
        ue = UncertaintyEngine(0.5, 0.7)
        categories = ["neg", "pos"]
        router = TieredRouter(t1, t2, ue, categories, entropy_threshold=0.5, conf_threshold=0.7, mode="STANDARD")

        res = router.process_sample("Sample text")
        self.assertEqual(res["tier"], 1)
        self.assertEqual(res["final_label"], ["neg"])

    def test_router_tier2_only_mode(self):
        t1 = MockTier1(idx=0, conf=0.5)
        t2 = MockTier2(labels=["pos"], conf=0.98)
        ue = UncertaintyEngine(0.5, 0.7)
        categories = ["neg", "pos"]
        router = TieredRouter(t1, t2, ue, categories, entropy_threshold=0.5, conf_threshold=0.7, mode="TIER2_ONLY")

        res = router.process_sample("Sample text")
        self.assertEqual(res["tier"], 2)
        self.assertEqual(res["final_label"], ["pos"])
        self.assertEqual(res["confidence"], 0.98)

if __name__ == "__main__":
    unittest.main()
