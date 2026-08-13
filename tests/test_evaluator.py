import unittest
from metrics.evaluator import PipelineEvaluator, calculate_multi_seed_stats

class TestEvaluator(unittest.TestCase):
    def test_evaluator_metrics_computation(self):
        evaluator = PipelineEvaluator()

        evaluator.add_result(
            prediction={"tier": 1, "t1_label": "pos", "final_label": ["pos"], "confidence": 0.9, "t1_confidence": 0.9},
            ground_truth="pos"
        )
        evaluator.add_result(
            prediction={"tier": 2, "t1_label": "neg", "final_label": ["pos"], "confidence": 0.85, "t1_confidence": 0.6},
            ground_truth="pos"
        )
        evaluator.add_result(
            prediction={"tier": 3, "t1_label": "neg", "final_label": ["neg"], "confidence": 1.0, "t1_confidence": 0.4},
            ground_truth="neg"
        )

        metrics = evaluator.calculate_metrics(categories=["neg", "pos"])
        self.assertEqual(metrics["total_samples"], 3)
        self.assertEqual(metrics["accuracy_final"], 1.0)
        self.assertIn("ece_final", metrics)
        self.assertIn("brier_score", metrics)

    def test_multi_seed_stats_computation(self):
        runs = [
            {"seed": 42, "accuracy_final": 0.90, "accuracy_t1": 0.85, "human_effort_ratio": 0.002, "picr": 25.0, "f1_macro": 0.90, "f1_weighted": 0.90, "ece_final": 0.05, "net_utility": 0.04},
            {"seed": 123, "accuracy_final": 0.92, "accuracy_t1": 0.86, "human_effort_ratio": 0.0, "picr": None, "f1_macro": 0.92, "f1_weighted": 0.92, "ece_final": 0.04, "net_utility": 0.06},
        ]

        stats = calculate_multi_seed_stats(runs)
        self.assertEqual(stats["valid_picr_runs"], "1/2")
        self.assertIn("ci95", stats)
        self.assertEqual(stats["mean"]["accuracy_final"], 0.91)

if __name__ == "__main__":
    unittest.main()
