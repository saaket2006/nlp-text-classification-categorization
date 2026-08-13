import unittest
import numpy as np
import math
from metrics.evaluator import PipelineEvaluator, calculate_multi_seed_stats

class TestCalibrationAndCost(unittest.TestCase):
    def test_multiclass_brier_score_computation(self):
        evaluator = PipelineEvaluator()
        
        # Add result 1: GT is class 'A' (index 0). Router predicts 'A' with 0.8 conf, other classes split 0.2
        # final_probs: [0.8, 0.1, 0.1]
        evaluator.add_result(
            prediction={
                "tier": 2, 
                "t1_label": "A", 
                "final_label": ["A"], 
                "confidence": 0.8, 
                "t1_confidence": 0.5,
                "final_probs": [0.8, 0.1, 0.1],
                "t1_probs": [0.5, 0.3, 0.2]
            },
            ground_truth="A"
        )
        
        # Add result 2: GT is class 'B' (index 1). Router predicts 'A' (incorrect).
        # final_probs: [0.7, 0.2, 0.1]
        evaluator.add_result(
            prediction={
                "tier": 2, 
                "t1_label": "A", 
                "final_label": ["A"], 
                "confidence": 0.7, 
                "t1_confidence": 0.6,
                "final_probs": [0.7, 0.2, 0.1],
                "t1_probs": [0.6, 0.2, 0.2]
            },
            ground_truth="B"
        )

        categories = ["A", "B", "C"]
        metrics = evaluator.calculate_metrics(categories=categories)
        
        # Expected Brier score calculation:
        # Sample 1 (GT: A [1, 0, 0], Pred: [0.8, 0.1, 0.1]):
        #   sum((pred - gt)^2) = (0.8-1)^2 + (0.1-0)^2 + (0.1-0)^2 = 0.04 + 0.01 + 0.01 = 0.06
        # Sample 2 (GT: B [0, 1, 0], Pred: [0.7, 0.2, 0.1]):
        #   sum((pred - gt)^2) = (0.7-0)^2 + (0.2-1)^2 + (0.1-0)^2 = 0.49 + 0.64 + 0.01 = 1.14
        # Mean Brier score = (0.06 + 1.14) / 2 = 0.60
        self.assertAlmostEqual(metrics["brier_score_final"], 0.60)
        
        # Expected Tier 1 Brier score:
        # Sample 1 (GT: A [1, 0, 0], T1: [0.5, 0.3, 0.2]):
        #   sum((T1 - gt)^2) = (0.5-1)^2 + (0.3-0)^2 + (0.2-0)^2 = 0.25 + 0.09 + 0.04 = 0.38
        # Sample 2 (GT: B [0, 1, 0], T1: [0.6, 0.2, 0.2]):
        #   sum((T1 - gt)^2) = (0.6-0)^2 + (0.2-1)^2 + (0.2-0)^2 = 0.36 + 0.64 + 0.04 = 1.04
        # Mean Tier 1 Brier score = (0.38 + 1.04) / 2 = 0.71
        self.assertAlmostEqual(metrics["brier_score_t1"], 0.71)

    def test_cost_latency_computation(self):
        evaluator = PipelineEvaluator()
        
        # Tier 1 run (latency: 0.01s)
        evaluator.add_result(
            prediction={"tier": 1, "t1_label": "A", "final_label": ["A"], "confidence": 0.9, "latency": 0.01},
            ground_truth="A"
        )
        # Tier 2 run (latency: 1.5s)
        evaluator.add_result(
            prediction={"tier": 2, "t1_label": "A", "final_label": ["B"], "confidence": 0.8, "latency": 1.5},
            ground_truth="B"
        )
        # Tier 3 run (latency: 0.1s)
        evaluator.add_result(
            prediction={"tier": 3, "t1_label": "A", "final_label": ["A"], "confidence": 1.0, "latency": 0.1},
            ground_truth="A"
        )
        
        # Configuration cost weights: T1=1.0, T2=10.0, T3=100.0
        metrics = evaluator.calculate_metrics(
            categories=["A", "B"],
            cost_t1=1.0,
            cost_t2=10.0,
            cost_t3=100.0
        )
        
        # Expected Latency:
        # Total latency = 0.01 + 1.5 + 0.1 = 1.61s
        # Average latency = 1.61 / 3 = 0.5367s
        self.assertAlmostEqual(metrics["total_latency"], 1.61)
        self.assertAlmostEqual(metrics["average_latency"], 1.61 / 3.0)
        
        # Expected Cost:
        # T1 count = 1, T2 count = 1, T3 count = 1
        # Total Cost = 1 * 1.0 + 1 * 10.0 + 1 * 100.0 = 111.0
        # Average Cost = 111.0 / 3 = 37.0
        self.assertAlmostEqual(metrics["total_compute_cost"], 111.0)
        self.assertAlmostEqual(metrics["average_compute_cost"], 37.0)

    def test_student_t_confidence_intervals(self):
        # We supply 5 runs to evaluate the Student-t distribution CI
        runs = [
            {"seed": 1, "accuracy_final": 0.90, "accuracy_t1": 0.85, "human_effort_ratio": 0.10, "picr": 1.0, "f1_macro": 0.90, "f1_weighted": 0.90, "ece_final": 0.05, "ece_t1": 0.06, "brier_score_final": 0.20, "brier_score_t1": 0.22, "net_utility": 0.08, "total_latency": 10.0, "average_latency": 0.10, "total_compute_cost": 100.0, "average_compute_cost": 1.0},
            {"seed": 2, "accuracy_final": 0.92, "accuracy_t1": 0.86, "human_effort_ratio": 0.12, "picr": 1.2, "f1_macro": 0.92, "f1_weighted": 0.92, "ece_final": 0.04, "ece_t1": 0.05, "brier_score_final": 0.18, "brier_score_t1": 0.20, "net_utility": 0.09, "total_latency": 11.0, "average_latency": 0.11, "total_compute_cost": 110.0, "average_compute_cost": 1.1},
            {"seed": 3, "accuracy_final": 0.88, "accuracy_t1": 0.84, "human_effort_ratio": 0.08, "picr": 0.8, "f1_macro": 0.88, "f1_weighted": 0.88, "ece_final": 0.06, "ece_t1": 0.07, "brier_score_final": 0.22, "brier_score_t1": 0.24, "net_utility": 0.07, "total_latency": 9.0, "average_latency": 0.09, "total_compute_cost": 90.0, "average_compute_cost": 0.9},
        ]
        
        stats = calculate_multi_seed_stats(runs)
        
        # Check standard statistics are populated
        self.assertEqual(stats["valid_picr_runs"], "3/3")
        self.assertAlmostEqual(stats["mean"]["accuracy_final"], 0.90)
        
        # Check standard deviation of accuracy_final (ddof=1)
        # values: [0.90, 0.92, 0.88]
        # mean: 0.90
        # squared diffs: (0.90-0.90)^2 + (0.92-0.90)^2 + (0.88-0.90)^2 = 0 + 0.0004 + 0.0004 = 0.0008
        # variance (ddof=1) = 0.0008 / 2 = 0.0004
        # stddev = sqrt(0.0004) = 0.02
        self.assertAlmostEqual(stats["std"]["accuracy_final"], 0.02)
        
        # For N=3, df=2, t_0.025,2 = 4.30265
        # CI95 = 4.30265 * (0.02 / sqrt(3)) = 4.30265 * 0.011547 = 0.04968
        self.assertAlmostEqual(stats["ci95"]["accuracy_final"], 4.302652729911275 * (0.02 / math.sqrt(3.0)))

if __name__ == "__main__":
    unittest.main()
