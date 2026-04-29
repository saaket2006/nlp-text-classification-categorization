import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score

class PipelineEvaluator:
    def __init__(self):
        self.results = []

    def add_result(self, prediction: dict, ground_truth: str):
        self.results.append({
            "prediction": prediction,
            "ground_truth": ground_truth
        })

    def calculate_metrics(self):
        if not self.results:
            return {}

        y_true = [r["ground_truth"] for r in self.results]
        # For Tier 1 label
        y_t1 = [r["prediction"]["predicted_labels"][0] if r["prediction"]["tier"] >= 1 else None for r in self.results]
        # For Final label (weighted by tier)
        y_final = [r["prediction"]["final_label"][0] if r["prediction"]["final_label"] else None for r in self.results]

        # Basic Stats
        total = len(self.results)
        tier_counts = {1: 0, 2: 0, 3: 0}
        for r in self.results:
            tier_counts[r["prediction"]["tier"]] += 1

        acc_t1 = accuracy_score(y_true, y_t1)
        acc_final = accuracy_score(y_true, y_final)
        f1_final = f1_score(y_true, y_final, average="weighted")

        # PICR calculation
        # PICR = ΔAccuracy / (Human Labels / Total Samples)
        # Here Human Labels = count of Tier 3
        human_labels_count = tier_counts[3]
        human_effort_ratio = human_labels_count / total if total > 0 else 0
        delta_acc = acc_final - acc_t1
        
        picr = delta_acc / human_effort_ratio if human_effort_ratio > 0 else 0.0

        metrics = {
            "total_samples": total,
            "tier_distribution": tier_counts,
            "accuracy_t1": float(acc_t1),
            "accuracy_final": float(acc_final),
            "f1_weighted": float(f1_final),
            "human_effort_ratio": float(human_effort_ratio),
            "picr": float(picr),
            "human_labels_count": human_labels_count
        }

        return metrics

    def save_report(self, filepath: str):
        metrics = self.calculate_metrics()
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
