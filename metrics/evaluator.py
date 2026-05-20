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
        # We use Tier 1's original prediction (t1_label) as the counterfactual baseline
        y_t1 = [r["prediction"]["t1_label"] if "t1_label" in r["prediction"] else None for r in self.results]
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
        raw_human_effort_ratio = human_labels_count / total if total > 0 else 0
        delta_acc = acc_final - acc_t1
        
        # Handle division by zero for infinite efficiency (Positive gain with zero human cost)
        if raw_human_effort_ratio == 0 and delta_acc > 0:
            # Use a virtual 0.25% human effort (0.5 samples equivalent) to keep PICR finite and realistic
            picr = delta_acc / 0.0025
        elif raw_human_effort_ratio == 0:
            picr = 0.0
        else:
            picr = delta_acc / raw_human_effort_ratio
            
        # PICR display string
        picr_display = f"{picr:.4f}"
        
        # PICR Status Interpretation
        if raw_human_effort_ratio == 0 and delta_acc > 0:
            picr_status = "PERFECT_EFFICIENCY"
        elif picr >= 2.0:
            picr_status = "STRONG"
        elif picr >= 1.0:
            picr_status = "ACCEPTABLE"
        elif picr > 0:
            picr_status = "BELOW_TARGET"
        else:
            picr_status = "NO_GAIN"

        metrics = {
            "total_samples": total,
            "tier_distribution": tier_counts,
            "accuracy_t1": float(acc_t1),
            "accuracy_final": float(acc_final),
            "f1_weighted": float(f1_final),
            "human_effort_ratio": float(raw_human_effort_ratio),
            "picr": float(picr) if picr != float('inf') else float('inf'), # JSON handles Infinity
            "picr_display": picr_display,
            "picr_status": picr_status,
            "picr_negative_warning": delta_acc < 0,
            "human_labels_count": human_labels_count
        }

        return metrics

    def save_report(self, filepath: str):
        metrics = self.calculate_metrics()
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
