import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from core.calibration import calculate_ece

class PipelineEvaluator:
    def __init__(self):
        self.results = []
        self.pre_al_t1_accuracy = None
        self.post_al_t1_accuracy = None

    def add_result(self, prediction: dict, ground_truth: str):
        self.results.append({
            "prediction": prediction,
            "ground_truth": ground_truth
        })

    def calculate_metrics(self, lambda_al: float = 0.5, annotation_cost_weight: float = 0.05, categories: list = None):
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

        # ECE Metric
        confidences = np.array([r["prediction"]["confidence"] for r in self.results])
        correctness = np.array([1 if y_f == y_t else 0 for y_f, y_t in zip(y_final, y_true)])
        ece = calculate_ece(confidences, correctness, np.ones_like(correctness))

        # Confusion Matrices
        if categories is None:
            labels_order = ["World", "Sports", "Business", "Sci/Tech"]
        else:
            labels_order = categories
        confusion_matrix_t1 = confusion_matrix(y_true, y_t1, labels=labels_order).tolist()
        confusion_matrix_final = confusion_matrix(y_true, y_final, labels=labels_order).tolist()
        
        y_true_escalated = [r["ground_truth"] for r in self.results if r["prediction"]["tier"] > 1]
        y_final_escalated = [r["prediction"]["final_label"][0] if r["prediction"]["final_label"] else None for r in self.results if r["prediction"]["tier"] > 1]
        if len(y_true_escalated) > 0:
            confusion_matrix_escalated = confusion_matrix(y_true_escalated, y_final_escalated, labels=labels_order).tolist()
        else:
            confusion_matrix_escalated = np.zeros((len(labels_order), len(labels_order))).tolist()

        # Per-Category F1
        report_final = classification_report(y_true, y_final, labels=labels_order, output_dict=True, zero_division=0)
        report_t1 = classification_report(y_true, y_t1, labels=labels_order, output_dict=True, zero_division=0)
        
        per_category_f1 = {}
        per_category_f1_t1 = {}
        for cat in labels_order:
            if cat in report_final:
                per_category_f1[cat] = {
                    "precision": float(report_final[cat]["precision"]),
                    "recall": float(report_final[cat]["recall"]),
                    "f1-score": float(report_final[cat]["f1-score"]),
                    "support": float(report_final[cat]["support"])
                }
            if cat in report_t1:
                per_category_f1_t1[cat] = {
                    "precision": float(report_t1[cat]["precision"]),
                    "recall": float(report_t1[cat]["recall"]),
                    "f1-score": float(report_t1[cat]["f1-score"]),
                    "support": float(report_t1[cat]["support"])
                }

        # Escalation breakdown by category
        escalation_by_category = {}
        for cat in labels_order:
            escalation_by_category[cat] = {
                "tier2": 0,
                "tier3": 0,
                "total": 0
            }
        for r in self.results:
            gt_cat = r["ground_truth"]
            tier = r["prediction"]["tier"]
            if gt_cat in escalation_by_category:
                escalation_by_category[gt_cat]["total"] += 1
                if tier == 2:
                    escalation_by_category[gt_cat]["tier2"] += 1
                elif tier == 3:
                    escalation_by_category[gt_cat]["tier3"] += 1

        # PICR calculation
        human_labels_count = tier_counts[3]
        raw_human_effort_ratio = human_labels_count / total if total > 0 else 0
        delta_acc = acc_final - acc_t1
        
        if raw_human_effort_ratio == 0:
            picr = None
            picr_display = "N/A"
            picr_status = "AUTONOMOUS" if delta_acc > 0 else "NO_GAIN"
            picr_negative_warning = False
        else:
            picr = delta_acc / raw_human_effort_ratio
            picr_display = f"{picr:.4f}"
            if picr >= 2.0:
                picr_status = "STRONG"
            elif picr >= 1.0:
                picr_status = "ACCEPTABLE"
            elif picr > 0:
                picr_status = "BELOW_TARGET"
            else:
                picr_status = "NO_GAIN"
            picr_negative_warning = delta_acc < 0

        # Compute delta_acc_al
        if self.pre_al_t1_accuracy is not None and self.post_al_t1_accuracy is not None:
            delta_acc_al = float(self.post_al_t1_accuracy - self.pre_al_t1_accuracy)
        else:
            delta_acc_al = 0.0

        # PICR-AL computation
        picr_al = (delta_acc + lambda_al * delta_acc_al) / (raw_human_effort_ratio + 0.001)
        picr_al = float(round(picr_al, 4))
        picr_al_display = f"{picr_al:.4f}"

        if picr_al >= 2.0:
            picr_al_status = "STRONG"
        elif picr_al >= 1.0:
            picr_al_status = "ACCEPTABLE"
        elif picr_al > 0:
            picr_al_status = "BELOW_TARGET"
        else:
            picr_al_status = "NO_GAIN"

        # Net Utility computation
        net_utility = delta_acc - (annotation_cost_weight * raw_human_effort_ratio)
        net_utility = float(round(net_utility, 4))
        net_utility_display = f"{net_utility:.4f}"

        if net_utility > 0:
            net_utility_status = "POSITIVE"
        elif net_utility == 0:
            net_utility_status = "BREAK_EVEN"
        else:
            net_utility_status = "NEGATIVE"

        metrics = {
            "total_samples": total,
            "tier_distribution": tier_counts,
            "accuracy_t1": float(acc_t1),
            "accuracy_final": float(acc_final),
            "f1_weighted": float(f1_final),
            "ece": float(ece),
            "human_effort_ratio": float(raw_human_effort_ratio),
            "picr": picr,
            "picr_display": picr_display,
            "picr_status": picr_status,
            "picr_negative_warning": picr_negative_warning,
            "tier2_autonomous_gain": float(delta_acc),
            "human_labels_count": human_labels_count,
            "confusion_matrix_t1": confusion_matrix_t1,
            "confusion_matrix_final": confusion_matrix_final,
            "confusion_matrix_escalated": confusion_matrix_escalated,
            "per_category_f1": per_category_f1,
            "per_category_f1_t1": per_category_f1_t1,
            "escalation_by_category": escalation_by_category,
            "picr_al": picr_al,
            "picr_al_display": picr_al_display,
            "picr_al_status": picr_al_status,
            "picr_al_lambda": float(lambda_al),
            "delta_acc_al": float(delta_acc_al),
            "net_utility": net_utility,
            "net_utility_display": net_utility_display,
            "net_utility_status": net_utility_status,
            "annotation_cost_weight": float(annotation_cost_weight)
        }

        if self.pre_al_t1_accuracy is not None:
            metrics["pre_al_t1_accuracy"] = float(self.pre_al_t1_accuracy)
        if self.post_al_t1_accuracy is not None:
            metrics["post_al_t1_accuracy"] = float(self.post_al_t1_accuracy)

        return metrics

    def save_report(self, filepath: str, lambda_al: float = 0.5, annotation_cost_weight: float = 0.05, categories: list = None):
        metrics = self.calculate_metrics(lambda_al=lambda_al, annotation_cost_weight=annotation_cost_weight, categories=categories)
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
