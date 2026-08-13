import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from core.calibration import calculate_ece
from scipy.stats import t

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

    def calculate_metrics(self, lambda_al: float = 0.5, annotation_cost_weight: float = 0.05, categories: list = None, cost_t1: float = 1.0, cost_t2: float = 200.0, cost_t3: float = 3000.0):
        if not self.results:
            return {}

        y_true = [r["ground_truth"] for r in self.results]
        # Tier 1 original prediction
        y_t1 = [r["prediction"]["t1_label"] if "t1_label" in r["prediction"] else None for r in self.results]
        # Final label (weighted by tier)
        y_final = [r["prediction"]["final_label"][0] if r["prediction"]["final_label"] else None for r in self.results]

        # Basic Stats
        total = len(self.results)
        tier_counts = {1: 0, 2: 0, 3: 0}
        for r in self.results:
            tier_counts[r["prediction"]["tier"]] += 1

        acc_t1 = accuracy_score(y_true, y_t1)
        acc_final = accuracy_score(y_true, y_final)
        f1_final_weighted = f1_score(y_true, y_final, average="weighted", zero_division=0)
        f1_final_macro = f1_score(y_true, y_final, average="macro", zero_division=0)
        f1_t1_macro = f1_score(y_true, y_t1, average="macro", zero_division=0)

        # Tier 1 Confidence & ECE
        t1_confidences = np.array([r["prediction"].get("t1_confidence", r["prediction"].get("confidence", 0.5)) for r in self.results])
        correctness_t1 = np.array([1 if y_t == y_gt else 0 for y_t, y_gt in zip(y_t1, y_true)])
        ece_t1 = calculate_ece(t1_confidences, correctness_t1, np.ones_like(correctness_t1))

        # Final System Confidence & ECE
        final_confidences = np.array([r["prediction"].get("confidence", 0.5) for r in self.results])
        correctness_final = np.array([1 if y_f == y_gt else 0 for y_f, y_gt in zip(y_final, y_true)])
        ece_final = calculate_ece(final_confidences, correctness_final, np.ones_like(correctness_final))

        # Latency & Cost Calculations
        total_latency = sum(r["prediction"].get("latency", 0.0) for r in self.results)
        average_latency = total_latency / total if total > 0 else 0.0
        
        t1_count = tier_counts[1]
        t2_count = tier_counts[2]
        t3_count = tier_counts[3]
        total_compute_cost = (t1_count * cost_t1) + (t2_count * cost_t2) + (t3_count * cost_t3)
        average_compute_cost = total_compute_cost / total if total > 0 else 0.0

        # Confusion Matrices
        if categories is None:
            labels_order = ["World", "Sports", "Business", "Sci/Tech"]
        else:
            labels_order = categories
            
        cat_to_id = {cat: i for i, cat in enumerate(labels_order)}
        
        # One-hot ground truth matrix and probability vectors for standard multiclass Brier score
        y_true_ids = [cat_to_id[gt] if gt in cat_to_id else 0 for gt in y_true]
        K = len(labels_order)
        y_true_one_hot = np.zeros((total, K))
        for i, val_id in enumerate(y_true_ids):
            y_true_one_hot[i, val_id] = 1.0

        t1_probs_list = []
        final_probs_list = []
        for r in self.results:
            pred = r["prediction"]
            # Tier 1 probs
            if "t1_probs" in pred:
                t1_probs_list.append(pred["t1_probs"])
            else:
                probs = [0.0] * K
                t1_lbl = pred.get("t1_label")
                if t1_lbl in cat_to_id:
                    probs[cat_to_id[t1_lbl]] = float(pred.get("t1_confidence", 1.0))
                t1_probs_list.append(probs)
                
            # Final probs
            if "final_probs" in pred:
                final_probs_list.append(pred["final_probs"])
            else:
                probs = [0.0] * K
                fin_lbl = pred.get("final_label", [None])[0]
                if fin_lbl in cat_to_id:
                    probs[cat_to_id[fin_lbl]] = float(pred.get("confidence", 1.0))
                final_probs_list.append(probs)
                
        t1_probs_matrix = np.array(t1_probs_list)
        final_probs_matrix = np.array(final_probs_list)
        
        brier_score_t1 = float(np.mean(np.sum((t1_probs_matrix - y_true_one_hot) ** 2, axis=1)))
        brier_score_final = float(np.mean(np.sum((final_probs_matrix - y_true_one_hot) ** 2, axis=1)))

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
        # 1. Uh = ΔAccuracy - (λ_cost * HumanEffortRatio)
        net_utility = delta_acc - (annotation_cost_weight * raw_human_effort_ratio)
        net_utility = float(round(net_utility, 4))
        net_utility_display = f"{net_utility:.4f}"

        if net_utility > 0:
            net_utility_status = "POSITIVE"
        elif net_utility == 0:
            net_utility_status = "BREAK_EVEN"
        else:
            net_utility_status = "NEGATIVE"

        # 2. Uc = ΔAccuracy - (λ_cost * NormalizedComputeCost)
        max_possible_cost = cost_t1 + cost_t3
        normalized_compute_cost = average_compute_cost / max_possible_cost if max_possible_cost > 0 else 0.0
        net_utility_cost = delta_acc - (annotation_cost_weight * normalized_compute_cost)
        net_utility_cost = float(round(net_utility_cost, 4))
        net_utility_cost_display = f"{net_utility_cost:.4f}"

        if net_utility_cost > 0:
            net_utility_cost_status = "POSITIVE"
        elif net_utility_cost == 0:
            net_utility_cost_status = "BREAK_EVEN"
        else:
            net_utility_cost_status = "NEGATIVE"

        metrics = {
            "total_samples": total,
            "tier_distribution": tier_counts,
            "accuracy_t1": float(acc_t1),
            "accuracy_final": float(acc_final),
            "f1_weighted": float(f1_final_weighted),
            "f1_macro": float(f1_final_macro),
            "f1_t1_macro": float(f1_t1_macro),
            "ece": float(ece_final),
            "ece_t1": float(ece_t1),
            "ece_final": float(ece_final),
            "brier_score": float(brier_score_final),
            "brier_score_t1": float(brier_score_t1),
            "brier_score_final": float(brier_score_final),
            "total_latency": float(total_latency),
            "average_latency": float(average_latency),
            "total_compute_cost": float(total_compute_cost),
            "average_compute_cost": float(average_compute_cost),
            "human_effort_ratio": float(raw_human_effort_ratio),
            "picr": picr,
            "picr_display": picr_display,
            "picr_status": picr_status,
            "picr_negative_warning": picr_negative_warning,
            "tier2_autonomous_gain": float(delta_acc),
            "human_labels_count": human_labels_count,
            "al_human_annotations": self.al_learning_curve[-1]["human_annotations"] if getattr(self, "al_learning_curve", None) else 0,
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
            "net_utility_cost": net_utility_cost,
            "net_utility_cost_display": net_utility_cost_display,
            "net_utility_cost_status": net_utility_cost_status,
            "annotation_cost_weight": float(annotation_cost_weight)
        }

        if self.pre_al_t1_accuracy is not None:
            metrics["pre_al_t1_accuracy"] = float(self.pre_al_t1_accuracy)
        if self.post_al_t1_accuracy is not None:
            metrics["post_al_t1_accuracy"] = float(self.post_al_t1_accuracy)

        return metrics

    def save_report(self, filepath: str, lambda_al: float = 0.5, annotation_cost_weight: float = 0.05, categories: list = None, cost_t1: float = 1.0, cost_t2: float = 200.0, cost_t3: float = 3000.0):
        metrics = self.calculate_metrics(
            lambda_al=lambda_al, 
            annotation_cost_weight=annotation_cost_weight, 
            categories=categories,
            cost_t1=cost_t1,
            cost_t2=cost_t2,
            cost_t3=cost_t3
        )
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

def calculate_multi_seed_stats(runs: list):
    """
    Computes rigorous multi-seed summary statistics including 95% Confidence Intervals (using Student-t) and Medians.
    """
    if not runs:
        return {}

    runs_summary = []
    for r in runs:
        runs_summary.append({
            "seed": r.get("seed"),
            "accuracy_final": r.get("accuracy_final"),
            "accuracy_t1": r.get("accuracy_t1"),
            "human_effort_ratio": r.get("human_effort_ratio"),
            "picr": r.get("picr"),
            "al_human_annotations": r.get("al_human_annotations")
        })

    keys = [
        "accuracy_final", "accuracy_t1", "human_effort_ratio", "f1_macro", 
        "f1_weighted", "ece_final", "ece_t1", "brier_score_final", "brier_score_t1", 
        "net_utility", "net_utility_cost", "al_human_annotations", "total_latency", "average_latency", "total_compute_cost", "average_compute_cost"
    ]
    mean_stats = {}
    std_stats = {}
    median_stats = {}
    ci95_stats = {}

    n_runs = len(runs)

    for key in keys:
        vals = [r[key] for r in runs if key in r and r[key] is not None]
        if len(vals) > 0:
            arr = np.array(vals, dtype=float)
            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            med = float(np.median(arr))
            
            # Use Student-t distribution critical value for 95% Confidence Interval
            df = len(arr) - 1
            t_val = float(t.ppf(0.975, df)) if df > 0 else 1.96
            ci = float(t_val * (s / np.sqrt(len(arr)))) if len(arr) > 1 else 0.0

            mean_stats[key] = m
            std_stats[key] = s
            median_stats[key] = med
            ci95_stats[key] = ci
        else:
            mean_stats[key] = None
            std_stats[key] = 0.0
            median_stats[key] = None
            ci95_stats[key] = 0.0

    # Handle PICR explicitly without dropping 0-human effort runs
    picr_vals = [r["picr"] for r in runs if "picr" in r and r["picr"] is not None]
    if picr_vals:
        arr_p = np.array(picr_vals, dtype=float)
        mean_stats["picr"] = float(np.mean(arr_p))
        std_stats["picr"] = float(np.std(arr_p, ddof=1)) if len(arr_p) > 1 else 0.0
        median_stats["picr"] = float(np.median(arr_p))
        
        df_p = len(arr_p) - 1
        t_val_p = float(t.ppf(0.975, df_p)) if df_p > 0 else 1.96
        ci95_stats["picr"] = float(t_val_p * (std_stats["picr"] / np.sqrt(len(arr_p)))) if len(arr_p) > 1 else 0.0
    else:
        mean_stats["picr"] = None
        std_stats["picr"] = 0.0
        median_stats["picr"] = None
        ci95_stats["picr"] = 0.0

    return {
        "runs": runs_summary,
        "valid_picr_runs": f"{len(picr_vals)}/{n_runs}",
        "mean": mean_stats,
        "std": std_stats,
        "median": median_stats,
        "ci95": ci95_stats
    }
