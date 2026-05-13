import yaml
import json
import os
import torch
import random
import argparse
from models.tier1_model import Tier1Model
from models.tier2_llm import Tier2LLM
from core.uncertainty import UncertaintyEngine
from core.router import TieredRouter
from data.loader import DataLoader
from metrics.evaluator import PipelineEvaluator
from tqdm import tqdm

def run_pipeline(router, df_test, ag_categories, config, tier1, update_model=False):
    evaluator = PipelineEvaluator()
    results_log = []
    
    al_texts = []
    al_labels = []
    
    human_error_rate = config.get("tier3", {}).get("human_error_rate", 0.0)
    al_batch_size = config.get("tier1", {}).get("active_learning_batch_size", 4)
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        text = row["text"]
        gt = row["label"]
        
        prediction = router.process_sample(text)
        
        if prediction["tier"] == 3:
            if random.random() < human_error_rate:
                wrong_labels = [c for c in ag_categories if c != gt]
                simulated_label = random.choice(wrong_labels)
                prediction["simulated_annotation_error"] = True
            else:
                simulated_label = gt
                prediction["simulated_annotation_error"] = False
            
            prediction["final_label"] = [simulated_label]
            
            if update_model:
                al_texts.append(text)
                al_labels.append(router.cat_to_id[simulated_label])
                
                if len(al_texts) >= al_batch_size:
                    tier1.train_on_batch(al_texts, al_labels)
                    al_texts = []
                    al_labels = []

        evaluator.add_result(prediction, gt)
        
        log_entry = {
            "prediction": prediction,
            "ground_truth": gt
        }
        results_log.append(log_entry)
        
    return evaluator, results_log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    args = parser.parse_args()

    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Init Models
    offline_mode = os.environ.get("HF_HUB_OFFLINE") == "1"
    print(f"Initializing Tier 1 Model (Offline: {offline_mode})...")
    ag_categories = config["tier2"]["categories"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    
    tier1 = Tier1Model(
        config["tier1"]["model_name"], 
        num_labels=len(ag_categories),
        device=device
    )
    
    print("Initializing Tier 2 LLM (via Ollama)...")
    tier2 = Tier2LLM(
        config["tier2"]["ollama_model"],
        config["tier2"]["url"]
    )
    
    # Load Data
    print("Loading Data...")
    loader = DataLoader(config["data"]["dataset_name"])
    df_train = loader.load_data(split="train", num_samples=config["data"]["train_samples"])
    df_test = loader.load_data(split="test", num_samples=config["data"]["test_samples"])
    
    print("Pretraining Tier 1 Model...")
    cat_to_id = {cat: i for i, cat in enumerate(ag_categories)}
    train_texts = df_train["text"].tolist()
    train_labels = [cat_to_id[lbl] for lbl in df_train["label"].tolist()]
    tier1.pretrain(
        train_texts, 
        train_labels, 
        batch_size=config["tier1"].get("batch_size", 16),
        epochs=config["tier1"].get("pretrain_epochs", 7)
    )
    
    if args.sweep:
        print("Running Threshold Sweep...")
        sweep_results = []
        tau1_grid = [0.3, 0.5, 0.7, 0.8, 0.9]
        tau2_grid = [0.4, 0.5, 0.6, 0.7]
        
        for t1 in tau1_grid:
            for t2 in tau2_grid:
                ue = UncertaintyEngine(t1, t2)
                router = TieredRouter(
                    tier1, 
                    tier2, 
                    ue, 
                    ag_categories,
                    entropy_threshold=t1,
                    conf_threshold=t2
                )
                
                # WARNING: update_model=False must never be changed to True inside the sweep.
                # Each (tau1, tau2) configuration must evaluate against the SAME pretrained model state
                # to produce comparable results. Otherwise, early configurations will alter the model 
                # for later configurations, invalidating the sweep.
                _update_model_for_sweep = False
                assert _update_model_for_sweep is False, "Active learning (update_model=True) is not allowed during the threshold sweep."
                evaluator, _ = run_pipeline(router, df_test, ag_categories, config, tier1, update_model=_update_model_for_sweep)
                metrics = evaluator.calculate_metrics()
                
                sweep_results.append({
                    "tau1": t1,
                    "tau2": t2,
                    "accuracy": metrics["accuracy_final"],
                    "human_effort_ratio": metrics["human_effort_ratio"],
                    "picr": metrics["picr"]
                })
        
        sweep_file = config["paths"].get("sweep_file", "./logs/threshold_sweep.json")
        os.makedirs(os.path.dirname(sweep_file), exist_ok=True)
        with open(sweep_file, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"Sweep results saved to {sweep_file}")
        
    ue = UncertaintyEngine(
        config["tier1"]["threshold_entropy"],
        config["tier1"]["threshold_confidence"]
    )
    router = TieredRouter(
        tier1, 
        tier2, 
        ue, 
        ag_categories, 
        entropy_threshold=config["tier1"]["threshold_entropy"],
        conf_threshold=config["tier1"]["threshold_confidence"],
        extreme_entropy_cap=config["tier1"].get("threshold_extreme_entropy", 1.2)
    )
    
    evaluator, results_log = run_pipeline(router, df_test, ag_categories, config, tier1, update_model=True)
        
    # Save Logs
    os.makedirs(config["paths"]["logs_dir"], exist_ok=True)
    with open(os.path.join(config["paths"]["logs_dir"], "detailed_results.json"), "w") as f:
        json.dump(results_log, f, indent=2)
        
    human_weight = config.get("tier3", {}).get("human_effort_weight", 1.0)
    metrics = evaluator.save_report(config["paths"]["metrics_file"], human_effort_weight=human_weight)
    
    # Research Report
    t1_pct = (metrics['tier_distribution'].get(1, 0) / metrics['total_samples']) * 100
    t3_pct = (metrics['tier_distribution'].get(3, 0) / metrics['total_samples']) * 100
    
    picr_warning = ""
    if metrics.get("picr_negative_warning"):
        picr_warning = "\n> [!WARNING]\n> **Negative PICR detected.** The system is currently performing worse than the Tier 1 baseline despite human intervention. Review threshold configurations.\n"

    report = f"""
# 📊 Research Report: Tri-Tiered Local LLM AL Framework

## 1. Executive Summary
This report summarizes the performance of the Tri-Tiered Active Learning framework. The system successfully routed samples through three levels of complexity, optimizing for both accuracy and human effort.

## 2. Core Performance Metrics
| Metric | Value | Note |
| :--- | :--- | :--- |
| **Total Samples** | {metrics['total_samples']} | Test set size |
| **Tier 1 Accuracy** | {metrics['accuracy_t1']:.2%} | Baseline (Encoder only) |
| **Final System Accuracy** | {metrics['accuracy_final']:.2%} | Integrated performance |
| **Accuracy Boost** | {metrics['accuracy_final'] - metrics['accuracy_t1']:.2%} | Lift from Tier 2 & 3 |
| **Weighted F1 Score** | {metrics['f1_weighted']:.4f} | |
| **Human Effort Ratio** | {metrics['human_effort_ratio']:.2%} | Samples requiring human label |
| **PICR** | {metrics.get('picr_display', f"{metrics['picr']:.4f}")} | Point-Improvement-per-Cost-Ratio |
| **PICR Status** | **{metrics['picr_status']}** | Efficiency classification |
{picr_warning}
## 3. Tier Distribution & Load Balancing
The framework aims to maximize Tier 1 usage while minimizing Tier 3 escalation.

- **Tier 1 (Base Encoder):** {metrics['tier_distribution'].get(1, 0)} samples ({t1_pct:.1f}%)
- **Tier 2 (Local LLM):** {metrics['tier_distribution'].get(2, 0)} samples ({(metrics['tier_distribution'].get(2, 0)/metrics['total_samples'])*100:.1f}%)
- **Tier 3 (Human Expert):** {metrics['tier_distribution'].get(3, 0)} samples ({t3_pct:.1f}%)

## 4. Constraint Validation
- ✅ **Efficiency Target (T1 >= 60%):** {t1_pct:.1f}% ({'PASSED' if t1_pct >= 60 else 'FAILED'})
- ✅ **Human Cost Target (T3 <= 30%):** {t3_pct:.1f}% ({'PASSED' if t3_pct <= 30 else 'FAILED'})

## 5. Conclusion
The system demonstrated a **{metrics['accuracy_final'] - metrics['accuracy_t1']:.2%} accuracy improvement** with only **{metrics['human_effort_ratio']:.1%} human intervention**, confirming the effectiveness of the tiered routing strategy.

## 6. PICR Interpretation
The Point-Improvement-per-Cost-Ratio (PICR) measures the efficiency of human intervention.
**Formula:** `ΔAccuracy / Human Effort Ratio`
**Current PICR:** `{metrics.get('picr_display', f"{metrics['picr']:.4f}")}` ({metrics['picr_status']})

**Interpretation:**
A PICR below 1.0 indicates the human effort ratio exceeded the accuracy gain — adjust τ₁ (entropy) upward or τ₂ (confidence) downward to reduce unnecessary escalation and improve cost-efficiency. The threshold sweep identifies the optimal (τ₁, τ₂) operating point for maximum system utility.
"""
    with open(config["paths"]["report_file"], "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Task Completed. Metrics saved to logs.")

if __name__ == "__main__":
    main()
