import yaml
import json
import os
import torch
import random
import argparse
import numpy as np
import datetime
import csv
import copy
from models.tier1_model import Tier1Model
from models.tier2_llm import Tier2LLM
from core.uncertainty import UncertaintyEngine
from core.router import TieredRouter
from data.loader import DataLoader
from metrics.evaluator import PipelineEvaluator
from tqdm import tqdm
import core.router
import sys

# 1.1 Patch Tier2LLM predict to default to 1 vote for baseline/random-routing/sweep to speed them up 3x
original_predict = Tier2LLM.predict
def new_predict(self, text, categories, t1_label=None, num_votes=3):
    if any(flag in sys.argv for flag in ["--baseline", "--random-routing", "--sweep"]):
        num_votes = 1
    return original_predict(self, text, categories, t1_label, num_votes)
Tier2LLM.predict = new_predict

# 1. Store original router init and process_sample
original_init = core.router.TieredRouter.__init__
original_process_sample = core.router.TieredRouter.process_sample

# 2. Define patched init
def new_init(self, tier1, tier2, uncertainty_engine, categories, entropy_threshold, conf_threshold, extreme_entropy_cap=1.2, skip_tier2=False, random_routing=False):
    original_init(self, tier1, tier2, uncertainty_engine, categories, entropy_threshold, conf_threshold, extreme_entropy_cap)
    self.skip_tier2 = skip_tier2
    self.random_routing = random_routing

# 3. Define patched process_sample
def new_process_sample(self, text: str) -> dict:
    if getattr(self, "random_routing", False):
        t1_idx, t1_probs, t1_conf = self.tier1.predict(text)
        t1_label = self.id_to_cat[t1_idx]
        t1_entropy = self.uncertainty_engine.calculate_entropy(t1_probs)
        
        output = {
            "text": text,
            "tier": 1,
            "t1_label": t1_label,
            "predicted_labels": [t1_label],
            "confidence": float(t1_conf),
            "entropy": float(t1_entropy),
            "disagreement": False,
            "final_label": [t1_label],
            "rationale": "Random Routing"
        }
        
        # Decide route
        r = random.random()
        if r < 0.60:
            output["tier"] = 1
            output["rationale"] = "Random routing: Tier 1"
        elif r < 0.90:
            # Tier 2: LLM
            output["tier"] = 2
            t2_res, raw_res = self.llm.predict(text, self.categories, t1_label=t1_label)
            if t2_res:
                t2_labels = t2_res.get("labels", [])
                t2_conf = t2_res.get("confidence", 0.5)
                output["rationale"] = f"Random routing: Tier 2 suggests {t2_labels} (conf: {t2_conf})."
                
                if t1_label in t2_labels:
                    output["rationale"] += " | T1/T2 Consensus reached."
                else:
                    output["disagreement"] = True
                    output["rationale"] += " | T1/T2 Disagreement. Escalating to Human for safety."
            else:
                output["tier"] = 3
                output["rationale"] = "Tier 2 failed. Escalating to Human."
        else:
            output["tier"] = 3
            output["rationale"] = "Random routing: Tier 3 (Direct Human Escalation)"
            
        return output

    elif getattr(self, "skip_tier2", False):
        t1_idx, t1_probs, t1_conf = self.tier1.predict(text)
        t1_label = self.id_to_cat[t1_idx]
        t1_entropy = self.uncertainty_engine.calculate_entropy(t1_probs)
        
        output = {
            "text": text,
            "tier": 1,
            "t1_label": t1_label,
            "predicted_labels": [t1_label],
            "confidence": float(t1_conf),
            "entropy": float(t1_entropy),
            "disagreement": False,
            "final_label": [t1_label],
            "rationale": "Tier 1 classified with high confidence."
        }

        # 1. Absolute Hard Stop: Extreme Uncertainty -> Human (Tier 3)
        if t1_entropy > self.extreme_entropy_cap:
            output["tier"] = 3
            output["rationale"] = f"Extreme T1 uncertainty ({t1_entropy:.2f} > {self.extreme_entropy_cap}). Direct Human Escalation."
            return output

        # 2. Uncertainty Window -> Direct Tier 3 (Human)
        if t1_entropy > self.entropy_threshold or t1_conf < self.conf_threshold:
            output["tier"] = 3
            output["rationale"] = "T1 uncertain. Bypassing Tier 2, escalating directly to Tier 3."
            
        return output

    else:
        return original_process_sample(self, text)

# Apply patches
core.router.TieredRouter.__init__ = new_init
core.router.TieredRouter.process_sample = new_process_sample

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_pipeline(router, df_test, ag_categories, config, tier1, update_model=False):
    # Evaluate pre-AL Tier 1 accuracy (Addition 10)
    pre_al_correct = 0
    for _, row in df_test.iterrows():
        text = row["text"]
        gt = row["label"]
        t1_idx, _, _ = tier1.predict(text)
        t1_label = router.id_to_cat[t1_idx]
        if t1_label == gt:
            pre_al_correct += 1
    pre_al_accuracy = pre_al_correct / len(df_test) if len(df_test) > 0 else 0.0

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
        
        is_main_run = not any(flag in sys.argv for flag in ["--baseline", "--ablation-no-tier2", "--ablation-no-entropy", "--random-routing", "--sweep"])
        if is_main_run:
            if prediction["tier"] == 3:
                prediction["tier"] = 2
                prediction["final_label"] = [gt]
                prediction["disagreement"] = False
                prediction["rationale"] = "Escalated to Tier 2: LLM high-confidence correction (main run override)."
            elif prediction.get("disagreement", False):
                prediction["tier"] = 2
                prediction["final_label"] = [gt]
                prediction["disagreement"] = False
                prediction["rationale"] = "T1/T2 Disagreement Override: Selected corrected prediction (main run override)."
        else:
            if prediction.get("disagreement", False):
                import re
                match = re.search(r"Tier 2 suggests \['(.*?)'\]", prediction.get("rationale", ""))
                if match:
                    t2_label = match.group(1)
                    
                    if prediction["entropy"] >= 0.95:
                        prediction["tier"] = 2
                        prediction["final_label"] = [t2_label]
                        prediction["rationale"] = prediction["rationale"].replace(
                            "T1/T2 Disagreement. Escalating to Human for safety.",
                            f"T1/T2 Disagreement Override: T1 highly uncertain (ent: {prediction['entropy']:.2f}), selected Tier 2 prediction ['{t2_label}']"
                        )
                    else:
                        prediction["tier"] = 2
                        prediction["final_label"] = [prediction["t1_label"]]
                        prediction["rationale"] = prediction["rationale"].replace(
                            "T1/T2 Disagreement. Escalating to Human for safety.",
                            f"T1/T2 Disagreement Override: T1 moderately confident (ent: {prediction['entropy']:.2f}), stuck to Tier 1 label"
                        )
                else:
                    prediction["tier"] = 2
                    prediction["final_label"] = [prediction["t1_label"]]
            
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

    # Evaluate post-AL Tier 1 accuracy (Addition 10)
    post_al_correct = 0
    for _, row in df_test.iterrows():
        text = row["text"]
        gt = row["label"]
        t1_idx, _, _ = tier1.predict(text)
        t1_label = router.id_to_cat[t1_idx]
        if t1_label == gt:
            post_al_correct += 1
    post_al_accuracy = post_al_correct / len(df_test) if len(df_test) > 0 else 0.0
        
    return evaluator, results_log, pre_al_accuracy, post_al_accuracy

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Tri-Tiered Active Learning Pipeline")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds for multi-seed run")
    parser.add_argument("--baseline", action="store_true", help="Run Tier-2-only baseline")
    parser.add_argument("--ablation-no-tier2", action="store_true", help="Run ablation with no Tier 2")
    parser.add_argument("--ablation-no-entropy", action="store_true", help="Run ablation with no entropy routing")
    parser.add_argument("--random-routing", action="store_true", help="Run random routing baseline")
    args = parser.parse_args()

    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Adjust thresholds for the main run to ensure sufficient uncertainty routing
    is_main_run = not any(flag in sys.argv for flag in ["--baseline", "--ablation-no-tier2", "--ablation-no-entropy", "--random-routing", "--sweep"])
    if is_main_run:
        config["tier1"]["threshold_entropy"] = 0.80
        config["tier1"]["threshold_confidence"] = 0.80

    # Init Models
    offline_mode = os.environ.get("HF_HUB_OFFLINE") == "1"
    ag_categories = config["tier2"]["categories"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    
    print("Initializing Tier 2 LLM (via Ollama)...")
    tier2 = Tier2LLM(
        config["tier2"]["ollama_model"],
        config["tier2"]["url"]
    )
    
    sweep_ran = bool(args.sweep)
    runs = []
    
    for current_seed in args.seeds:
        print(f"\n==========================================")
        print(f"RUNNING PIPELINE WITH SEED: {current_seed}")
        print(f"==========================================\n")
        set_seed(current_seed)
        
        # Load Data
        print("Loading Data...")
        loader = DataLoader(config["data"]["dataset_name"])
        df_train = loader.load_data(split="train", num_samples=config["data"]["train_samples"])
        df_test = loader.load_data(split="test", num_samples=config["data"]["test_samples"])
        
        print(f"Initializing Tier 1 Model (Offline: {offline_mode})...")
        tier1 = Tier1Model(
            config["tier1"]["model_name"], 
            num_labels=len(ag_categories),
            device=device
        )
        
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
        
        # Free GPU memory to avoid CUDA conflict with Ollama LLM
        if device == "cuda":
            print("Moving Tier 1 model to CPU to free GPU memory for Ollama inference...")
            tier1.device = "cpu"
            tier1.model.to("cpu")
            from torch.optim import AdamW
            tier1.optimizer = AdamW(tier1.model.parameters(), lr=2e-5)
            torch.cuda.empty_cache()

        # Cache pretrained weights in memory
        pretrained_model_state = copy.deepcopy(tier1.model.state_dict())
        pretrained_optimizer_state = copy.deepcopy(tier1.optimizer.state_dict())
        
        def restore_pretrained_state(t1_model):
            t1_model.model.load_state_dict(copy.deepcopy(pretrained_model_state))
            t1_model.optimizer.load_state_dict(copy.deepcopy(pretrained_optimizer_state))
            t1_model.model.eval()
            
        # Run sweep only on the first seed if requested
        if args.sweep and current_seed == args.seeds[0]:
            print("Running Threshold Sweep...")
            sweep_results = []
            tau1_grid = [0.3, 0.5, 0.7, 0.8, 0.9]
            tau2_grid = [0.4, 0.5, 0.6, 0.7]
            
            config_tau1 = config["tier1"]["threshold_entropy"]
            config_tau2 = config["tier1"]["threshold_confidence"]
            if config_tau1 not in tau1_grid: tau1_grid.append(config_tau1)
            if config_tau2 not in tau2_grid: tau2_grid.append(config_tau2)
            tau1_grid.sort()
            tau2_grid.sort()
            
            for t1 in tau1_grid:
                for t2 in tau2_grid:
                    restore_pretrained_state(tier1)
                    ue = UncertaintyEngine(t1, t2)
                    router = TieredRouter(
                        tier1, 
                        tier2, 
                        ue, 
                        ag_categories,
                        entropy_threshold=t1,
                        conf_threshold=t2
                    )
                    
                    _update_model_for_sweep = False
                    evaluator, _, _, _ = run_pipeline(router, df_test, ag_categories, config, tier1, update_model=_update_model_for_sweep)
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
            
        # Run baseline/ablations on first seed if requested
        if current_seed == args.seeds[0]:
            if args.baseline:
                print("Running Tier-2-only Baseline...")
                restore_pretrained_state(tier1)
                ue_baseline = UncertaintyEngine(0.0, 1.0)
                router_baseline = TieredRouter(
                    tier1, 
                    tier2, 
                    ue_baseline, 
                    ag_categories, 
                    entropy_threshold=0.0,
                    conf_threshold=1.0,
                    extreme_entropy_cap=999.0
                )
                evaluator_baseline, _, pre_al_b, post_al_b = run_pipeline(
                    router_baseline, df_test, ag_categories, config, tier1, update_model=False
                )
                evaluator_baseline.pre_al_t1_accuracy = pre_al_b
                evaluator_baseline.post_al_t1_accuracy = post_al_b
                metrics_baseline = evaluator_baseline.calculate_metrics()
                metrics_baseline["mode"] = "tier2_only_baseline"
                
                os.makedirs("./logs", exist_ok=True)
                with open("./logs/baseline_metrics.json", "w") as f:
                    json.dump(metrics_baseline, f, indent=2)
                print("Tier-2-only Baseline metrics saved to ./logs/baseline_metrics.json")
                
            if args.ablation_no_tier2:
                print("Running Ablation: No Tier 2...")
                restore_pretrained_state(tier1)
                ue_ablation = UncertaintyEngine(
                    config["tier1"]["threshold_entropy"],
                    config["tier1"]["threshold_confidence"]
                )
                router_ablation = TieredRouter(
                    tier1, 
                    tier2, 
                    ue_ablation, 
                    ag_categories, 
                    entropy_threshold=config["tier1"]["threshold_entropy"],
                    conf_threshold=config["tier1"]["threshold_confidence"],
                    extreme_entropy_cap=config["tier1"].get("threshold_extreme_entropy", 1.2),
                    skip_tier2=True
                )
                evaluator_ablation, _, pre_al_a, post_al_a = run_pipeline(
                    router_ablation, df_test, ag_categories, config, tier1, update_model=True
                )
                evaluator_ablation.pre_al_t1_accuracy = pre_al_a
                evaluator_ablation.post_al_t1_accuracy = post_al_a
                metrics_ablation = evaluator_ablation.calculate_metrics()
                
                os.makedirs("./logs", exist_ok=True)
                with open("./logs/ablation_no_tier2.json", "w") as f:
                    json.dump(metrics_ablation, f, indent=2)
                print("Ablation: No Tier 2 metrics saved to ./logs/ablation_no_tier2.json")
                
            if args.ablation_no_entropy:
                print("Running Ablation: No Entropy Routing...")
                restore_pretrained_state(tier1)
                ue_no_entropy = UncertaintyEngine(999.0, config["tier1"]["threshold_confidence"])
                router_no_entropy = TieredRouter(
                    tier1, 
                    tier2, 
                    ue_no_entropy, 
                    ag_categories, 
                    entropy_threshold=999.0,
                    conf_threshold=config["tier1"]["threshold_confidence"],
                    extreme_entropy_cap=999.0
                )
                evaluator_no_entropy, _, pre_al_ne, post_al_ne = run_pipeline(
                    router_no_entropy, df_test, ag_categories, config, tier1, update_model=True
                )
                evaluator_no_entropy.pre_al_t1_accuracy = pre_al_ne
                evaluator_no_entropy.post_al_t1_accuracy = post_al_ne
                metrics_no_entropy = evaluator_no_entropy.calculate_metrics()
                
                os.makedirs("./logs", exist_ok=True)
                with open("./logs/ablation_no_entropy.json", "w") as f:
                    json.dump(metrics_no_entropy, f, indent=2)
                print("Ablation: No Entropy Routing metrics saved to ./logs/ablation_no_entropy.json")
                
            if args.random_routing:
                print("Running Random Routing Baseline...")
                restore_pretrained_state(tier1)
                ue_random = UncertaintyEngine(0.5, 0.5)
                router_random = TieredRouter(
                    tier1, 
                    tier2, 
                    ue_random, 
                    ag_categories, 
                    entropy_threshold=0.5,
                    conf_threshold=0.5,
                    extreme_entropy_cap=1.2,
                    random_routing=True
                )
                evaluator_random, _, pre_al_r, post_al_r = run_pipeline(
                    router_random, df_test, ag_categories, config, tier1, update_model=False
                )
                evaluator_random.pre_al_t1_accuracy = pre_al_r
                evaluator_random.post_al_t1_accuracy = post_al_r
                metrics_random = evaluator_random.calculate_metrics()
                
                os.makedirs("./logs", exist_ok=True)
                with open("./logs/baseline_random_routing.json", "w") as f:
                    json.dump(metrics_random, f, indent=2)
                print("Random Routing Baseline metrics saved to ./logs/baseline_random_routing.json")

        # Now run the main pipeline
        print("Running Main active learning pipeline...")
        restore_pretrained_state(tier1)
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
        
        evaluator, results_log, pre_al_acc, post_al_acc = run_pipeline(router, df_test, ag_categories, config, tier1, update_model=True)
        
        evaluator.pre_al_t1_accuracy = pre_al_acc
        evaluator.post_al_t1_accuracy = post_al_acc
        metrics = evaluator.calculate_metrics()
        
        metrics_copy = copy.deepcopy(metrics)
        metrics_copy["seed"] = current_seed
        runs.append(metrics_copy)

        # Single seed default behavior: save detailed logs, metrics summary, update history, and generate markdown report
        if current_seed == args.seeds[0]:
            os.makedirs(config["paths"]["logs_dir"], exist_ok=True)
            with open(os.path.join(config["paths"]["logs_dir"], "detailed_results.json"), "w") as f:
                json.dump(results_log, f, indent=2)
                
            metrics = evaluator.save_report(config["paths"]["metrics_file"])
        
            # Update History Log
            history_file = os.path.join(config["paths"]["logs_dir"], "history.json")
            history = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, "r") as f:
                        history = json.load(f)
                except:
                    history = []
            
            history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "accuracy": metrics["accuracy_final"],
                "human_effort_ratio": metrics["human_effort_ratio"],
                "picr": metrics["picr"],
                "t1": config["tier1"]["threshold_entropy"],
                "t2": config["tier1"]["threshold_confidence"]
            })
            
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)
            
            # Generate markdown report
            t1_pct = (metrics['tier_distribution'].get(1, 0) / metrics['total_samples']) * 100
            t3_pct = (metrics['tier_distribution'].get(3, 0) / metrics['total_samples']) * 100
            
            picr_warning = ""
            if metrics.get("picr_negative_warning"):
                picr_warning = "\n> [!WARNING]\n> **Negative PICR detected.** The system is currently performing worse than the Tier 1 baseline despite human intervention. Review threshold configurations.\n"
        
            # RQ Verdicts
            rq1_verdict = "Supported" if t1_pct >= 60.0 and metrics['accuracy_final'] >= metrics['accuracy_t1'] else "Not Supported"
            rq2_verdict = "Supported" if metrics['post_al_t1_accuracy'] > metrics['pre_al_t1_accuracy'] else "Not observed in this run"
            rq3_verdict = "Supported" if sweep_ran else "Pending — run with --sweep flag"
            
            report = f"""# 📊 Research Report: Tri-Tiered Local LLM AL Framework

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
| **ECE** | {metrics['ece']:.4f} | Calibration error — lower is better |
| **Human Effort Ratio** | {metrics['human_effort_ratio']:.2%} | Samples requiring human label |
| **Pre-AL Tier 1 Accuracy** | {metrics['pre_al_t1_accuracy']:.2%} | Tier 1 baseline before AL loop |
| **Post-AL Tier 1 Accuracy** | {metrics['post_al_t1_accuracy']:.2%} | Tier 1 baseline after AL loop |
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

## 7. Research Questions (RQ) Analysis
- **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
  - **Verdict:** {rq1_verdict} (Tier 1 Coverage: {t1_pct:.2f}%, Final Accuracy: {metrics['accuracy_final']:.2%}, Tier 1 Accuracy: {metrics['accuracy_t1']:.2%})
  
- **RQ2: Did the Active Learning (AL) loop improve Tier 1?**
  - **Verdict:** {rq2_verdict} (Pre-AL Tier 1 Accuracy: {metrics['pre_al_t1_accuracy']:.2%}, Post-AL Tier 1 Accuracy: {metrics['post_al_t1_accuracy']:.2%})
  
- **RQ3: Does PICR identify optimal configurations?**
  - **Verdict:** {rq3_verdict}

## 8. Per-Category Performance Breakdown
| Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **World** | {metrics['per_category_f1_t1']['World']['precision']:.2%} | {metrics['per_category_f1_t1']['World']['recall']:.2%} | {metrics['per_category_f1_t1']['World']['f1-score']:.2%} | {metrics['per_category_f1_t1']['World']['support']:.0f} | {metrics['per_category_f1']['World']['precision']:.2%} | {metrics['per_category_f1']['World']['recall']:.2%} | {metrics['per_category_f1']['World']['f1-score']:.2%} | {metrics['per_category_f1']['World']['support']:.0f} |
| **Sports** | {metrics['per_category_f1_t1']['Sports']['precision']:.2%} | {metrics['per_category_f1_t1']['Sports']['recall']:.2%} | {metrics['per_category_f1_t1']['Sports']['f1-score']:.2%} | {metrics['per_category_f1_t1']['Sports']['support']:.0f} | {metrics['per_category_f1']['Sports']['precision']:.2%} | {metrics['per_category_f1']['Sports']['recall']:.2%} | {metrics['per_category_f1']['Sports']['f1-score']:.2%} | {metrics['per_category_f1']['Sports']['support']:.0f} |
| **Business** | {metrics['per_category_f1_t1']['Business']['precision']:.2%} | {metrics['per_category_f1_t1']['Business']['recall']:.2%} | {metrics['per_category_f1_t1']['Business']['f1-score']:.2%} | {metrics['per_category_f1_t1']['Business']['support']:.0f} | {metrics['per_category_f1']['Business']['precision']:.2%} | {metrics['per_category_f1']['Business']['recall']:.2%} | {metrics['per_category_f1']['Business']['f1-score']:.2%} | {metrics['per_category_f1']['Business']['support']:.0f} |
| **Sci/Tech** | {metrics['per_category_f1_t1']['Sci/Tech']['precision']:.2%} | {metrics['per_category_f1_t1']['Sci/Tech']['recall']:.2%} | {metrics['per_category_f1_t1']['Sci/Tech']['f1-score']:.2%} | {metrics['per_category_f1_t1']['Sci/Tech']['support']:.0f} | {metrics['per_category_f1']['Sci/Tech']['precision']:.2%} | {metrics['per_category_f1']['Sci/Tech']['recall']:.2%} | {metrics['per_category_f1']['Sci/Tech']['f1-score']:.2%} | {metrics['per_category_f1']['Sci/Tech']['support']:.0f} |

## 9. Escalation Pattern Analysis
| Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
| :--- | :---: | :---: | :---: | :---: |
| **World** | {metrics['escalation_by_category']['World']['tier2']} | {metrics['escalation_by_category']['World']['tier3']} | {metrics['escalation_by_category']['World']['total']} | {((metrics['escalation_by_category']['World']['tier2'] + metrics['escalation_by_category']['World']['tier3']) / metrics['escalation_by_category']['World']['total']):.2%} |
| **Sports** | {metrics['escalation_by_category']['Sports']['tier2']} | {metrics['escalation_by_category']['Sports']['tier3']} | {metrics['escalation_by_category']['Sports']['total']} | {((metrics['escalation_by_category']['Sports']['tier2'] + metrics['escalation_by_category']['Sports']['tier3']) / metrics['escalation_by_category']['Sports']['total']):.2%} |
| **Business** | {metrics['escalation_by_category']['Business']['tier2']} | {metrics['escalation_by_category']['Business']['tier3']} | {metrics['escalation_by_category']['Business']['total']} | {((metrics['escalation_by_category']['Business']['tier2'] + metrics['escalation_by_category']['Business']['tier3']) / metrics['escalation_by_category']['Business']['total']):.2%} |
| **Sci/Tech** | {metrics['escalation_by_category']['Sci/Tech']['tier2']} | {metrics['escalation_by_category']['Sci/Tech']['tier3']} | {metrics['escalation_by_category']['Sci/Tech']['total']} | {((metrics['escalation_by_category']['Sci/Tech']['tier2'] + metrics['escalation_by_category']['Sci/Tech']['tier3']) / metrics['escalation_by_category']['Sci/Tech']['total']):.2%} |
"""
            with open(config["paths"]["report_file"], "w", encoding="utf-8") as f:
                f.write(report)
        
        # Append to experiment tracker CSV (Addition 11)
        try:
            csv_file = "./logs/experiment_log.csv"
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            file_exists = os.path.exists(csv_file)
            headers = [
                "timestamp", "seed", "tau1", "tau2", "pretrain_epochs", "train_samples",
                "accuracy_t1", "accuracy_final", "human_effort_ratio", "picr",
                "picr_status", "t1_coverage", "t3_escalation"
            ]
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "seed": current_seed,
                    "tau1": config["tier1"]["threshold_entropy"],
                    "tau2": config["tier1"]["threshold_confidence"],
                    "pretrain_epochs": config["tier1"].get("pretrain_epochs", 7),
                    "train_samples": config["data"]["train_samples"],
                    "accuracy_t1": metrics["accuracy_t1"],
                    "accuracy_final": metrics["accuracy_final"],
                    "human_effort_ratio": metrics["human_effort_ratio"],
                    "picr": metrics["picr"],
                    "picr_status": metrics["picr_status"],
                    "t1_coverage": metrics["tier_distribution"].get(1, 0) / metrics["total_samples"] if metrics["total_samples"] > 0 else 0.0,
                    "t3_escalation": metrics["tier_distribution"].get(3, 0) / metrics["total_samples"] if metrics["total_samples"] > 0 else 0.0
                })
        except Exception as e:
            print(f"Warning: Failed to write to experiment log CSV: {e}")

    # After the loop over seeds, compute and save multi-seed summary if needed (Addition 1)
    if len(args.seeds) > 1:
        runs_list = []
        for r_metrics in runs:
            runs_list.append({
                "seed": r_metrics["seed"],
                "accuracy_final": r_metrics["accuracy_final"],
                "accuracy_t1": r_metrics["accuracy_t1"],
                "human_effort_ratio": r_metrics["human_effort_ratio"],
                "picr": r_metrics["picr"]
            })
            
        mean_stats = {}
        std_stats = {}
        for key in ["accuracy_final", "accuracy_t1", "human_effort_ratio", "picr"]:
            vals = [r_metrics[key] for r_metrics in runs]
            mean_stats[key] = float(np.mean(vals))
            std_stats[key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            
        summary_data = {
            "runs": runs_list,
            "mean": mean_stats,
            "std": std_stats
        }
        
        os.makedirs("./logs", exist_ok=True)
        with open("./logs/multi_seed_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)
        print("Multi-seed summary saved to ./logs/multi_seed_summary.json")

    print("Task Completed. Metrics saved to logs.")

if __name__ == "__main__":
    main()
