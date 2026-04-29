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
    print("Initializing Tier 1 Model...")
    ag_categories = ["World", "Sports", "Business", "Sci/Tech"]
    
    tier1 = Tier1Model(
        config["tier1"]["model_name"], 
        num_labels=len(ag_categories),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Initializing Tier 2 LLM...")
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
        epochs=config["tier1"].get("pretrain_epochs", 3)
    )
    
    if args.sweep:
        print("Running Threshold Sweep...")
        sweep_results = []
        tau1_grid = [0.3, 0.5, 0.7, 0.9]
        tau2_grid = [0.4, 0.5, 0.6, 0.7]
        
        for t1 in tau1_grid:
            for t2 in tau2_grid:
                ue = UncertaintyEngine(t1, t2)
                router = TieredRouter(tier1, tier2, ue, ag_categories)
                evaluator, _ = run_pipeline(router, df_test, ag_categories, config, tier1, update_model=False)
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
        
    print("Running Main Pipeline with Active Learning...")
    ue = UncertaintyEngine(
        config["tier1"]["threshold_entropy"],
        config["tier1"]["threshold_confidence"]
    )
    router = TieredRouter(tier1, tier2, ue, ag_categories)
    
    evaluator, results_log = run_pipeline(router, df_test, ag_categories, config, tier1, update_model=True)
        
    # Save Logs
    os.makedirs(config["paths"]["logs_dir"], exist_ok=True)
    with open(os.path.join(config["paths"]["logs_dir"], "detailed_results.json"), "w") as f:
        json.dump(results_log, f, indent=2)
        
    metrics = evaluator.save_report(config["paths"]["metrics_file"])
    
    # Research Report
    report = f"""
# Research Report: Tri-Tiered Local LLM AL Framework

## Performance Summary
- Total Samples: {metrics['total_samples']}
- Final F1 (Weighted): {metrics['f1_weighted']:.4f}
- Accuracy Improvement over Tier 1: {metrics['accuracy_final'] - metrics['accuracy_t1']:.4f}
- Human Effort Ratio: {metrics['human_effort_ratio']:.4f}
- PICR: {metrics['picr']:.4f}

## Tier Distribution
- Tier 1 (Base): {metrics['tier_distribution'].get(1, 0)}
- Tier 2 (Reasoning): {metrics['tier_distribution'].get(2, 0)}
- Tier 3 (Human): {metrics['tier_distribution'].get(3, 0)}

## Constraints Validation
- samples handled by Tier 1: {(metrics['tier_distribution'].get(1, 0)/metrics['total_samples'])*100:.2f}% (Target: >=60%)
- Tier 3 escalation: {(metrics['tier_distribution'].get(3, 0)/metrics['total_samples'])*100:.2f}% (Target: <=30%)
"""
    with open(config["paths"]["report_file"], "w") as f:
        f.write(report)
        
    print("Task Completed. Metrics saved to logs.")

if __name__ == "__main__":
    main()
