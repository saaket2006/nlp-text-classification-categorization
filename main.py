import yaml
import json
import os
import torch
from models.tier1_model import Tier1Model
from models.tier2_llm import Tier2LLM
from core.uncertainty import UncertaintyEngine
from core.router import TieredRouter
from data.loader import DataLoader
from metrics.evaluator import PipelineEvaluator
from tqdm import tqdm

def main():
    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Init Models
    print("Initializing Tier 1 Model...")
    # For AG News, we have 4 categories
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
    
    ue = UncertaintyEngine(
        config["tier1"]["threshold_entropy"],
        config["tier1"]["threshold_confidence"]
    )
    
    router = TieredRouter(tier1, tier2, ue, ag_categories)
    
    # Load Data
    print("Loading Data...")
    loader = DataLoader(config["data"]["dataset_name"])
    df_test = loader.load_data(split="test", num_samples=config["data"]["test_samples"])
    
    evaluator = PipelineEvaluator()
    
    # Run Pipeline
    print("Running Pipeline...")
    results_log = []
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        text = row["text"]
        gt = row["label"]
        
        prediction = router.process_sample(text)
        evaluator.add_result(prediction, gt)
        
        log_entry = {
            "prediction": prediction,
            "ground_truth": gt
        }
        results_log.append(log_entry)
        
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
- Tier 1 (Base): {metrics['tier_distribution'][1]}
- Tier 2 (Reasoning): {metrics['tier_distribution'][2]}
- Tier 3 (Human): {metrics['tier_distribution'][3]}

## Constraints Validation
- samples handled by Tier 1: {(metrics['tier_distribution'][1]/metrics['total_samples'])*100:.2f}% (Target: >=60%)
- Tier 3 escalation: {(metrics['tier_distribution'][3]/metrics['total_samples'])*100:.2f}% (Target: <=30%)
"""
    with open(config["paths"]["report_file"], "w") as f:
        f.write(report)
        
    print("Task Completed. Metrics saved to logs.")

if __name__ == "__main__":
    main()
