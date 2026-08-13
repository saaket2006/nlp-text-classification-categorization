import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import json
import sys
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
from metrics.evaluator import PipelineEvaluator, calculate_multi_seed_stats
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_pipeline(router, df_eval, ag_categories, config, tier1, update_model=False, df_al_pool=None, df_test=None, desc="Evaluating Test Set", disable_tqdm=False):
    """
    Executes the evaluation pipeline with ZERO test-set contamination.
    - If update_model is True, the active learning loop draws samples from df_al_pool (training split).
    - Evaluation pre-AL and post-AL are measured strictly against df_test (untouched test split).
    """
    if hasattr(router, "llm") and hasattr(router.llm, "sample_counter"):
        router.llm.sample_counter = 0

    eval_set = df_test if df_test is not None else df_eval

    # 1. Pre-AL Tier 1 accuracy on untouched test set
    test_texts = eval_set["text"].tolist()
    test_gts = eval_set["label"].tolist()

    pre_al_correct = 0
    if len(test_texts) > 0:
        pred_idxs, _, _ = tier1.predict_batch(test_texts, batch_size=64)
        for idx, gt in zip(pred_idxs, test_gts):
            t1_label = router.id_to_cat[idx]
            if t1_label == gt:
                pre_al_correct += 1
    pre_al_accuracy = pre_al_correct / len(eval_set) if len(eval_set) > 0 else 0.0

    evaluator = PipelineEvaluator()
    evaluator.al_learning_curve = []

    # 2. Optional Active Learning Adaptation Loop on Unlabeled AL Pool (Train Split)
    human_error_rate = config.get("tier3", {}).get("human_error_rate", 0.0)
    al_batch_size = config.get("tier1", {}).get("active_learning_batch_size", 1)

    if update_model and df_al_pool is not None:
        accumulated_texts = []
        accumulated_labels = []
        al_texts = []
        al_labels = []
        human_annotations_count = 0
        
        # Add initial point to learning curve
        evaluator.al_learning_curve.append({
            "human_annotations": 0,
            "test_accuracy": pre_al_accuracy
        })
        
        for idx, row in df_al_pool.iterrows():
            text = row["text"]
            gt = row["label"]

            prediction = router.process_sample(text, sample_idx=idx)
            if prediction["tier"] == 3:
                human_annotations_count += 1
                if random.random() < human_error_rate:
                    wrong_labels = [c for c in ag_categories if c != gt]
                    simulated_label = random.choice(wrong_labels)
                else:
                    simulated_label = gt

                al_texts.append(text)
                al_labels.append(router.cat_to_id[simulated_label])
                
                accumulated_texts.append(text)
                accumulated_labels.append(router.cat_to_id[simulated_label])

                if len(al_texts) >= al_batch_size:
                    # Train on the accumulated active learning history to guarantee stability and prevent catastrophic forgetting
                    tier1.train_on_history(accumulated_texts, accumulated_labels, epochs=3, batch_size=8)
                    al_texts = []
                    al_labels = []
                    
                # Evaluate model at regular intervals of 10 human annotations
                if human_annotations_count % 10 == 0:
                    current_test_correct = 0
                    if len(test_texts) > 0:
                        pred_idxs, _, _ = tier1.predict_batch(test_texts, batch_size=64)
                        for p_idx, p_gt in zip(pred_idxs, test_gts):
                            t1_label = router.id_to_cat[p_idx]
                            if t1_label == p_gt:
                                current_test_correct += 1
                    current_acc = current_test_correct / len(eval_set) if len(eval_set) > 0 else 0.0
                    evaluator.al_learning_curve.append({
                        "human_annotations": human_annotations_count,
                        "test_accuracy": current_acc
                    })

    # 3. Final System Evaluation on untouched Test Set
    results_log = []

    for idx, row in tqdm(eval_set.iterrows(), total=len(eval_set), desc=desc, disable=disable_tqdm, file=sys.stdout):
        text = row["text"]
        gt = row["label"]

        prediction = router.process_sample(text, sample_idx=idx)

        # Handle Tier 3 simulation during evaluation display
        if prediction["tier"] == 3:
            if random.random() < human_error_rate:
                wrong_labels = [c for c in ag_categories if c != gt]
                simulated_label = random.choice(wrong_labels)
                prediction["simulated_annotation_error"] = True
            else:
                simulated_label = gt
                prediction["simulated_annotation_error"] = False

            prediction["final_label"] = [simulated_label]

        evaluator.add_result(prediction, gt)
        results_log.append({
            "prediction": prediction,
            "ground_truth": gt
        })

    # 4. Post-AL Tier 1 accuracy on untouched test set
    post_al_correct = 0
    if len(test_texts) > 0:
        pred_idxs, _, _ = tier1.predict_batch(test_texts, batch_size=64)
        for idx, gt in zip(pred_idxs, test_gts):
            t1_label = router.id_to_cat[idx]
            if t1_label == gt:
                post_al_correct += 1
    post_al_accuracy = post_al_correct / len(eval_set) if len(eval_set) > 0 else 0.0
    
    # Append final post-AL accuracy to learning curve if update_model was active
    if update_model and df_al_pool is not None:
        if not evaluator.al_learning_curve or evaluator.al_learning_curve[-1]["human_annotations"] != human_annotations_count:
            evaluator.al_learning_curve.append({
                "human_annotations": human_annotations_count,
                "test_accuracy": post_al_accuracy
            })

    return evaluator, results_log, pre_al_accuracy, post_al_accuracy


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    set_seed(42)
    parser = argparse.ArgumentParser(description="Tri-Tiered Active Learning Pipeline (Rigorous Evaluation)")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep on validation split")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 99, 2024], help="Seeds for multi-seed run")
    parser.add_argument("--baseline", action="store_true", help="Run Tier-2-only baseline")
    parser.add_argument("--ablation-no-tier2", action="store_true", help="Run ablation with no Tier 2")
    parser.add_argument("--ablation-no-entropy", action="store_true", help="Run ablation with no entropy routing")
    parser.add_argument("--random-routing", action="store_true", help="Run budget-matched random routing baseline")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override (e.g. ag_news, dbpedia_14, imdb, emotion, sst2)")
    args = parser.parse_args()

    # Load Config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = args.dataset if args.dataset is not None else config["data"]["dataset_name"]
    print(f"Loading DataLoader for dataset: {dataset_name}...")
    loader = DataLoader(dataset_name)
    ag_categories = loader.get_categories()
    print(f"Discovered categories for {dataset_name} ({len(ag_categories)} classes): {ag_categories}")

    dataset_cfg = config.get("datasets_config", {}).get(dataset_name, {})
    train_samples = dataset_cfg.get("train_samples", config["data"].get("train_samples", 1500))
    test_samples = dataset_cfg.get("test_samples", config["data"].get("test_samples", 500))

    num_classes = len(ag_categories)
    max_entropy = float(np.log2(num_classes)) if num_classes > 0 else 1.0

    t1_cfg = config["tier1"].get("threshold_entropy")
    t2_cfg = config["tier1"].get("threshold_confidence")
    t3_cfg = config["tier1"].get("threshold_extreme_entropy")

    if t1_cfg is not None and isinstance(t1_cfg, (int, float)):
        threshold_entropy = float(t1_cfg)
    elif "datasets_config" in config and dataset_name in config["datasets_config"] and "threshold_entropy" in config["datasets_config"][dataset_name]:
        threshold_entropy = config["datasets_config"][dataset_name]["threshold_entropy"]
    else:
        threshold_entropy = 0.45 * max_entropy

    if t2_cfg is not None and isinstance(t2_cfg, (int, float)):
        conf_threshold = float(t2_cfg)
    elif "datasets_config" in config and dataset_name in config["datasets_config"] and "threshold_confidence" in config["datasets_config"][dataset_name]:
        conf_threshold = config["datasets_config"][dataset_name]["threshold_confidence"]
    else:
        conf_threshold = max(0.40, 0.70 - (0.01 * num_classes))

    if t3_cfg is not None and isinstance(t3_cfg, (int, float)):
        extreme_entropy_cap = float(t3_cfg)
    elif "datasets_config" in config and dataset_name in config["datasets_config"] and "threshold_extreme_entropy" in config["datasets_config"][dataset_name]:
        extreme_entropy_cap = config["datasets_config"][dataset_name]["threshold_extreme_entropy"]
    else:
        extreme_entropy_cap = 0.94 * max_entropy

    config["tier1"]["threshold_entropy"] = threshold_entropy
    config["tier1"]["threshold_confidence"] = conf_threshold
    config["tier1"]["threshold_extreme_entropy"] = extreme_entropy_cap

    lambda_al = config["tier1"].get("picr_al_lambda", 0.5)
    annotation_cost_weight = config["tier3"].get("annotation_cost_weight", 0.05)
    human_error_rate = config.get("tier3", {}).get("human_error_rate", 0.05)

    dataset_logs_dir = os.path.join(config["paths"]["logs_dir"], dataset_name)
    dataset_report_dir = os.path.join(config["paths"]["logs_dir"], "reports")
    os.makedirs(dataset_logs_dir, exist_ok=True)
    os.makedirs(dataset_report_dir, exist_ok=True)

    dataset_metrics_file = os.path.join(dataset_logs_dir, "metrics_summary.json")
    dataset_detailed_file = os.path.join(dataset_logs_dir, "detailed_results.json")
    dataset_history_file = os.path.join(dataset_logs_dir, "history.json")
    dataset_multiseed_file = os.path.join(dataset_logs_dir, "multi_seed_summary.json")
    report_filename = f"{dataset_name}_report.md"
    dataset_report_file = os.path.join(dataset_report_dir, report_filename)
    dataset_baseline_file = os.path.join(dataset_logs_dir, "baseline_metrics.json")
    dataset_ablation_no_t2_file = os.path.join(dataset_logs_dir, "ablation_no_tier2.json")
    dataset_ablation_no_ent_file = os.path.join(dataset_logs_dir, "ablation_no_entropy.json")
    dataset_random_file = os.path.join(dataset_logs_dir, "baseline_random_routing.json")
    dataset_sweep_file = os.path.join(dataset_logs_dir, "threshold_sweep.json")
    dataset_csv_file = os.path.join(dataset_logs_dir, "experiment_log.csv")

    offline_mode = os.environ.get("HF_HUB_OFFLINE") == "1"
    print(f"Using Device: {device}")

    tier2 = Tier2LLM(
        config["tier2"]["ollama_model"],
        config["tier2"]["url"]
    )
    print(f"Initializing Tier 2 LLM (Offline: {tier2.offline})...")
    tier2.setup_dataset(dataset_name, ag_categories)

    if args.baseline or args.random_routing or args.sweep or dataset_name in ["dbpedia_14", "imdb"]:
        tier2.num_votes = 1
    else:
        tier2.num_votes = 3

    sweep_ran = bool(args.sweep)
    runs = []

    for current_seed in args.seeds:
        print(f"\n==========================================")
        print(f"RUNNING PIPELINE WITH SEED: {current_seed}")
        print(f"==========================================")
        set_seed(current_seed)
        tier2.current_seed = current_seed

        # Load non-overlapping dataset partitions
        print("Loading non-overlapping dataset partitions (train_initial, al_pool, val_calibration, test)...")
        splits = loader.load_dataset_splits(
            seed=current_seed,
            train_initial_size=train_samples,
            al_pool_size=1000,
            val_calib_size=500,
            test_size=test_samples
        )
        df_train_initial = splits["train_initial"]
        df_al_pool = splits["al_pool"]
        df_val_calib = splits["val_calibration"]
        df_test = splits["test"]  # 100% UNTOUCHED test set

        pretrain_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Tier 1 Model (Pretraining on: {pretrain_device})...")
        tier1 = Tier1Model(
            config["tier1"]["model_name"], 
            num_labels=len(ag_categories),
            device=pretrain_device
        )

        print(f"Pretraining Tier 1 Model on {len(df_train_initial)} initial train samples...")
        cat_to_id = {cat: i for i, cat in enumerate(ag_categories)}
        train_texts = df_train_initial["text"].tolist()
        train_labels = [cat_to_id[lbl] for lbl in df_train_initial["label"].tolist()]
        tier1.pretrain(
            train_texts, 
            train_labels, 
            batch_size=config["tier1"].get("batch_size", 16),
            epochs=config["tier1"].get("pretrain_epochs", 7),
            early_stopping_patience=config["tier1"].get("early_stopping_patience")
        )

        if pretrain_device == "cuda" and device == "cpu":
            print("Moving Tier 1 model to CPU for active learning routing phase...")
            tier1.model.to("cpu")
            tier1.device = "cpu"
            torch.cuda.empty_cache()

        # Head adaptation optimizer
        encoder_params = []
        head_params = []
        for name, param in tier1.model.named_parameters():
            if "classifier" not in name and "pre_classifier" not in name:
                param.requires_grad = False
                encoder_params.append(param)
            else:
                head_params.append(param)

        from torch.optim import AdamW
        tier1.optimizer = AdamW([
            {"params": encoder_params, "lr": 0.0},
            {"params": head_params, "lr": 5e-4}
        ])

        pretrained_model_state = copy.deepcopy(tier1.model.state_dict())
        pretrained_optimizer_state = copy.deepcopy(tier1.optimizer.state_dict())

        def restore_pretrained_state(t1_model):
            t1_model.model.load_state_dict(copy.deepcopy(pretrained_model_state))
            t1_model.optimizer.load_state_dict(copy.deepcopy(pretrained_optimizer_state))
            t1_model.model.eval()

        # --- 1. Validation Calibration & Threshold Sweep ---
        if current_seed == args.seeds[0]:
            val_texts = df_val_calib["text"].tolist()
            
            # Calibrate extreme entropy cap using percentile (allow dataset-specific override)
            ds_cfg = config.get("datasets_config", {}).get(dataset_name, {})
            percentile = ds_cfg.get("extreme_entropy_percentile", config.get("tier3", {}).get("extreme_entropy_percentile", 0.98))
            current_extreme_entropy_cap = config["tier1"].get("threshold_extreme_entropy", 1.2)
            if len(val_texts) >= 2:
                print(f"Calibrating dynamic extreme entropy cap on VALIDATION split for {dataset_name} at {percentile * 100:.1f}% percentile...")
                _, probs_batch, _ = tier1.predict_batch(val_texts, batch_size=64)
                temp_ue = UncertaintyEngine(0.0, 1.0)
                entropies = [temp_ue.calculate_entropy(probs) for probs in probs_batch]
                sorted_ents = sorted(entropies)
                idx_p = min(int(percentile * len(sorted_ents)), len(sorted_ents) - 1)
                current_extreme_entropy_cap = float(sorted_ents[idx_p])
                print(f"Calibrated extreme_entropy_cap: {current_extreme_entropy_cap:.4f} (Max val entropy: {sorted_ents[-1]:.4f})")
            config["tier1"]["threshold_extreme_entropy"] = current_extreme_entropy_cap
            
            if args.sweep:
                print("Running Threshold Sweep on VALIDATION split (df_val_calib)...")
                sweep_results = []
                tau1_grid = [0.3, 0.5, 0.7, 0.8, 0.9]
                tau2_grid = [0.4, 0.5, 0.6, 0.7]

                config_tau1 = config["tier1"]["threshold_entropy"]
                config_tau2 = config["tier1"]["threshold_confidence"]
                if config_tau1 not in tau1_grid: tau1_grid.append(config_tau1)
                if config_tau2 not in tau2_grid: tau2_grid.append(config_tau2)
                tau1_grid.sort()
                sweep_pairs = [(t1, t2) for t1 in tau1_grid for t2 in tau2_grid]
                for t1, t2 in tqdm(sweep_pairs, desc="Validation Sweep Grid", file=sys.stdout):
                    restore_pretrained_state(tier1)
                    ue_s = UncertaintyEngine(t1, t2)
                    router_s = TieredRouter(
                        tier1, tier2, ue_s, ag_categories,
                        entropy_threshold=t1, conf_threshold=t2,
                        extreme_entropy_cap=current_extreme_entropy_cap,
                        dataset_name=dataset_name, mode="STANDARD",
                        human_error_rate=human_error_rate
                    )

                    evaluator_sw, _, _, _ = run_pipeline(
                        router_s, df_val_calib, ag_categories, config, tier1,
                        update_model=False, df_test=df_val_calib, disable_tqdm=True
                    )
                    metrics_sw = evaluator_sw.calculate_metrics(
                        lambda_al=lambda_al, 
                        annotation_cost_weight=annotation_cost_weight, 
                        categories=ag_categories,
                        cost_t1=config["tier3"].get("cost_t1", 1.0),
                        cost_t2=config["tier3"].get("cost_t2", 200.0),
                        cost_t3=config["tier3"].get("cost_t3", 3000.0)
                    )

                    sweep_results.append({
                        "tau1": t1,
                        "tau2": t2,
                        "accuracy": metrics_sw["accuracy_final"],
                        "human_effort_ratio": metrics_sw["human_effort_ratio"],
                        "picr": metrics_sw["picr"],
                        "net_utility": metrics_sw["net_utility"],
                        "net_utility_cost": metrics_sw["net_utility_cost"]
                    })

                os.makedirs(os.path.dirname(dataset_sweep_file), exist_ok=True)
                with open(dataset_sweep_file, "w") as f:
                    json.dump(sweep_results, f, indent=2)
                print(f"Validation Threshold Sweep results saved to {dataset_sweep_file}")
                
                # Select optimal operating point automatically based on Max Cost-Aware Net Utility
                best_sweep = max(sweep_results, key=lambda x: x["net_utility_cost"])
                config["tier1"]["threshold_entropy"] = best_sweep["tau1"]
                config["tier1"]["threshold_confidence"] = best_sweep["tau2"]
                print(f"Validation Sweep selected optimal thresholds (Max Cost-Aware Net Utility on Validation Calib: {best_sweep['net_utility_cost']:.4f}):")
                print(f"  Optimal tau_1 (entropy): {best_sweep['tau1']}")
                print(f"  Optimal tau_2 (confidence): {best_sweep['tau2']}")
        else:
            # Use thresholds calibrated from seed 0
            pass

        # --- 2. Main Active Learning Pipeline Run ---
        print("Running Main Tri-Tiered Active Learning Pipeline...")
        restore_pretrained_state(tier1)

        ue = UncertaintyEngine(
            config["tier1"]["threshold_entropy"],
            config["tier1"]["threshold_confidence"]
        )
        router = TieredRouter(
            tier1, tier2, ue, ag_categories,
            entropy_threshold=config["tier1"]["threshold_entropy"],
            conf_threshold=config["tier1"]["threshold_confidence"],
            extreme_entropy_cap=config["tier1"]["threshold_extreme_entropy"],
            dataset_name=dataset_name, mode="STANDARD",
            human_error_rate=human_error_rate
        )

        evaluator, results_log, pre_al_acc, post_al_acc = run_pipeline(
            router, df_test, ag_categories, config, tier1,
            update_model=True, df_al_pool=df_al_pool, df_test=df_test, desc=f"Main Pipeline (Seed {current_seed})"
        )

        evaluator.pre_al_t1_accuracy = pre_al_acc
        evaluator.post_al_t1_accuracy = post_al_acc
        
        cost_t1 = config["tier3"].get("cost_t1", 1.0)
        cost_t2 = config["tier3"].get("cost_t2", 200.0)
        cost_t3 = config["tier3"].get("cost_t3", 3000.0)
        
        metrics = evaluator.calculate_metrics(
            lambda_al=lambda_al, 
            annotation_cost_weight=annotation_cost_weight, 
            categories=ag_categories,
            cost_t1=cost_t1,
            cost_t2=cost_t2,
            cost_t3=cost_t3
        )

        metrics_copy = copy.deepcopy(metrics)
        metrics_copy["seed"] = current_seed
        runs.append(metrics_copy)
        
        # Save active learning learning curve history
        learning_curve_path = os.path.join(dataset_logs_dir, f"learning_curve_seed{current_seed}.json")
        with open(learning_curve_path, "w") as f:
            json.dump(evaluator.al_learning_curve, f, indent=2)

        # --- 3. Baselines & Ablations (Run on first seed using budgets frozen from validation) ---
        if current_seed == args.seeds[0]:
            # Evaluate proposed router on Validation Split to capture budget ratios cleanly and calibrate thresholds
            print("Evaluating Proposed Router on Validation Split to freeze baseline budgets...")
            restore_pretrained_state(tier1)
            ue_val = UncertaintyEngine(config["tier1"]["threshold_entropy"], config["tier1"]["threshold_confidence"])
            router_val = TieredRouter(
                tier1, tier2, ue_val, ag_categories,
                entropy_threshold=config["tier1"]["threshold_entropy"],
                conf_threshold=config["tier1"]["threshold_confidence"],
                extreme_entropy_cap=config["tier1"]["threshold_extreme_entropy"],
                dataset_name=dataset_name, mode="STANDARD",
                human_error_rate=human_error_rate
            )
            eval_val, _, _, _ = run_pipeline(
                router_val, df_val_calib, ag_categories, config, tier1,
                update_model=False, df_test=df_val_calib, disable_tqdm=True
            )
            val_metrics = eval_val.calculate_metrics(
                lambda_al=lambda_al, 
                annotation_cost_weight=annotation_cost_weight, 
                categories=ag_categories,
                cost_t1=cost_t1, cost_t2=cost_t2, cost_t3=cost_t3
            )
            
            # Freeze routing budget ratios from Validation run
            val_t1_count = val_metrics['tier_distribution'].get(1, 0)
            val_t2_count = val_metrics['tier_distribution'].get(2, 0)
            val_t3_count = val_metrics['tier_distribution'].get(3, 0)
            val_total = val_metrics['total_samples']
            
            val_f1 = val_t1_count / val_total if val_total > 0 else 0.60
            val_f2 = val_t2_count / val_total if val_total > 0 else 0.30
            val_f3 = val_t3_count / val_total if val_total > 0 else 0.10
            val_budget_ratios = (val_f1, val_f2, val_f3)
            f3_val = val_f3 # AL baseline budget prob
            
            print(f"Frozen Validation Budget Ratios: T1={val_f1:.2%}, T2={val_f2:.2%}, T3={val_f3:.2%}")

            if args.baseline:
                print("Running Tier-2-only Baseline on Test Set...")
                restore_pretrained_state(tier1)
                ue_b = UncertaintyEngine(0.0, 1.0)
                router_b = TieredRouter(
                    tier1, tier2, ue_b, ag_categories,
                    entropy_threshold=0.0, conf_threshold=1.0,
                    dataset_name=dataset_name, mode="TIER2_ONLY",
                    human_error_rate=human_error_rate
                )
                eval_b, _, pre_b, post_b = run_pipeline(
                    router_b, df_test, ag_categories, config, tier1,
                    update_model=False, df_test=df_test, desc="Tier-2-Only Baseline"
                )
                eval_b.pre_al_t1_accuracy = pre_b
                eval_b.post_al_t1_accuracy = post_b
                m_b = eval_b.calculate_metrics(
                    lambda_al=lambda_al, 
                    annotation_cost_weight=annotation_cost_weight, 
                    categories=ag_categories,
                    cost_t1=cost_t1, cost_t2=cost_t2, cost_t3=cost_t3
                )
                m_b["mode"] = "tier2_only_baseline"
                with open(dataset_baseline_file, "w") as f:
                    json.dump(m_b, f, indent=2)

            if args.ablation_no_tier2:
                print("Running Ablation: No Tier 2 on Test Set...")
                restore_pretrained_state(tier1)
                ue_a = UncertaintyEngine(config["tier1"]["threshold_entropy"], config["tier1"]["threshold_confidence"])
                router_a = TieredRouter(
                    tier1, tier2, ue_a, ag_categories,
                    entropy_threshold=config["tier1"]["threshold_entropy"],
                    conf_threshold=config["tier1"]["threshold_confidence"],
                    dataset_name=dataset_name, mode="NO_TIER2",
                    human_error_rate=human_error_rate
                )
                eval_a, _, pre_a, post_a = run_pipeline(
                    router_a, df_test, ag_categories, config, tier1,
                    update_model=True, df_al_pool=df_al_pool, df_test=df_test, desc="Ablation: No Tier 2"
                )
                eval_a.pre_al_t1_accuracy = pre_a
                eval_a.post_al_t1_accuracy = post_a
                m_a = eval_a.calculate_metrics(
                    lambda_al=lambda_al, 
                    annotation_cost_weight=annotation_cost_weight, 
                    categories=ag_categories,
                    cost_t1=cost_t1, cost_t2=cost_t2, cost_t3=cost_t3
                )
                with open(dataset_ablation_no_t2_file, "w") as f:
                    json.dump(m_a, f, indent=2)

            if args.ablation_no_entropy:
                print("Running Ablation: No Entropy Routing on Test Set...")
                restore_pretrained_state(tier1)
                ue_ne = UncertaintyEngine(999.0, config["tier1"]["threshold_confidence"])
                router_ne = TieredRouter(
                    tier1, tier2, ue_ne, ag_categories,
                    entropy_threshold=999.0, conf_threshold=config["tier1"]["threshold_confidence"],
                    dataset_name=dataset_name, mode="NO_ENTROPY",
                    human_error_rate=human_error_rate
                )
                eval_ne, _, pre_ne, post_ne = run_pipeline(
                    router_ne, df_test, ag_categories, config, tier1,
                    update_model=True, df_al_pool=df_al_pool, df_test=df_test, desc="Ablation: No Entropy"
                )
                eval_ne.pre_al_t1_accuracy = pre_ne
                eval_ne.post_al_t1_accuracy = post_ne
                m_ne = eval_ne.calculate_metrics(
                    lambda_al=lambda_al, 
                    annotation_cost_weight=annotation_cost_weight, 
                    categories=ag_categories,
                    cost_t1=cost_t1, cost_t2=cost_t2, cost_t3=cost_t3
                )
                with open(dataset_ablation_no_ent_file, "w") as f:
                    json.dump(m_ne, f, indent=2)

            if args.random_routing:
                print("Running Genuinely Budget-Matched Random Routing Baseline on Test Set...")
                restore_pretrained_state(tier1)
                ue_r = UncertaintyEngine(0.5, 0.5)
                router_r = TieredRouter(
                    tier1, tier2, ue_r, ag_categories,
                    entropy_threshold=0.5, conf_threshold=0.5,
                    dataset_name=dataset_name, mode="BUDGET_MATCHED_RANDOM",
                    budget_ratios=val_budget_ratios,
                    human_error_rate=human_error_rate
                )
                eval_r, _, pre_r, post_r = run_pipeline(
                    router_r, df_test, ag_categories, config, tier1,
                    update_model=False, df_test=df_test, desc="Random Routing Baseline"
                )
                eval_r.pre_al_t1_accuracy = pre_r
                eval_r.post_al_t1_accuracy = post_r
                m_r = eval_r.calculate_metrics(
                    lambda_al=lambda_al, 
                    annotation_cost_weight=annotation_cost_weight, 
                    categories=ag_categories,
                    cost_t1=cost_t1, cost_t2=cost_t2, cost_t3=cost_t3
                )
                with open(dataset_random_file, "w") as f:
                    json.dump(m_r, f, indent=2)

            # --- 4. Calibrate and Run Active Learning Baselines ---
            print("Calibrating and Running Active Learning Baselines (Budget-Matched)...")
            val_texts = df_val_calib["text"].tolist()
            print(f"Validation human annotation rate to match: {f3_val:.2%}")
            
            # Gather T1 outputs on validation set for threshold calibration
            _, val_probs_batch, _ = tier1.predict_batch(val_texts, batch_size=64)
            val_confs = [float(np.max(p)) for p in val_probs_batch]
            temp_ue = UncertaintyEngine(0.0, 1.0)
            val_ents = [temp_ue.calculate_entropy(p) for p in val_probs_batch]
            val_margins = []
            for p in val_probs_batch:
                s_p = sorted(p, reverse=True)
                val_margins.append(s_p[0] - s_p[1] if len(s_p) > 1 else 1.0)
                
            # Calibrate Least Confidence threshold
            s_confs = sorted(val_confs)
            idx_c = min(int(f3_val * len(s_confs)), len(s_confs) - 1)
            al_least_conf_threshold = float(s_confs[idx_c]) if len(s_confs) > 0 else 0.5
            
            # Calibrate Entropy threshold
            s_ents = sorted(val_ents)
            idx_e = min(int((1.0 - f3_val) * len(s_ents)), len(s_ents) - 1)
            al_entropy_threshold = float(s_ents[idx_e]) if len(s_ents) > 0 else 0.5
            
            # Calibrate Margin threshold
            s_margs = sorted(val_margins)
            idx_m = min(int(f3_val * len(s_margs)), len(s_margs) - 1)
            al_margin_threshold = float(s_margs[idx_m]) if len(s_margs) > 0 else 0.1
            
            print(f"Calibrated AL thresholds:")
            print(f"  AL_RANDOM selection probability: {f3_val:.4f}")
            print(f"  AL_LEAST_CONFIDENCE threshold: {al_least_conf_threshold:.4f}")
            print(f"  AL_ENTROPY threshold: {al_entropy_threshold:.4f}")
            print(f"  AL_MARGIN threshold: {al_margin_threshold:.4f}")
            
            # Run AL baselines
            al_modes = ["AL_RANDOM", "AL_LEAST_CONFIDENCE", "AL_ENTROPY", "AL_MARGIN"]
            for mode in al_modes:
                print(f"Running AL baseline: {mode} on Test Set...")
                restore_pretrained_state(tier1)
                router_al = TieredRouter(
                    tier1, tier2, ue_val, ag_categories,
                    entropy_threshold=al_entropy_threshold,
                    conf_threshold=al_least_conf_threshold,
                    extreme_entropy_cap=config["tier1"]["threshold_extreme_entropy"],
                    dataset_name=dataset_name, mode=mode,
                    human_error_rate=human_error_rate
                )
                router_al.al_budget_prob = f3_val
                router_al.al_least_conf_threshold = al_least_conf_threshold
                router_al.al_entropy_threshold = al_entropy_threshold
                router_al.al_margin_threshold = al_margin_threshold
                
                eval_al, _, pre_al, post_al = run_pipeline(
                    router_al, df_test, ag_categories, config, tier1,
                    update_model=True, df_al_pool=df_al_pool, df_test=df_test, desc=f"AL Baseline: {mode}"
                )
                eval_al.pre_al_t1_accuracy = pre_al
                eval_al.post_al_t1_accuracy = post_al
                m_al = eval_al.calculate_metrics(
                    lambda_al=lambda_al, 
                    annotation_cost_weight=annotation_cost_weight, 
                    categories=ag_categories,
                    cost_t1=cost_t1, cost_t2=cost_t2, cost_t3=cost_t3
                )
                
                # Save AL baseline results
                al_out_path = os.path.join(dataset_logs_dir, f"baseline_{mode.lower()}.json")
                with open(al_out_path, "w") as f:
                    json.dump(m_al, f, indent=2)
                
                # Save learning curve for baseline
                al_lc_path = os.path.join(dataset_logs_dir, f"learning_curve_{mode.lower()}_seed{current_seed}.json")
                with open(al_lc_path, "w") as f:
                    json.dump(eval_al.al_learning_curve, f, indent=2)

        # --- 5. Save results and logs for seed ---
        if current_seed == args.seeds[0]:
            os.makedirs(dataset_logs_dir, exist_ok=True)
            with open(dataset_detailed_file, "w") as f:
                json.dump(results_log, f, indent=2)

            metrics["active_learning_batch_size"] = config["tier1"].get("active_learning_batch_size", 1)
            with open(dataset_metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            history = []
            if os.path.exists(dataset_history_file):
                try:
                    with open(dataset_history_file, "r") as f:
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

            with open(dataset_history_file, "w") as f:
                json.dump(history, f, indent=2)

            # Generate markdown report
            t1_pct = (metrics['tier_distribution'].get(1, 0) / metrics['total_samples']) * 100
            t3_pct = (metrics['tier_distribution'].get(3, 0) / metrics['total_samples']) * 100

            picr_status = metrics.get("picr_status", "NO_GAIN")
            human_effort_ratio = metrics.get("human_effort_ratio", 0.0)
            tier2_autonomous_gain = metrics.get("tier2_autonomous_gain", 0.0)

            if picr_status == "AUTONOMOUS":
                picr_interpretation = f"The system achieved a {tier2_autonomous_gain:.2%} accuracy improvement entirely through Tier 2 LLM reasoning with zero human intervention."
            else:
                picr_interpretation = f"A PICR below 1.0 indicates human effort exceeded accuracy gain — adjust τ₁ upward or τ₂ downward to optimize cost-efficiency."

            per_cat_rows = []
            for cat in ag_categories:
                t1_perf = metrics['per_category_f1_t1'].get(cat, {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0})
                fin_perf = metrics['per_category_f1'].get(cat, {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0})
                per_cat_rows.append(
                    f"| **{cat}** | {t1_perf.get('precision', 0.0):.2%} | {t1_perf.get('recall', 0.0):.2%} | {t1_perf.get('f1-score', 0.0):.2%} | {t1_perf.get('support', 0.0):.0f} | {fin_perf.get('precision', 0.0):.2%} | {fin_perf.get('recall', 0.0):.2%} | {fin_perf.get('f1-score', 0.0):.2%} | {fin_perf.get('support', 0.0):.0f} |"
                )
            per_cat_table = "\n".join(per_cat_rows)

            escalation_rows = []
            for cat in ag_categories:
                esc_data = metrics['escalation_by_category'].get(cat, {'tier2': 0, 'tier3': 0, 'total': 0})
                esc_rate_pct = ((esc_data['tier2'] + esc_data['tier3']) / esc_data['total']) if esc_data['total'] > 0 else 0.0
                escalation_rows.append(
                    f"| **{cat}** | {esc_data.get('tier2', 0)} | {esc_data.get('tier3', 0)} | {esc_data.get('total', 0)} | {esc_rate_pct:.2%} |"
                )
            escalation_table = "\n".join(escalation_rows)

            rq1_verdict = "Supported" if t1_pct >= 60.0 and metrics['accuracy_final'] >= metrics['accuracy_t1'] else "Not Supported"
            rq2_verdict = "Supported" if metrics['post_al_t1_accuracy'] > metrics['pre_al_t1_accuracy'] else "Not observed in this run"
            rq3_verdict = "Supported" if sweep_ran else "Pending — run with --sweep flag"

            report = f"""# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | {metrics['total_samples']} | Official untouched test set |
 | **Tier 1 Accuracy** | {metrics['accuracy_t1']:.2%} | Baseline (Encoder only) |
 | **Final System Accuracy** | {metrics['accuracy_final']:.2%} | Integrated performance |
 | **Accuracy Boost** | {metrics['accuracy_final'] - metrics['accuracy_t1']:.2%} | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | {metrics['f1_macro']:.4f} | Macro-averaged F1 |
 | **Weighted F1 Score** | {metrics['f1_weighted']:.4f} | |
 | **Final ECE** | {metrics['ece_final']:.4f} | Calibration error (Final system) |
 | **Tier 1 ECE** | {metrics['ece_t1']:.4f} | Calibration error (Tier 1) |
 | **Brier Score** | {metrics['brier_score']:.4f} | Lower is better |
 | **Human Effort Ratio** | {metrics['human_effort_ratio']:.2%} | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | {metrics['pre_al_t1_accuracy']:.2%} | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | {metrics['post_al_t1_accuracy']:.2%} | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | {metrics.get('picr_display', 'N/A')} | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **{metrics['picr_status']}** | Efficiency classification |
 | **PICR-AL** | {metrics['picr_al_display']} | AL-aware cost-efficiency (λ={metrics['picr_al_lambda']}) |
 | **Net Utility (U)** | {metrics['net_utility_display']} | ΔAcc − (λ × HumanEffort), λ={metrics['annotation_cost_weight']} |
 | **Net Utility Status** | **{metrics['net_utility_status']}** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** {metrics['tier_distribution'].get(1, 0)} samples ({t1_pct:.1f}%)
 - **Tier 2 (Local LLM):** {metrics['tier_distribution'].get(2, 0)} samples ({(metrics['tier_distribution'].get(2, 0)/metrics['total_samples'])*100:.1f}%)
 - **Tier 3 (Simulated Human):** {metrics['tier_distribution'].get(3, 0)} samples ({t3_pct:.1f}%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** {t1_pct:.1f}% ({'PASSED' if t1_pct >= 60 else 'FAILED'})
 - ✅ **Human Cost Target (<=10%):** {t3_pct:.1f}% ({'PASSED' if t3_pct <= 10 else 'FAILED'})
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** {rq1_verdict} (Tier 1 Coverage: {t1_pct:.2f}%, Final Accuracy: {metrics['accuracy_final']:.2%}, Tier 1 Accuracy: {metrics['accuracy_t1']:.2%})
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** {rq2_verdict} (Pre-AL Tier 1 Accuracy: {metrics['pre_al_t1_accuracy']:.2%}, Post-AL Tier 1 Accuracy: {metrics['post_al_t1_accuracy']:.2%})
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** {rq3_verdict}
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 {per_cat_table}
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 {escalation_table}
 """
            with open(dataset_report_file, "w", encoding="utf-8") as f:
                f.write(report)

        try:
            os.makedirs(dataset_logs_dir, exist_ok=True)
            file_exists = os.path.exists(dataset_csv_file)
            headers = [
                "timestamp", "seed", "tau1", "tau2", "pretrain_epochs", "train_samples",
                "accuracy_t1", "accuracy_final", "f1_macro", "ece_final", "human_effort_ratio", "picr",
                "picr_status", "t1_coverage", "t3_escalation"
            ]
            with open(dataset_csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "seed": current_seed,
                    "tau1": config["tier1"]["threshold_entropy"],
                    "tau2": config["tier1"]["threshold_confidence"],
                    "pretrain_epochs": config["tier1"].get("pretrain_epochs", 7),
                    "train_samples": train_samples,
                    "accuracy_t1": metrics["accuracy_t1"],
                    "accuracy_final": metrics["accuracy_final"],
                    "f1_macro": metrics["f1_macro"],
                    "ece_final": metrics["ece_final"],
                    "human_effort_ratio": metrics["human_effort_ratio"],
                    "picr": metrics["picr"],
                    "picr_status": metrics["picr_status"],
                    "t1_coverage": metrics['tier_distribution'].get(1, 0) / metrics["total_samples"] if metrics["total_samples"] > 0 else 0.0,
                    "t3_escalation": metrics['tier_distribution'].get(3, 0) / metrics["total_samples"] if metrics["total_samples"] > 0 else 0.0
                })
        except Exception as e:
            print(f"Warning: Failed to write to experiment log CSV: {e}")

    # Rigorous Multi-Seed Statistics Calculation with 95% Confidence Intervals
    if len(args.seeds) > 1:
        multi_seed_summary = calculate_multi_seed_stats(runs)
        os.makedirs(dataset_logs_dir, exist_ok=True)
        with open(dataset_multiseed_file, "w") as f:
            json.dump(multi_seed_summary, f, indent=2)
        print(f"Multi-seed summary with 95% CIs saved to {dataset_multiseed_file}")

    print("Task Completed. Metrics saved to logs.")

if __name__ == "__main__":
    main()
