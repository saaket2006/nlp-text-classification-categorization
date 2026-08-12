import os
import sys
import subprocess
import json
import argparse
import datetime
import platform

def get_reproducibility_manifest(ollama_model="qwen2.5:3b"):
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        },
        "git": {},
        "ollama": {
            "model": ollama_model
        },
        "python_packages": {}
    }
    
    # Try to get Git commit hash
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
        manifest["git"] = {
            "commit": commit,
            "branch": branch
        }
    except Exception:
        manifest["git"] = "Not a git repository or git not installed"
        
    # Get Python package versions from requirements.txt or installed list
    try:
        packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8").splitlines()
        manifest["python_packages"] = {p.split("==")[0]: p.split("==")[1] for p in packages if "==" in p}
    except Exception:
        manifest["python_packages"] = "Failed to list packages"

    return manifest

def main():
    parser = argparse.ArgumentParser(description="Tri-Tiered Local LLM AL Framework - Reproducibility Entry Point")
    parser.add_argument("--datasets", nargs="+", default=["ag_news", "dbpedia_14", "imdb", "emotion"],
                        help="List of datasets to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 99, 2024],
                        help="List of seeds to evaluate")
    parser.add_argument("--sweep", action="store_true", default=True,
                        help="Run validation calibration sweep")
    parser.add_argument("--baseline", action="store_true", default=True,
                        help="Run Tier-2-only baseline")
    parser.add_argument("--ablation-no-tier2", action="store_true", default=True,
                        help="Run ablation with no Tier 2")
    parser.add_argument("--ablation-no-entropy", action="store_true", default=True,
                        help="Run ablation with no entropy routing")
    parser.add_argument("--random-routing", action="store_true", default=True,
                        help="Run budget-matched random routing baseline")
    args = parser.parse_args()

    print("======================================================================")
    print("      TRI-TIERED LOCAL LLM ACTIVE LEARNING FRAMEWORK RUNNER          ")
    print("======================================================================")
    
    # Save reproducibility manifest
    manifest = get_reproducibility_manifest()
    os.makedirs("logs", exist_ok=True)
    manifest_path = "logs/reproducibility_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved reproducibility manifest to {manifest_path}")

    # Build commands
    base_cmd = [sys.executable, "main.py", "--seeds"] + [str(s) for s in args.seeds]
    
    if args.sweep:
        base_cmd.append("--sweep")
    if args.baseline:
        base_cmd.append("--baseline")
    if args.ablation_no_tier2:
        base_cmd.append("--ablation-no-tier2")
    if args.ablation_no_entropy:
        base_cmd.append("--ablation-no-entropy")
    if args.random_routing:
        base_cmd.append("--random-routing")

    for ds in args.datasets:
        print(f"\n>>> Running experiments for dataset: {ds.upper()}...")
        cmd = base_cmd + ["--dataset", ds]
        print(f"Command: {' '.join(cmd)}")
        
        # Execute main.py via subprocess
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed run for {ds} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running pipeline for {ds}: {e}")
            
    # Aggregated Summary Table Generator
    print("\n======================================================================")
    print("                    EXPERIMENTAL RUNS SUMMARY                         ")
    print("======================================================================")
    
    headers = [
        "Dataset", "T1 Acc (Pre-AL)", "T1 Acc (Post-AL)", "Final Acc", "F1 Macro", 
        "ECE T1", "ECE Final", "Brier Final", "Human Effort %", "Net Utility", "Avg Cost"
    ]
    
    rows = []
    for ds in args.datasets:
        summary_path = f"logs/{ds}/multi_seed_summary.json"
        metrics_path = f"logs/{ds}/metrics_summary.json"
        
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                stats = json.load(f)
            mean = stats.get("mean", {})
            ci = stats.get("ci95", {})
            
            def fmt(key, percent=False):
                val = mean.get(key)
                c_val = ci.get(key)
                if val is None or c_val is None:
                    return "N/A"
                if percent:
                    return f"{val:.2%} ± {c_val:.2%}"
                return f"{val:.4f} ± {c_val:.4f}"

            rows.append([
                ds,
                fmt("accuracy_t1", percent=True),
                # Post-AL pre-AL is not in multi_seed_summary by default, retrieve from metrics_summary if available
                "N/A", # Will populate below
                fmt("accuracy_final", percent=True),
                fmt("f1_macro"),
                fmt("ece_t1"),
                fmt("ece_final"),
                fmt("brier_score_final"),
                fmt("human_effort_ratio", percent=True),
                fmt("net_utility"),
                fmt("average_compute_cost")
            ])
            
            # Try to pull pre-AL/post-AL means if we can
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics_single = json.load(f)
                # If we have single run, print it to console
        elif os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                m = json.load(f)
            rows.append([
                ds,
                f"{m.get('accuracy_t1', 0.0):.2%}",
                f"{m.get('post_al_t1_accuracy', 0.0):.2%}",
                f"{m.get('accuracy_final', 0.0):.2%}",
                f"{m.get('f1_macro', 0.0):.4f}",
                f"{m.get('ece_t1', 0.0):.4f}",
                f"{m.get('ece_final', 0.0):.4f}",
                f"{m.get('brier_score_final', 0.0):.4f}",
                f"{m.get('human_effort_ratio', 0.0):.2%}",
                f"{m.get('net_utility', 0.0):.4f}",
                f"{m.get('average_compute_cost', 0.0):.2f}"
            ])
        else:
            rows.append([ds] + ["N/A"] * (len(headers) - 1))
            
    # Print Markdown Table
    col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    
    row_fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep_row = "-|-".join("-" * w for w in col_widths)
    
    print("\n" + row_fmt.format(*headers))
    print(sep_row)
    for r in rows:
        print(row_fmt.format(*r))
        
    print("\nReproduction run completed successfully.")

if __name__ == "__main__":
    main()
