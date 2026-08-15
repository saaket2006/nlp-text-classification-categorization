import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from PIL import Image
from matplotlib.patches import FancyArrowPatch
from core.calibration import reliability_diagram_data

class PDFReport(FPDF):
    def header(self):
        # Draw header only after the cover page
        if self.page_no() > 1:
            self.set_text_color(100, 116, 139) # Slate-500
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Tri-Tiered Active Learning Framework - Empirical Research Report", 0, 1, "R")
            self.line(10, 15, 200, 15)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139) # Slate-500
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def chapter_title(self, label):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(30, 41, 59) # Slate-900 (#1e293b)
        self.cell(0, 10, label, 0, 1, "L")
        self.ln(4)

    def add_summary_table(self, summary_data):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 41, 59) # Slate-900
        self.set_text_color(255, 255, 255)
        
        headers = ["Dataset", "T1 Acc (Pre)", "T1 Acc (Post)", "Final Acc", "Human Effort %", "Net Util (Human)", "Net Util (Cost)", "Avg Cost"]
        col_widths = [32, 22, 22, 20, 24, 25, 25, 20]
        
        # Header Row
        for h, w in zip(headers, col_widths):
            self.cell(w, 8, h, 1, 0, "C", True)
        self.ln()
        
        # Data Rows
        self.set_text_color(51, 65, 85) # Slate-700
        self.set_font("Helvetica", "", 8)
        fill = False
        for row in summary_data:
            self.set_fill_color(248, 250, 252 if fill else 255) # Alternate gray/white
            self.cell(col_widths[0], 7, row[0], 1, 0, "L", True)
            self.cell(col_widths[1], 7, row[1], 1, 0, "C", True)
            self.cell(col_widths[2], 7, row[2], 1, 0, "C", True)
            self.cell(col_widths[3], 7, row[3], 1, 0, "C", True)
            self.cell(col_widths[4], 7, row[4], 1, 0, "C", True)
            self.cell(col_widths[5], 7, row[5], 1, 0, "C", True)
            self.cell(col_widths[6], 7, row[6], 1, 0, "C", True)
            self.cell(col_widths[7], 7, row[7], 1, 0, "C", True)
            self.ln()
            fill = not fill
        self.ln(10)

def generate_all_plots_for_dataset(dataset_name, dataset_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load all required JSONs
    metrics_file = os.path.join(dataset_path, "metrics_summary.json")
    if not os.path.exists(metrics_file):
        print(f"Warning: No metrics_summary.json found for {dataset_name}")
        return None
        
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    detailed_results = None
    detailed_file = os.path.join(dataset_path, "detailed_results.json")
    if os.path.exists(detailed_file):
        with open(detailed_file, "r") as f:
            detailed_results = json.load(f)
            
    baseline_metrics = None
    baseline_file = os.path.join(dataset_path, "baseline_metrics.json")
    if os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            baseline_metrics = json.load(f)
            
    ablation_no_tier2 = None
    ablation_file = os.path.join(dataset_path, "ablation_no_tier2.json")
    if os.path.exists(ablation_file):
        with open(ablation_file, "r") as f:
            ablation_no_tier2 = json.load(f)
            
    ablation_no_entropy = None
    ablation_ent_file = os.path.join(dataset_path, "ablation_no_entropy.json")
    if os.path.exists(ablation_ent_file):
        with open(ablation_ent_file, "r") as f:
            ablation_no_entropy = json.load(f)
            
    random_routing = None
    rr_file = os.path.join(dataset_path, "baseline_random_routing.json")
    if os.path.exists(rr_file):
        with open(rr_file, "r") as f:
            random_routing = json.load(f)
            
    sweep_data = None
    sweep_file = os.path.join(dataset_path, "threshold_sweep.json")
    if os.path.exists(sweep_file):
        with open(sweep_file, "r") as f:
            sweep_data = json.load(f)
            
    history_data = None
    hist_file = os.path.join(dataset_path, "history.json")
    if os.path.exists(hist_file):
        with open(hist_file, "r") as f:
            history_data = json.load(f)
            
    # Read active learning curve
    lc_data = []
    for fn in os.listdir(dataset_path):
        if fn.startswith("learning_curve_seed") and fn.endswith(".json"):
            with open(os.path.join(dataset_path, fn), "r") as f:
                lc_data = json.load(f)
            break
            
    # Set clean plotting style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    # ── TIER COLORS ──
    COLORS = {
        "primary": "#818cf8",     # Indigo
        "success": "#34d399",     # Emerald
        "danger": "#f87171",      # Red
        "warning": "#fbbf24",     # Amber
        "info": "#60a5fa",        # Blue
        "purple": "#a78bfa",      # Purple
        "pink": "#f472b6",        # Pink
        "slate": "#94a3b8",       # Slate
    }
    
    plot_files = {}

    # 1. Routing Distribution (Donut)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    t1_count = metrics["tier_distribution"].get("1", metrics["tier_distribution"].get(1, 0))
    t2_count = metrics["tier_distribution"].get("2", metrics["tier_distribution"].get(2, 0))
    t3_count = metrics["tier_distribution"].get("3", metrics["tier_distribution"].get(3, 0))
    total = metrics["total_samples"]
    
    labels = ['Tier 1 (Autonomous)', 'Tier 2 (LLM Assisted)', 'Tier 3 (Human Expert)']
    sizes = [t1_count, t2_count, t3_count]
    non_zero = [(l, s) for l, s in zip(labels, sizes) if s > 0]
    p_labels, p_sizes = zip(*non_zero) if non_zero else (labels, [100, 0, 0])
    colors_matched = [COLORS[labels.index(l) == 0 and "info" or (labels.index(l) == 1 and "success" or "danger")] for l in p_labels]
    
    wedges, texts, autotexts = ax.pie(
        p_sizes, labels=p_labels, autopct='%1.1f%%',
        startangle=140, colors=colors_matched,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )
    plt.setp(texts, size=8, weight="bold", color="#1e293b")
    plt.setp(autotexts, size=8, weight="bold", color="white")
    ax.set_title("Request Routing Distribution", fontsize=10, weight="bold", pad=12)
    plt.tight_layout()
    plot1_path = os.path.join(out_dir, f"{dataset_name}_plot_dist.png")
    plt.savefig(plot1_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["dist"] = plot1_path

    # 2. Sample Flow Diagram (horizontal arrows flow)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    t3_via_t2 = 0
    t3_direct = 0
    if detailed_results:
        for r in detailed_results:
            pred = r.get("prediction", {})
            if pred.get("tier") == 3:
                if "Extreme" in pred.get("rationale", ""):
                    t3_direct += 1
                else:
                    t3_via_t2 += 1
        if t3_via_t2 + t3_direct != t3_count:
            t3_via_t2 = max(0, t3_count - t3_direct)
            t3_direct = t3_count - t3_via_t2
    else:
        t3_via_t2 = t3_count
        
    # Draw flowchart boxes
    # Input
    ax.text(1, 8.5, f"Input\n({total})", bbox=dict(boxstyle="round,pad=0.5", fc=COLORS["slate"], ec="none"), ha="center", va="center", color="white", weight="bold", fontsize=8)
    # Tier 1
    ax.text(4, 8.5, f"Tier 1 (Encoder)\n({total})", bbox=dict(boxstyle="round,pad=0.5", fc=COLORS["info"], ec="none"), ha="center", va="center", color="white", weight="bold", fontsize=8)
    # Tier 2
    ax.text(4, 4.5, f"Tier 2 (LLM)\n({t2_count + t3_via_t2})", bbox=dict(boxstyle="round,pad=0.5", fc=COLORS["success"], ec="none"), ha="center", va="center", color="white", weight="bold", fontsize=8)
    # Tier 3
    ax.text(4, 1.0, f"Tier 3 (Human)\n({t3_count})", bbox=dict(boxstyle="round,pad=0.5", fc=COLORS["danger"], ec="none"), ha="center", va="center", color="white", weight="bold", fontsize=8)
    # Final Output
    ax.text(8.5, 8.5, f"Final Output\n({total})", bbox=dict(boxstyle="round,pad=0.5", fc=COLORS["purple"], ec="none"), ha="center", va="center", color="white", weight="bold", fontsize=8)

    # Connections
    def draw_arrow(start, end, label):
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12, color='#94a3b8', lw=1.2)
        ax.add_patch(arrow)
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        ax.text(mid_x, mid_y + 0.25, label, fontsize=7, color="#475569", weight="bold", ha="center")
        
    draw_arrow((1.8, 8.5), (3.0, 8.5), f"{total}")
    draw_arrow((5.0, 8.5), (7.7, 8.5), f"{t1_count}")
    draw_arrow((4, 7.7), (4, 5.3), f"{t2_count + t3_via_t2 + t3_direct}")
    draw_arrow((4, 3.7), (4, 1.8), f"{t3_via_t2}")
    
    # Custom arrows for side channels
    arrow2_out = FancyArrowPatch((5.0, 4.5), (8.0, 8.1), arrowstyle='-|>', mutation_scale=12, color='#94a3b8', lw=1.2)
    ax.add_patch(arrow2_out)
    ax.text(6.8, 6.0, f"{t2_count}", fontsize=7, color="#475569", weight="bold", ha="center")
    
    arrow3_out = FancyArrowPatch((5.0, 1.0), (8.2, 7.7), arrowstyle='-|>', mutation_scale=12, color='#94a3b8', lw=1.2)
    ax.add_patch(arrow3_out)
    ax.text(6.9, 3.5, f"{t3_count}", fontsize=7, color="#475569", weight="bold", ha="center")

    ax.set_title("Request Flow Pipeline (Sankey)", fontsize=10, weight="bold", pad=12)
    plt.tight_layout()
    plot_flow_path = os.path.join(out_dir, f"{dataset_name}_plot_flow.png")
    plt.savefig(plot_flow_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["flow"] = plot_flow_path

    # 3. Accuracy Waterfall Chart
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if detailed_results:
        t1_correct = sum(1 for r in detailed_results if r["prediction"]["t1_label"] == r["ground_truth"])
        t2_corrections = sum(1 for r in detailed_results if r["prediction"]["t1_label"] != r["ground_truth"] and r["prediction"]["final_label"][0] == r["ground_truth"] and r["prediction"]["tier"] == 2)
        t3_corrections = sum(1 for r in detailed_results if r["prediction"]["t1_label"] != r["ground_truth"] and r["prediction"]["final_label"][0] == r["ground_truth"] and r["prediction"]["tier"] == 3)
        acc_t1 = t1_correct / total
        llm_gain = t2_corrections / total
        human_gain = t3_corrections / total
        acc_final = metrics["accuracy_final"]
        
        x = [0, 1, 2, 3]
        heights = [acc_t1, llm_gain, human_gain, acc_final]
        bottoms = [0, acc_t1, acc_t1 + llm_gain, 0]
        colors_wf = [COLORS["slate"], COLORS["success"], COLORS["warning"], COLORS["primary"]]
        
        bars = ax.bar(x, heights, bottom=bottoms, color=colors_wf, width=0.5, edgecolor="none", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["T1 Base", "T2 Gain", "T3 Gain", "Final Acc"], fontsize=8, weight="bold")
        ax.set_ylabel("Accuracy", fontsize=8, weight="bold")
        ax.set_ylim(0.0, 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        
        # Draw connector lines
        ax.plot([0.25, 0.75], [acc_t1, acc_t1], color="gray", linestyle=":", lw=1)
        ax.plot([1.25, 1.75], [acc_t1 + llm_gain, acc_t1 + llm_gain], color="gray", linestyle=":", lw=1)
        ax.plot([2.25, 2.75], [acc_final, acc_final], color="gray", linestyle=":", lw=1)
        
        for idx, bar in enumerate(bars):
            h = heights[idx]
            b = bottoms[idx]
            val = h if idx < 3 else h
            sign = "+" if idx in [1, 2] else ""
            ax.text(bar.get_x() + bar.get_width()/2.0, b + h + 0.01, f"{sign}{val:.1%}", ha="center", va="bottom", fontsize=8, weight="bold")
    else:
        ax.text(0.5, 0.5, "Detailed Results Not Available", ha='center', va='center', fontsize=12, color='#64748b')
        ax.set_axis_off()
        
    ax.set_title("Accuracy Waterfall Study", fontsize=10, weight="bold", pad=12)
    plt.tight_layout()
    plot_wf_path = os.path.join(out_dir, f"{dataset_name}_plot_waterfall.png")
    plt.savefig(plot_wf_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["waterfall"] = plot_wf_path

    # 4. Efficiency Frontier Chart
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if sweep_data:
        effort = [pt["human_effort_ratio"] for pt in sweep_data]
        acc = [pt["accuracy"] for pt in sweep_data]
        ax.scatter(effort, acc, color=COLORS["info"], alpha=0.5, s=25, label="Sweep Points", zorder=3)
        
    if history_data:
        h_effort = [pt["human_effort_ratio"] for pt in history_data]
        h_acc = [pt["accuracy"] for pt in history_data]
        ax.scatter(h_effort, h_acc, color=COLORS["purple"], alpha=0.7, s=20, label="Trial History", zorder=3)
        
    # Theoretical curve
    x_range = np.linspace(0, 0.3, 30)
    y_base = metrics["accuracy_t1"]
    y_theoretical = y_base + (1.0 - y_base) * (1 - np.exp(-10 * x_range))
    ax.plot(x_range, y_theoretical, linestyle="--", color=COLORS["slate"], label="Theoretical Limit", lw=1.5, zorder=2)
    
    # Current point
    ax.scatter([metrics["human_effort_ratio"]], [metrics["accuracy_final"]], color=COLORS["danger"], marker="*", s=160, edgecolor="white", linewidth=1.5, label="Current Run", zorder=5)
    
    ax.set_xlabel("Human Effort Ratio", fontsize=8, weight="bold")
    ax.set_ylabel("Accuracy", fontsize=8, weight="bold")
    ax.set_xlim(-0.02, 0.32)
    ax.set_ylim(min(y_theoretical) - 0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Efficiency Frontier Analysis", fontsize=10, weight="bold", pad=12)
    plt.tight_layout()
    plot_ef_path = os.path.join(out_dir, f"{dataset_name}_plot_frontier.png")
    plt.savefig(plot_ef_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["frontier"] = plot_ef_path

    # 5. Per-Category F1 Score
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if "per_category_f1" in metrics and "per_category_f1_t1" in metrics:
        categories = list(metrics["per_category_f1"].keys())
        # Truncate long category names
        display_cats = [c[:10] for c in categories]
        f1_t1 = [metrics["per_category_f1_t1"].get(c, {}).get("f1-score", 0) for c in categories]
        f1_final = [metrics["per_category_f1"].get(c, {}).get("f1-score", 0) for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, f1_t1, width, label="Tier 1 F1", color=COLORS["slate"], zorder=3)
        ax.bar(x + width/2, f1_final, width, label="Final F1", color=COLORS["primary"], zorder=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(display_cats, fontsize=7, rotation=15, weight="bold")
        ax.set_ylabel("F1 Score", fontsize=8, weight="bold")
        ax.set_ylim(0.0, 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Per-Category F1 Data Missing", ha='center', va='center', fontsize=12, color='#64748b')
        ax.set_axis_off()
        
    ax.set_title("Per-Category F1: T1 vs Final", fontsize=10, weight="bold", pad=12)
    plt.tight_layout()
    plot_f1_path = os.path.join(out_dir, f"{dataset_name}_plot_f1.png")
    plt.savefig(plot_f1_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["f1"] = plot_f1_path

    # 6. Precision & Recall Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    if "per_category_f1" in metrics and "per_category_f1_t1" in metrics:
        categories = list(metrics["per_category_f1"].keys())
        display_cats = [c[:8] for c in categories]
        prec_t1 = [metrics["per_category_f1_t1"].get(c, {}).get("precision", 0) for c in categories]
        prec_final = [metrics["per_category_f1"].get(c, {}).get("precision", 0) for c in categories]
        rec_t1 = [metrics["per_category_f1_t1"].get(c, {}).get("recall", 0) for c in categories]
        rec_final = [metrics["per_category_f1"].get(c, {}).get("recall", 0) for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        # Precision
        ax1.bar(x - width/2, prec_t1, width, label="Tier 1", color=COLORS["slate"], zorder=3)
        ax1.bar(x + width/2, prec_final, width, label="Final", color=COLORS["success"], zorder=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_cats, fontsize=7, rotation=15, weight="bold")
        ax1.set_ylabel("Precision", fontsize=8, weight="bold")
        ax1.set_ylim(0.0, 1.1)
        ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax1.legend(fontsize=7)
        ax1.set_title("Precision Comparison", fontsize=9, weight="bold")
        
        # Recall
        ax2.bar(x - width/2, rec_t1, width, label="Tier 1", color=COLORS["slate"], zorder=3)
        ax2.bar(x + width/2, rec_final, width, label="Final", color=COLORS["warning"], zorder=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_cats, fontsize=7, rotation=15, weight="bold")
        ax2.set_ylabel("Recall", fontsize=8, weight="bold")
        ax2.set_ylim(0.0, 1.1)
        ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax2.legend(fontsize=7)
        ax2.set_title("Recall Comparison", fontsize=9, weight="bold")
    else:
        ax1.text(0.5, 0.5, "Precision & Recall Data Missing", ha='center', va='center')
        ax2.set_axis_off()
        
    plt.tight_layout()
    plot_pr_path = os.path.join(out_dir, f"{dataset_name}_plot_pr.png")
    plt.savefig(plot_pr_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["pr"] = plot_pr_path

    # 7. Escalation Pattern Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    if "escalation_by_category" in metrics:
        esc = metrics["escalation_by_category"]
        esc_cats = list(esc.keys())
        display_cats = [c[:8] for c in esc_cats]
        t2_esc = [esc[c]["tier2"] for c in esc_cats]
        t3_esc = [esc[c]["tier3"] for c in esc_cats]
        t2_rate = [esc[c]["tier2"] / esc[c]["total"] if esc[c]["total"] > 0 else 0 for c in esc_cats]
        t3_rate = [esc[c]["tier3"] / esc[c]["total"] if esc[c]["total"] > 0 else 0 for c in esc_cats]
        
        x = np.arange(len(esc_cats))
        # Counts Subplot
        ax1.bar(x, t2_esc, label="Tier 2 (LLM)", color=COLORS["success"], zorder=3)
        ax1.bar(x, t3_esc, bottom=t2_esc, label="Tier 3 (Human)", color=COLORS["danger"], zorder=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_cats, fontsize=7, rotation=15, weight="bold")
        ax1.set_ylabel("Escalation Counts", fontsize=8, weight="bold")
        ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax1.legend(fontsize=7)
        ax1.set_title("Escalation Counts by Class", fontsize=9, weight="bold")
        
        # Rates Subplot
        ax2.bar(x, t2_rate, label="Tier 2 Rate", color=COLORS["success"], zorder=3)
        ax2.bar(x, t3_rate, bottom=t2_rate, label="Tier 3 Rate", color=COLORS["danger"], zorder=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_cats, fontsize=7, rotation=15, weight="bold")
        ax2.set_ylabel("Escalation Rate", fontsize=8, weight="bold")
        ax2.set_ylim(0.0, 1.1)
        ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax2.legend(fontsize=7)
        ax2.set_title("Escalation Rates by Class", fontsize=9, weight="bold")
    else:
        ax1.text(0.5, 0.5, "Escalation Profile Missing", ha='center', va='center')
        ax2.set_axis_off()
        
    plt.tight_layout()
    plot_esc_path = os.path.join(out_dir, f"{dataset_name}_plot_escalation.png")
    plt.savefig(plot_esc_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["escalation"] = plot_esc_path

    # 8. Active Learning adaptation Curve (plot3)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if lc_data:
        annotations = [pt["human_annotations"] for pt in lc_data]
        accuracies = [pt["test_accuracy"] for pt in lc_data]
        ax.plot(annotations, accuracies, marker='o', color='#3b82f6', linewidth=2, markersize=5, zorder=3)
        ax.set_xlabel("Human Annotations", fontsize=8, weight="bold")
        ax.set_ylabel("Test Accuracy", fontsize=8, weight="bold")
        ax.set_ylim(min(accuracies) - 0.03, max(accuracies) + 0.03)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        for i, acc in enumerate(accuracies):
            ax.text(annotations[i], acc + 0.005, f"{acc:.1%}", fontsize=7, weight="bold", ha='center')
    else:
        ax.text(0.5, 0.5, "No Active Learning Updates (0 Escalated)", ha='center', va='center', fontsize=9, color='#64748b')
        ax.set_axis_off()
    ax.set_title("AL Tracking (Accuracy vs Annotations)", fontsize=10, weight="bold", pad=12)
    plt.tight_layout()
    plot_al_path = os.path.join(out_dir, f"{dataset_name}_plot_al_curve.png")
    plt.savefig(plot_al_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["al_curve"] = plot_al_path

    # 9. Reliability Diagram
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if detailed_results:
        confidences = np.array([r["prediction"]["confidence"] for r in detailed_results])
        predictions = np.array([r["prediction"]["predicted_labels"][0] for r in detailed_results])
        labels = np.array([r["ground_truth"] for r in detailed_results])
        bin_lowers, bin_accs, bin_confs = reliability_diagram_data(confidences, predictions, labels)
        
        x_bins = [f"{bl:.1f}" for bl in bin_lowers]
        ax.bar(x_bins, bin_accs, alpha=0.8, color=COLORS["success"], label="Accuracy", zorder=3)
        ax.plot(x_bins, bin_confs, marker="o", color=COLORS["warning"], linestyle=":", markersize=4, label="Avg Confidence", zorder=4)
        ax.plot([0, 9], [0.05, 0.95], color=COLORS["danger"], linestyle="--", lw=1, label="Perfect Calibration", zorder=2)
        
        ax.set_xlabel("Confidence Bin", fontsize=8, weight="bold")
        ax.set_ylabel("Accuracy / Confidence", fontsize=8, weight="bold")
        ax.set_ylim(0.0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Detailed Results Not Available", ha='center', va='center')
        ax.set_axis_off()
        
    ax.set_title("Calibration Reliability Diagram", fontsize=10, weight="bold", pad=12)
    plt.tight_layout()
    plot_rel_path = os.path.join(out_dir, f"{dataset_name}_plot_reliability.png")
    plt.savefig(plot_rel_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["reliability"] = plot_rel_path

    # 10. Confidence Distribution Histogram
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if detailed_results:
        conf_vals = [r["prediction"]["confidence"] for r in detailed_results]
        ax.hist(conf_vals, bins=15, color=COLORS["primary"], alpha=0.8, edgecolor="none", zorder=3)
        ax.set_xlabel("Confidence Score", fontsize=8, weight="bold")
        ax.set_ylabel("Frequency Count", fontsize=8, weight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    else:
        ax.text(0.5, 0.5, "Confidence Distribution Missing", ha='center', va='center')
        ax.set_axis_off()
        
    ax.set_title("Confidence Distribution Histogram", fontsize=10, weight="bold", pad=12)
    plt.tight_layout()
    plot_cd_path = os.path.join(out_dir, f"{dataset_name}_plot_conf_dist.png")
    plt.savefig(plot_cd_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["conf_dist"] = plot_cd_path

    # 11. Confusion Matrices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.5))
    cm_labels = list(metrics.get("per_category_f1", {}).keys())
    if not cm_labels:
        cm_labels = ["Class 0", "Class 1"]
        
    # Truncate labels for confusion matrix rendering
    trunc_labels = [l[:6] for l in cm_labels]
    
    def render_cm_heatmap(ax_obj, cm_data, title, cmap):
        if cm_data is not None:
            cm_arr = np.array(cm_data)
            im = ax_obj.imshow(cm_arr, interpolation='nearest', cmap=cmap)
            ax_obj.set_title(title, fontsize=10, weight="bold")
            ax_obj.set_xticks(np.arange(len(trunc_labels)))
            ax_obj.set_yticks(np.arange(len(trunc_labels)))
            ax_obj.set_xticklabels(trunc_labels, fontsize=7, rotation=30)
            ax_obj.set_yticklabels(trunc_labels, fontsize=7)
            # Add text inside cells
            for i in range(len(trunc_labels)):
                for j in range(len(trunc_labels)):
                    val = cm_arr[i, j]
                    color = "white" if val > cm_arr.max()/2.0 else "black"
                    ax_obj.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8, weight="bold")
        else:
            ax_obj.text(0.5, 0.5, "Matrix Missing", ha='center', va='center')
            ax_obj.set_axis_off()
            
    render_cm_heatmap(ax1, metrics.get("confusion_matrix_t1"), "Tier 1 Conf Matrix", plt.cm.Blues)
    render_cm_heatmap(ax2, metrics.get("confusion_matrix_final"), "Final System Conf Matrix", plt.cm.Greens)
    render_cm_heatmap(ax3, metrics.get("confusion_matrix_escalated"), "Escalated Conf Matrix", plt.cm.Reds)
    
    plt.tight_layout()
    plot_cm_path = os.path.join(out_dir, f"{dataset_name}_plot_confusion.png")
    plt.savefig(plot_cm_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["confusion"] = plot_cm_path

    # 12. Baselines & Ablation Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    configs = {"Main": metrics}
    if baseline_metrics: configs["T2-only"] = baseline_metrics
    if ablation_no_tier2: configs["No T2"] = ablation_no_tier2
    if ablation_no_entropy: configs["No Ent"] = ablation_no_entropy
    if random_routing: configs["Random"] = random_routing
    
    config_names = list(configs.keys())
    # Accuracy Comparison
    accs = [configs[n]["accuracy_final"] for n in config_names]
    ax1.bar(config_names, accs, color=[COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["pink"], COLORS["info"]][:len(config_names)], width=0.5, zorder=3)
    ax1.set_ylabel("Final Accuracy", fontsize=8, weight="bold")
    ax1.set_ylim(0.0, 1.15)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax1.set_title("Accuracy Across Baselines", fontsize=9, weight="bold")
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=8, weight="bold")
        
    # Human Effort Comparison
    efforts = [configs[n]["human_effort_ratio"] for n in config_names]
    ax2.bar(config_names, efforts, color=COLORS["danger"], width=0.5, zorder=3)
    ax2.set_ylabel("Human Effort Ratio", fontsize=8, weight="bold")
    ax2.set_ylim(0.0, max(efforts) * 1.3 if max(efforts) > 0 else 0.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax2.set_title("Human Effort Across Baselines", fontsize=9, weight="bold")
    for i, v in enumerate(efforts):
        ax2.text(i, v + (max(efforts)*0.02 if max(efforts)>0 else 0.005), f"{v:.2%}", ha="center", va="bottom", fontsize=8, weight="bold")
        
    plt.tight_layout()
    plot_comp_path = os.path.join(out_dir, f"{dataset_name}_plot_comparison.png")
    plt.savefig(plot_comp_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["comparison"] = plot_comp_path

    # 13. PICR ROI Heatmap from Sweep
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if sweep_data:
        df_sweep = pd.DataFrame(sweep_data)
        if "tau1" in df_sweep.columns and "tau2" in df_sweep.columns and "picr" in df_sweep.columns:
            pivot = df_sweep.pivot_table(values="picr", index="tau2", columns="tau1", aggfunc="mean", fill_value=0)
            
            # Draw heatmap
            im = ax.imshow(pivot.values, aspect='auto', cmap='magma', interpolation='nearest', origin='lower')
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], fontsize=6, rotation=30)
            ax.set_yticklabels([f"{r:.2f}" for r in pivot.index], fontsize=6)
            
            ax.set_xlabel("Tier 1 threshold (tau 1)", fontsize=8, weight="bold")
            ax.set_ylabel("Tier 2 threshold (tau 2)", fontsize=8, weight="bold")
            fig.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, "tau1/tau2/picr columns missing", ha='center', va='center')
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Threshold Sweep Sweep Data Missing", ha='center', va='center')
        ax.set_axis_off()
        
    ax.set_title("PICR ROI Heatmap (Threshold Sweep)", fontsize=10, weight="bold", pad=12)
    plt.tight_layout()
    plot_hm_path = os.path.join(out_dir, f"{dataset_name}_plot_heatmap.png")
    plt.savefig(plot_hm_path, dpi=180, bbox_inches='tight')
    plt.close()
    plot_files["heatmap"] = plot_hm_path

    return {
        "metrics": metrics,
        "plots": plot_files
    }

def main():
    logs_dir = "./logs"
    out_pdf = "./logs/research_report.pdf"
    tmp_plots_dir = "./logs/tmp_plots"
    
    print("=========================================================")
    # Find all dataset subfolders in logs/
    datasets = []
    if os.path.exists(logs_dir):
        for f in os.listdir(logs_dir):
            p = os.path.join(logs_dir, f)
            if os.path.isdir(p) and f != "tmp_plots" and f != "reports" and f != "tier2_errors":
                datasets.append(f)
                
    if not datasets:
        print("Error: No dataset directories found inside the logs/ directory.")
        return
        
    print(f"Discovered datasets: {datasets}")
    
    # 1. Compile plots and read metrics
    compiled_data = {}
    summary_rows = []
    
    for ds in datasets:
        print(f"Generating charts and parsing metrics for dataset: {ds.upper()}")
        ds_path = os.path.join(logs_dir, ds)
        res = generate_all_plots_for_dataset(ds, ds_path, tmp_plots_dir)
        if res:
            compiled_data[ds] = res
            m = res["metrics"]
            
            # Format row for summary table
            pre_t1 = m.get("pre_al_t1_accuracy", m.get("accuracy_t1", 0.0))
            post_t1 = m.get("post_al_t1_accuracy", m.get("accuracy_t1", 0.0))
            final_acc = m.get("accuracy_final", 0.0)
            human_effort = m.get("human_effort_ratio", 0.0)
            net_util = m.get("net_utility", 0.0)
            net_util_cost = m.get("net_utility_cost", 0.0)
            avg_cost = m.get("average_compute_cost", 0.0)
            
            summary_rows.append([
                ds.upper(),
                f"{pre_t1:.2%}",
                f"{post_t1:.2%}",
                f"{final_acc:.2%}",
                f"{human_effort:.2%}",
                f"{net_util:.4f}",
                f"{net_util_cost:.4f}",
                f"{avg_cost:.2f}"
            ])
            
    if not compiled_data:
        print("Error: No metrics parsed successfully. PDF generation aborted.")
        return
        
    # Generate Cross-Dataset charts (Comparative synthesis at the end)
    print("Generating Cross-Dataset comparison charts...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.5))
    ds_names = [r[0] for r in summary_rows]
    
    # Chart 1: Final Routed Accuracy vs Tier 1 Accuracy
    pre_accs = [float(r[1].replace("%",""))/100.0 for r in summary_rows]
    post_accs = [float(r[3].replace("%",""))/100.0 for r in summary_rows]
    x = np.arange(len(ds_names))
    width = 0.35
    ax1.bar(x - width/2, pre_accs, width, label="T1 Pre-AL", color="#64748b", zorder=3)
    ax1.bar(x + width/2, post_accs, width, label="Routed Final", color="#818cf8", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_names, fontsize=7, rotation=15, weight="bold")
    ax1.set_ylabel("Accuracy", fontsize=8, weight="bold")
    ax1.set_ylim(0.0, 1.15)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax1.legend(fontsize=7)
    ax1.set_title("System Accuracy Comparison", fontsize=9, weight="bold")
    
    # Chart 2: Human Effort Ratio comparison
    efforts = [float(r[4].replace("%",""))/100.0 for r in summary_rows]
    ax2.bar(ds_names, efforts, color="#f87171", width=0.5, zorder=3)
    ax2.set_ylabel("Human Effort Ratio", fontsize=8, weight="bold")
    ax2.set_ylim(0.0, max(efforts)*1.3 if max(efforts)>0 else 0.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax2.set_title("Human Effort Workload", fontsize=9, weight="bold")
    for i, v in enumerate(efforts):
        ax2.text(i, v + (max(efforts)*0.02 if max(efforts)>0 else 0.005), f"{v:.2%}", ha="center", va="bottom", fontsize=8, weight="bold")
        
    # Chart 3: Net Utility comparisons
    utils = [float(r[5]) for r in summary_rows]
    ax3.bar(ds_names, utils, color="#34d399", width=0.5, zorder=3)
    ax3.set_ylabel("Net Utility (Human)", fontsize=8, weight="bold")
    ax3.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax3.set_title("Net Utility Output", fontsize=9, weight="bold")
    for i, v in enumerate(utils):
        ax3.text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=8, weight="bold")
        
    plt.tight_layout()
    cross_plot_path = os.path.join(tmp_plots_dir, "cross_dataset_plot.png")
    plt.savefig(cross_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Generate PDF Report
    print("Generating FPDF Research Report Document...")
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Cover Page
    pdf.add_page()
    
    # Title Banner Background
    pdf.set_fill_color(30, 41, 59) # Slate-900
    pdf.rect(0, 0, 210, 100, "F")
    
    pdf.set_y(35)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "TRI-TIERED LOCAL LLM", 0, 1, "C")
    pdf.cell(0, 12, "ACTIVE LEARNING FRAMEWORK", 0, 1, "C")
    
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(148, 163, 184) # Slate-400
    pdf.cell(0, 8, "Empirical Evaluation & Cost-Efficient Routing Report", 0, 1, "C")
    
    # Project info block
    pdf.set_y(115)
    pdf.set_text_color(30, 41, 59) # Slate-900
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Research Summary", 0, 1, "L")
    pdf.line(10, 125, 200, 125)
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85) # Slate-700
    intro_text = (
        "This research report compiles results from the empirical evaluation of the Tri-Tiered Active Learning Router. "
        "The tiered framework dynamically routes incoming text classification queries to appropriate computation layers "
        "(Autonomous DistilBERT, Assisted Qwen2.5:3B LLM, or Human Expert Annotator) based on uncertainty and entropy thresholds. "
        "By leveraging dynamic validation threshold sweeps and experience replay active learning, the framework achieves "
        "significant reductions in human expert workloads while maximizing model accuracy and utility."
    )
    pdf.multi_cell(0, 5.5, intro_text)
    pdf.ln(8)
    
    pdf.add_summary_table(summary_rows)
    
    # Pages 2+: Dataset Specific Reports
    for ds, data in compiled_data.items():
        m = data["metrics"]
        plots = data["plots"]
        
        # ── PAGE 1: Metrics & Routing ──
        pdf.add_page()
        pdf.chapter_title(f"Dataset Analysis: {ds.upper()}")
        
        # Display dataset key metrics inside a clean table
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "Performance & Budget Summary Metrics", 0, 1, "L")
        pdf.ln(2)
        
        col_w_label = 55
        col_w_val = 35
        
        metrics_table = [
            ("Total Evaluated Samples", f"{m.get('total_samples', 0)}"),
            ("Human Effort Ratio", f"{m.get('human_effort_ratio', 0.0):.2%}"),
            ("Pre-AL T1 Accuracy", f"{m.get('pre_al_t1_accuracy', m.get('accuracy_t1', 0.0)):.2%}"),
            ("Post-AL T1 Accuracy", f"{m.get('post_al_t1_accuracy', m.get('accuracy_t1', 0.0)):.2%}"),
            ("Routed Final System Accuracy", f"{m.get('accuracy_final', 0.0):.2%}"),
            ("Final System F1 Macro", f"{m.get('f1_macro', 0.0):.4f}"),
            ("Brier Score (Final)", f"{m.get('brier_score_final', 0.0):.4f}"),
            ("Final System ECE", f"{m.get('ece_final', 0.0):.4f}"),
            ("Net Utility (Human-Effort)", f"{m.get('net_utility', 0.0):.4f}"),
            ("Net Utility (Cost-Aware)", f"{m.get('net_utility_cost', 0.0):.4f}"),
            ("Average Compute Cost", f"{m.get('average_compute_cost', 0.0):.2f}"),
            ("Active Learning Escalations", f"{m.get('al_human_annotations', 0)} sample(s)")
        ]
        
        # Render a clean two-column grid for key metrics
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(51, 65, 85)
        for i in range(0, len(metrics_table), 2):
            k1, v1 = metrics_table[i]
            pdf.cell(col_w_label, 6, f"{k1}:", 0, 0, "L")
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(col_w_val, 6, v1, 0, 0, "L")
            pdf.set_font("Helvetica", "", 9)
            
            if i + 1 < len(metrics_table):
                k2, v2 = metrics_table[i+1]
                pdf.cell(col_w_label, 6, f"{k2}:", 0, 0, "L")
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(col_w_val, 6, v2, 0, 1, "L")
                pdf.set_font("Helvetica", "", 9)
            else:
                pdf.ln()
                
        pdf.ln(8)
        
        # Embed Plot 1 (Dist) and Plot 2 (Flow) Side by Side
        y_plots1 = pdf.get_y()
        pdf.image(plots["dist"], x=10, y=y_plots1, w=92)
        pdf.image(plots["flow"], x=108, y=y_plots1, w=92)
        
        # Move cursor below the plots
        pdf.set_y(y_plots1 + 82)
        
        # Embed Waterfall and Frontier Side by Side
        y_plots2 = pdf.get_y()
        pdf.image(plots["waterfall"], x=10, y=y_plots2, w=92)
        pdf.image(plots["frontier"], x=108, y=y_plots2, w=92)
        
        # ── PAGE 2: Categories & Active Learning Curve ──
        pdf.add_page()
        pdf.chapter_title(f"Category & Escalation Profile: {ds.upper()}")
        
        y_plots3 = pdf.get_y()
        pdf.image(plots["f1"], x=10, y=y_plots3, w=92)
        pdf.image(plots["al_curve"], x=108, y=y_plots3, w=92)
        
        pdf.set_y(y_plots3 + 82)
        
        # Embed precision/recall & escalation subplots
        y_plots4 = pdf.get_y()
        pdf.image(plots["pr"], x=10, y=y_plots4, w=190)
        pdf.set_y(y_plots4 + 85)
        pdf.image(plots["escalation"], x=10, y=pdf.get_y(), w=190)
        
        # ── PAGE 3: Calibration & Validation Studies ──
        pdf.add_page()
        pdf.chapter_title(f"Calibration & Validation Studies: {ds.upper()}")
        
        y_plots5 = pdf.get_y()
        pdf.image(plots["reliability"], x=10, y=y_plots5, w=92)
        pdf.image(plots["conf_dist"], x=108, y=y_plots5, w=92)
        
        pdf.set_y(y_plots5 + 82)
        
        # Embed Confusion matrix subplot (wide 3 panel)
        y_plots6 = pdf.get_y()
        pdf.image(plots["confusion"], x=10, y=y_plots6, w=190)
        
        pdf.set_y(y_plots6 + 68)
        
        # Embed Baselines Comparison & Heatmap Side by Side
        y_plots7 = pdf.get_y()
        pdf.image(plots["comparison"], x=10, y=y_plots7, w=92)
        pdf.image(plots["heatmap"], x=108, y=y_plots7, w=92)
        
    # Page Last: Cross Dataset Comparison Page
    pdf.add_page()
    pdf.chapter_title("Cross-Dataset Comparison Synthesis")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    pdf.ln(8)
    
    # Embed comparative summary plots
    pdf.image(cross_plot_path, x=10, y=pdf.get_y(), w=190)
        
    # 3. Clean up temporary plots
    print("Writing PDF File...")
    pdf.output(out_pdf)
    
    # Delete temporary plots from disk
    print("Cleaning up temporary plots files...")
    for fn in os.listdir(tmp_plots_dir):
        os.remove(os.path.join(tmp_plots_dir, fn))
    os.rmdir(tmp_plots_dir)
    
    print("=========================================================")
    print(f"SUCCESS: PDF Research Report successfully generated and saved to:")
    print(f"  {os.path.abspath(out_pdf)}")
    print("=========================================================")

if __name__ == "__main__":
    main()
