import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# Add the project root to sys.path to allow importing from core, models, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration import reliability_diagram_data

# Page Config
st.set_page_config(
    page_title="Tri-Tiered LLM AL Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.main { background-color: #0e1117; color: #fafafa; font-family: 'Inter', sans-serif; }

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1f2e 0%, #1e2130 100%);
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid rgba(99, 102, 241, 0.15);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.82rem !important; font-weight: 500 !important; letter-spacing: 0.03em; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background-color: #1e2130;
    border-radius: 10px 10px 0 0;
    padding: 10px 24px;
    color: #94a3b8;
    font-weight: 500;
    border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
    background-color: #1e2130 !important;
    color: #818cf8 !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
    border-bottom: 2px solid #818cf8 !important;
}

/* Subheaders */
.stSubheader, h2, h3 { color: #c7d2fe !important; font-weight: 600 !important; }

/* Section dividers */
hr { border-color: rgba(99, 102, 241, 0.1) !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# Color Palette
COLORS = {
    "primary": "#818cf8",     # Indigo
    "success": "#34d399",     # Emerald
    "danger": "#f87171",      # Red
    "warning": "#fbbf24",     # Amber
    "info": "#60a5fa",        # Blue
    "purple": "#a78bfa",      # Purple
    "pink": "#f472b6",        # Pink
    "slate": "#94a3b8",       # Slate
    "bg": "rgba(0,0,0,0)",
    "card": "#1e2130",
}
TIER_COLORS = [COLORS["info"], COLORS["success"], COLORS["danger"]]
CATEGORY_COLORS = ["#818cf8", "#34d399", "#fbbf24", "#f472b6"]

PLOT_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color="#e2e8f0", family="Inter"),
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(bgcolor="rgba(30,33,48,0.8)", bordercolor="rgba(99,102,241,0.2)", borderwidth=1),
)

# Helper Functions
LOGS_DIR = "./logs"

def load_json(filename):
    filepath = os.path.join(LOGS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

def get_tier_counts(metrics):
    dist = metrics["tier_distribution"]
    return (
        dist.get("1", dist.get(1, 0)),
        dist.get("2", dist.get(2, 0)),
        dist.get("3", dist.get(3, 0)),
    )

# Header
st.markdown("""
<div style="text-align:center; padding: 10px 0 5px 0;">
    <h1 style="color:#c7d2fe; font-size:2.2rem; font-weight:700; margin-bottom:0;">
        🛡️ Tri-Tiered Local LLM Framework
    </h1>
    <p style="color:#94a3b8; font-size:1rem; margin-top:4px;">
        Research Dashboard — Active Learning &amp; Automated Categorization
    </p>
</div>
""", unsafe_allow_html=True)

# Load Data
metrics = load_json("metrics_summary.json")
detailed_results = load_json("detailed_results.json")

if metrics is None:
    st.error("⚠️ No metrics found. Please run `python main.py` first.")
    st.stop()

t1_count, t2_count, t3_count = get_tier_counts(metrics)
total = metrics["total_samples"]

# Load optional data
baseline_metrics = load_json("baseline_metrics.json")
ablation_no_tier2 = load_json("ablation_no_tier2.json")
ablation_no_entropy = load_json("ablation_no_entropy.json")
random_routing = load_json("baseline_random_routing.json")
sweep_data = load_json("threshold_sweep.json")
history_data = load_json("history.json")
multi_seed = load_json("multi_seed_summary.json")

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Run Summary")
    st.markdown(f"**Total Samples:** {total}")
    st.markdown(f"**Tier 1 Accuracy:** {metrics['accuracy_t1']:.2%}")
    st.markdown(f"**Final Accuracy:** {metrics['accuracy_final']:.2%}")
    picr_val = metrics.get('picr_display', "N/A")
    st.markdown(f"**PICR:** {picr_val}")
    st.markdown(f"**Status:** `{metrics['picr_status']}`")
    st.markdown("---")

    st.markdown("### 📂 Available Data")
    data_flags = {
        "Main Metrics": True,
        "Detailed Results": detailed_results is not None,
        "Baseline (T2-only)": baseline_metrics is not None,
        "Ablation (No T2)": ablation_no_tier2 is not None,
        "Ablation (No Entropy)": ablation_no_entropy is not None,
        "Random Routing": random_routing is not None,
        "Threshold Sweep": sweep_data is not None,
        "Experiment History": history_data is not None,
        "Multi-Seed Summary": multi_seed is not None,
    }
    for name, available in data_flags.items():
        icon = "✅" if available else "❌"
        st.markdown(f"{icon} {name}")

# KPI Banner
# KPI Banner
st.markdown("### 📈 Core Performance Metrics")
r1_1, r1_2, r1_3, r1_4 = st.columns(4)

picr_status = metrics.get("picr_status", "NO_GAIN")
status_delta = None
delta_color = "off"
if picr_status == "AUTONOMOUS":
    status_delta = "Autonomous Mode"
    delta_color = "off"
elif picr_status in ["PERFECT_EFFICIENCY", "STRONG"]:
    status_delta = "High Efficiency"
    delta_color = "normal"
elif picr_status == "BELOW_TARGET" or picr_status == "NO_GAIN":
    status_delta = "Low Efficiency"
    delta_color = "inverse"

r1_1.metric("Final Accuracy", f"{metrics['accuracy_final']:.2%}")
r1_2.metric("Accuracy Boost", f"+{(metrics['accuracy_final'] - metrics['accuracy_t1']):.2%}")
t2_autonomous_gain = metrics.get("tier2_autonomous_gain", 0.0)
r1_3.metric("T2 Autonomous Gain", f"{t2_autonomous_gain:.2%}")
r1_4.metric("Weighted F1", f"{metrics['f1_weighted']:.4f}")

st.markdown("### ⚙️ Operational & Efficiency Metrics")
r2_1, r2_2, r2_3, r2_4, r2_5 = st.columns(5)

r2_1.metric("Human Effort", f"{metrics['human_effort_ratio']:.2%}")

picr_display = metrics.get("picr_display", "N/A")
if picr_display == "N/A":
    r2_2.metric("PICR Score", "N/A", delta="Autonomous Mode", delta_color="off")
else:
    picr_score_val = metrics.get("picr")
    picr_score_str = f"{picr_score_val:.4f}" if picr_score_val is not None else "N/A"
    r2_2.metric("PICR Score", picr_score_str)

r2_3.metric("PICR Status", picr_status, delta=status_delta, delta_color=delta_color)

# PICR-AL card
picr_al_status = metrics.get("picr_al_status", "")
picr_al_delta_color = "off"
if picr_al_status in ["STRONG", "ACCEPTABLE"]:
    picr_al_delta_color = "normal"
elif picr_al_status in ["BELOW_TARGET", "NO_GAIN"]:
    picr_al_delta_color = "inverse"
r2_4.metric("PICR-AL", metrics.get('picr_al_display', 'N/A'), delta=metrics.get('picr_al_status', ''), delta_color=picr_al_delta_color)

# Net Utility card
net_utility_status = metrics.get("net_utility_status", "")
net_utility_delta_color = "off"
if net_utility_status == "POSITIVE":
    net_utility_delta_color = "normal"
elif net_utility_status == "NEGATIVE":
    net_utility_delta_color = "inverse"
elif net_utility_status == "BREAK_EVEN":
    net_utility_delta_color = "off"
r2_5.metric("Net Utility (U)", metrics.get('net_utility_display', 'N/A'), delta=net_utility_status, delta_color=net_utility_delta_color)

# Conditional banners
if metrics.get("picr_negative_warning"):
    st.error("⚠️ **Negative PICR:** Pipeline accuracy is below Tier 1 baseline. Review threshold configuration.")
if picr_status == "AUTONOMOUS":
    st.success("🤖 Autonomous Mode: Accuracy improved through Tier 2 alone — no human intervention required. PICR is not applicable.")
elif picr_status == "PERFECT_EFFICIENCY":
    st.success("✨ **Perfect Efficiency:** Accuracy improved with zero human intervention.")

if net_utility_status == "NEGATIVE":
    st.warning("⚠️ Net Utility is negative — the annotation cost exceeded the accuracy gain at this operating point. Consider adjusting escalation thresholds.")
elif net_utility_status == "POSITIVE":
    st.success("✅ Net Utility is positive — the system adds measurable value after accounting for human annotation cost.")

st.markdown("---")

# Tabs
tab_overview, tab_categories, tab_calibration, tab_comparison, tab_history, tab_logs = st.tabs([
    "📊 Overview",
    "🏷️ Per-Category",
    "🎯 Calibration & Confusion",
    "⚔️ Baselines & Ablations",
    "📈 Experiment History",
    "📝 Decision Logs",
])

# TAB 1: OVERVIEW
with tab_overview:
    col_left, col_right = st.columns(2)

    # ── Routing Distribution Donut ──
    with col_left:
        st.subheader("Routing Distribution")
        df_dist = pd.DataFrame({
            "Tier": ["Tier 1 (Encoder)", "Tier 2 (LLM)", "Tier 3 (Human)"],
            "Count": [t1_count, t2_count, t3_count],
        })
        fig = px.pie(
            df_dist, values="Count", names="Tier", hole=0.55,
            color_discrete_sequence=TIER_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent+value", textfont_size=13)
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Sankey Flow ──
    with col_right:
        st.subheader("Sample Flow (Sankey)")
        # Calculate dynamic flow values to satisfy conservation:
        # 0: Input, 1: Tier 1, 2: Tier 2, 3: Tier 3, 4: Final Output
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
            # Reconciliation
            if t3_via_t2 + t3_direct != t3_count:
                t3_via_t2 = max(0, t3_count - t3_direct)
                t3_direct = t3_count - t3_via_t2
        else:
            t3_direct = 0
            t3_via_t2 = t3_count

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15, thickness=20,
                line=dict(color="#1e293b", width=0.5),
                label=["Input", "Tier 1", "Tier 2", "Tier 3", "Final Output"],
                color=[COLORS["slate"], COLORS["info"], COLORS["success"], COLORS["danger"], COLORS["purple"]],
            ),
            link=dict(
                source=[0, 1, 1, 1, 2, 2, 3],
                target=[1, 4, 2, 3, 4, 3, 4],
                value=[total, t1_count, t2_count + t3_via_t2, t3_direct, t2_count, t3_via_t2, t3_count],
                color=["rgba(148,163,184,0.3)"] * 7,
            ),
        )])
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


    col_left2, col_right2 = st.columns(2)

    # ── Accuracy Waterfall ──
    with col_left2:
        st.subheader("Accuracy Waterfall")
        if detailed_results:
            t1_correct = sum(1 for r in detailed_results if r["prediction"]["t1_label"] == r["ground_truth"])
            t2_corrections = sum(1 for r in detailed_results
                                 if r["prediction"]["t1_label"] != r["ground_truth"]
                                 and r["prediction"]["final_label"][0] == r["ground_truth"]
                                 and r["prediction"]["tier"] == 2)
            t3_corrections = sum(1 for r in detailed_results
                                 if r["prediction"]["t1_label"] != r["ground_truth"]
                                 and r["prediction"]["final_label"][0] == r["ground_truth"]
                                 and r["prediction"]["tier"] == 3)
            acc_t1 = t1_correct / total
            llm_gain = t2_corrections / total
            human_gain = t3_corrections / total

            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["T1 Baseline", "T2 (LLM) Gain", "T3 (Human) Gain", "Final Accuracy"],
                textposition="outside",
                text=[f"{acc_t1:.1%}", f"+{llm_gain:.1%}", f"+{human_gain:.1%}", f"{metrics['accuracy_final']:.1%}"],
                y=[acc_t1, llm_gain, human_gain, metrics["accuracy_final"]],
                connector={"line": {"color": "rgba(99,102,241,0.3)"}},
                increasing={"marker": {"color": COLORS["success"]}},
                decreasing={"marker": {"color": COLORS["danger"]}},
                totals={"marker": {"color": COLORS["primary"]}},
            ))
            fig.update_layout(**PLOT_LAYOUT, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the pipeline with detailed results to see the accuracy waterfall.")

    # ── Efficiency Frontier ──
    with col_right2:
        st.subheader("Efficiency Frontier")
        fig = go.Figure()

        # Sweep points
        if sweep_data:
            effort = [pt["human_effort_ratio"] for pt in sweep_data]
            acc = [pt["accuracy"] for pt in sweep_data]
            fig.add_trace(go.Scatter(x=effort, y=acc, mode="markers", name="Sweep Points",
                                     marker=dict(size=7, color=COLORS["info"], opacity=0.4)))

        # History
        if history_data:
            h_effort = [pt["human_effort_ratio"] for pt in history_data]
            h_acc = [pt["accuracy"] for pt in history_data]
            fig.add_trace(go.Scatter(x=h_effort, y=h_acc, mode="markers", name="Trial History",
                                     marker=dict(size=6, color=COLORS["purple"], opacity=0.7, symbol="circle")))

        # Theoretical frontier
        x_range = np.linspace(0, 0.3, 30)
        y_base = metrics["accuracy_t1"]
        y_theoretical = y_base + (1.0 - y_base) * (1 - np.exp(-10 * x_range))
        fig.add_trace(go.Scatter(x=x_range, y=y_theoretical, mode="lines", name="Theoretical Limit",
                                 line=dict(dash="dash", color=COLORS["slate"])))

        # Current operating point
        fig.add_trace(go.Scatter(
            x=[metrics["human_effort_ratio"]], y=[metrics["accuracy_final"]],
            mode="markers", name="Current Run",
            marker=dict(size=16, color=COLORS["danger"], symbol="star", line=dict(width=2, color="white")),
        ))

        fig.update_layout(**PLOT_LAYOUT, xaxis_title="Human Effort Ratio", yaxis_title="Accuracy", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # ── AL Improvement Tracking ──
    if "pre_al_t1_accuracy" in metrics and "post_al_t1_accuracy" in metrics:
        st.subheader("Active Learning Improvement Tracking")
        al_col1, al_col2, al_col3 = st.columns(3)
        al_col1.metric("Pre-AL Tier 1 Accuracy", f"{metrics['pre_al_t1_accuracy']:.2%}")
        al_col2.metric("Post-AL Tier 1 Accuracy", f"{metrics['post_al_t1_accuracy']:.2%}")
        delta = metrics["post_al_t1_accuracy"] - metrics["pre_al_t1_accuracy"]
        al_col3.metric("AL Improvement", f"{delta:+.2%}")

        # Contextual explanation banners
        human_labels = metrics.get("human_labels_count", 0)
        al_batch_size = metrics.get("active_learning_batch_size", 1)
        if delta == 0:
            if human_labels == 0:
                st.info(
                    "💡 **Why is Active Learning Improvement 0.00%?**\n\n"
                    "No samples were escalated to Tier 3 (human expert) in this run. "
                    "The Tier 1 model was confident enough on all samples, so no new training data was collected.\n\n"
                    "**This is actually a positive signal** — it means the model is performing well autonomously."
                )
            elif human_labels < al_batch_size:
                st.info(
                    "💡 **Why is Active Learning Improvement 0.00%?**\n\n"
                    f"Only **{human_labels}** sample(s) were escalated to Tier 3 (human expert), "
                    f"but the active learning batch size is **{al_batch_size}**. "
                    "The model only retrains after accumulating a full batch of escalated samples.\n\n"
                    "**How to trigger AL retraining:**\n"
                    "- Increase the test dataset size to naturally encounter more uncertain boundary cases\n"
                    "- Or reduce `active_learning_batch_size` in `config.yaml` (not recommended — can destabilize training)"
                )
            else:
                st.warning(
                    "⚠️ **Active Learning ran but produced no measurable improvement.**\n\n"
                    f"**{human_labels}** samples were used for retraining, "
                    "but the model's test accuracy did not change. "
                    "This can happen when the escalated samples are outliers that don't generalize to the broader test set."
                )
        elif delta > 0:
            st.success(
                f"✅ **Active Learning improved Tier 1 accuracy by {delta:+.2%}!**\n\n"
                f"The model was retrained on **{human_labels}** escalated sample(s) during the pipeline run, "
                "successfully learning from high-uncertainty boundary cases identified by the routing system."
            )
        else:
            st.error(
                f"⚠️ **Active Learning caused a regression of {delta:+.2%}.**\n\n"
                f"The model was retrained on **{human_labels}** escalated sample(s), "
                "but this led to a slight accuracy decrease. "
                "Consider increasing the human error rate guard or reviewing the quality of escalated samples."
            )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Pre-AL", "Post-AL"],
            y=[metrics["pre_al_t1_accuracy"], metrics["post_al_t1_accuracy"]],
            marker_color=[COLORS["slate"], COLORS["success"] if delta >= 0 else COLORS["danger"]],
            text=[f"{metrics['pre_al_t1_accuracy']:.2%}", f"{metrics['post_al_t1_accuracy']:.2%}"],
            textposition="outside", textfont=dict(color="#e2e8f0"),
        ))
        fig.update_layout(**PLOT_LAYOUT, yaxis_title="Tier 1 Accuracy", yaxis_tickformat=".0%",
                          yaxis_range=[max(0, metrics["pre_al_t1_accuracy"] - 0.05), 1.0])
        st.plotly_chart(fig, use_container_width=True)

    # ── Constraint Validation ──
    st.subheader("Constraint Validation")
    cv1, cv2 = st.columns(2)
    t1_ratio = t1_count / total
    t3_ratio = t3_count / total
    if t1_ratio >= 0.6:
        cv1.success(f"✅ Efficiency Target Met — Tier 1 handled {t1_ratio:.1%} of samples (target ≥ 60%)")
    else:
        cv1.warning(f"⚠️ Efficiency Target Missed — Tier 1 handled {t1_ratio:.1%} of samples (target ≥ 60%)")
    if t3_ratio <= 0.3:
        cv2.success(f"✅ Human Cost Target Met — Tier 3 handled {t3_ratio:.1%} of samples (target ≤ 30%)")
    else:
        cv2.warning(f"⚠️ Human Cost Target Exceeded — Tier 3 handled {t3_ratio:.1%} of samples (target ≤ 30%)")


# TAB 2: PER-CATEGORY

with tab_categories:
    categories = ["World", "Sports", "Business", "Sci/Tech"]

    # ── Per-Category F1 Grouped Bar ──
    st.subheader("Per-Category F1 Score: Tier 1 vs Final")
    if "per_category_f1" in metrics and "per_category_f1_t1" in metrics:
        f1_t1 = [metrics["per_category_f1_t1"].get(c, {}).get("f1-score", 0) for c in categories]
        f1_final = [metrics["per_category_f1"].get(c, {}).get("f1-score", 0) for c in categories]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Tier 1 F1", x=categories, y=f1_t1, marker_color=COLORS["slate"],
                             text=[f"{v:.2%}" for v in f1_t1], textposition="outside"))
        fig.add_trace(go.Bar(name="Final F1", x=categories, y=f1_final, marker_color=COLORS["primary"],
                             text=[f"{v:.2%}" for v in f1_final], textposition="outside"))
        fig.update_layout(**PLOT_LAYOUT, barmode="group", yaxis_title="F1 Score", yaxis_tickformat=".0%",
                          yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Per-category F1 data is not available.")

    col_prec, col_rec = st.columns(2)

    # ── Precision Comparison ──
    with col_prec:
        st.subheader("Precision: Tier 1 vs Final")
        if "per_category_f1" in metrics:
            prec_t1 = [metrics["per_category_f1_t1"].get(c, {}).get("precision", 0) for c in categories]
            prec_final = [metrics["per_category_f1"].get(c, {}).get("precision", 0) for c in categories]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Tier 1", x=categories, y=prec_t1, marker_color=COLORS["slate"]))
            fig.add_trace(go.Bar(name="Final", x=categories, y=prec_final, marker_color=COLORS["success"]))
            fig.update_layout(**PLOT_LAYOUT, barmode="group", yaxis_title="Precision", yaxis_tickformat=".0%",
                              yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)

    # ── Recall Comparison ──
    with col_rec:
        st.subheader("Recall: Tier 1 vs Final")
        if "per_category_f1" in metrics:
            rec_t1 = [metrics["per_category_f1_t1"].get(c, {}).get("recall", 0) for c in categories]
            rec_final = [metrics["per_category_f1"].get(c, {}).get("recall", 0) for c in categories]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Tier 1", x=categories, y=rec_t1, marker_color=COLORS["slate"]))
            fig.add_trace(go.Bar(name="Final", x=categories, y=rec_final, marker_color=COLORS["warning"]))
            fig.update_layout(**PLOT_LAYOUT, barmode="group", yaxis_title="Recall", yaxis_tickformat=".0%",
                              yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)

    # ── Escalation Pattern Analysis ──
    st.subheader("Escalation Pattern Analysis")
    if "escalation_by_category" in metrics:
        esc = metrics["escalation_by_category"]
        esc_cats = list(esc.keys())
        t2_esc = [esc[c]["tier2"] for c in esc_cats]
        t3_esc = [esc[c]["tier3"] for c in esc_cats]
        totals = [esc[c]["total"] for c in esc_cats]
        t2_rate = [esc[c]["tier2"] / esc[c]["total"] if esc[c]["total"] > 0 else 0 for c in esc_cats]
        t3_rate = [esc[c]["tier3"] / esc[c]["total"] if esc[c]["total"] > 0 else 0 for c in esc_cats]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Escalation Counts", "Escalation Rate (%)"),
                            specs=[[{"type": "bar"}, {"type": "bar"}]])
        fig.add_trace(go.Bar(name="Tier 2", x=esc_cats, y=t2_esc, marker_color=COLORS["success"]), row=1, col=1)
        fig.add_trace(go.Bar(name="Tier 3", x=esc_cats, y=t3_esc, marker_color=COLORS["danger"]), row=1, col=1)
        fig.add_trace(go.Bar(name="Tier 2 Rate", x=esc_cats, y=t2_rate, marker_color=COLORS["success"],
                             text=[f"{r:.1%}" for r in t2_rate], textposition="outside", showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(name="Tier 3 Rate", x=esc_cats, y=t3_rate, marker_color=COLORS["danger"],
                             text=[f"{r:.1%}" for r in t3_rate], textposition="outside", showlegend=False), row=1, col=2)
        fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=400)
        fig.update_yaxes(tickformat=".0%", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        # Escalation table
        st.markdown("**Detailed Escalation Table**")
        esc_df = pd.DataFrame({
            "Category": esc_cats,
            "Total Samples": totals,
            "Tier 2 Escalations": t2_esc,
            "Tier 3 Escalations": t3_esc,
            "Escalation Rate": [f"{(t2_esc[i] + t3_esc[i]) / totals[i]:.2%}" if totals[i] > 0 else "0.00%" for i in range(len(esc_cats))],
        })
        st.dataframe(esc_df, use_container_width=True, hide_index=True)
    else:
        st.info("Escalation data is not available.")


# TAB 3: CALIBRATION & CONFUSION

with tab_calibration:
    col_rel, col_ece = st.columns(2)

    # ── Reliability Diagram ──
    with col_rel:
        st.subheader("Reliability Diagram")
        if detailed_results:
            confidences = np.array([r["prediction"]["confidence"] for r in detailed_results])
            predictions = np.array([r["prediction"]["predicted_labels"][0] for r in detailed_results])
            labels = np.array([r["ground_truth"] for r in detailed_results])
            bin_lowers, bin_accs, bin_confs = reliability_diagram_data(confidences, predictions, labels)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"{bl:.1f}" for bl in bin_lowers], y=bin_accs, name="Accuracy",
                                 marker_color=COLORS["success"], opacity=0.8))
            fig.add_trace(go.Scatter(x=[f"{bl:.1f}" for bl in bin_lowers], y=bin_confs, mode="lines+markers",
                                     name="Avg Confidence", line=dict(color=COLORS["warning"], dash="dot"),
                                     marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=["0.0", "0.9"], y=[0, 0.95], mode="lines",
                                     line=dict(dash="dash", color=COLORS["danger"], width=1),
                                     name="Perfect Calibration"))
            fig.update_layout(**PLOT_LAYOUT, xaxis_title="Confidence Bin", yaxis_title="Accuracy / Confidence",
                              yaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Detailed results needed for the reliability diagram.")

    # ── ECE Display ──
    with col_ece:
        st.subheader("Calibration Metrics")
        ece_val = metrics.get("ece", 0)
        st.metric("Expected Calibration Error (ECE)", f"{ece_val:.4f}")
        if ece_val < 0.05:
            st.success("🎯 Excellent calibration — ECE < 5%")
        elif ece_val < 0.10:
            st.info("✅ Good calibration — ECE < 10%")
        else:
            st.warning("⚠️ Calibration could be improved — ECE ≥ 10%")

        # Confidence distribution histogram
        if detailed_results:
            st.subheader("Confidence Distribution")
            conf_vals = [r["prediction"]["confidence"] for r in detailed_results]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=conf_vals, nbinsx=20, marker_color=COLORS["primary"], opacity=0.8,
                                       name="Confidence"))
            fig.update_layout(**PLOT_LAYOUT, xaxis_title="Confidence", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrices ──
    st.subheader("Confusion Matrices")
    cm_labels = ["World", "Sports", "Business", "Sci/Tech"]
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("**Tier 1 Predictions**")
        if "confusion_matrix_t1" in metrics:
            cm_t1 = np.array(metrics["confusion_matrix_t1"])
            fig = px.imshow(cm_t1, x=cm_labels, y=cm_labels, text_auto=True,
                            color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"))
            fig.update_layout(**PLOT_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True)

    with cm_col2:
        st.markdown("**Final System Predictions**")
        if "confusion_matrix_final" in metrics:
            cm_final = np.array(metrics["confusion_matrix_final"])
            fig = px.imshow(cm_final, x=cm_labels, y=cm_labels, text_auto=True,
                            color_continuous_scale="Greens", labels=dict(x="Predicted", y="Actual", color="Count"))
            fig.update_layout(**PLOT_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Escalated Confusion Matrix ──
    if "confusion_matrix_escalated" in metrics:
        st.markdown("**Escalated Samples Only**")
        cm_esc = np.array(metrics["confusion_matrix_escalated"])
        if cm_esc.sum() > 0:
            fig = px.imshow(cm_esc, x=cm_labels, y=cm_labels, text_auto=True,
                            color_continuous_scale="Reds", labels=dict(x="Predicted", y="Actual", color="Count"))
            fig.update_layout(**PLOT_LAYOUT, height=400, width=500)
            st.plotly_chart(fig, use_container_width=False)
        else:
            st.info("No escalated samples in this run.")


# TAB 4: BASELINES & ABLATIONS

with tab_comparison:
    st.subheader("Baselines & Ablation Study Comparison")

    # Collect all available configurations
    configs = {}
    configs["Main Pipeline"] = metrics
    if baseline_metrics:
        configs["Tier-2-Only Baseline"] = baseline_metrics
    if ablation_no_tier2:
        configs["Ablation: No Tier 2"] = ablation_no_tier2
    if ablation_no_entropy:
        configs["Ablation: No Entropy"] = ablation_no_entropy
    if random_routing:
        configs["Random Routing"] = random_routing

    if len(configs) > 1:
        config_names = list(configs.keys())

        # ── Accuracy Comparison Bar ──
        acc_vals = [configs[n]["accuracy_final"] for n in config_names]
        acc_t1_vals = [configs[n]["accuracy_t1"] for n in config_names]
        bar_colors = [COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["pink"], COLORS["info"]]

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("**Final Accuracy Comparison**")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=config_names, y=acc_t1_vals, name="Tier 1 Accuracy",
                                 marker_color=COLORS["slate"], text=[f"{v:.2%}" for v in acc_t1_vals],
                                 textposition="outside"))
            fig.add_trace(go.Bar(x=config_names, y=acc_vals, name="Final Accuracy",
                                 marker_color=[bar_colors[i % len(bar_colors)] for i in range(len(config_names))],
                                 text=[f"{v:.2%}" for v in acc_vals], textposition="outside"))
            fig.update_layout(**PLOT_LAYOUT, barmode="group", yaxis_title="Accuracy", yaxis_tickformat=".0%",
                              yaxis_range=[0, 1.1], height=450)
            st.plotly_chart(fig, use_container_width=True)

        with comp_col2:
            st.markdown("**Human Effort & PICR Comparison**")
            effort_vals = [configs[n]["human_effort_ratio"] for n in config_names]
            picr_vals = [configs[n]["picr"] for n in config_names]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=config_names, y=effort_vals, name="Human Effort",
                                 marker_color=COLORS["danger"], opacity=0.7,
                                 text=[f"{v:.2%}" for v in effort_vals], textposition="outside"),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=config_names, y=picr_vals, name="PICR", mode="lines+markers",
                                     marker=dict(size=10, color=COLORS["warning"]),
                                     line=dict(color=COLORS["warning"], width=2)),
                          secondary_y=True)
            fig.update_layout(**PLOT_LAYOUT, height=450)
            fig.update_yaxes(title_text="Human Effort", tickformat=".0%", secondary_y=False)
            fig.update_yaxes(title_text="PICR Score", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        # ── Tier Distribution Comparison ──
        st.markdown("**Tier Distribution Across Configurations**")
        tier_data = []
        for name in config_names:
            c = configs[name]
            dist = c["tier_distribution"]
            t = c["total_samples"]
            tier_data.append({
                "Config": name,
                "Tier 1 %": (dist.get("1", dist.get(1, 0)) / t * 100) if t > 0 else 0,
                "Tier 2 %": (dist.get("2", dist.get(2, 0)) / t * 100) if t > 0 else 0,
                "Tier 3 %": (dist.get("3", dist.get(3, 0)) / t * 100) if t > 0 else 0,
            })
        df_tier = pd.DataFrame(tier_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Tier 1", x=df_tier["Config"], y=df_tier["Tier 1 %"], marker_color=COLORS["info"]))
        fig.add_trace(go.Bar(name="Tier 2", x=df_tier["Config"], y=df_tier["Tier 2 %"], marker_color=COLORS["success"]))
        fig.add_trace(go.Bar(name="Tier 3", x=df_tier["Config"], y=df_tier["Tier 3 %"], marker_color=COLORS["danger"]))
        fig.update_layout(**PLOT_LAYOUT, barmode="stack", yaxis_title="Percentage (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary Table ──
        st.markdown("**Summary Table**")
        summary_rows = []
        for name in config_names:
            c = configs[name]
            summary_rows.append({
                "Configuration": name,
                "Total Samples": c["total_samples"],
                "T1 Accuracy": f"{c['accuracy_t1']:.2%}",
                "Final Accuracy": f"{c['accuracy_final']:.2%}",
                "F1 (Weighted)": f"{c['f1_weighted']:.4f}",
                "ECE": f"{c.get('ece', 0):.4f}",
                "Human Effort": f"{c['human_effort_ratio']:.2%}",
                "PICR": c.get("picr_display", "N/A"),
                "Status": c.get("picr_status", "N/A"),
                "PICR-AL": c.get("picr_al_display", "N/A"),
                "Net Utility": c.get("net_utility_display", "N/A"),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run baselines and ablations (`--baseline`, `--ablation-no-tier2`, `--ablation-no-entropy`, `--random-routing`) to see comparisons.")

    # ── PICR ROI Heatmap from Sweep ──
    st.subheader("PICR ROI Heatmap (Threshold Sweep)")
    if sweep_data:
        df_sweep = pd.DataFrame(sweep_data)
        pivot = df_sweep.pivot_table(values="picr", index="tau2", columns="tau1", aggfunc="mean", fill_value=0)
        fig = go.Figure(data=go.Heatmap(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.values,
            colorscale="Viridis",
            colorbar=dict(title="PICR"),
            text=pivot.values,
            texttemplate="%{text:.2f}",
        ))
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="τ₁ (Entropy Threshold)", yaxis_title="τ₂ (Confidence Threshold)",
                          height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run with `--sweep` to generate the PICR ROI heatmap.")

    # ── Multi-Seed Summary ──
    if multi_seed:
        st.subheader("Multi-Seed Reproducibility")
        ms_col1, ms_col2 = st.columns(2)
        with ms_col1:
            st.markdown("**Per-Seed Results**")
            st.dataframe(pd.DataFrame(multi_seed["runs"]), use_container_width=True, hide_index=True)
        with ms_col2:
            st.markdown("**Aggregated Statistics**")
            agg_data = []
            for key in ["accuracy_final", "accuracy_t1", "human_effort_ratio", "picr"]:
                agg_data.append({
                    "Metric": key,
                    "Mean": f"{multi_seed['mean'][key]:.4f}",
                    "Std Dev": f"{multi_seed['std'][key]:.4f}",
                })
            st.dataframe(pd.DataFrame(agg_data), use_container_width=True, hide_index=True)


# TAB 5: EXPERIMENT HISTORY

with tab_history:
    st.subheader("Experiment History")

    # ── CSV Log ──
    csv_path = os.path.join(LOGS_DIR, "experiment_log.csv")
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        st.markdown("**Experiment Log (CSV)**")
        st.dataframe(df_csv, use_container_width=True, hide_index=True)

        if len(df_csv) > 1:
            hist_col1, hist_col2 = st.columns(2)

            with hist_col1:
                st.markdown("**Accuracy Over Runs**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df_csv["accuracy_t1"], mode="lines+markers", name="Tier 1 Accuracy",
                                         line=dict(color=COLORS["slate"]), marker=dict(size=7)))
                fig.add_trace(go.Scatter(y=df_csv["accuracy_final"], mode="lines+markers", name="Final Accuracy",
                                         line=dict(color=COLORS["primary"]), marker=dict(size=7)))
                fig.update_layout(**PLOT_LAYOUT, xaxis_title="Run #", yaxis_title="Accuracy", yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

            with hist_col2:
                st.markdown("**PICR Over Runs**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df_csv["picr"], mode="lines+markers", name="PICR",
                                         line=dict(color=COLORS["warning"], width=2),
                                         marker=dict(size=8, color=COLORS["warning"])))
                fig.add_hline(y=3.0, line_dash="dash", line_color=COLORS["danger"],
                              annotation_text="PICR Target (3.0)", annotation_position="top left")
                fig.update_layout(**PLOT_LAYOUT, xaxis_title="Run #", yaxis_title="PICR Score")
                st.plotly_chart(fig, use_container_width=True)

            # ── Tier Coverage Over Runs ──
            st.markdown("**Tier 1 Coverage & Tier 3 Escalation Over Runs**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df_csv["t1_coverage"], mode="lines+markers", name="Tier 1 Coverage",
                                     line=dict(color=COLORS["info"]), marker=dict(size=7)))
            fig.add_trace(go.Scatter(y=df_csv["t3_escalation"], mode="lines+markers", name="Tier 3 Escalation",
                                     line=dict(color=COLORS["danger"]), marker=dict(size=7)))
            fig.add_hline(y=0.60, line_dash="dash", line_color=COLORS["success"],
                          annotation_text="Min T1 Target (60%)", annotation_position="top left")
            fig.add_hline(y=0.30, line_dash="dash", line_color=COLORS["danger"],
                          annotation_text="Max T3 Target (30%)", annotation_position="bottom left")
            fig.update_layout(**PLOT_LAYOUT, xaxis_title="Run #", yaxis_title="Ratio", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No experiment log CSV found. Run the pipeline to generate experiment history.")

    # ── History JSON ──
    if history_data and len(history_data) > 1:
        st.subheader("PICR History (JSON Log)")
        df_hist = pd.DataFrame(history_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["picr"], mode="lines+markers",
                                 name="PICR", line=dict(color=COLORS["purple"], width=2),
                                 marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["accuracy"], mode="lines+markers",
                                 name="Final Accuracy", line=dict(color=COLORS["success"]),
                                 marker=dict(size=6), yaxis="y2"))
        fig.update_layout(
            **PLOT_LAYOUT,
            xaxis_title="Timestamp",
            yaxis=dict(title="PICR Score"),
            yaxis2=dict(title="Accuracy", overlaying="y", side="right", tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)


# TAB 6: DECISION LOGS

with tab_logs:
    st.subheader("Sample Decision Logs")
    if detailed_results:
        # Filters
        filt_col1, filt_col2 = st.columns(2)
        with filt_col1:
            tier_filter = st.multiselect("Filter by Tier", [1, 2, 3], default=[1, 2, 3])
        with filt_col2:
            n_rows = st.slider("Number of rows to display", 10, min(200, len(detailed_results)), 30)

        filtered = [r for r in detailed_results if r["prediction"]["tier"] in tier_filter][:n_rows]

        df_logs = pd.DataFrame([
            {
                "Text": r["prediction"]["text"][:120] + "..." if len(r["prediction"]["text"]) > 120 else r["prediction"]["text"],
                "Tier": f"Tier {r['prediction']['tier']}",
                "T1 Label": r["prediction"].get("t1_label", "N/A"),
                "Final Label": r["prediction"]["final_label"][0] if r["prediction"]["final_label"] else "N/A",
                "Ground Truth": r["ground_truth"],
                "Correct": "✅" if r["prediction"]["final_label"][0] == r["ground_truth"] else "❌",
                "Confidence": f"{r['prediction']['confidence']:.3f}",
                "Entropy": f"{r['prediction'].get('entropy', 0):.3f}",
                "Rationale": r["prediction"].get("rationale", "N/A")[:80],
            } for r in filtered
        ])
        st.dataframe(df_logs, use_container_width=True, hide_index=True)

        # ── Entropy vs Confidence scatter ──
        st.subheader("Entropy vs Confidence (Colored by Tier)")
        scatter_data = []
        for r in detailed_results:
            scatter_data.append({
                "Confidence": r["prediction"]["confidence"],
                "Entropy": r["prediction"].get("entropy", 0),
                "Tier": f"Tier {r['prediction']['tier']}",
                "Correct": r["prediction"]["final_label"][0] == r["ground_truth"],
            })
        df_scatter = pd.DataFrame(scatter_data)
        fig = px.scatter(df_scatter, x="Confidence", y="Entropy", color="Tier",
                         color_discrete_map={"Tier 1": COLORS["info"], "Tier 2": COLORS["success"], "Tier 3": COLORS["danger"]},
                         opacity=0.6, hover_data=["Correct"])
        fig.update_layout(**PLOT_LAYOUT, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the pipeline to generate detailed decision logs.")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.85rem; padding: 10px 0;">
    🛡️ Tri-Tiered Local LLM AL Framework — IEEE Conference Submission Dashboard<br/>
    Built with Streamlit • Plotly • scikit-learn • HuggingFace Transformers • Ollama
</div>
""", unsafe_allow_html=True)
