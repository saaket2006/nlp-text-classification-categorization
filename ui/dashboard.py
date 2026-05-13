import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add the project root to sys.path to allow importing from core, models, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration import reliability_diagram_data

st.set_page_config(
    page_title="Tri-Tiered LLM AL Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stMetric {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3d4156;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stSubheader { color: #60a5fa; font-weight: 600; margin-top: 2rem; }
    .stMarkdown h3 { color: #94a3b8; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Tri-Tiered Local LLM Framework")
st.markdown("### Research Dashboard: Active Learning & Automated Categorization")

# Sidebar
st.sidebar.header("Configuration")
logs_dir = "./logs"
metrics_file = os.path.join(logs_dir, "metrics_summary.json")
results_file = os.path.join(logs_dir, "detailed_results.json")

if not os.path.exists(metrics_file):
    st.error("No metrics found. Please run main.py first.")
else:
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    with open(results_file, "r") as f:
        detailed_results = json.load(f)

    # Top Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Final Accuracy", f"{metrics['accuracy_final']:.2%}")
    col2.metric("Human Effort", f"{metrics['human_effort_ratio']:.2%}")
    
    # PICR Status Delta Logic
    picr_status = metrics.get('picr_status', 'NO_GAIN')
    if picr_status in ["PERFECT_EFFICIENCY", "STRONG"]:
        status_delta = "High Efficiency"
        delta_color = "normal"
    elif picr_status == "ACCEPTABLE":
        status_delta = None
        delta_color = "off"
    else:
        status_delta = "-Low Efficiency"
        delta_color = "normal"

    col3.metric("PICR Score", metrics.get('picr_display', str(metrics.get('picr', 0.0))))
    col4.metric("PICR Status", picr_status, delta=status_delta, delta_color=delta_color)
    col5.metric("F1 Score", f"{metrics['f1_weighted']:.4f}")

    # Conditional Banners
    if metrics.get("picr_negative_warning"):
        st.error("⚠️ Negative PICR: pipeline accuracy is below Tier 1 baseline. Review threshold configuration.")
    if picr_status == "PERFECT_EFFICIENCY":
        st.info("✨ Perfect Efficiency: accuracy improved with zero human intervention.")

    # Layout - Row 1: Distribution & Flow
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Routing Distribution")
        dist = metrics["tier_distribution"]
        t1_count = dist.get("1", dist.get(1, 0))
        t2_count = dist.get("2", dist.get(2, 0))
        t3_count = dist.get("3", dist.get(3, 0))
        df_dist = pd.DataFrame({
            "Tier": ["Tier 1 (Encoder)", "Tier 2 (LLM)", "Tier 3 (Human)"],
            "Count": [t1_count, t2_count, t3_count]
        })
        fig = px.pie(df_dist, values='Count', names='Tier', hole=0.5, 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        st.subheader("Sample Flow (Sankey)")
        total = metrics["total_samples"]
        # Nodes: Input (0), T1 (1), T2 (2), T3 (3), Final (4)
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15, thickness = 20, line = dict(color = "black", width = 0.5),
              label = ["Input", "Tier 1", "Tier 2", "Tier 3", "Final Output"],
              color = ["#94a3b8", "#60a5fa", "#34d399", "#f87171", "#a78bfa"]
            ),
            link = dict(
              source = [0, 1, 1, 1, 2, 2, 3],
              target = [1, 4, 2, 3, 4, 3, 4],
              value = [total, t1_count, t2_count, 0, t2_count, 0, t3_count]
            ))])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: ROI & Performance Frontier
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Accuracy Waterfall")
        # EXACT CALCULATION from detailed_results
        total = metrics["total_samples"]
        t1_correct = sum(1 for r in detailed_results if r["prediction"]["t1_label"] == r["ground_truth"])
        
        # Samples where T1 was WRONG but T2/T3 corrected it
        t2_corrections = 0
        t3_corrections = 0
        
        for r in detailed_results:
            t1_wrong = r["prediction"]["t1_label"] != r["ground_truth"]
            if t1_wrong:
                final_correct = r["prediction"]["final_label"][0] == r["ground_truth"]
                if final_correct:
                    if r["prediction"]["tier"] == 2:
                        t2_corrections += 1
                    elif r["prediction"]["tier"] == 3:
                        t3_corrections += 1
        
        acc_t1 = t1_correct / total
        llm_gain = t2_corrections / total
        human_gain = t3_corrections / total
        acc_final = metrics["accuracy_final"]
        
        fig = go.Figure(go.Waterfall(
            name = "Accuracy", orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["T1 Base", "T2 (LLM) Gain", "T3 (Human) Gain", "Final Accuracy"],
            textposition = "outside",
            text = [f"{acc_t1:.1%}", f"+{llm_gain:.1%}", f"+{human_gain:.1%}", f"{acc_final:.1%}"],
            y = [acc_t1, llm_gain, human_gain, acc_final],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            increasing = {"marker":{"color":"#34d399"}},
            decreasing = {"marker":{"color":"#f87171"}},
            totals = {"marker":{"color":"#60a5fa"}}
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with row2_col2:
        st.subheader("Efficiency Frontier")
        sweep_file = os.path.join(logs_dir, "threshold_sweep.json")
        history_file = os.path.join(logs_dir, "history.json")
        fig = go.Figure()
        
        # 1. Plot Sweep Points (if available)
        if os.path.exists(sweep_file):
            with open(sweep_file, "r") as f:
                sweep_data = json.load(f)
            effort = [pt['human_effort_ratio'] for pt in sweep_data]
            acc = [pt['accuracy'] for pt in sweep_data]
            fig.add_trace(go.Scatter(x=effort, y=acc, mode='markers', name='Sweep Points', marker=dict(size=8, color='#60a5fa', opacity=0.4)))
        
        # 2. Plot Historical Trials
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                hist_data = json.load(f)
            h_effort = [pt['human_effort_ratio'] for pt in hist_data]
            h_acc = [pt['accuracy'] for pt in hist_data]
            fig.add_trace(go.Scatter(x=h_effort, y=h_acc, mode='markers', name='Trial History', marker=dict(size=6, color='#a78bfa', opacity=0.8, symbol='circle')))

        # 3. Plot Theoretical Frontier (Simulated)
        x_range = np.linspace(0, 0.3, 20)
        y_base = metrics['accuracy_t1']
        y_theoretical = y_base + (1.0 - y_base) * (1 - np.exp(-10 * x_range))
        fig.add_trace(go.Scatter(x=x_range, y=y_theoretical, mode='lines', name='Theoretical Limit', line=dict(dash='dash', color='#4b5563')))
            
        # 4. Plot Current Config
        fig.add_trace(go.Scatter(x=[metrics['human_effort_ratio']], y=[metrics['accuracy_final']], mode='markers', marker=dict(size=15, color='#f87171', symbol='star', line=dict(width=2, color='white')), name='Current Trial'))
        
        fig.update_layout(xaxis_title="Human Effort (Ratio)", yaxis_title="Accuracy", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: ROI & Calibration
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        st.subheader("PICR ROI Heatmap")
        if os.path.exists(sweep_file):
            with open(sweep_file, "r") as f:
                sweep_data = json.load(f)
            df_sweep = pd.DataFrame(sweep_data)
            fig = px.density_heatmap(df_sweep, x="human_effort_ratio", y="accuracy", z="picr", 
                                     color_continuous_scale='Viridis', labels={'picr':'PICR Score'})
        else:
            st.info("Run a threshold sweep to see the PICR ROI Heatmap.")
            # Scale placeholder to reach current PICR (e.g. 3.0)
            max_picr = max(1.0, metrics.get('picr', 1.0))
            z = np.random.rand(10, 10) * max_picr
            fig = px.imshow(z, labels=dict(x="tau_1", y="tau_2", color="PICR"), 
                            x=np.linspace(0.3, 0.9, 10), y=np.linspace(0.4, 0.7, 10),
                            color_continuous_scale='Viridis')
        
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with row3_col2:
        st.subheader("Reliability Diagram")
        confidences = np.array([r["prediction"]["confidence"] for r in detailed_results])
        predictions = np.array([r["prediction"]["predicted_labels"][0] for r in detailed_results])
        labels = np.array([r["ground_truth"] for r in detailed_results])
        bin_lowers, bin_accs, bin_confs = reliability_diagram_data(confidences, predictions, labels)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_lowers, y=bin_accs, name="Accuracy", marker_color='#34d399', offsetgroup=0))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash='dash', color='#f87171'), name="Perfect Calibration"))
        fig.update_layout(xaxis_title="Confidence", yaxis_title="Accuracy", barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    # row 3
    st.subheader("Sample Decision Logs")
    df_logs = pd.DataFrame([
        {
            "Text": r["prediction"]["text"][:100] + "...",
            "Tier": f"Tier {r['prediction']['tier']}",
            "Predicted": ", ".join(r["prediction"]["predicted_labels"]),
            "GT": r["ground_truth"],
            "Confidence": f"{r['prediction']['confidence']:.2f}",
            "Entropy": f"{r['prediction'].get('entropy', 0):.2f}"
        } for r in detailed_results[:20]
    ])
    st.dataframe(df_logs, use_container_width=True)

    st.markdown("---")
    st.info("The framework target: ≥60% handled by Tier 1 and ≤30% escalation to Tier 3.")
    t1_ratio = t1_count / metrics["total_samples"]
    t3_ratio = t3_count / metrics["total_samples"]
    
    col1, col2 = st.columns(2)
    if t1_ratio >= 0.6:
        col1.success(f"Tier 1 Target Met: {t1_ratio:.1%}")
    else:
        col1.warning(f"Tier 1 Target Below: {t1_ratio:.1%}")
        
    if t3_ratio <= 0.3:
        col2.success(f"Tier 3 Target Met: {t3_ratio:.1%}")
    else:
        col2.warning(f"Tier 3 Target Exceeded: {t3_ratio:.1%}")
