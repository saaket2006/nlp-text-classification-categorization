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

st.set_page_config(page_title="Tri-Tiered LLM AL Dashboard", layout="wide")

st.title("🛡️ A Tri-Tiered Local LLM Framework for Active Learning")
st.markdown("### Cost-Efficient Text Classification & Automated Categorization")

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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Accuracy", f"{metrics['accuracy_final']:.2%}")
    col2.metric("Human Effort", f"{metrics['human_effort_ratio']:.2%}")
    col3.metric("PICR", f"{metrics['picr']:.2f}")
    col4.metric("F1 Score", f"{metrics['f1_weighted']:.4f}")

    # Layout
    row1_col1, row1_col2 = st.columns(2)

    # 1. Routing Distribution (Donut Chart)
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
        st.plotly_chart(fig, use_container_width=True)

    # 2. Efficiency Frontier
    with row1_col2:
        st.subheader("Efficiency Frontier")
        sweep_file = os.path.join(logs_dir, "threshold_sweep.json")
        fig = go.Figure()
        
        if os.path.exists(sweep_file):
            with open(sweep_file, "r") as f:
                sweep_data = json.load(f)
            
            effort = [pt['human_effort_ratio'] for pt in sweep_data]
            acc = [pt['accuracy'] for pt in sweep_data]
            
            fig.add_trace(go.Scatter(
                x=effort, 
                y=acc, 
                mode='markers', 
                name='Sweep Points',
                marker=dict(size=8, color='blue', opacity=0.6)
            ))
            
        fig.add_trace(go.Scatter(
            x=[metrics['human_effort_ratio']], 
            y=[metrics['accuracy_final']], 
            mode='markers', 
            marker=dict(size=15, color='red', symbol='star'), 
            name='Current Config'
        ))
        
        fig.update_layout(xaxis_title="Human Effort (Ratio)", yaxis_title="Accuracy")
        st.plotly_chart(fig, use_container_width=True)

    # row 2
    row2_col1, row2_col2 = st.columns(2)

    # 3. Reliability Diagram
    with row2_col1:
        st.subheader("Reliability Diagram (Calibration)")
        confidences = np.array([r["prediction"]["confidence"] for r in detailed_results])
        predictions = np.array([r["prediction"]["predicted_labels"][0] for r in detailed_results])
        labels = np.array([r["ground_truth"] for r in detailed_results])
        
        bin_lowers, bin_accs, bin_confs = reliability_diagram_data(confidences, predictions, labels)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_lowers, y=bin_accs, name="Accuracy", offsetgroup=0))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash='dash'), name="Perfectly Calibrated"))
        fig.update_layout(xaxis_title="Confidence", yaxis_title="Accuracy", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    # 4. Detailed Decision Log
    with row2_col2:
        st.subheader("Sample Decision Logs")
        df_logs = pd.DataFrame([
            {
                "Text": r["prediction"]["text"][:100] + "...",
                "Tier": r["prediction"]["tier"],
                "Predicted": r["prediction"]["predicted_labels"],
                "GT": r["ground_truth"],
                "Confidence": r["prediction"]["confidence"]
            } for r in detailed_results[:10]
        ])
        st.table(df_logs)

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
