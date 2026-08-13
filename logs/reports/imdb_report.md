# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 78.20% | Baseline (Encoder only) |
 | **Final System Accuracy** | 87.60% | Integrated performance |
 | **Accuracy Boost** | 9.40% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8749 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8751 | |
 | **Final ECE** | 0.0664 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0296 | Calibration error (Tier 1) |
 | **Brier Score** | 0.2333 | Lower is better |
 | **Human Effort Ratio** | 2.40% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 79.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 78.20% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 3.9167 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 3.5200 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0928 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 275 samples (55.0%)
 - **Tier 2 (Local LLM):** 213 samples (42.6%)
 - **Tier 3 (Simulated Human):** 12 samples (2.4%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 55.0% (FAILED)
 - ✅ **Human Cost Target (<=10%):** 2.4% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Not Supported (Tier 1 Coverage: 55.00%, Final Accuracy: 87.60%, Tier 1 Accuracy: 78.20%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 79.40%, Post-AL Tier 1 Accuracy: 78.20%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **neg** | 76.56% | 82.28% | 79.32% | 254 | 82.88% | 95.28% | 88.64% | 254 |
| **pos** | 80.18% | 73.98% | 76.96% | 246 | 94.23% | 79.67% | 86.34% | 246 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **neg** | 104 | 5 | 254 | 42.91% |
| **pos** | 109 | 7 | 246 | 47.15% |
 