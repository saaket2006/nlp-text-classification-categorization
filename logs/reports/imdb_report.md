# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 79.80% | Baseline (Encoder only) |
 | **Final System Accuracy** | 84.80% | Integrated performance |
 | **Accuracy Boost** | 5.00% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8480 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8480 | |
 | **Final ECE** | 0.1192 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0981 | Calibration error (Tier 1) |
 | **Brier Score** | 0.2946 | Lower is better |
 | **Human Effort Ratio** | 1.80% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 79.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 79.80% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 2.7778 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 2.7368 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0491 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 304 samples (60.8%)
 - **Tier 2 (Local LLM):** 187 samples (37.4%)
 - **Tier 3 (Simulated Human):** 9 samples (1.8%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 60.8% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 1.8% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 60.80%, Final Accuracy: 84.80%, Tier 1 Accuracy: 79.80%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Supported (Pre-AL Tier 1 Accuracy: 79.40%, Post-AL Tier 1 Accuracy: 79.80%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **neg** | 88.83% | 68.90% | 77.61% | 254 | 86.18% | 83.46% | 84.80% | 254 |
| **pos** | 73.93% | 91.06% | 81.60% | 246 | 83.46% | 86.18% | 84.80% | 246 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **neg** | 113 | 7 | 254 | 47.24% |
| **pos** | 74 | 2 | 246 | 30.89% |
 