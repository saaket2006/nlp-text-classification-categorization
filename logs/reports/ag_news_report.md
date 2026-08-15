# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 87.60% | Baseline (Encoder only) |
 | **Final System Accuracy** | 89.20% | Integrated performance |
 | **Accuracy Boost** | 1.60% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8931 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8919 | |
 | **Final ECE** | 0.0564 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0246 | Calibration error (Tier 1) |
 | **Brier Score** | 0.2045 | Lower is better |
 | **Human Effort Ratio** | 0.40% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 87.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 87.60% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 4.0000 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 3.4000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0158 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 414 samples (82.8%)
 - **Tier 2 (Local LLM):** 84 samples (16.8%)
 - **Tier 3 (Simulated Human):** 2 samples (0.4%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 82.8% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.4% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 82.80%, Final Accuracy: 89.20%, Tier 1 Accuracy: 87.60%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Supported (Pre-AL Tier 1 Accuracy: 87.40%, Post-AL Tier 1 Accuracy: 87.60%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **World** | 89.91% | 81.67% | 85.59% | 120 | 93.46% | 83.33% | 88.11% | 120 |
| **Sports** | 97.52% | 97.52% | 97.52% | 121 | 96.80% | 100.00% | 98.37% | 121 |
| **Business** | 83.21% | 85.07% | 84.13% | 134 | 86.47% | 85.82% | 86.14% | 134 |
| **Sci/Tech** | 81.20% | 86.40% | 83.72% | 125 | 81.48% | 88.00% | 84.62% | 125 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **World** | 21 | 1 | 120 | 18.33% |
| **Sports** | 3 | 1 | 121 | 3.31% |
| **Business** | 32 | 0 | 134 | 23.88% |
| **Sci/Tech** | 28 | 0 | 125 | 22.40% |
 