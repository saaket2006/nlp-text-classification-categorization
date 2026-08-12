# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 86.80% | Baseline (Encoder only) |
 | **Final System Accuracy** | 89.00% | Integrated performance |
 | **Accuracy Boost** | 2.20% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8913 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8902 | |
 | **Final ECE** | 0.0764 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0554 | Calibration error (Tier 1) |
 | **Brier Score** | 0.2066 | Lower is better |
 | **Human Effort Ratio** | 0.60% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 87.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 86.80% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 3.6667 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 2.7143 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0217 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 429 samples (85.8%)
 - **Tier 2 (Local LLM):** 68 samples (13.6%)
 - **Tier 3 (Simulated Human):** 3 samples (0.6%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 85.8% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.6% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 85.80%, Final Accuracy: 89.00%, Tier 1 Accuracy: 86.80%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 87.40%, Post-AL Tier 1 Accuracy: 86.80%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **World** | 94.79% | 75.83% | 84.26% | 120 | 95.10% | 80.83% | 87.39% | 120 |
| **Sports** | 96.75% | 98.35% | 97.54% | 121 | 97.56% | 99.17% | 98.36% | 121 |
| **Business** | 83.58% | 83.58% | 83.58% | 134 | 85.40% | 87.31% | 86.35% | 134 |
| **Sci/Tech** | 76.19% | 89.60% | 82.35% | 125 | 80.43% | 88.80% | 84.41% | 125 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **World** | 16 | 3 | 120 | 15.83% |
| **Sports** | 5 | 0 | 121 | 4.13% |
| **Business** | 24 | 0 | 134 | 17.91% |
| **Sci/Tech** | 23 | 0 | 125 | 18.40% |
 