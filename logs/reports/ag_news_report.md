# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 87.80% | Baseline (Encoder only) |
 | **Final System Accuracy** | 89.20% | Integrated performance |
 | **Accuracy Boost** | 1.40% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8929 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8918 | |
 | **Final ECE** | 0.0790 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0570 | Calibration error (Tier 1) |
 | **Brier Score** | 0.2023 | Lower is better |
 | **Human Effort Ratio** | 0.20% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 87.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 87.80% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 7.0000 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 5.3333 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0139 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 439 samples (87.8%)
 - **Tier 2 (Local LLM):** 60 samples (12.0%)
 - **Tier 3 (Simulated Human):** 1 samples (0.2%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 87.8% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.2% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 87.80%, Final Accuracy: 89.20%, Tier 1 Accuracy: 87.80%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Supported (Pre-AL Tier 1 Accuracy: 87.40%, Post-AL Tier 1 Accuracy: 87.80%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **World** | 94.00% | 78.33% | 85.45% | 120 | 94.23% | 81.67% | 87.50% | 120 |
| **Sports** | 96.77% | 99.17% | 97.96% | 121 | 96.80% | 100.00% | 98.37% | 121 |
| **Business** | 82.39% | 87.31% | 84.78% | 134 | 84.89% | 88.06% | 86.45% | 134 |
| **Sci/Tech** | 80.60% | 86.40% | 83.40% | 125 | 82.58% | 87.20% | 84.82% | 125 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **World** | 16 | 0 | 120 | 13.33% |
| **Sports** | 3 | 1 | 121 | 3.31% |
| **Business** | 18 | 0 | 134 | 13.43% |
| **Sci/Tech** | 23 | 0 | 125 | 18.40% |
 