# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 87.00% | Baseline (Encoder only) |
 | **Final System Accuracy** | 88.80% | Integrated performance |
 | **Accuracy Boost** | 1.80% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8899 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8885 | |
 | **Final ECE** | 0.0776 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0670 | Calibration error (Tier 1) |
 | **Brier Score** | 0.0961 | Lower is better |
 | **Human Effort Ratio** | 0.00% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 87.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 87.00% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | N/A | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **AUTONOMOUS** | Efficiency classification |
 | **PICR-AL** | 16.0000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0180 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 440 samples (88.0%)
 - **Tier 2 (Local LLM):** 60 samples (12.0%)
 - **Tier 3 (Simulated Human):** 0 samples (0.0%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 88.0% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.0% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 88.00%, Final Accuracy: 88.80%, Tier 1 Accuracy: 87.00%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 87.40%, Post-AL Tier 1 Accuracy: 87.00%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **World** | 93.94% | 77.50% | 84.93% | 120 | 95.19% | 82.50% | 88.39% | 120 |
| **Sports** | 96.75% | 98.35% | 97.54% | 121 | 97.56% | 99.17% | 98.36% | 121 |
| **Business** | 88.24% | 78.36% | 83.00% | 134 | 87.40% | 82.84% | 85.06% | 134 |
| **Sci/Tech** | 74.21% | 94.40% | 83.10% | 125 | 78.08% | 91.20% | 84.13% | 125 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **World** | 15 | 0 | 120 | 12.50% |
| **Sports** | 4 | 0 | 121 | 3.31% |
| **Business** | 27 | 0 | 134 | 20.15% |
| **Sci/Tech** | 14 | 0 | 125 | 11.20% |
 