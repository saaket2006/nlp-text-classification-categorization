# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 86.20% | Baseline (Encoder only) |
 | **Final System Accuracy** | 86.80% | Integrated performance |
 | **Accuracy Boost** | 0.60% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8705 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8689 | |
 | **Final ECE** | 0.0778 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0426 | Calibration error (Tier 1) |
 | **Brier Score** | 0.2255 | Lower is better |
 | **Human Effort Ratio** | 0.00% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 87.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 86.20% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | N/A | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **AUTONOMOUS** | Efficiency classification |
 | **PICR-AL** | 0.0000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0060 | ΔAcc − (λ × HumanEffort), λ=0.05 |
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
   - **Verdict:** Supported (Tier 1 Coverage: 88.00%, Final Accuracy: 86.80%, Tier 1 Accuracy: 86.20%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 87.40%, Post-AL Tier 1 Accuracy: 86.20%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **World** | 89.09% | 81.67% | 85.22% | 120 | 90.83% | 82.50% | 86.46% | 120 |
| **Sports** | 99.15% | 96.69% | 97.91% | 121 | 98.31% | 95.87% | 97.07% | 121 |
| **Business** | 83.08% | 80.60% | 81.82% | 134 | 84.38% | 80.60% | 82.44% | 134 |
| **Sci/Tech** | 76.06% | 86.40% | 80.90% | 125 | 76.55% | 88.80% | 82.22% | 125 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **World** | 17 | 0 | 120 | 14.17% |
| **Sports** | 6 | 0 | 121 | 4.96% |
| **Business** | 18 | 0 | 134 | 13.43% |
| **Sci/Tech** | 19 | 0 | 125 | 15.20% |
 