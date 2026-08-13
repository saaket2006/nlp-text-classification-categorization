# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 54.00% | Baseline (Encoder only) |
 | **Final System Accuracy** | 58.00% | Integrated performance |
 | **Accuracy Boost** | 4.00% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.3787 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.5648 | |
 | **Final ECE** | 0.2555 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0741 | Calibration error (Tier 1) |
 | **Brier Score** | 0.7228 | Lower is better |
 | **Human Effort Ratio** | 6.60% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 58.80% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 54.00% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 0.6061 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **BELOW_TARGET** | Efficiency classification |
 | **PICR-AL** | 0.2388 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0367 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 34 samples (6.8%)
 - **Tier 2 (Local LLM):** 433 samples (86.6%)
 - **Tier 3 (Simulated Human):** 33 samples (6.6%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 6.8% (FAILED)
 - ✅ **Human Cost Target (<=10%):** 6.6% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Not Supported (Tier 1 Coverage: 6.80%, Final Accuracy: 58.00%, Tier 1 Accuracy: 54.00%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 58.80%, Post-AL Tier 1 Accuracy: 54.00%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **sadness** | 60.55% | 45.83% | 52.17% | 144 | 60.78% | 64.58% | 62.63% | 144 |
| **joy** | 56.08% | 86.01% | 67.89% | 193 | 63.95% | 77.20% | 69.95% | 193 |
| **love** | 20.00% | 13.33% | 16.00% | 30 | 24.24% | 26.67% | 25.40% | 30 |
| **anger** | 45.65% | 32.81% | 38.18% | 64 | 52.27% | 35.94% | 42.59% | 64 |
| **fear** | 46.43% | 24.07% | 31.71% | 54 | 50.00% | 24.07% | 32.50% | 54 |
| **surprise** | 0.00% | 0.00% | 0.00% | 15 | 40.00% | 26.67% | 32.00% | 15 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **sadness** | 132 | 11 | 144 | 99.31% |
| **joy** | 156 | 5 | 193 | 83.42% |
| **love** | 29 | 1 | 30 | 100.00% |
| **anger** | 58 | 6 | 64 | 100.00% |
| **fear** | 49 | 5 | 54 | 100.00% |
| **surprise** | 9 | 5 | 15 | 93.33% |
 