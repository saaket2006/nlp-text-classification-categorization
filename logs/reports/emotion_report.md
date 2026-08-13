# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 48.00% | Baseline (Encoder only) |
 | **Final System Accuracy** | 54.00% | Integrated performance |
 | **Accuracy Boost** | 6.00% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.2630 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.4811 | |
 | **Final ECE** | 0.3278 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.2492 | Calibration error (Tier 1) |
 | **Brier Score** | 0.7486 | Lower is better |
 | **Human Effort Ratio** | 0.00% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 58.80% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 48.00% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | N/A | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **AUTONOMOUS** | Efficiency classification |
 | **PICR-AL** | 6.0000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0600 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 188 samples (37.6%)
 - **Tier 2 (Local LLM):** 312 samples (62.4%)
 - **Tier 3 (Simulated Human):** 0 samples (0.0%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 37.6% (FAILED)
 - ✅ **Human Cost Target (<=10%):** 0.0% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Not Supported (Tier 1 Coverage: 37.60%, Final Accuracy: 54.00%, Tier 1 Accuracy: 48.00%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 58.80%, Post-AL Tier 1 Accuracy: 48.00%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **sadness** | 52.44% | 29.86% | 38.05% | 144 | 59.06% | 52.08% | 55.35% | 144 |
| **joy** | 46.62% | 96.37% | 62.84% | 193 | 53.35% | 90.67% | 67.18% | 193 |
| **love** | 0.00% | 0.00% | 0.00% | 30 | 12.50% | 3.33% | 5.26% | 30 |
| **anger** | 77.78% | 10.94% | 19.18% | 64 | 65.00% | 20.31% | 30.95% | 64 |
| **fear** | 44.44% | 7.41% | 12.70% | 54 | 45.45% | 9.26% | 15.38% | 54 |
| **surprise** | 0.00% | 0.00% | 0.00% | 15 | 20.00% | 6.67% | 10.00% | 15 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **sadness** | 123 | 0 | 144 | 85.42% |
| **joy** | 52 | 0 | 193 | 26.94% |
| **love** | 13 | 0 | 30 | 43.33% |
| **anger** | 62 | 0 | 64 | 96.88% |
| **fear** | 50 | 0 | 54 | 92.59% |
| **surprise** | 12 | 0 | 15 | 80.00% |
 