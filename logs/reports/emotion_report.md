# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 60.20% | Baseline (Encoder only) |
 | **Final System Accuracy** | 62.80% | Integrated performance |
 | **Accuracy Boost** | 2.60% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.3994 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.6114 | |
 | **Final ECE** | 0.3067 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0311 | Calibration error (Tier 1) |
 | **Brier Score** | 0.7066 | Lower is better |
 | **Human Effort Ratio** | 1.00% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 58.80% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 60.20% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 2.6000 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 3.0000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0255 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 3 samples (0.6%)
 - **Tier 2 (Local LLM):** 492 samples (98.4%)
 - **Tier 3 (Simulated Human):** 5 samples (1.0%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 0.6% (FAILED)
 - ✅ **Human Cost Target (<=10%):** 1.0% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Not Supported (Tier 1 Coverage: 0.60%, Final Accuracy: 62.80%, Tier 1 Accuracy: 60.20%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Supported (Pre-AL Tier 1 Accuracy: 58.80%, Post-AL Tier 1 Accuracy: 60.20%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **sadness** | 53.81% | 78.47% | 63.84% | 144 | 56.48% | 84.72% | 67.78% | 144 |
| **joy** | 70.09% | 81.35% | 75.30% | 193 | 78.38% | 75.13% | 76.72% | 193 |
| **love** | 0.00% | 0.00% | 0.00% | 30 | 33.33% | 30.00% | 31.58% | 30 |
| **anger** | 56.00% | 21.88% | 31.46% | 64 | 67.86% | 29.69% | 41.30% | 64 |
| **fear** | 48.57% | 31.48% | 38.20% | 54 | 60.00% | 27.78% | 37.97% | 54 |
| **surprise** | 0.00% | 0.00% | 0.00% | 15 | 22.22% | 26.67% | 24.24% | 15 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **sadness** | 141 | 3 | 144 | 100.00% |
| **joy** | 190 | 0 | 193 | 98.45% |
| **love** | 30 | 0 | 30 | 100.00% |
| **anger** | 63 | 1 | 64 | 100.00% |
| **fear** | 54 | 0 | 54 | 100.00% |
| **surprise** | 14 | 1 | 15 | 100.00% |
 