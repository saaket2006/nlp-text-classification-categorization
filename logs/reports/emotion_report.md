# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 59.60% | Baseline (Encoder only) |
 | **Final System Accuracy** | 62.60% | Integrated performance |
 | **Accuracy Boost** | 3.00% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.4303 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.5971 | |
 | **Final ECE** | 0.2355 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0401 | Calibration error (Tier 1) |
 | **Brier Score** | 0.6239 | Lower is better |
 | **Human Effort Ratio** | 0.40% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 58.80% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 59.60% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 7.5000 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 6.8000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0298 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 67 samples (13.4%)
 - **Tier 2 (Local LLM):** 431 samples (86.2%)
 - **Tier 3 (Simulated Human):** 2 samples (0.4%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 13.4% (FAILED)
 - ✅ **Human Cost Target (<=10%):** 0.4% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Not Supported (Tier 1 Coverage: 13.40%, Final Accuracy: 62.60%, Tier 1 Accuracy: 59.60%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Supported (Pre-AL Tier 1 Accuracy: 58.80%, Post-AL Tier 1 Accuracy: 59.60%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **sadness** | 54.46% | 76.39% | 63.58% | 144 | 59.09% | 81.25% | 68.42% | 144 |
| **joy** | 66.12% | 82.90% | 73.56% | 193 | 72.43% | 80.31% | 76.17% | 193 |
| **love** | 0.00% | 0.00% | 0.00% | 30 | 33.33% | 16.67% | 22.22% | 30 |
| **anger** | 63.16% | 18.75% | 28.92% | 64 | 59.26% | 25.00% | 35.16% | 64 |
| **fear** | 44.44% | 29.63% | 35.56% | 54 | 48.65% | 33.33% | 39.56% | 54 |
| **surprise** | 0.00% | 0.00% | 0.00% | 15 | 22.22% | 13.33% | 16.67% | 15 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **sadness** | 136 | 1 | 144 | 95.14% |
| **joy** | 138 | 0 | 193 | 71.50% |
| **love** | 28 | 0 | 30 | 93.33% |
| **anger** | 64 | 0 | 64 | 100.00% |
| **fear** | 52 | 1 | 54 | 98.15% |
| **surprise** | 13 | 0 | 15 | 86.67% |
 