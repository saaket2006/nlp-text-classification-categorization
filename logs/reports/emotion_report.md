# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 58.80% | Baseline (Encoder only) |
 | **Final System Accuracy** | 59.80% | Integrated performance |
 | **Accuracy Boost** | 1.00% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.3433 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.5758 | |
 | **Final ECE** | 0.2330 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0357 | Calibration error (Tier 1) |
 | **Brier Score** | 0.2699 | Lower is better |
 | **Human Effort Ratio** | 0.00% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 58.80% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 58.80% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | N/A | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **AUTONOMOUS** | Efficiency classification |
 | **PICR-AL** | 10.0000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0100 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 60 samples (12.0%)
 - **Tier 2 (Local LLM):** 440 samples (88.0%)
 - **Tier 3 (Simulated Human):** 0 samples (0.0%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 12.0% (FAILED)
 - ✅ **Human Cost Target (<=10%):** 0.0% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Not Supported (Tier 1 Coverage: 12.00%, Final Accuracy: 59.80%, Tier 1 Accuracy: 58.80%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 58.80%, Post-AL Tier 1 Accuracy: 58.80%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **sadness** | 56.11% | 70.14% | 62.35% | 144 | 57.67% | 75.69% | 65.47% | 144 |
| **joy** | 65.16% | 82.38% | 72.77% | 193 | 70.48% | 76.68% | 73.45% | 193 |
| **love** | 16.67% | 3.33% | 5.56% | 30 | 31.82% | 23.33% | 26.92% | 30 |
| **anger** | 46.67% | 21.88% | 29.79% | 64 | 54.84% | 26.56% | 35.79% | 64 |
| **fear** | 47.50% | 35.19% | 40.43% | 54 | 46.15% | 33.33% | 38.71% | 54 |
| **surprise** | 0.00% | 0.00% | 0.00% | 15 | 0.00% | 0.00% | 0.00% | 15 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **sadness** | 140 | 0 | 144 | 97.22% |
| **joy** | 140 | 0 | 193 | 72.54% |
| **love** | 29 | 0 | 30 | 96.67% |
| **anger** | 64 | 0 | 64 | 100.00% |
| **fear** | 54 | 0 | 54 | 100.00% |
| **surprise** | 13 | 0 | 15 | 86.67% |
 