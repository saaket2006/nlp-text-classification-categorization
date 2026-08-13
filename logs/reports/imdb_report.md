# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 76.40% | Baseline (Encoder only) |
 | **Final System Accuracy** | 87.20% | Integrated performance |
 | **Accuracy Boost** | 10.80% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8704 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8706 | |
 | **Final ECE** | 0.0851 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0732 | Calibration error (Tier 1) |
 | **Brier Score** | 0.2382 | Lower is better |
 | **Human Effort Ratio** | 1.40% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 79.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 76.40% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 7.7143 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 6.2000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.1073 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 250 samples (50.0%)
 - **Tier 2 (Local LLM):** 243 samples (48.6%)
 - **Tier 3 (Simulated Human):** 7 samples (1.4%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 50.0% (FAILED)
 - ✅ **Human Cost Target (<=10%):** 1.4% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Not Supported (Tier 1 Coverage: 50.00%, Final Accuracy: 87.20%, Tier 1 Accuracy: 76.40%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 79.40%, Post-AL Tier 1 Accuracy: 76.40%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **neg** | 71.52% | 88.98% | 79.30% | 254 | 81.46% | 96.85% | 88.49% | 254 |
| **pos** | 84.78% | 63.41% | 72.56% | 246 | 95.96% | 77.24% | 85.59% | 246 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **neg** | 101 | 1 | 254 | 40.16% |
| **pos** | 142 | 6 | 246 | 60.16% |
 