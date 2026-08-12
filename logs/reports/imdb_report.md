# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 79.40% | Baseline (Encoder only) |
 | **Final System Accuracy** | 85.80% | Integrated performance |
 | **Accuracy Boost** | 6.40% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8576 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8578 | |
 | **Final ECE** | 0.0802 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0290 | Calibration error (Tier 1) |
 | **Brier Score** | 0.1247 | Lower is better |
 | **Human Effort Ratio** | 0.00% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 79.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 79.40% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | N/A | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **AUTONOMOUS** | Efficiency classification |
 | **PICR-AL** | 64.0000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0640 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 302 samples (60.4%)
 - **Tier 2 (Local LLM):** 198 samples (39.6%)
 - **Tier 3 (Simulated Human):** 0 samples (0.0%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 60.4% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.0% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 60.40%, Final Accuracy: 85.80%, Tier 1 Accuracy: 79.40%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 79.40%, Post-AL Tier 1 Accuracy: 79.40%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **neg** | 82.97% | 74.80% | 78.67% | 254 | 83.76% | 89.37% | 86.48% | 254 |
| **pos** | 76.38% | 84.15% | 80.08% | 246 | 88.21% | 82.11% | 85.05% | 246 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **neg** | 105 | 0 | 254 | 41.34% |
| **pos** | 93 | 0 | 246 | 37.80% |
 