# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 500 | Official untouched test set |
 | **Tier 1 Accuracy** | 79.40% | Baseline (Encoder only) |
 | **Final System Accuracy** | 87.40% | Integrated performance |
 | **Accuracy Boost** | 8.00% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.8734 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.8736 | |
 | **Final ECE** | 0.0719 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0211 | Calibration error (Tier 1) |
 | **Brier Score** | 0.2509 | Lower is better |
 | **Human Effort Ratio** | 3.60% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 79.40% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 79.40% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 2.2222 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 2.1622 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0782 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 257 samples (51.4%)
 - **Tier 2 (Local LLM):** 225 samples (45.0%)
 - **Tier 3 (Simulated Human):** 18 samples (3.6%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 51.4% (FAILED)
 - ✅ **Human Cost Target (<=10%):** 3.6% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Not Supported (Tier 1 Coverage: 51.40%, Final Accuracy: 87.40%, Tier 1 Accuracy: 79.40%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 79.40%, Post-AL Tier 1 Accuracy: 79.40%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Pending — run with --sweep flag
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **neg** | 83.26% | 74.41% | 78.59% | 254 | 84.23% | 92.52% | 88.18% | 254 |
| **pos** | 76.19% | 84.55% | 80.15% | 246 | 91.40% | 82.11% | 86.51% | 246 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **neg** | 126 | 9 | 254 | 53.15% |
| **pos** | 99 | 9 | 246 | 43.90% |
 