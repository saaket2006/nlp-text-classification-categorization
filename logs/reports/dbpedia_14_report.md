# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 1000 | Official untouched test set |
 | **Tier 1 Accuracy** | 97.30% | Baseline (Encoder only) |
 | **Final System Accuracy** | 98.20% | Integrated performance |
 | **Accuracy Boost** | 0.90% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.9828 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.9819 | |
 | **Final ECE** | 0.0154 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0159 | Calibration error (Tier 1) |
 | **Brier Score** | 0.0387 | Lower is better |
 | **Human Effort Ratio** | 0.20% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 97.60% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 97.30% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 4.5000 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 2.5000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0089 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 954 samples (95.4%)
 - **Tier 2 (Local LLM):** 44 samples (4.4%)
 - **Tier 3 (Simulated Human):** 2 samples (0.2%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 95.4% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.2% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 95.40%, Final Accuracy: 98.20%, Tier 1 Accuracy: 97.30%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 97.60%, Post-AL Tier 1 Accuracy: 97.30%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **Company** | 98.75% | 89.77% | 94.05% | 88 | 98.77% | 90.91% | 94.67% | 88 |
| **EducationalInstitution** | 94.19% | 97.59% | 95.86% | 83 | 95.29% | 97.59% | 96.43% | 83 |
| **Artist** | 100.00% | 96.10% | 98.01% | 77 | 100.00% | 97.40% | 98.68% | 77 |
| **Athlete** | 98.46% | 100.00% | 99.22% | 64 | 98.46% | 100.00% | 99.22% | 64 |
| **OfficeHolder** | 96.34% | 97.53% | 96.93% | 81 | 97.56% | 98.77% | 98.16% | 81 |
| **MeanOfTransportation** | 95.45% | 98.44% | 96.92% | 64 | 95.45% | 98.44% | 96.92% | 64 |
| **Building** | 94.23% | 96.08% | 95.15% | 51 | 96.15% | 98.04% | 97.09% | 51 |
| **NaturalPlace** | 96.39% | 100.00% | 98.16% | 80 | 98.77% | 100.00% | 99.38% | 80 |
| **Village** | 100.00% | 95.83% | 97.87% | 48 | 100.00% | 100.00% | 100.00% | 48 |
| **Animal** | 98.57% | 100.00% | 99.28% | 69 | 100.00% | 100.00% | 100.00% | 69 |
| **Plant** | 98.72% | 98.72% | 98.72% | 78 | 100.00% | 100.00% | 100.00% | 78 |
| **Album** | 100.00% | 98.28% | 99.13% | 58 | 100.00% | 98.28% | 99.13% | 58 |
| **Film** | 94.94% | 98.68% | 96.77% | 76 | 96.20% | 100.00% | 98.06% | 76 |
| **WrittenWork** | 97.56% | 96.39% | 96.97% | 83 | 98.78% | 97.59% | 98.18% | 83 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **Company** | 14 | 0 | 88 | 15.91% |
| **EducationalInstitution** | 5 | 0 | 83 | 6.02% |
| **Artist** | 5 | 0 | 77 | 6.49% |
| **Athlete** | 0 | 0 | 64 | 0.00% |
| **OfficeHolder** | 4 | 1 | 81 | 6.17% |
| **MeanOfTransportation** | 0 | 0 | 64 | 0.00% |
| **Building** | 3 | 0 | 51 | 5.88% |
| **NaturalPlace** | 2 | 0 | 80 | 2.50% |
| **Village** | 1 | 1 | 48 | 4.17% |
| **Animal** | 1 | 0 | 69 | 1.45% |
| **Plant** | 2 | 0 | 78 | 2.56% |
| **Album** | 0 | 0 | 58 | 0.00% |
| **Film** | 3 | 0 | 76 | 3.95% |
| **WrittenWork** | 4 | 0 | 83 | 4.82% |
 