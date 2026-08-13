# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 1000 | Official untouched test set |
 | **Tier 1 Accuracy** | 96.40% | Baseline (Encoder only) |
 | **Final System Accuracy** | 97.80% | Integrated performance |
 | **Accuracy Boost** | 1.40% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.9790 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.9779 | |
 | **Final ECE** | 0.0156 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0144 | Calibration error (Tier 1) |
 | **Brier Score** | 0.0440 | Lower is better |
 | **Human Effort Ratio** | 0.30% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 97.60% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 96.40% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 4.6667 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 2.0000 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0139 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 926 samples (92.6%)
 - **Tier 2 (Local LLM):** 71 samples (7.1%)
 - **Tier 3 (Simulated Human):** 3 samples (0.3%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 92.6% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.3% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 92.60%, Final Accuracy: 97.80%, Tier 1 Accuracy: 96.40%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 97.60%, Post-AL Tier 1 Accuracy: 96.40%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **Company** | 98.73% | 88.64% | 93.41% | 88 | 98.77% | 90.91% | 94.67% | 88 |
| **EducationalInstitution** | 94.19% | 97.59% | 95.86% | 83 | 95.29% | 97.59% | 96.43% | 83 |
| **Artist** | 100.00% | 96.10% | 98.01% | 77 | 100.00% | 97.40% | 98.68% | 77 |
| **Athlete** | 96.97% | 100.00% | 98.46% | 64 | 98.46% | 100.00% | 99.22% | 64 |
| **OfficeHolder** | 97.56% | 98.77% | 98.16% | 81 | 97.56% | 98.77% | 98.16% | 81 |
| **MeanOfTransportation** | 95.45% | 98.44% | 96.92% | 64 | 95.45% | 98.44% | 96.92% | 64 |
| **Building** | 96.00% | 94.12% | 95.05% | 51 | 96.15% | 98.04% | 97.09% | 51 |
| **NaturalPlace** | 97.47% | 96.25% | 96.86% | 80 | 98.77% | 100.00% | 99.38% | 80 |
| **Village** | 92.31% | 100.00% | 96.00% | 48 | 100.00% | 100.00% | 100.00% | 48 |
| **Animal** | 98.57% | 100.00% | 99.28% | 69 | 100.00% | 100.00% | 100.00% | 69 |
| **Plant** | 100.00% | 98.72% | 99.35% | 78 | 100.00% | 100.00% | 100.00% | 78 |
| **Album** | 98.28% | 98.28% | 98.28% | 58 | 98.28% | 98.28% | 98.28% | 58 |
| **Film** | 97.06% | 86.84% | 91.67% | 76 | 96.00% | 94.74% | 95.36% | 76 |
| **WrittenWork** | 88.17% | 98.80% | 93.18% | 83 | 95.29% | 97.59% | 96.43% | 83 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **Company** | 15 | 2 | 88 | 19.32% |
| **EducationalInstitution** | 4 | 0 | 83 | 4.82% |
| **Artist** | 8 | 0 | 77 | 10.39% |
| **Athlete** | 0 | 0 | 64 | 0.00% |
| **OfficeHolder** | 4 | 1 | 81 | 6.17% |
| **MeanOfTransportation** | 0 | 0 | 64 | 0.00% |
| **Building** | 5 | 0 | 51 | 9.80% |
| **NaturalPlace** | 9 | 0 | 80 | 11.25% |
| **Village** | 0 | 0 | 48 | 0.00% |
| **Animal** | 1 | 0 | 69 | 1.45% |
| **Plant** | 2 | 0 | 78 | 2.56% |
| **Album** | 1 | 0 | 58 | 1.72% |
| **Film** | 18 | 0 | 76 | 23.68% |
| **WrittenWork** | 4 | 0 | 83 | 4.82% |
 