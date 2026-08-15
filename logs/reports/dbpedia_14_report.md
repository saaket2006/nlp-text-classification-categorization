# 📊 Framework Report: Tri-Tiered Local LLM AL Framework (Zero Test Leakage Protocol)
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework under a strict, zero test-set contamination protocol.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Test Samples** | 1000 | Official untouched test set |
 | **Tier 1 Accuracy** | 97.90% | Baseline (Encoder only) |
 | **Final System Accuracy** | 98.30% | Integrated performance |
 | **Accuracy Boost** | 0.40% | Lift from Tier 2 & 3 |
 | **Macro F1 Score** | 0.9835 | Macro-averaged F1 |
 | **Weighted F1 Score** | 0.9830 | |
 | **Final ECE** | 0.0143 | Calibration error (Final system) |
 | **Tier 1 ECE** | 0.0085 | Calibration error (Tier 1) |
 | **Brier Score** | 0.0353 | Lower is better |
 | **Human Effort Ratio** | 0.20% | Samples requiring simulated human label |
 | **Pre-AL Tier 1 Accuracy** | 97.60% | Tier 1 baseline before AL loop (evaluated on test) |
 | **Post-AL Tier 1 Accuracy** | 97.90% | Tier 1 baseline after AL loop (evaluated on test) |
 | **PICR** | 2.0000 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 1.8333 | AL-aware cost-efficiency (λ=0.5) |
 | **Net Utility (U)** | 0.0039 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 
 ## 3. Tier Distribution & Load Balancing
 - **Tier 1 (Base Encoder):** 932 samples (93.2%)
 - **Tier 2 (Local LLM):** 66 samples (6.6%)
 - **Tier 3 (Simulated Human):** 2 samples (0.2%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 93.2% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.2% (PASSED)
 
 ## 5. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 93.20%, Final Accuracy: 98.30%, Tier 1 Accuracy: 97.90%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1 on untouched test set?**
   - **Verdict:** Supported (Pre-AL Tier 1 Accuracy: 97.60%, Post-AL Tier 1 Accuracy: 97.90%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported
 
 ## 6. Per-Category Performance Breakdown
 | Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
 | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | **Company** | 98.82% | 95.45% | 97.11% | 88 | 98.81% | 94.32% | 96.51% | 88 |
| **EducationalInstitution** | 95.29% | 97.59% | 96.43% | 83 | 95.29% | 97.59% | 96.43% | 83 |
| **Artist** | 100.00% | 97.40% | 98.68% | 77 | 100.00% | 97.40% | 98.68% | 77 |
| **Athlete** | 98.46% | 100.00% | 99.22% | 64 | 98.46% | 100.00% | 99.22% | 64 |
| **OfficeHolder** | 97.56% | 98.77% | 98.16% | 81 | 97.56% | 98.77% | 98.16% | 81 |
| **MeanOfTransportation** | 100.00% | 98.44% | 99.21% | 64 | 96.92% | 98.44% | 97.67% | 64 |
| **Building** | 94.12% | 94.12% | 94.12% | 51 | 96.08% | 96.08% | 96.08% | 51 |
| **NaturalPlace** | 96.39% | 100.00% | 98.16% | 80 | 98.77% | 100.00% | 99.38% | 80 |
| **Village** | 100.00% | 95.83% | 97.87% | 48 | 100.00% | 100.00% | 100.00% | 48 |
| **Animal** | 98.57% | 100.00% | 99.28% | 69 | 100.00% | 100.00% | 100.00% | 69 |
| **Plant** | 100.00% | 98.72% | 99.35% | 78 | 100.00% | 100.00% | 100.00% | 78 |
| **Album** | 100.00% | 98.28% | 99.13% | 58 | 100.00% | 98.28% | 99.13% | 58 |
| **Film** | 93.83% | 100.00% | 96.82% | 76 | 96.20% | 100.00% | 98.06% | 76 |
| **WrittenWork** | 98.75% | 95.18% | 96.93% | 83 | 98.77% | 96.39% | 97.56% | 83 |
 
 ## 7. Escalation Pattern Analysis
 | Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
 | :--- | :---: | :---: | :---: | :---: |
 | **Company** | 8 | 0 | 88 | 9.09% |
| **EducationalInstitution** | 7 | 0 | 83 | 8.43% |
| **Artist** | 12 | 0 | 77 | 15.58% |
| **Athlete** | 0 | 0 | 64 | 0.00% |
| **OfficeHolder** | 7 | 1 | 81 | 9.88% |
| **MeanOfTransportation** | 0 | 0 | 64 | 0.00% |
| **Building** | 8 | 0 | 51 | 15.69% |
| **NaturalPlace** | 4 | 0 | 80 | 5.00% |
| **Village** | 1 | 1 | 48 | 4.17% |
| **Animal** | 1 | 0 | 69 | 1.45% |
| **Plant** | 2 | 0 | 78 | 2.56% |
| **Album** | 0 | 0 | 58 | 0.00% |
| **Film** | 3 | 0 | 76 | 3.95% |
| **WrittenWork** | 13 | 0 | 83 | 15.66% |
 