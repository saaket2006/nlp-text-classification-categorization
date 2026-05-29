# 📊 Framework Report: Tri-Tiered Local LLM AL Framework
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework. The system successfully routed samples through three levels of complexity, optimizing for both accuracy and human effort.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Samples** | 1000 | Test set size |
 | **Tier 1 Accuracy** | 97.50% | Baseline (Encoder only) |
 | **Final System Accuracy** | 98.40% | Integrated performance |
 | **Accuracy Boost** | 0.90% | Lift from Tier 2 & 3 |
 | **Weighted F1 Score** | 0.9840 | |
 | **ECE** | 0.0123 | Calibration error — lower is better |
 | **Human Effort Ratio** | 0.10% | Samples requiring human label |
 | **Pre-AL Tier 1 Accuracy** | 97.60% | Tier 1 baseline before AL loop |
 | **Post-AL Tier 1 Accuracy** | 97.70% | Tier 1 baseline after AL loop |
 | **PICR** | 9.0000 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 4.7500 | AL-aware cost-efficiency (λ=0.5) |
 | **PICR-AL Status** | **STRONG** | Rewards AL loop progression |
 | **Net Utility (U)** | 0.0090 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 | **Tier 2 Autonomous Gain** | 0.90% | Accuracy lift from LLM with zero human cost |
 
 ## 3. Tier Distribution & Load Balancing
 The framework aims to maximize Tier 1 usage while minimizing Tier 3 escalation.
 
 - **Tier 1 (Base Encoder):** 962 samples (96.2%)
 - **Tier 2 (Local LLM):** 37 samples (3.7%)
 - **Tier 3 (Human Expert):** 1 samples (0.1%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 96.2% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.1% (PASSED)
 
 ## 5. Conclusion
 The system demonstrated a **0.90% accuracy improvement** with only **0.1% human intervention**, confirming the effectiveness of the tiered routing strategy.
 
 ## 6. PICR Interpretation
 The Point-Improvement-per-Cost-Ratio (PICR) measures the efficiency of human intervention.
 **Formula:** `ΔAccuracy / Human Effort Ratio`
 **Current PICR:** `9.0000` (STRONG) Note: PICR is based on only 1 Tier 3 sample(s). Run with --seeds for statistically robust PICR estimates.
 
 **Interpretation:**
 A PICR below 1.0 indicates the human effort ratio exceeded the accuracy gain — adjust τ₁ (entropy) upward or τ₂ (confidence) downward to reduce unnecessary escalation and improve cost-efficiency. The threshold sweep identifies the optimal (τ₁, τ₂) operating point for maximum system utility.
 
 ## 6b. PICR-AL Interpretation
 **Formula:** (ΔAccuracy + λ · ΔAcc_AL) / (Human Effort Ratio + ε)
 **λ:** 0.5 | **ε:** 0.001 | **ΔAcc_AL:** 0.0010
 **PICR-AL:** 4.7500 (STRONG)
 
 PICR-AL extends PICR by incorporating active learning progression into the numerator.
 A configuration that escalates only 1 sample scores lower under PICR-AL than under PICR
 if that sample fails to trigger a retraining batch — exposing the metric gaming behaviour
 of near-zero human escalation strategies.
 
 ## 6c. Net Utility Interpretation
 **Formula:** ΔAccuracy − (λ_cost × Human Effort Ratio)
 **λ_cost:** 0.05 | **Net Utility:** 0.0090 (POSITIVE)
 
 Net Utility is additive and cannot be gamed by minimising human escalation. A POSITIVE
 result confirms the system adds measurable value after accounting for annotation cost.
 A NEGATIVE result means the human effort cost exceeded the accuracy gain at this
 operating point — reduce escalation thresholds or increase Tier 2 autonomy.
 
 ## 7. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 96.20%, Final Accuracy: 98.40%, Tier 1 Accuracy: 97.50%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1?**
   - **Verdict:** Supported (Pre-AL Tier 1 Accuracy: 97.60%, Post-AL Tier 1 Accuracy: 97.70%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Pending — run with --sweep flag

## 8. Per-Category Performance Breakdown
| Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Company** | 98.80% | 93.18% | 95.91% | 88 | 98.82% | 95.45% | 97.11% | 88 |
| **EducationalInstitution** | 93.10% | 97.59% | 95.29% | 83 | 96.43% | 97.59% | 97.01% | 83 |
| **Artist** | 98.67% | 96.10% | 97.37% | 77 | 98.68% | 97.40% | 98.04% | 77 |
| **Athlete** | 98.46% | 100.00% | 99.22% | 64 | 98.46% | 100.00% | 99.22% | 64 |
| **OfficeHolder** | 96.30% | 96.30% | 96.30% | 81 | 97.53% | 97.53% | 97.53% | 81 |
| **MeanOfTransportation** | 100.00% | 98.44% | 99.21% | 64 | 100.00% | 98.44% | 99.21% | 64 |
| **Building** | 94.23% | 96.08% | 95.15% | 51 | 96.08% | 96.08% | 96.08% | 51 |
| **NaturalPlace** | 96.39% | 100.00% | 98.16% | 80 | 97.56% | 100.00% | 98.77% | 80 |
| **Village** | 100.00% | 95.83% | 97.87% | 48 | 100.00% | 100.00% | 100.00% | 48 |
| **Animal** | 98.57% | 100.00% | 99.28% | 69 | 100.00% | 100.00% | 100.00% | 69 |
| **Plant** | 100.00% | 98.72% | 99.35% | 78 | 100.00% | 100.00% | 100.00% | 78 |
| **Album** | 100.00% | 98.28% | 99.13% | 58 | 100.00% | 98.28% | 99.13% | 58 |
| **Film** | 94.94% | 98.68% | 96.77% | 76 | 96.20% | 100.00% | 98.06% | 76 |
| **WrittenWork** | 97.56% | 96.39% | 96.97% | 83 | 98.78% | 97.59% | 98.18% | 83 |

## 9. Escalation Pattern Analysis
| Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Company** | 6 | 0 | 88 | 6.82% |
| **EducationalInstitution** | 4 | 0 | 83 | 4.82% |
| **Artist** | 6 | 0 | 77 | 7.79% |
| **Athlete** | 0 | 0 | 64 | 0.00% |
| **OfficeHolder** | 5 | 1 | 81 | 7.41% |
| **MeanOfTransportation** | 0 | 0 | 64 | 0.00% |
| **Building** | 1 | 0 | 51 | 1.96% |
| **NaturalPlace** | 2 | 0 | 80 | 2.50% |
| **Village** | 2 | 0 | 48 | 4.17% |
| **Animal** | 0 | 0 | 69 | 0.00% |
| **Plant** | 2 | 0 | 78 | 2.56% |
| **Album** | 0 | 0 | 58 | 0.00% |
| **Film** | 3 | 0 | 76 | 3.95% |
| **WrittenWork** | 6 | 0 | 83 | 7.23% |
