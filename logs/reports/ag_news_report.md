# 📊 Framework Report: Tri-Tiered Local LLM AL Framework
 
 ## 1. Executive Summary
 This report summarizes the performance of the Tri-Tiered Active Learning framework. The system successfully routed samples through three levels of complexity, optimizing for both accuracy and human effort.
 
 ## 2. Core Performance Metrics
 | Metric | Value | Note |
 | :--- | :--- | :--- |
 | **Total Samples** | 500 | Test set size |
 | **Tier 1 Accuracy** | 87.20% | Baseline (Encoder only) |
 | **Final System Accuracy** | 88.80% | Integrated performance |
 | **Accuracy Boost** | 1.60% | Lift from Tier 2 & 3 |
 | **Weighted F1 Score** | 0.8884 | |
 | **ECE** | 0.0457 | Calibration error — lower is better |
 | **Human Effort Ratio** | 0.20% | Samples requiring human label |
 | **Pre-AL Tier 1 Accuracy** | 87.40% | Tier 1 baseline before AL loop |
 | **Post-AL Tier 1 Accuracy** | 87.40% | Tier 1 baseline after AL loop |
 | **PICR** | 8.0000 | Point-Improvement-per-Cost-Ratio |
 | **PICR Status** | **STRONG** | Efficiency classification |
 | **PICR-AL** | 5.3333 | AL-aware cost-efficiency (λ=0.5) |
 | **PICR-AL Status** | **STRONG** | Rewards AL loop progression |
 | **Net Utility (U)** | 0.0159 | ΔAcc − (λ × HumanEffort), λ=0.05 |
 | **Net Utility Status** | **POSITIVE** | POSITIVE = system adds value after annotation cost |
 | **Tier 2 Autonomous Gain** | 1.60% | Accuracy lift from LLM with zero human cost |
 
 ## 3. Tier Distribution & Load Balancing
 The framework aims to maximize Tier 1 usage while minimizing Tier 3 escalation.
 
 - **Tier 1 (Base Encoder):** 425 samples (85.0%)
 - **Tier 2 (Local LLM):** 74 samples (14.8%)
 - **Tier 3 (Human Expert):** 1 samples (0.2%)
 
 ## 4. Constraint Validation
 - ✅ **Efficiency Target (>=60%):** 85.0% (PASSED)
 - ✅ **Human Cost Target (<=10%):** 0.2% (PASSED)
 
 ## 5. Conclusion
 The system demonstrated a **1.60% accuracy improvement** with only **0.2% human intervention**, confirming the effectiveness of the tiered routing strategy.
 
 ## 6. PICR Interpretation
 The Point-Improvement-per-Cost-Ratio (PICR) measures the efficiency of human intervention.
 **Formula:** `ΔAccuracy / Human Effort Ratio`
 **Current PICR:** `8.0000` (STRONG) Note: PICR is based on only 1 Tier 3 sample(s). Run with --seeds for statistically robust PICR estimates.
 
 **Interpretation:**
 A PICR below 1.0 indicates the human effort ratio exceeded the accuracy gain — adjust τ₁ (entropy) upward or τ₂ (confidence) downward to reduce unnecessary escalation and improve cost-efficiency. The threshold sweep identifies the optimal (τ₁, τ₂) operating point for maximum system utility.
 
 ## 6b. PICR-AL Interpretation
 **Formula:** (ΔAccuracy + λ · ΔAcc_AL) / (Human Effort Ratio + ε)
 **λ:** 0.5 | **ε:** 0.001 | **ΔAcc_AL:** 0.0000
 **PICR-AL:** 5.3333 (STRONG)
 
 PICR-AL extends PICR by incorporating active learning progression into the numerator.
 A configuration that escalates only 1 sample scores lower under PICR-AL than under PICR
 if that sample fails to trigger a retraining batch — exposing the metric gaming behaviour
 of near-zero human escalation strategies.
 
 ## 6c. Net Utility Interpretation
 **Formula:** ΔAccuracy − (λ_cost × Human Effort Ratio)
 **λ_cost:** 0.05 | **Net Utility:** 0.0159 (POSITIVE)
 
 Net Utility is additive and cannot be gamed by minimising human escalation. A POSITIVE
 result confirms the system adds measurable value after accounting for annotation cost.
 A NEGATIVE result means the human effort cost exceeded the accuracy gain at this
 operating point — reduce escalation thresholds or increase Tier 2 autonomy.
 
 ## 7. Research Questions (RQ) Analysis
 - **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
   - **Verdict:** Supported (Tier 1 Coverage: 85.00%, Final Accuracy: 88.80%, Tier 1 Accuracy: 87.20%)
   
 - **RQ2: Did the Active Learning (AL) loop improve Tier 1?**
   - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 87.40%, Post-AL Tier 1 Accuracy: 87.40%)
   
 - **RQ3: Does PICR identify optimal configurations?**
   - **Verdict:** Supported

## 8. Per-Category Performance Breakdown
| Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **World** | 90.65% | 80.83% | 85.46% | 120 | 93.46% | 83.33% | 88.11% | 120 |
| **Sports** | 97.52% | 97.52% | 97.52% | 121 | 98.36% | 99.17% | 98.77% | 121 |
| **Business** | 85.27% | 82.09% | 83.65% | 134 | 85.07% | 85.07% | 85.07% | 134 |
| **Sci/Tech** | 77.62% | 88.80% | 82.84% | 125 | 80.29% | 88.00% | 83.97% | 125 |

## 9. Escalation Pattern Analysis
| Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
| :--- | :---: | :---: | :---: | :---: |
| **World** | 17 | 0 | 120 | 14.17% |
| **Sports** | 4 | 0 | 121 | 3.31% |
| **Business** | 33 | 0 | 134 | 24.63% |
| **Sci/Tech** | 20 | 1 | 125 | 16.80% |
