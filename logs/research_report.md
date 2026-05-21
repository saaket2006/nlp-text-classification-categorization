# 📊 Research Report: Tri-Tiered Local LLM AL Framework

## 1. Executive Summary
This report summarizes the performance of the Tri-Tiered Active Learning framework. The system successfully routed samples through three levels of complexity, optimizing for both accuracy and human effort.

## 2. Core Performance Metrics
| Metric | Value | Note |
| :--- | :--- | :--- |
| **Total Samples** | 1000 | Test set size |
| **Tier 1 Accuracy** | 90.50% | Baseline (Encoder only) |
| **Final System Accuracy** | 91.60% | Integrated performance |
| **Accuracy Boost** | 1.10% | Lift from Tier 2 & 3 |
| **Weighted F1 Score** | 0.9161 | |
| **ECE** | 0.0567 | Calibration error — lower is better |
| **Human Effort Ratio** | 0.00% | Samples requiring human label |
| **Pre-AL Tier 1 Accuracy** | 90.50% | Tier 1 baseline before AL loop |
| **Post-AL Tier 1 Accuracy** | 90.50% | Tier 1 baseline after AL loop |
| **PICR** | 4.4000 | Point-Improvement-per-Cost-Ratio |
| **PICR Status** | **PERFECT_EFFICIENCY** | Efficiency classification |

## 3. Tier Distribution & Load Balancing
The framework aims to maximize Tier 1 usage while minimizing Tier 3 escalation.

- **Tier 1 (Base Encoder):** 949 samples (94.9%)
- **Tier 2 (Local LLM):** 51 samples (5.1%)
- **Tier 3 (Human Expert):** 0 samples (0.0%)

## 4. Constraint Validation
- ✅ **Efficiency Target (T1 >= 60%):** 94.9% (PASSED)
- ✅ **Human Cost Target (T3 <= 30%):** 0.0% (PASSED)

## 5. Conclusion
The system demonstrated a **1.10% accuracy improvement** with only **0.0% human intervention**, confirming the effectiveness of the tiered routing strategy.

## 6. PICR Interpretation
The Point-Improvement-per-Cost-Ratio (PICR) measures the efficiency of human intervention.
**Formula:** `ΔAccuracy / Human Effort Ratio`
**Current PICR:** `4.4000` (PERFECT_EFFICIENCY)

**Interpretation:**
A PICR below 1.0 indicates the human effort ratio exceeded the accuracy gain — adjust τ₁ (entropy) upward or τ₂ (confidence) downward to reduce unnecessary escalation and improve cost-efficiency. The threshold sweep identifies the optimal (τ₁, τ₂) operating point for maximum system utility.

## 7. Research Questions (RQ) Analysis
- **RQ1: Did uncertainty routing reduce human effort without sacrificing accuracy?**
  - **Verdict:** Supported (Tier 1 Coverage: 94.90%, Final Accuracy: 91.60%, Tier 1 Accuracy: 90.50%)
  
- **RQ2: Did the Active Learning (AL) loop improve Tier 1?**
  - **Verdict:** Not observed in this run (Pre-AL Tier 1 Accuracy: 90.50%, Post-AL Tier 1 Accuracy: 90.50%)
  
- **RQ3: Does PICR identify optimal configurations?**
  - **Verdict:** Pending — run with --sweep flag

## 8. Per-Category Performance Breakdown
| Category | Tier 1 Precision | Tier 1 Recall | Tier 1 F1 | Tier 1 Support | Final Precision | Final Recall | Final F1 | Final Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **World** | 95.47% | 87.22% | 91.16% | 266 | 95.93% | 88.72% | 92.19% | 266 |
| **Sports** | 96.40% | 97.97% | 97.18% | 246 | 96.41% | 98.37% | 97.38% | 246 |
| **Business** | 88.19% | 84.96% | 86.54% | 246 | 89.21% | 87.40% | 88.30% | 246 |
| **Sci/Tech** | 82.59% | 92.15% | 87.11% | 242 | 85.11% | 92.15% | 88.49% | 242 |

## 9. Escalation Pattern Analysis
| Category | Tier 2 Escalations | Tier 3 Escalations | Total Samples | Escalation Rate |
| :--- | :---: | :---: | :---: | :---: |
| **World** | 13 | 0 | 266 | 4.89% |
| **Sports** | 6 | 0 | 246 | 2.44% |
| **Business** | 15 | 0 | 246 | 6.10% |
| **Sci/Tech** | 17 | 0 | 242 | 7.02% |
