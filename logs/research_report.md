
# 📊 Research Report: Tri-Tiered Local LLM AL Framework

## 1. Executive Summary
This report summarizes the performance of the Tri-Tiered Active Learning framework. The system successfully routed samples through three levels of complexity, optimizing for both accuracy and human effort.

## 2. Core Performance Metrics
| Metric | Value | Note |
| :--- | :--- | :--- |
| **Total Samples** | 200 | Test set size |
| **Tier 1 Accuracy** | 87.00% | Baseline (Encoder only) |
| **Final System Accuracy** | 88.50% | Integrated performance |
| **Accuracy Boost** | 1.50% | Lift from Tier 2 & 3 |
| **Weighted F1 Score** | 0.8853 | |
| **Human Effort Ratio** | 0.50% | Samples requiring human label |
| **PICR** | 3.0000 | Point-Improvement-per-Cost-Ratio |
| **PICR Status** | **STRONG** | Efficiency classification |

## 3. Tier Distribution & Load Balancing
The framework aims to maximize Tier 1 usage while minimizing Tier 3 escalation.

- **Tier 1 (Base Encoder):** 189 samples (94.5%)
- **Tier 2 (Local LLM):** 10 samples (5.0%)
- **Tier 3 (Human Expert):** 1 samples (0.5%)

## 4. Constraint Validation
- ✅ **Efficiency Target (T1 >= 60%):** 94.5% (PASSED)
- ✅ **Human Cost Target (T3 <= 30%):** 0.5% (PASSED)

## 5. Conclusion
The system demonstrated a **1.50% accuracy improvement** with only **0.5% human intervention**, confirming the effectiveness of the tiered routing strategy.

## 6. PICR Interpretation
The Point-Improvement-per-Cost-Ratio (PICR) measures the efficiency of human intervention.
**Formula:** `ΔAccuracy / Human Effort Ratio`
**Current PICR:** `3.0000` (STRONG)

**Interpretation:**
A PICR below 1.0 indicates the human effort ratio exceeded the accuracy gain — adjust τ₁ (entropy) upward or τ₂ (confidence) downward to reduce unnecessary escalation and improve cost-efficiency. The threshold sweep identifies the optimal (τ₁, τ₂) operating point for maximum system utility.
