# Walkthrough - Conference Readiness and Methodological Rigor

We have completed the implementation of the full conference readiness updates (P0/P1/P2 checklist) to ensure the tri-tiered active learning framework is methodologically sound, rigorously evaluated, and fully reproducible.

---

## Architectural & Methodological Enhancements

### 1. Robust Validation Threshold Calibration & Cap
- **Percentile-based Extreme Entropy Cap Calibration**: Instead of the second-highest entropy, `config.yaml` now introduces `extreme_entropy_percentile` (default `0.98`). We compute this percentile on the validation split (`df_val_calib`), avoiding outliers and matching the statistics of the target distribution.
- **Validation Threshold Sweep**: Automatically sweeps the grid of thresholds $(\tau_1, \tau_2)$ on `df_val_calib`, evaluates the optimization metric **Net Utility** ($U = \Delta\text{Accuracy} - (\lambda_{cost} \times \text{Human Effort Ratio})$), and freezes the optimal parameters to prevent test set contamination.
- **Config-level cost parameters**: Added parameters to configure cost per tier (`cost_t1`, `cost_t2`, `cost_t3`).

### 2. Fair Budget-Matched Baselines
- **Genuinely Budget-Matched Random Routing**: The random routing baseline dynamically queries the standard pipeline's first run to retrieve the exact sample routing distribution ratio ($f_1, f_2, f_3$) and uses it to sample decision tiers, ensuring a fair, budget-matched comparison.
- **Calibrated Active Learning Baselines**: Implemented standard query strategies (`AL_RANDOM`, `AL_LEAST_CONFIDENCE`, `AL_ENTROPY`, `AL_MARGIN`) inside the router. Their thresholds are dynamically calibrated on the validation calibration split to match the exact human annotation budget ($f_3$) of the proposed router.

### 3. Rigorous Evaluation Metrics
- **Multiclass Brier Score**: Added standard multiclass Brier score calculation over probability vectors for both the Tier 1 model and the final routed system.
- **ECE Breakdowns**: ECE is separately calculated for the Tier 1 model and the final system.
- **Student-t Confidence Intervals**: Replaced the fixed $z$-critical value ($1.96$) with Student-t distribution critical values (`scipy.stats.t.ppf`) to compute rigorous $95\%$ Confidence Intervals for small sample sizes.
- **Cost & Latency Modeling**: Tracks actual query latency and compute costs based on tier weights.

### 4. Learning Curve Tracking
- The pipeline tracks the test accuracy at regular intervals (every 10 annotations) as simulated feedback is received, allowing users to plot training efficiency curves.

### 5. Reproducibility & Environment Manifest
- **`run_experiments.py`**: A single CLI entry point that runs experiments across multiple seeds and datasets, compiling results into a clean markdown table.
- **`reproducibility_manifest.json`**: Captures environment details including Python version, platform architecture, git commit, Ollama model, and package versions to guarantee exact reproduction.

---

## Verification Results

### Static Compilation
- Successfully ran py_compile on all updated files and verified they are free of syntax/import issues:
  - `main.py`
  - `run_experiments.py`
  - `tests/test_calibration_and_cost.py`

### Unit Tests
- Implemented extensive unit tests in `tests/test_calibration_and_cost.py` covering:
  - Multiclass Brier score calculation
  - Latency & Cost calculations
  - Student-t 95% Confidence Interval calculations
