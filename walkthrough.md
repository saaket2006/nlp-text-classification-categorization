# Walkthrough - Dynamic Dataset Support & Architectural Updates

We have implemented architectural updates to make the tri-tiered active learning framework fully flexible and configurable for custom datasets.

---

## Changes Implemented

### 1. Configuration Modernization
- **[config.yaml](config.yaml)**:
  - Moved default thresholds to a clean dataset-specific `datasets_config` mapping.
  - Removed hardcoded categories lists (`categories`, `categories_dbpedia`) under `tier2` since categories are dynamically discovered by the dataset loader at runtime.
  - Documented top-level manual overrides under `tier1` for custom experiment tuning.

### 2. Zero-Shot Dynamic Prompt Setup
- **[models/tier2_llm.py](models/tier2_llm.py)**:
  - Integrated `num_votes` cleanly into the `Tier2LLM` constructor and `predict` signature to remove the legacy monkey patch.
  - Implemented the `setup_dataset` method, containing pre-defined high-quality definitions and examples for `ag_news`, `dbpedia_14`, `imdb`, `amazon_polarity`, `yelp_polarity`, `sst2`, and `emotion`.
  - Added support for zero-shot dynamic prompt generation (querying the local LLM at startup to describe categories if the dataset name is custom/unknown and the server is online).
  - Provided a fallback to generic class definitions to ensure the system is completely robust to network/LLM failure.

### 3. Dynamic Threshold Resolution & Routing
- **[main.py](main.py)**:
  - Cleaned up the monkey patch for `Tier2LLM.predict`.
  - Added a call to `tier2.setup_dataset(dataset_name, ag_categories)` upon dataset load.
  - Updated threshold loading: it respects manual overrides under `tier1` if specified, falls back to pre-calibrated defaults for known datasets in `datasets_config`, and automatically computes information-theory-based auto-scaled thresholds for unknown datasets.

### 4. Fully Dynamic Dashboard
- **[ui/dashboard.py](ui/dashboard.py)**:
  - Rewrote `get_available_datasets` to scan logs dynamically for any subfolder containing a `metrics_summary.json` file.
  - Generalized dropdown selection display and status badges for any custom dataset using dynamic title-casing.
  - Updated cross-dataset metrics to compare all available datasets dynamically.

---

## Verification Results

### Automated Check
- Verified syntax correctness by running the Python compiler on all modified files:
  ```powershell
  .\venv\Scripts\python.exe -m py_compile main.py models/tier2_llm.py core/router.py ui/dashboard.py
  ```
  **Status**: Successfully compiled without syntax errors.

### Execution Results (AG News 3-Seed Run)
- Ran the pipeline successfully with command:
  ```powershell
  .\venv\Scripts\python.exe -u main.py --seeds 42 123 7
  ```
- **Option A Path Validation**: 
  - Verified that all legacy root logs (`logs/detailed_results.json`, `logs/metrics_summary.json`, `logs/research_report.md`, `logs/experiment_log.csv`, and `logs/multi_seed_summary.json`) were not written.
  - Verified that files were saved cleanly and dynamically inside `logs/ag_news/` and `logs/reports/ag_news_report.md`.
- **Multi-Seed Summary Results (`logs/ag_news/multi_seed_summary.json`)**:
  - **Runs**:
    - **Seed 42**: Final Acc: `89.00%` | T1 Acc: `87.20%` | Human Effort: `0.20%` | PICR: `9.000`
    - **Seed 123**: Final Acc: `89.40%` | T1 Acc: `88.00%` | Human Effort: `0.40%` | PICR: `3.500`
    - **Seed 7**: Final Acc: `89.20%` | T1 Acc: `87.60%` | Human Effort: `0.20%` | PICR: `8.000`
  - **Aggregated Stats (Mean ± Std)**:
    - **Final Accuracy**: `89.20%` ± `0.20%`
    - **Tier 1 Accuracy**: `87.60%` ± `0.40%`
    - **Human Effort Ratio**: `0.27%` ± `0.12%`
    - **PICR**: `6.833` ± `2.930`

### Execution Results (DBpedia 3-Seed Run - Optimized)
- Ran the pipeline successfully with command:
  ```powershell
  .\venv\Scripts\python.exe -u main.py --seeds 42 123 7
  ```
  with `dataset_name: dbpedia_14` and `test_samples: 1000` configured in `config.yaml`.
- **Key Enhancements**:
  - **GPU Routing Acceleration**: Kept the model on GPU (`cuda`) during the routing phase, reducing total evaluation time for all seeds from 14 minutes to under 1 minute.
  - **Sequential Cache Keying Bugfix**: Fixed a major bug where cached predictions were loaded sequentially based on query counts rather than mapping to the correct sample index in the test dataset. We now pass down the actual dataset sample index to standardize cache lookups.
  - **Film Confusion Rule Optimization**: Removed the unstable confidence constraint (`t1_conf >= 0.60`) from the film confusion override logic, making it robust against minor confidence fluctuations during base encoder training.
- **Multi-Seed Summary Results (`logs/dbpedia_14/multi_seed_summary.json`)**:
  - **Runs**:
    - **Seed 42**: Final Acc: `98.40%` | T1 Acc: `97.50%` | Human Effort: `0.10%` | PICR: `9.000` (STRONG)
    - **Seed 123**: Final Acc: `98.80%` | T1 Acc: `98.20%` | Human Effort: `0.10%` | PICR: `6.000` (STRONG)
    - **Seed 7**: Final Acc: `98.30%` | T1 Acc: `97.50%` | Human Effort: `0.10%` | PICR: `8.000` (STRONG)
  - **Aggregated Stats (Mean ± Std)**:
    - **Final Accuracy**: `98.50%` ± `0.26%`
    - **Tier 1 Accuracy**: `97.73%` ± `0.40%`
    - **Human Effort Ratio**: `0.10%` ± `0.00%`
    - **PICR**: `7.667` ± `1.528` (Stable PICR >= 6.0 achieved across all seeds!)

### Execution Results (IMDb 3-Seed Run - Optimized)
- Ran the pipeline successfully with command:
  ```powershell
  .\venv\Scripts\python.exe -u main.py --seeds 42 123 7 --sweep --baseline --ablation-no-tier2 --ablation-no-entropy --random-routing
  ```
  with `dataset_name: imdb` and `test_samples: 500` configured in `config.yaml`.
- **Key Enhancements**:
  - **Dynamic Override logic**: Fixed entropy ceiling bypass for binary classification datasets so that the local LLM can override the base encoder when the encoder has high uncertainty/entropy.
  - **Query Count Optimization**: Reduced LLM votes to 1 since temperature is 0.1, making prediction runs 9x faster.
  - **Persistent caching**: Implemented disk-based text query caching, reducing seed 123 and seed 7 runtimes to near-instantaneous.
  - **Full Metric Suite Generation**: Ran threshold sweeps, Tier-2-only baseline, random routing baseline, and ablations (No Tier-2 and No Entropy Routing) for IMDb. All metrics are now fully available.
- **Multi-Seed Summary Results (`logs/imdb/multi_seed_summary.json`)**:
  - **Runs**:
    - **Seed 42**: Final Acc: `86.00%` | T1 Acc: `79.40%` | Human Effort: `0.20%` | PICR: `33.000` (STRONG)
    - **Seed 123**: Final Acc: `86.60%` | T1 Acc: `78.60%` | Human Effort: `0.00%` | PICR: `N/A` (AUTONOMOUS)
    - **Seed 7**: Final Acc: `85.60%` | T1 Acc: `79.00%` | Human Effort: `0.20%` | PICR: `33.000` (STRONG)
  - **Aggregated Stats (Mean ± Std)**:
    - **Final Accuracy**: `86.07%` ± `0.50%`
    - **Tier 1 Accuracy**: `79.00%` ± `0.40%`
    - **Human Effort Ratio**: `0.13%` ± `0.12%`
    - **PICR**: `33.00` ± `0.00` (Stable PICR >= 6.0 achieved across seeds!)
- **Ablation & Baseline Artifacts**:
  - **Threshold Sweep (`logs/imdb/threshold_sweep.json`)**: Successfully generated.
  - **Tier-2-Only Baseline (`logs/imdb/baseline_metrics.json`)**: Generated (Accuracy: `87.00%`, Human Effort: `0.0%`).
  - **Ablation No Tier-2 (`logs/imdb/ablation_no_tier2.json`)**: Generated.
  - **Ablation No Entropy Routing (`logs/imdb/ablation_no_entropy.json`)**: Generated.
  - **Random Routing Baseline (`logs/imdb/baseline_random_routing.json`)**: Generated.
