# 🛡️ Tri-Tiered Local LLM Active Learning Framework

### Cost-Efficient Text Classification & Automated Categorization

The **Tri-Tiered Local LLM Active Learning (AL) Framework** is a performance-optimized system designed for high-accuracy text classification while minimizing computational costs and human intervention. By combining fast encoder models with powerful local LLMs and human-in-the-loop escalation, it achieves an optimal balance between efficiency and reliability.

> [!TIP]
> **Dynamic Dataset Support & Recent Execution Results**: For a detailed walkthrough of the dynamic dataset support architectural changes, zero-shot dynamic prompt configurations, and latest multi-seed execution results on **AG News**, **DBpedia-14**, and **IMDb**, please refer to the [walkthrough.md](./walkthrough.md) file.

---

## 🚀 Key Features

- **Tri-Tiered Routing Architecture**:
  - **Tier 1 (Base Encoder)**: Rapid classification using lightweight encoder models (DistilBERT). Handles the majority of "easy" samples.
  - **Tier 2 (Reasoning LLM)**: Local LLM integration via **Ollama** (Qwen 2.5:3b) for samples requiring deeper semantic understanding or reasoning. Receives Tier 1 label hints for context-aware classification.
  - **Tier 3 (Human Expertise)**: Escalation to human reviewers for highly ambiguous or high-entropy cases.
- **Intelligent Uncertainty Engine**: Dynamic routing based on entropy and confidence thresholds (τ₁, τ₂).
- **Active Learning Loop**: Online fine-tuning of the Tier 1 encoder during evaluation using human-labeled samples.
- **Multi-Seed Reproducibility**: Run across multiple seeds with aggregated mean/std statistics.
- **Comprehensive Ablation Suite**: Tier-2-only baseline, no-Tier-2 ablation, no-entropy-routing ablation, and random routing baseline.
- **Threshold Sweep**: Automated grid search over (τ₁, τ₂) to find optimal operating points.
- **Performance Metrics**: PICR (Point-Improvement-per-Cost-Ratio), Accuracy, F1-Score, ECE (Expected Calibration Error), per-category breakdowns, and confusion matrices.
- **Interactive Streamlit Dashboard**: Premium dark-theme dashboard with 6 tabbed views and 15+ interactive Plotly charts.

---

## 📁 Project Structure

```
├── core/                   # Routing, Uncertainty, and Calibration logic
│   ├── router.py           # TieredRouter — entropy/confidence-based routing
│   ├── uncertainty.py      # UncertaintyEngine — entropy calculation
│   └── calibration.py      # ECE metric and reliability diagram utilities
├── models/                 # Tier 1 (HuggingFace) and Tier 2 (Ollama) implementations
│   ├── tier1_model.py      # DistilBERT encoder with online fine-tuning
│   └── tier2_llm.py        # Ollama LLM with majority-vote and Tier 1 hint
├── data/                   # Data loading and preprocessing utilities
│   └── loader.py           # Dataset loader (dynamic dataset/category support)
├── metrics/                # Pipeline evaluation and report generation
│   └── evaluator.py        # PipelineEvaluator — PICR, F1, confusion matrices
├── ui/                     # Streamlit-based visualization dashboard
│   └── dashboard.py        # Premium interactive dashboard (6 tabs, 15+ charts)
├── logs/                   # Execution logs, metrics, and research reports
├── config.yaml             # Centralized configuration (models, thresholds, data)
├── main.py                 # Main entry point with CLI flags for all modes
├── run_dashboard.bat       # One-click dashboard launcher (Windows)
├── walkthrough.md          # Architectural walkthrough and verification results
└── requirements.txt        # Project dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/nlp-text-classification-categorization.git
cd nlp-text-classification-categorization
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 3. Setup Ollama (Tier 2)
Download and install [Ollama](https://ollama.com/). Then pull the required model:
```bash
ollama pull qwen2.5:3b
```
Ensure Ollama is running before executing the pipeline:
```bash
ollama serve
```

### 4. GPU Acceleration (NVIDIA)
For best performance, ensure you have the CUDA version of PyTorch (installed automatically by `requirements.txt` for CUDA 12.1). To reinstall manually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🏃 Usage

### 1. Run the Main Pipeline
```bash
python main.py
```

### 2. Run with CLI Flags

| Flag | Description |
| :--- | :--- |
| `--seeds 42 123 456` | Multi-seed reproducibility run |
| `--baseline` | Tier-2-only baseline (all samples → LLM) |
| `--ablation-no-tier2` | Ablation: bypass Tier 2, escalate directly to Tier 3 |
| `--ablation-no-entropy` | Ablation: disable entropy routing |
| `--random-routing` | Random routing baseline (60% T1, 30% T2, 10% T3) |
| `--sweep` | Threshold sweep over (τ₁, τ₂) grid |

**Examples:**
```bash
# Multi-seed run
python main.py --seeds 42 123 456

# Run all baselines and ablations
python main.py --baseline --ablation-no-tier2 --ablation-no-entropy --random-routing

# Run threshold sweep
python main.py --sweep

# Offline mode (cached models/data)
$env:HF_HUB_OFFLINE=1; python main.py
```

### 3. Launch the Dashboard
```bash
streamlit run ui/dashboard.py
```
Or use the batch file:
```bash
./run_dashboard.bat
```

---

## 📊 Dashboard Features

The interactive Streamlit dashboard provides a comprehensive view of all pipeline results across **6 tabbed views**:

| Tab | Contents |
| :--- | :--- |
| **📊 Overview** | KPI banner, routing donut chart, Sankey flow, accuracy waterfall, efficiency frontier, AL improvement tracking, constraint validation |
| **🏷️ Per-Category** | F1/Precision/Recall comparison (T1 vs Final), escalation pattern analysis with rates and counts |
| **🎯 Calibration & Confusion** | Reliability diagram, ECE display, confidence distribution histogram, confusion matrices (T1, Final, Escalated) |
| **⚔️ Baselines & Ablations** | Side-by-side accuracy/effort/PICR comparison, tier distribution stacked bar, summary table, PICR ROI heatmap, multi-seed stats |
| **📈 Experiment History** | Accuracy/PICR/coverage trends over runs from experiment log CSV and history JSON |
| **📝 Decision Logs** | Filterable sample-level decision table, entropy vs confidence scatter plot colored by tier |

---

## 📊 Performance Benchmarks (Latest Run — 1000 Test Samples)

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Total Samples** | 1000 | Test set size |
| **Training Samples** | 2500 | AG News dataset |
| **Tier 1 Accuracy** | 90.50% | DistilBERT baseline |
| **Final System Accuracy** | 91.10% | Integrated performance |
| **Accuracy Boost** | +0.60% | Lift from Tier 2 consensus overrides |
| **Weighted F1 Score** | 0.9112 | |
| **ECE** | 0.0539 | Calibration error |
| **Human Effort Ratio** | 0.10% | Only 1 human escalation out of 1000 samples |
| **PICR** | **6.0000** | **STRONG** |

### Scalability Results

| Test Set | T1 Accuracy | Final Accuracy | Boost | Tier 2 Escalations | Tier 3 | PICR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 300 | 88.67% | 90.67% | +2.00% | 19 (6.3%) | 0 | **8.0000** | PERFECT_EFFICIENCY |
| 500 | 89.80% | 91.00% | +1.20% | 30 (6.0%) | 0 | **4.8000** | PERFECT_EFFICIENCY |
| 1000 | 90.50% | 91.10% | +0.60% | 48 (4.8%) | 1 (0.1%) | **6.0000** | STRONG |

### Routing Strategy
- **Tier 1 Coverage:** ~95.1% of all samples handled by the fast encoder
- **Tier 2 (LLM) Escalation:** ~4.8% of uncertain samples routed to Tier 2 (Qwen 2.5)
- **Tier 3 (Human):** 0.1% — only 1 sample escalated to human intervention
- **PICR > 3.0** consistently across all test set sizes

### 🧮 Point-Improvement-per-Cost-Ratio (PICR)
The **PICR** measures the return on investment of human annotation (Tier 3) by comparing the accuracy improvement against the human effort required:

$$\text{PICR} = \frac{\Delta\text{Accuracy}}{\text{Human Effort Ratio}} = \frac{\text{Final Accuracy} - \text{Tier 1 Accuracy}}{\text{Tier 3 Sample Count} / \text{Total Sample Count}}$$

*   **Status Classifications**:
    *   `AUTONOMOUS`: System achieved gain entirely through Tier 2 LLM reasoning with zero human intervention.
    *   `PERFECT_EFFICIENCY` / `STRONG`: Highly efficient routing where the accuracy boost vastly outperforms human cost (PICR >= 4.0).
    *   `ACCEPTABLE` / `BELOW_TARGET`: Modest efficiency.
    *   `NO_GAIN`: The accuracy gain did not offset or justify the human cost (or was negative).

---

## ⚙️ Configuration

All settings are centralized in `config.yaml`:

```yaml
project_name: "Tri-Tiered Local LLM AL Framework"
seed: 42

tier1:
  model_name: "distilbert-base-uncased"
  max_length: 128
  batch_size: 16
  active_learning_batch_size: 1
  pretrain_epochs: 7
  early_stopping_patience: 0    # 0 = disabled; set to 2 to enable
  picr_al_lambda: 0.5           # λ — AL progression weight for PICR-AL
  # threshold_entropy: 0.72     # optional manual override for τ₁
  # threshold_confidence: 0.75  # optional manual override for τ₂

tier2:
  ollama_model: "qwen2.5:3b"
  url: "http://localhost:11434"

tier3:
  human_error_rate: 0.05
  annotation_cost_weight: 0.05  # λ_cost — annotation cost weight

data:
  dataset_name: "imdb"
  train_samples: 2500
  test_samples: 500

datasets_config:
  ag_news:
    threshold_entropy: 1.00
    threshold_confidence: 0.75
    threshold_extreme_entropy: 1.80
    train_samples: 2500
    test_samples: 500
  dbpedia_14:
    threshold_entropy: 0.40
    threshold_confidence: 0.95
    threshold_extreme_entropy: 1.60
    train_samples: 2500
    test_samples: 1000
  imdb:
    threshold_entropy: 0.72
    threshold_confidence: 0.75
    threshold_extreme_entropy: 0.95
    train_samples: 2500
    test_samples: 500
```


### Key Parameters
| Parameter | Description |
| :--- | :--- |
| `threshold_entropy` | Sensitivity for Tier 2 escalation (higher = fewer escalations) |
| `threshold_confidence` | Minimum confidence to stay in Tier 1 |
| `threshold_extreme_entropy` | Absolute cap for direct Tier 3 escalation |
| `pretrain_epochs` | Number of pretraining epochs for the DistilBERT encoder |
| `human_error_rate` | Simulated human annotation error rate for Tier 3 |
| `train_samples` / `test_samples` | Default dataset split sizes |
| `datasets_config` | Dataset-specific configuration mappings (thresholds, samples) |
| `annotation_cost_weight` | Relative cost of human effort in Net Utility calculations (λ_cost) |
| `picr_al_lambda` | Active learning progression weight in PICR-AL calculations (λ) |

---

## 📄 Output Files

The framework organizes execution logs dynamically under dataset-specific directories to support concurrent dataset evaluations:

| File Pattern | Description |
| :--- | :--- |
| `logs/<dataset_name>/metrics_summary.json` | Core metrics (accuracy, F1, PICR, Net Utility, confusion matrices) |
| `logs/<dataset_name>/detailed_results.json` | Per-sample details (tier, prediction, confidence, entropy, rationale) |
| `logs/reports/<dataset_name>_report.md` | Auto-generated markdown research report with Research Questions (RQ) analysis |
| `logs/<dataset_name>/experiment_log.csv` | Historical run tracker with timestamps, threshold settings, and key outcomes |
| `logs/<dataset_name>/history.json` | Historical trial log for the efficiency frontier plotting |
| `logs/<dataset_name>/baseline_metrics.json` | Baseline metrics from Tier-2-only evaluation |
| `logs/<dataset_name>/ablation_no_tier2.json` | Ablation results with no Tier 2 reasoning |
| `logs/<dataset_name>/ablation_no_entropy.json` | Ablation results with entropy routing disabled |
| `logs/<dataset_name>/baseline_random_routing.json` | Baseline results with random routing |
| `logs/<dataset_name>/threshold_sweep.json` | Swept threshold configurations over the search grid |
| `logs/<dataset_name>/multi_seed_summary.json` | Aggregated statistics (mean and standard deviation) across multi-seed runs |

---

## 🔧 Dependencies

```
torch (CUDA 12.1)
transformers
ollama
streamlit
pandas
numpy
scikit-learn
plotly
scipy
datasets
tqdm
pyyaml
```

