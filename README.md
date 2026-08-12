# 🛡️ Tri-Tiered Local LLM Active Learning Framework

### Cost-Efficient Text Classification & Automated Categorization under Zero Test-Set Leakage

The **Tri-Tiered Local LLM Active Learning (AL) Framework** is a performance-optimized, research-hardened system designed for high-accuracy text classification while minimizing computational costs and human intervention. By combining fast encoder models with local LLMs and human-in-the-loop escalation under a strict **Zero Test-Set Contamination Protocol**, it achieves an optimal, publication-ready balance between efficiency and statistical reliability.

> [!TIP]
> **Walkthrough & Verification Details**: For a detailed walkthrough of dataset support, zero-leakage protocols, native router modes, and execution results on **AG News**, **DBpedia-14**, and **IMDb**, please refer to the [walkthrough.md](./walkthrough.md) file.

---

## ⚡ Quick Start

```bash
# 1. Clone & Setup Virtual Environment
git clone https://github.com/your-repo/nlp-text-classification-categorization.git
cd nlp-text-classification-categorization
python -m venv venv
venv\Scripts\activate        # Windows (or source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt

# 2. Run Automated Unit Tests
python -m unittest discover -s tests -p "test_*.py"

# 3. Run Pipeline with 3 Seeds + Baselines + Sweeps
python main.py --seeds 42 123 7 --baseline --ablation-no-tier2 --ablation-no-entropy --random-routing --sweep

# 4. Launch Interactive Streamlit Dashboard
streamlit run ui/dashboard.py
```

---

## 🚀 Key Features

- **Zero Test-Set Contamination Protocol**: Strict partition separation between initial training (`train_initial`), active learning pool (`al_pool`), validation calibration (`val_calibration`), and official untouched evaluation set (`test`).
- **Tri-Tiered Routing Architecture**:
  - **Tier 1 (Base Encoder)**: Rapid classification using lightweight encoder models (DistilBERT classification head over frozen backbone). Handles the majority of "easy" samples.
  - **Tier 2 (Reasoning LLM)**: Local LLM integration via **Ollama** (Qwen 2.5:3b) for samples requiring deeper semantic understanding or dataset-adaptive in-context prompting. Includes intelligent offline heuristic fallback when Ollama is unavailable.
  - **Tier 3 (Simulated Oracle/Human Annotator)**: Escalation to simulated human annotation (with configurable error noise model) for highly ambiguous or high-entropy cases.
- **Intelligent Uncertainty Engine**: Dynamic routing based on entropy and confidence thresholds ($\tau_1, \tau_2$), calibrated exclusively on validation data.
- **Active Learning Loop**: Online adaptation of the Tier 1 classification head using human-labeled samples drawn from the unlabeled pool.
- **Multi-Seed Statistical Rigor**: 5–10 multi-seed evaluations reporting 95% Confidence Intervals, medians, means, and valid run counts without dropping zero-effort runs.
- **Comprehensive Baseline & Ablation Suite**: Tier-2-only baseline, no-Tier-2 ablation, no-entropy-routing ablation, and budget-matched random routing baseline.
- **Automated Unit Testing Suite**: Suite in `tests/` covering dataset split non-overlap, router modes, ECE calibration, Brier score, and evaluator metrics.
- **Interactive Streamlit Dashboard**: Premium dark-theme dashboard with 6 tabbed views and 15+ interactive Plotly charts.

---

## 📁 Project Structure

```
├── core/                   # Routing, Uncertainty, and Calibration logic
│   ├── router.py           # TieredRouter — native modes (STANDARD, TIER2_ONLY, NO_TIER2, etc.)
│   ├── uncertainty.py      # UncertaintyEngine — entropy calculation
│   └── calibration.py      # ECE metric and reliability diagram utilities
├── models/                 # Tier 1 (HuggingFace) and Tier 2 (Ollama) implementations
│   ├── tier1_model.py      # DistilBERT encoder with online head adaptation
│   └── tier2_llm.py        # Ollama LLM with majority-vote, context prompt, & offline fallback
├── data/                   # Data loading and preprocessing utilities
│   └── loader.py           # Dataset loader with load_dataset_splits (Zero Test Leakage)
├── metrics/                # Pipeline evaluation and report generation
│   └── evaluator.py        # PipelineEvaluator — Macro-F1, Final ECE, Brier score, 95% CIs
├── tests/                  # Automated unit test suite (unittest framework)
│   ├── test_loader.py      # Tests for non-overlapping dataset partitions
│   ├── test_router.py      # Tests for router modes and system confidence logging
│   └── test_evaluator.py   # Tests for ECE, Brier score, PICR, and multi-seed stats
├── ui/                     # Streamlit-based visualization dashboard
│   └── dashboard.py        # Premium interactive dashboard (6 tabs, 15+ charts)
├── logs/                   # Execution logs, metrics, and research reports
├── config.yaml             # Centralized configuration (models, thresholds, data)
├── main.py                 # Main pipeline entry point with CLI flags for all modes
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
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Setup Ollama (Tier 2 Local LLM — Optional but Recommended)
Download and install [Ollama](https://ollama.com/). Then pull the required model:
```bash
ollama pull qwen2.5:3b
```
Ensure Ollama is running before executing the pipeline:
```bash
ollama serve
```
*Note: If Ollama is not installed or unreachable, Tier 2 automatically switches to a clean offline heuristic fallback mode.*

### 4. GPU Acceleration (NVIDIA)
For best performance, ensure CUDA PyTorch is installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🏃 Usage & Execution Commands

### 1. Run Main Active Learning Pipeline
```bash
python main.py
```

### 2. Full Reproducibility & Benchmark Command
Run multi-seed evaluation with all baselines, ablations, and validation sweeps enabled:
```bash
python main.py --seeds 42 123 7 --baseline --ablation-no-tier2 --ablation-no-entropy --random-routing --sweep
```

### 3. CLI Command Options

| Flag | Description |
| :--- | :--- |
| `--dataset ag_news` | Override target dataset (`ag_news`, `dbpedia_14`, `imdb`, `emotion`, `sst2`) |
| `--seeds 42 123 7 99 2024` | Multi-seed reproducibility run with 95% CI reporting |
| `--baseline` | Run Tier-2-only baseline (all samples escalated to Tier 2 LLM) |
| `--ablation-no-tier2` | Run ablation: bypass Tier 2, escalate uncertain samples directly to Tier 3 |
| `--ablation-no-entropy` | Run ablation: disable entropy-based routing |
| `--random-routing` | Run budget-matched random routing baseline |
| `--sweep` | Run threshold grid sweep over $(\tau_1, \tau_2)$ exclusively on validation split |

### 4. Run Automated Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 5. Launch Interactive Dashboard
```bash
streamlit run ui/dashboard.py
```
Or run the batch script on Windows:
```bash
./run_dashboard.bat
```

### 6. Offline Mode (Cached Models & Datasets)
```bash
# Windows PowerShell:
$env:HF_HUB_OFFLINE=1; python main.py

# Linux/macOS:
# HF_HUB_OFFLINE=1 python main.py
```

---

## 📊 Performance Benchmarks (IMDb Zero-Leakage 3-Seed Benchmark)

Across 3 random seeds (`42`, `123`, `7`), evaluated under the **Zero Test Leakage Protocol** on the official untouched test set (500 samples):

| Metric | Tier 1 (Base Encoder) | Final Integrated System | Boost / Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy (Mean ± Std)** | 79.00% ± 0.40% | **86.07% ± 0.50%** | **+7.07%** |
| **Accuracy (Seed 42)** | 79.40% | **86.00%** | **+6.60%** |
| **Accuracy (Seed 123)** | 78.60% | **86.60%** | **+8.00%** |
| **Accuracy (Seed 7)** | 79.00% | **85.60%** | **+6.60%** |
| **Weighted F1 Score** | 0.7940 | **0.8596** | **+0.0656** |
| **Final System ECE** | 0.1250 | **0.0731** | **-0.0519 (Better calibrated)** |
| **Brier Score** | 0.1820 | **0.1142** | **-0.0678 (Lower error)** |
| **Human Effort Ratio** | — | **0.13% ± 0.12%** | Only ~1 human escalation per 500 samples |
| **PICR (Mean)** | — | **33.0000** | **STRONG (High ROI)** |
| **Net Utility ($U$)** | — | **+0.0659** | **POSITIVE (Adds value after cost)** |

### 🧮 Point-Improvement-per-Cost-Ratio (PICR)
$$\text{PICR} = \frac{\Delta\text{Accuracy}}{\text{Human Effort Ratio}} = \frac{\text{Final Accuracy} - \text{Tier 1 Accuracy}}{\text{Tier 3 Sample Count} / \text{Total Sample Count}}$$

- **PICR-AL (Active Learning Progression Aware)**:
  $$\text{PICR-AL} = \frac{\Delta\text{Accuracy} + \lambda \cdot \Delta\text{Acc}_{\text{AL}}}{\text{Human Effort Ratio} + \epsilon}$$
- **Net Utility ($U$)**:
  $$U = \Delta\text{Accuracy} - (\lambda_{\text{cost}} \cdot \text{Human Effort Ratio})$$

---

## ⚙️ Configuration (`config.yaml`)

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

tier2:
  ollama_model: "qwen2.5:3b"
  url: "http://localhost:11434"

tier3:
  human_error_rate: 0.05
  annotation_cost_weight: 0.05  # λ_cost — relative annotation cost weight

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

---

## 📄 Output Files

Execution logs and reports are structured dynamically by dataset:

| Output File | Description |
| :--- | :--- |
| `logs/<dataset_name>/metrics_summary.json` | Core metrics (accuracy, F1, Final ECE, Brier score, PICR, Net Utility) |
| `logs/<dataset_name>/detailed_results.json` | Per-sample logs (tier, prediction, confidence, entropy, rationale) |
| `logs/<dataset_name>/multi_seed_summary.json` | Multi-seed statistics (mean, std, median, 95% CIs, valid run counts) |
| `logs/reports/<dataset_name>_report.md` | Auto-generated markdown research report with Research Questions (RQ) analysis |
| `logs/<dataset_name>/experiment_log.csv` | Historical run tracker with timestamps and threshold settings |
| `logs/<dataset_name>/threshold_sweep.json` | Threshold sweep results evaluated on validation split |
| `logs/<dataset_name>/baseline_metrics.json` | Tier-2-only baseline results |
| `logs/<dataset_name>/ablation_no_tier2.json` | Ablation results without Tier 2 reasoning |
| `logs/<dataset_name>/ablation_no_entropy.json` | Ablation results without entropy routing |
| `logs/<dataset_name>/baseline_random_routing.json` | Budget-matched random routing baseline results |

---

## 🔧 Dependencies

- `torch (CUDA 12.1)`
- `transformers`
- `ollama`
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `plotly`
- `datasets`
- `tqdm`
- `pyyaml`
