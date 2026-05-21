# 🛡️ Tri-Tiered Local LLM Active Learning Framework

### Cost-Efficient Text Classification & Automated Categorization — IEEE Conference Submission

The **Tri-Tiered Local LLM Active Learning (AL) Framework** is a performance-optimized system designed for high-accuracy text classification while minimizing computational costs and human intervention. By combining fast encoder models with powerful local LLMs and human-in-the-loop escalation, it achieves an optimal balance between efficiency and reliability.

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
│   └── loader.py           # AG News dataset loader
├── metrics/                # Pipeline evaluation and report generation
│   └── evaluator.py        # PipelineEvaluator — PICR, F1, confusion matrices
├── ui/                     # Streamlit-based visualization dashboard
│   └── dashboard.py        # Premium interactive dashboard (6 tabs, 15+ charts)
├── logs/                   # Execution logs, metrics, and research reports
├── config.yaml             # Centralized configuration (models, thresholds, data)
├── main.py                 # Main entry point with CLI flags for all modes
├── run_dashboard.bat       # One-click dashboard launcher (Windows)
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
| **Final System Accuracy** | 91.60% | Integrated performance |
| **Accuracy Boost** | +1.10% | Lift from Tier 2 |
| **Weighted F1 Score** | 0.9161 | |
| **ECE** | 0.0567 | Calibration error |
| **Human Effort Ratio** | 0.00% | Zero human escalation |
| **PICR** | **4.4000** | **PERFECT_EFFICIENCY** |

### Scalability Results

| Test Set | T1 Accuracy | Final Accuracy | Boost | Tier 2 Escalations | Tier 3 | PICR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 300 | 88.67% | 90.67% | +2.00% | 19 (6.3%) | 0 | **8.0000** | PERFECT_EFFICIENCY |
| 500 | 89.80% | 91.00% | +1.20% | 30 (6.0%) | 0 | **4.8000** | PERFECT_EFFICIENCY |
| 1000 | 90.50% | 91.60% | +1.10% | 51 (5.1%) | 0 | **4.4000** | PERFECT_EFFICIENCY |

### Routing Strategy
- **Tier 1 Coverage:** ~94% of all samples handled by the fast encoder
- **Tier 2 (LLM) Escalation:** ~5-6% of uncertain samples improved by Qwen 2.5
- **Tier 3 (Human):** 0% — zero human intervention required
- **PICR > 3.0** consistently across all test set sizes

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
  threshold_entropy: 1.10        # τ₁ — entropy threshold for Tier 2 escalation
  threshold_confidence: 0.50     # τ₂ — confidence threshold
  threshold_extreme_entropy: 10.0
  active_learning_batch_size: 1
  pretrain_epochs: 7
  early_stopping_patience: 7

tier2:
  ollama_model: "qwen2.5:3b"
  url: "http://localhost:11434"
  categories: ["World", "Sports", "Business", "Sci/Tech"]

tier3:
  human_error_rate: 0.05

data:
  dataset_name: "ag_news"
  train_samples: 2500
  test_samples: 1000
```

### Key Parameters
| Parameter | Description |
| :--- | :--- |
| `threshold_entropy` | Sensitivity for Tier 2 escalation (higher = fewer escalations) |
| `threshold_confidence` | Minimum confidence to stay in Tier 1 |
| `threshold_extreme_entropy` | Absolute cap for direct Tier 3 escalation |
| `pretrain_epochs` | Number of pretraining epochs for the DistilBERT encoder |
| `human_error_rate` | Simulated human annotation error rate for Tier 3 |
| `train_samples` / `test_samples` | Dataset split sizes |

---

## 📄 Output Files

| File | Description |
| :--- | :--- |
| `logs/metrics_summary.json` | Core metrics (accuracy, F1, PICR, confusion matrices, per-category F1) |
| `logs/detailed_results.json` | Per-sample prediction details with tier, labels, confidence, entropy, rationale |
| `logs/research_report.md` | Auto-generated IEEE-style research report with RQ analysis |
| `logs/experiment_log.csv` | Experiment tracker with timestamps, configs, and key metrics |
| `logs/history.json` | Historical trial log for efficiency frontier visualization |
| `logs/baseline_metrics.json` | Tier-2-only baseline results |
| `logs/ablation_no_tier2.json` | No-Tier-2 ablation results |
| `logs/ablation_no_entropy.json` | No-entropy-routing ablation results |
| `logs/baseline_random_routing.json` | Random routing baseline results |
| `logs/threshold_sweep.json` | Threshold sweep grid results |
| `logs/multi_seed_summary.json` | Multi-seed aggregated statistics (mean, std) |

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

