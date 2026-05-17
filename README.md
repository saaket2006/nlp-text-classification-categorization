# 🛡️ Tri-Tiered Local LLM AL Framework

### Cost-Efficient Text Classification & Automated Categorization

The **Tri-Tiered Local LLM Active Learning (AL) Framework** is a performance-optimized system designed for high-accuracy text classification while minimizing computational costs and human intervention. By combining fast encoder models with powerful local LLMs and human-in-the-loop escalation, it achieves an optimal balance between efficiency and reliability.

---

## 🚀 Key Features

- **Tri-Tiered Routing Architecture**:
  - **Tier 1 (Base Model)**: Rapid classification using lightweight encoder models (e.g., DistilBERT). Handles the majority of "easy" samples.
  - **Tier 2 (Reasoning LLM)**: Local LLM integration via **Ollama** (e.g., Qwen 2.5, Llama 3) for samples requiring deeper semantic understanding or reasoning.
  - **Tier 3 (Human Expertise)**: Escalation to human reviewers for highly ambiguous or high-entropy cases.
- **Intelligent Uncertainty Engine**: Dynamic routing based on entropy and confidence thresholds ($\tau_1$, $\tau_2$).
- **Performance Metrics**: Comprehensive tracking of Accuracy, F1-Score, **PICR** (Performance Improvement to Cost Ratio), and Human Effort Ratio.
- **Interactive Visualization**: A built-in Streamlit dashboard featuring:
    - Routing Distribution (Donut Charts)
    - Efficiency Frontier analysis
    - Reliability Diagrams (Calibration analysis)
    - Detailed decision logs

---

## 📁 Project Structure

```bash
├── core/               # Routing, Uncertainty, and Calibration logic
├── models/             # Tier 1 (HF) and Tier 2 (Ollama) implementations
├── data/               # Data loading and preprocessing utilities
├── metrics/            # Pipeline evaluation and report generation
├── ui/                 # Streamlit-based visualization dashboard
├── logs/               # Execution logs, metrics, and research reports
├── config.yaml         # Centralized configuration (models, thresholds, data)
├── main.py             # Main entry point to run the classification pipeline
└── requirements.txt    # Project dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/nlp-text-classification-categorization.git
cd nlp-text-classification-categorization
```

### 2. Install Dependencies
Ensure you have Python 3.9+ and run:
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama (Tier 2)
Download and install [Ollama](https://ollama.com/). Then pull the required model specified in `config.yaml`:
```bash
ollama pull qwen2.5:3b
```

### 4. GPU Acceleration (NVIDIA)
For best performance, ensure you have the CUDA version of PyTorch (installed automatically by the current `requirements.txt` for CUDA 12.1). If you need to reinstall:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🏃 Usage

### 1. Run the Pipeline
Execute the main script to process the dataset:
```bash
python main.py
```
**Offline Mode**: To run without an internet connection (using cached models/data), set the environment variable:
```bash
$env:HF_HUB_OFFLINE=1; python main.py
```

### 2. Launch the Dashboard
Visualize results and model performance:
```bash
./run_dashboard.bat
```
or
```bash
streamlit run ui/dashboard.py
```

---

## 📊 Performance Benchmarks (Latest Run)
The framework is optimized for **real-world production environments** with a strict 1:1 human-to-accuracy cost ratio.

| Metric | Current Value | Target / Status |
| :--- | :--- | :--- |
| **PICR** | **3.0000** | **STRONG** (Target > 2.0) |
| **Accuracy Boost** | **+1.5%** | Measured against T1 Baseline |
| **Human Effort Ratio** | **0.5%** | Only 1 in 200 samples escalated |
| **Tier 1 Load** | **89.5%** | Maximizing speed and cost-savings |

### 🛠️ Optimized Routing Strategy
To achieve a PICR > 2.0 under realistic conditions, the following logic is implemented:
- **Uncertainty Capture ($\tau_1 = 0.65$)**: Borderline samples are escalated to Tier 2 for LLM reasoning.
- **Smart Override ($Conf \ge 0.85$)**: The Tier 2 LLM (Qwen 2.5) is permitted to override Tier 1 if its confidence is high, yielding "Free Gains" without human cost.
- **Safety Hard Stop ($\tau_{max} = 1.40$)**: Extremely chaotic or contradictory samples are sent directly to Human Experts to maintain system integrity.
- **Realistic Weighting**: Efficiency is calculated using a `human_effort_weight: 1.0`, ensuring the PICR accurately reflects real-world operational costs.

---

## ⚙️ Configuration

Modify `config.yaml` to adjust model parameters and routing thresholds:
- `threshold_entropy`: Adjusts the sensitivity for Tier 2 escalation.
- `threshold_extreme_entropy`: Absolute cap for direct Tier 3 human escalation.
- `threshold_confidence`: Sets the bar for Tier 1 automated acceptance.
- `human_effort_weight`: Set to `1.0` for realistic ROI analysis.
- `model_name`: Switch between different HuggingFace or Ollama models.

