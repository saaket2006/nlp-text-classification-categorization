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

---

## 🏃 Usage

### 1. Run the Pipeline
Execute the main script to process the dataset through the tiered framework:
```bash
python main.py
```
This will generate execution logs and a `research_report.md` in the `logs/` directory.

### 2. Launch the Dashboard
Visualize the results and model performance:
```bash
streamlit run ui/dashboard.py
```

---

## ⚙️ Configuration

Modify `config.yaml` to adjust model parameters and routing thresholds:
- `threshold_entropy`: Adjusts the sensitivity for Tier 2 escalation.
- `threshold_confidence`: Sets the bar for Tier 1 automated acceptance.
- `model_name`: Switch between different HuggingFace or Ollama models.

---

## 📊 Target Benchmarks

The framework is optimized to meet the following operational constraints:
- **Automation Target**: ≥60% of samples handled by Tier 1.
- **Escalation Constraint**: ≤30% of samples requiring Tier 3 (Human) intervention.

---

## 📜 License
[MIT License](LICENSE)
