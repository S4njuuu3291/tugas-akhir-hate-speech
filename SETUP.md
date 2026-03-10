# Project Structure & Setup Guide

## Directory Organization

```
/DATA              - Processed datasets (ignored - too large)
/DOCS              - Documentation files
/EXPLAINABILITY    - SHAP explanation implementations & results
/MODELS            - Trained model configs and tokenizer files
/SKRIPSI           - Thesis documents and references
/z_APP             - Application code
/_ARCHIVE          - Legacy code and previous implementations (ignored)
```

## Key Source Files

Main implementations are located in:
- `_ARCHIVE/manual_shap/` - SHAP explainability implementations
- `_ARCHIVE/tugas_akhir/` - Main project code
- `_ARCHIVE/data/` - Sample datasets and preprocessing utilities

## Environment Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/S4njuuu3291/tugas-akhir-hate-speech.git
cd tugas-akhir-hate-speech
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Focus

- **Hate Speech Detection**: Multilabel classification for in Indonesian social media
- **Explainability**: Using SHAP (SHapley Additive exPlanations) for model interpretability
- **Models**: BERT-based (IndoBERT), SVM, Logistic Regression baselines

## Note

Large files such as:
- Trained model weights (*.h5, *.keras)
- Datasets (*.csv)
- Embeddings (*.npy)
- PDFs and documentation

Are excluded from git tracking due to size. Keep them locally or use Git LFS for tracking.

## Getting Started

1. Review notebooks in the project
2. Install required dependencies
3. Download/prepare datasets
4. Run notebooks or Python scripts for training/evaluation
