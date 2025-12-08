# Authorship Classification Project

This repository implements two complementary approaches to predict the author of text fragments from four novelists: a TF-IDF + linear classifier baseline and a fine-tuned DistilBERT transformer. Cross-validation highlights that both pipelines generalize well (linear classification: 0.9441 ± 0.0013 accuracy; DistilBERT: 0.9607 ± 0.0014), and the accompanying notebooks walk through the full experimentation details.

## Project map
- **Exploratory analysis:** [`data_exploration.ipynb`](data_exploration.ipynb)
- **Pipeline notebooks:** [`run_linear_classifier.ipynb`](run_linear_classifier.ipynb), [`run_distilbert_classifier.ipynb`](run_distilbert_classifier.ipynb)
- **Pipeline scripts:** [`run_linear_classifier.py`](run_linear_classifier.py), [`run_distilbert_classifier.py`](run_distilbert_classifier.py)
- **Model outputs:** [`results/result_linear_classification.csv`](results/result_linear_classification.csv), [`results/result_distilbert_classification.csv`](results/result_distilbert_classification.csv)
- **Final submission:** [`result.csv`](result.csv)

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Execute a pipeline:
   - Linear baseline: `python run_linear_classifier.py` (writes predictions to `results/result_linear_classification.csv`)
   - DistilBERT: `python run_distilbert_classifier.py` (writes predictions to `results/result_distilbert_classification.csv`)

## Experiment tracking
Visualize DistilBERT training and cross-validation logs with TensorBoard:

```bash
tensorboard --logdir_spec="cv:results/distilbert_cv,final:results/distilbert_final/logs"
```

## Additional notes
- Place `train.csv` and `test.csv` in the `data/` directory before running the pipelines.
- Results are reproducible with the provided random seeds and configurations embedded in the pipeline modules.
- Consider reviewing the notebooks before running the scripts to understand feature engineering decisions, evaluation splits, and how to adapt the pipelines for new experiments.
