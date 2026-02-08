# Kaggle Titanic Project

## Directory Structure
- data/raw: Raw data (train.csv, test.csv, gender_submission.csv)
- data/processed: Processed intermediate data
- notebooks: EDA and experiment notebooks
- src: Reusable code (data, features, models, evaluation, utils)
- scripts: Training and submission scripts
- submissions: Submission files
- reports: Experiment logs and iteration notes
- configs: Config files
- models: Trained artifacts (optional)
- logs: Training logs (optional)

## Quick Start
1. Download the Kaggle data and place it under data/raw
2. Run cross-validation:
   - python scripts/run_train.py
3. Generate a submission file:
   - python scripts/make_submission.py

## Configuration
- configs/base.yaml: Unified feature, model, and CV settings

## Notebook
- notebooks/titanic-baseline-eda-submission.ipynb

## Iteration Notes
- reports/experiment-log.md
