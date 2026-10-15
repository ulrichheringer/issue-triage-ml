# issue-triage-ml

Finds likely duplicate issues with TF-IDF and estimates resolution time with Ridge regression.

## Data

The project generates deterministic synthetic data locally in `pipeline.py`. It is not a public benchmark and contains no real user issues. The evaluation groups related duplicate pairs and resolution templates so examples based on the same case do not appear in both training and test sets.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib --input '{"issue_a":"login timeout after password reset","issue_b":"sign in request times out","issue":"query times out under load"}'
```

## Evaluation results

With seed 42, duplicate detection reached 0.6231 accuracy and 0.4842 F1. Resolution-time estimates had an MAE of 2.9106 hours, an RMSE of 3.4901 hours, and an R² of 0.0202. The low R² on a template-grouped holdout shows that this baseline does not generalize well to unseen descriptions. These results apply only to the synthetic data.

## Limitations

The vocabulary and data distribution are small and artificial. Real use would require appropriately licensed historical issues and evaluation by time or project to reduce leakage between related issues.
