# Titanic Survival Prediction — Classification Project

A complete, working ML pipeline: data cleaning → feature engineering →
model training → evaluation. Built to demonstrate the exact skills listed
in most AI/ML Engineer job postings: working with structured data,
preparing features, training/evaluating models, and interpreting results.

## What this project does

Predicts whether a Titanic passenger survived, using their ticket class,
age, fare, family situation, and other passenger data.

## How to run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 project.py
```

Outputs land in `outputs/`: a confusion matrix, a feature importance
chart, and a results summary.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 83.8% | 80.3% | 76.8% | 78.5% |
| Random Forest | 81.6% | 81.0% | 68.1% | 74.0% |

Logistic Regression performed slightly better here — likely because the
dataset is small (891 rows) and the relationships are fairly linear
(e.g. being female or in 1st class strongly increased survival odds),
which favors a simpler model over a more complex one.

## What each pipeline step does, in plain English

1. **Load data** — read the raw CSV (891 passengers, 12 columns).
2. **Clean data** — handled 3 different kinds of missingness differently:
   - `Age` (20% missing): filled with the median age *within each ticket
     class*, since 1st class skewed older than 3rd class.
   - `Embarked` (2 missing): filled with the most common port.
   - `Cabin` (77% missing): too sparse to fill in reliably, so instead of
     dropping the column entirely, converted it into a yes/no
     "do we know the cabin" flag, which still carries signal.
3. **Feature engineering** — created new columns that carry more
   predictive information than the raw data alone:
   - `FamilySize` / `IsAlone` — family size affected survival odds.
   - `Title` — extracted "Mr / Mrs / Miss / Master / Rare" from the name
     text, which captures age, gender, and social status more usefully
     than the raw name.
   - Converted categories (`Sex`, `Embarked`, `Title`) into numbers,
     since models can't read text directly.
4. **Train/test split** — held out 20% of the data the models never see
   during training, so we can check performance on unseen data.
5. **Train models** — trained two different model types (Logistic
   Regression and Random Forest) to compare approaches.
6. **Evaluate** — measured accuracy, precision, recall, and F1, plus a
   confusion matrix (visualizing false positives/negatives) and a
   feature importance chart (which factors mattered most).


## Files

- `project.py` — full pipeline, heavily commented
- `data/titanic.csv` — source dataset
- `outputs/confusion_matrix.png` — visualizes prediction errors
- `outputs/feature_importance.png` — which features mattered most
- `outputs/results_summary.txt` — metrics in plain text
