"""
TITANIC SURVIVAL PREDICTION
============================
A beginner ML classification project covering the full pipeline:
data cleaning -> feature preparation -> model training -> evaluation.

WHY THIS PROJECT:
The dataset is structured (tabular) data with realistic messiness:
missing values, mixed data types, and a text field (Name) that needs
feature engineering. It mirrors the "clean data, prepare features,
train and evaluate a model" workflow described in most AI/ML Engineer
job postings.

Run with: python3 project.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it saves plots without needing a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

pd.set_option("display.width", 120)

# -----------------------------------------------------------------
# STEP 1: LOAD THE DATA
# -----------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv("data/titanic.csv")
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.\n")
print("First 3 rows:")
print(df.head(3))
print("\nMissing values per column:")
print(df.isnull().sum())

# -----------------------------------------------------------------
# STEP 2: DATA CLEANING
# -----------------------------------------------------------------
# Real-world datasets are almost never complete. Here we have missing
# Age, Cabin, and Embarked values. Each gets a different treatment
# because "missing" means something different for each column.
print("\n" + "=" * 60)
print("STEP 2: Cleaning data")
print("=" * 60)

# Age: missing for ~20% of passengers. Fill with the median AGE
# *within each passenger class*, since 1st class passengers skewed
# older than 3rd class -- a single overall median would be less accurate.
df["Age"] = df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))

# Embarked: only 2 missing values. Fill with the most common port.
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin: missing for ~77% of passengers -- too sparse to reliably impute.
# Instead of dropping it, we turn "do we know the cabin?" into a
# feature itself, since knowing the cabin correlates with ticket class.
df["HasCabin"] = df["Cabin"].notnull().astype(int)

print("Missing values after cleaning:")
print(df[["Age", "Embarked", "Cabin"]].isnull().sum())
print(f"(Cabin itself is dropped later -- HasCabin flag keeps its signal)")

# -----------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING / PREPARATION
# -----------------------------------------------------------------
# Raw columns aren't always the most useful signal for a model.
# Here we derive new features that carry more predictive information.
print("\n" + "=" * 60)
print("STEP 3: Feature engineering")
print("=" * 60)

# FamilySize: siblings/spouses + parents/children + self.
# Being completely alone or in a very large family both hurt survival odds.
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Title: extracted from the Name field (e.g. "Mr.", "Mrs.", "Master.").
# Title captures age, gender, and social status more compactly than
# the raw name text, which a model can't use directly.
df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
# Collapse rare titles into a single "Rare" bucket so the model isn't
# trying to learn from titles that appear only once or twice.
common_titles = ["Mr", "Mrs", "Miss", "Master"]
df["Title"] = df["Title"].apply(lambda t: t if t in common_titles else "Rare")

# Encode categorical columns as numbers (models need numeric input).
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)

print("New/derived columns: FamilySize, IsAlone, HasCabin, Title_*, Embarked_*")
print(f"Dataset shape after feature engineering: {df.shape}")

# -----------------------------------------------------------------
# STEP 4: SELECT FEATURES AND SPLIT DATA
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Preparing train/test split")
print("=" * 60)

# Drop columns that don't help prediction: identifiers and raw text
# fields we've already extracted signal from.
drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
X = df.drop(columns=drop_cols)
y = df["Survived"]

feature_names = X.columns.tolist()
print(f"Features used ({len(feature_names)}): {feature_names}")

# 80/20 train/test split. stratify=y keeps the survival ratio
# consistent between train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training rows: {len(X_train)}  |  Test rows: {len(X_test)}")

# Scale numeric features so no single feature (like Fare, which ranges
# up to 500+) dominates a model just because of its raw magnitude.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# STEP 5: TRAIN MODELS
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Training models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)  # tree models don't need scaling
        preds = model.predict(X_test)

    results[name] = {
        "model": model,
        "preds": preds,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    print(f"\n{name} trained.")

# -----------------------------------------------------------------
# STEP 6: EVALUATE
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Evaluation")
print("=" * 60)

for name, r in results.items():
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {r['accuracy']:.3f}")
    print(f"Precision: {r['precision']:.3f}")
    print(f"Recall:    {r['recall']:.3f}")
    print(f"F1 Score:  {r['f1']:.3f}")

best_name = max(results, key=lambda n: results[n]["accuracy"])
print(f"\nBest model: {best_name} (accuracy {results[best_name]['accuracy']:.3f})")

# Confusion matrix for the best model
cm = confusion_matrix(y_test, results[best_name]["preds"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# Feature importance (Random Forest only -- it exposes this directly)
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=True).tail(10)

plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#1F3864")
plt.title("Top 10 Most Important Features (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

print("\nSaved outputs/confusion_matrix.png and outputs/feature_importance.png")

# Save a results summary to file for the README
with open("outputs/results_summary.txt", "w") as f:
    f.write("TITANIC SURVIVAL PREDICTION - RESULTS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    for name, r in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Accuracy:  {r['accuracy']:.3f}\n")
        f.write(f"  Precision: {r['precision']:.3f}\n")
        f.write(f"  Recall:    {r['recall']:.3f}\n")
        f.write(f"  F1 Score:  {r['f1']:.3f}\n\n")
    f.write(f"Best model: {best_name}\n\n")
    f.write("Top 5 most important features (Random Forest):\n")
    for feat, imp in importances.sort_values(ascending=False).head(5).items():
        f.write(f"  {feat}: {imp:.3f}\n")

print("\nDone. See outputs/results_summary.txt for the full report.")
