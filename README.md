# YieldPulse
Predictive Modeling for Agriculture

## Overview

YieldPulse is a machine learning project focused on predicting crop types based on soil and environmental features. The goal is to identify the most informative features for crop classification and build accurate predictive models to assist farmers and agricultural planners.

---

## Task: Identify the Best Predictive Feature for Crop Classification

### Objective

Find the single feature in the dataset that has the **strongest predictive performance** for classifying crop types (`crop`).

### Dataset

The dataset typically includes the following features used to predict the target variable `crop`:

| Feature       | Description                          |
|---------------|--------------------------------------|
| `N`           | Nitrogen content in soil             |
| `P`           | Phosphorus content in soil           |
| `K`           | Potassium content in soil            |
| `temperature` | Average temperature (°C)             |
| `humidity`    | Relative humidity (%)                |
| `ph`          | pH value of the soil                 |
| `rainfall`    | Annual rainfall (mm)                 |

**Target variable:** `crop` — the type of crop suitable for the given conditions.

---

## Methodology

For each feature in the dataset, train a classifier using **only that feature** and evaluate its performance. The feature that yields the highest evaluation score is identified as the best predictive feature.

### Steps

1. Load the dataset.
2. For each individual feature, train a `DecisionTreeClassifier` using only that feature.
3. Evaluate using **accuracy score** via cross-validation (or train/test split).
4. Record the best-performing feature and its score.
5. Store the result in a dictionary called `best_predictive_feature`.

### Code

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv("crop_data.csv")

features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
target = "crop"

X = df[features]
y = df[target]

best_feature = None
best_score = 0.0

for feature in features:
    X_single = X[[feature]]
    clf = DecisionTreeClassifier(random_state=42)
    scores = cross_val_score(clf, X_single, y, cv=5, scoring="accuracy")
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_feature = feature

# Dictionary containing the best predictive feature and its evaluation score
best_predictive_feature = {best_feature: best_score}

print("Best Predictive Feature:", best_predictive_feature)
```

### Result

The variable `best_predictive_feature` is a dictionary where:
- **Key** — the name of the feature with the highest predictive accuracy for `crop`
- **Value** — the mean cross-validated accuracy score for that feature

**Example output:**
```python
best_predictive_feature = {"rainfall": 0.5743}
```

> **Note:** The actual best feature and score will depend on the dataset used. Run the script above with your data to obtain the precise result.

---

## Requirements

```
pandas
scikit-learn
```

Install dependencies:

```bash
pip install pandas scikit-learn
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
