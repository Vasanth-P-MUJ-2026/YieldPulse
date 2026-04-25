# Identifying the Best Soil Feature for Crop Prediction
# -------------------------------------------------------
# A farmer can only afford to measure ONE of the four soil properties:
#   N  – Nitrogen content ratio
#   P  – Phosphorous content ratio
#   K  – Potassium content ratio
#   ph – pH value
#
# We use logistic regression to evaluate each feature individually and
# pick the one that yields the highest weighted F1-score.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# ── 1. Load data ──────────────────────────────────────────────────────────────
crops = pd.read_csv("soil_measures.csv")

# ── 2. Evaluate each soil feature independently ───────────────────────────────
features = ["N", "P", "K", "ph"]
feature_performance = {}

for feature in features:
    X = crops[[feature]]
    y = crops["crop"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=2000)
    log_reg.fit(X_train_scaled, y_train)

    y_pred = log_reg.predict(X_test_scaled)
    score = f1_score(y_test, y_pred, average="weighted")
    feature_performance[feature] = score
    print(f"F1-score for {feature}: {score:.4f}")

# ── 3. Identify the best predictive feature ───────────────────────────────────
best_feature_name = max(feature_performance, key=feature_performance.get)
best_feature_score = feature_performance[best_feature_name]

best_predictive_feature = {best_feature_name: best_feature_score}
print(f"\nBest predictive feature: {best_predictive_feature}")
