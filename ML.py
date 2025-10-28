# ============================================================
# 🚗 GENERATIVE AI for Predictive Maintenance in EVs
# ML Model (Classification + Regression) with Evaluation
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression

# ============================================================
# 1️⃣ CLASSIFICATION MODEL – Predict Failure / No Failure
# ============================================================

print("\n==============================")
print("🔹 CLASSIFICATION MODEL")
print("==============================")

# Generate synthetic binary classification data (replace with your dataset)
# df = pd.read_csv("C:\Users\Jeyabala\Downloads\EV_d.csv")  # Example: your dataset file
# X = df.drop(columns=["failure"])
# y = df["failure"]
X_cls, y_cls = make_classification(
    n_samples=5000,
    n_features=30,
    n_informative=8,
    weights=[0.95, 0.05],
    random_state=42
)

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42
)

# Train model
clf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)
clf.fit(X_train_c, y_train_c)

# Predictions
y_pred_c = clf.predict(X_test_c)
y_proba_c = clf.predict_proba(X_test_c)[:, 1]

# Evaluation metrics
acc = accuracy_score(y_test_c, y_pred_c)
prec = precision_score(y_test_c, y_pred_c, zero_division=0)
rec = recall_score(y_test_c, y_pred_c, zero_division=0)
f1 = f1_score(y_test_c, y_pred_c, zero_division=0)
roc = roc_auc_score(y_test_c, y_proba_c)
cm = confusion_matrix(y_test_c, y_pred_c)

# Display metrics
print(f"✅ Accuracy       : {acc:.4f}")
print(f"✅ Precision      : {prec:.4f}")
print(f"✅ Recall         : {rec:.4f}")
print(f"✅ F1 Score       : {f1:.4f}")
print(f"✅ ROC-AUC        : {roc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nDetailed Report:\n", classification_report(y_test_c, y_pred_c, zero_division=0))

# Interpretation
if acc > 0.8:
    print("🌟 Model performs well with accuracy above 80%.")
else:
    print("⚠️ Accuracy below 80% — consider tuning model or more data balancing.")

# ============================================================
# 2️⃣ REGRESSION MODEL – Predict Remaining Useful Life (RUL)
# ============================================================

print("\n==============================")
print("🔹 REGRESSION MODEL")
print("==============================")

# Generate synthetic regression data (replace with your dataset)
# df_rul = pd.read_csv("EV_RUL.csv")
# X_rul = df_rul.drop(columns=["RUL"])
# y_rul = df_rul["RUL"]
X_reg, y_reg = make_regression(
    n_samples=3000, n_features=20, noise=10.0, random_state=42
)

# Split data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train regression model
reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
reg.fit(X_train_r, y_train_r)

# Predictions
y_pred_r = reg.predict(X_test_r)

# Evaluation metrics
mse = mean_squared_error(y_test_r, y_pred_r)
mae = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, y_pred_r)

# Display metrics
print(f"✅ MAE   : {mae:.4f}")
print(f"✅ MSE   : {mse:.4f}")
print(f"✅ RMSE  : {rmse:.4f}")
print(f"✅ R²    : {r2:.4f}")

# Interpretation
if r2 > 0.8:
    print("🌟 Regression model explains over 80% of variance — strong performance.")
else:
    print("⚠️ R² below 0.8 — model may need tuning or more features.")

# ============================================================
# 3️⃣ FINAL CONCLUSION
# ============================================================
print("\n==============================")
print("🔸 SUMMARY")
print("==============================")
print(f"Classification Accuracy : {acc:.2%}")
print(f"Classification Recall   : {rec:.2%}")
print(f"Regression MAE          : {mae:.2f}")
print(f"Regression R² Score     : {r2:.2f}")

if acc > 0.8 and r2 > 0.8:
    print("\n✅ Both models perform well. Suitable for deployment in EV Predictive Maintenance system.")
else:
    print("\n⚙️ Consider tuning hyperparameters or adding features for better accuracy and generalization.")
