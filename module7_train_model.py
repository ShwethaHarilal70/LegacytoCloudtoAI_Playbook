"""
Module 7: Train the XGBoost Fraud Detection Model
===================================================
Run this script ONCE to generate synthetic training data
and train your ML model. It saves:
  - fraud_model.pkl   (the trained XGBoost model)
  - scaler.pkl        (the feature scaler)
  - training_data.csv (so students can inspect the data)

Usage:
    pip install xgboost scikit-learn pandas
    python module7_train_model.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

# ─── STEP 1: Generate synthetic fraud data ───────────────────────────────────
print("Step 1: Generating synthetic training data...")

np.random.seed(42)
N = 2000  # total transactions

data = []
for i in range(N):
    is_fraud = np.random.random() < 0.2  # 20% fraud rate

    if is_fraud:
        is_vpn        = np.random.random() < 0.75   # fraud often uses VPN
        is_tor        = np.random.random() < 0.30   # fraud sometimes uses TOR
        is_new_device = np.random.random() < 0.70   # fraud often uses new device
        hour          = np.random.choice([0,1,2,3,4,23], p=[.2,.2,.2,.15,.15,.1])
        amount        = np.random.uniform(500, 5000) # fraud = higher amounts
        failed_logins = np.random.randint(2, 8)      # fraud = more failed logins
    else:
        is_vpn        = np.random.random() < 0.05
        is_tor        = np.random.random() < 0.01
        is_new_device = np.random.random() < 0.15
        hour          = np.random.randint(8, 22)     # legit = business hours
        amount        = np.random.uniform(10, 500)
        failed_logins = np.random.randint(0, 2)

    data.append({
        "is_vpn":        int(is_vpn),
        "is_tor":        int(is_tor),
        "is_new_device": int(is_new_device),
        "hour_of_day":   hour,
        "amount":        round(amount, 2),
        "failed_logins": failed_logins,
        "is_fraud":      int(is_fraud)
    })

df = pd.DataFrame(data)
df.to_csv("training_data.csv", index=False)
print(f"   Generated {N} transactions ({df['is_fraud'].sum()} fraud, {(~df['is_fraud'].astype(bool)).sum()} legitimate)")

# ─── STEP 2: Prepare features ────────────────────────────────────────────────
print("\nStep 2: Preparing features...")

FEATURE_COLS = ["is_vpn", "is_tor", "is_new_device", "hour_of_day", "amount", "failed_logins"]
X = df[FEATURE_COLS]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── STEP 3: Train XGBoost model ─────────────────────────────────────────────
print("\nStep 3: Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=4,   # handles class imbalance (4x more legit than fraud)
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ─── STEP 4: Evaluate ────────────────────────────────────────────────────────
print("\nStep 4: Model evaluation:")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# ─── STEP 5: Save model and scaler ───────────────────────────────────────────
print("Step 5: Saving model and scaler...")

with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nDone! Files saved:")
print("   fraud_model.pkl   - trained XGBoost model")
print("   scaler.pkl        - feature scaler")
print("   training_data.csv - synthetic training data")
print("\nNext: run main.py to start the API with ML scoring enabled.")
