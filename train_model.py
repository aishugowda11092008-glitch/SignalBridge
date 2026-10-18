
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model input features
FEATURES = [
    "rainfall_mm",
    "wind_kmh",
    "humidity_pct",
    "river_rise_cm",
    "historical_incidents",
]

# --------------------------------------------------
# 1. GENERATE DEMONSTRATION DATA
# These are synthetic records, NOT real disaster data.
# --------------------------------------------------

rng = np.random.default_rng(42)
n = 3000

df = pd.DataFrame({
    "rainfall_mm": rng.uniform(0, 120, n),
    "wind_kmh": rng.uniform(0, 120, n),
    "humidity_pct": rng.uniform(20, 100, n),
    "river_rise_cm": rng.uniform(0, 150, n),
    "historical_incidents": rng.integers(0, 11, n),
})

# Illustrative scoring rule to generate training labels.
# These thresholds are not validated emergency criteria.
score = (
    (df["rainfall_mm"] >= 50).astype(int)
    + (df["wind_kmh"] >= 60).astype(int)
    + (df["river_rise_cm"] >= 60).astype(int)
    + (df["historical_incidents"] >= 5).astype(int)
)

df["risk_level"] = np.select(
    [score >= 3, score >= 1],
    ["High", "Moderate"],
    default="Low",
)

# Save demonstration dataset
data_path = DATA_DIR / "demo_training_data.csv"
df.to_csv(data_path, index=False)

print("SignalBridge AI Training")
print("=" * 40)
print(f"Dataset saved: {data_path}")
print(f"Total records: {len(df)}")

# --------------------------------------------------
# 2. SPLIT DATA FOR TRAINING AND TESTING
# --------------------------------------------------

X = df[FEATURES]
y = df["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# 3. TRAIN RANDOM FOREST CLASSIFIER
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 4. EVALUATE THE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Demo test accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0,
))

# --------------------------------------------------
# 5. SAVE THE TRAINED MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "risk_model.joblib"

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "data_type": "synthetic_demo_only",
    },
    model_path,
)

print(f"\nModel saved successfully: {model_path}")
print("\nTraining completed!")
print(
    "IMPORTANT: This model uses synthetic data and "
    "must not be used for real disaster warnings."
)
