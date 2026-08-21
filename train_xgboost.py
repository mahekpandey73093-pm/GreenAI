import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "processed" / "greenai_features.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "green_cover_xgboost.pkl"


# ==========================================
# LOAD DATA
# ==========================================

print("\n==========================================")
print("        GreenAI XGBoost Training")
print("==========================================")

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==========================================
# FEATURES & TARGET
# ==========================================

features = [
    "ndvi_mean",
    "ndvi_min",
    "ndvi_max",
    "builtup_percentage",
    "water_percentage",
    "temperature",
    "rainfall"
]

target = "green_cover_percentage"


X = df[features]
y = df[target]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# XGBOOST MODEL
# ==========================================

print("\nTraining XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n==========================================")
print("             MODEL RESULTS")
print("==========================================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, MODEL_PATH)

print("\n==========================================")
print("Model saved successfully!")
print("Location:", MODEL_PATH)
print("==========================================")


# ==========================================
# SAMPLE PREDICTION
# ==========================================

sample = X_test.iloc[[0]]

prediction = model.predict(sample)[0]

print("\nSample Prediction")
print("---------------------------")

print("Input:")
print(sample)

print(
    f"\nPredicted Green Cover: {prediction:.2f}%"
)

print(
    f"Actual Green Cover: {y_test.iloc[0]:.2f}%"
)