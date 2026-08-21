import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "dataset/processed/greenai_features.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "green_cover_ann.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "green_cover_ann_scaler.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)


print("\n==========================================")
print("        GreenAI ANN Training")
print("==========================================\n")


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "year",
    "ndvi_mean",
    "ndvi_min",
    "ndvi_max",
    "builtup_percentage",
    "water_percentage",
    "temperature",
    "rainfall"
]

TARGET = "green_cover_percentage"


X = df[FEATURES]
y = df[TARGET]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================================================
# FEATURE SCALING
# =========================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================================================
# BUILD ANN
# =========================================================

print("\nBuilding ANN model...")

model = Sequential([
    
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="linear"
    )
])


# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\nANN Architecture:")
model.summary()


# =========================================================
# EARLY STOPPING
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# =========================================================
# TRAIN
# =========================================================

print("\n==========================================")
print("             TRAINING ANN")
print("==========================================\n")

history = model.fit(
    X_train_scaled,
    y_train,

    validation_split=0.20,

    epochs=100,

    batch_size=32,

    callbacks=[early_stopping],

    verbose=1
)


# =========================================================
# PREDICTION
# =========================================================

print("\nEvaluating model...")

predictions = model.predict(
    X_test_scaled,
    verbose=0
).flatten()


# =========================================================
# MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# =========================================================
# RESULTS
# =========================================================

print("\n==========================================")
print("             ANN RESULTS")
print("==========================================")

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R²   : {r2:.4f}")


# =========================================================
# SAVE MODEL
# =========================================================

model.save(
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print("\n==========================================")
print("       MODEL SAVED SUCCESSFULLY")
print("==========================================")

print(
    "ANN Model :",
    os.path.abspath(MODEL_PATH)
)

print(
    "Scaler    :",
    os.path.abspath(SCALER_PATH)
)


# =========================================================
# SAMPLE PREDICTION
# =========================================================

sample = X_test.iloc[[0]]

sample_scaled = scaler.transform(
    sample
)

sample_prediction = model.predict(
    sample_scaled,
    verbose=0
)[0][0]

actual_value = y_test.iloc[0]


print("\n==========================================")
print("          SAMPLE PREDICTION")
print("==========================================")

print("\nInput:")

print(sample.to_string(index=False))

print(
    f"\nPredicted Green Cover : "
    f"{sample_prediction:.2f}%"
)

print(
    f"Actual Green Cover    : "
    f"{actual_value:.2f}%"
)

print("\n==========================================")
print("             ANN COMPLETE")
print("==========================================\n")