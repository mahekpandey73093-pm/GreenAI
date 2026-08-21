import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# GreenAI - CNN Training
# ============================================================

print("=" * 55)
print("             GreenAI CNN Training")
print("=" * 55)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "cnn", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "cnn", "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "green_vs_urban_cnn.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

print("\nTrain directory:")
print(TRAIN_DIR)

print("\nTest directory:")
print(TEST_DIR)

# ------------------------------------------------------------
# Check folders
# ------------------------------------------------------------

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Testing folder not found: {TEST_DIR}")

# ------------------------------------------------------------
# Data Augmentation
# ------------------------------------------------------------

print("\nPreparing image generators...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# ------------------------------------------------------------
# Training data
# ------------------------------------------------------------

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=42
)

# ------------------------------------------------------------
# Validation data
# ------------------------------------------------------------

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=42
)

# ------------------------------------------------------------
# Test data
# ------------------------------------------------------------

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass mapping:")
print(train_generator.class_indices)

print("\nTraining images  :", train_generator.samples)
print("Validation images:", validation_generator.samples)
print("Testing images   :", test_generator.samples)

# ------------------------------------------------------------
# Build CNN
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("             CNN Architecture")
print("=" * 55)

model = models.Sequential([
    
    # Input
    layers.Input(shape=(128, 128, 3)),

    # CNN Block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # CNN Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # CNN Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convert feature maps to vector
    layers.Flatten(),

    # Fully connected layers
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Binary output
    layers.Dense(1, activation="sigmoid")
])

model.summary()

# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("             TRAINING CNN")
print("=" * 55)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("             CNN EVALUATION")
print("=" * 55)

test_loss, test_accuracy = model.evaluate(test_generator)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

probabilities = model.predict(test_generator)

predictions = (probabilities >= 0.5).astype(int).ravel()

true_labels = test_generator.classes

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(true_labels, predictions)

precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    true_labels,
    predictions
)

print("\n" + "=" * 55)
print("             CNN RESULTS")
print("=" * 55)

print(f"\nAccuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

class_names = list(test_generator.class_indices.keys())

print(
    classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        zero_division=0
    )
)

# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------

model.save(MODEL_PATH)

print("=" * 55)
print("        CNN MODEL SAVED SUCCESSFULLY")
print("=" * 55)

print("\nModel:")
print(MODEL_PATH)

print("\nCNN training complete!")