from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
import io
import os


# ==========================================
# GreenAI API
# ==========================================

app = FastAPI(
    title="GreenAI API",
    description="Green Cover Prediction and Green/Urban Image Classification API",
    version="1.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# MODEL PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANN_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "green_cover_ann.keras"
)

ANN_SCALER_PATH = os.path.join(
    BASE_DIR, "models", "green_cover_ann_scaler.pkl"
)

CNN_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "green_vs_urban_cnn.keras"
)


# ==========================================
# LOAD MODELS
# ==========================================

print("Loading GreenAI models...")

ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)

ann_scaler = joblib.load(ANN_SCALER_PATH)

cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

print("ANN model loaded successfully.")
print("ANN scaler loaded successfully.")
print("CNN model loaded successfully.")


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "GreenAI API is running successfully",
        "models": [
            "ANN - Green Cover Prediction",
            "CNN - Green vs Urban Classification"
        ]
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "ann_model": "loaded",
        "cnn_model": "loaded"
    }


# ==========================================
# CNN IMAGE PREDICTION
# ==========================================

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Resize according to CNN training
        image = image.resize((128, 128))

        # Convert image to numpy
        image_array = np.array(image)

        # Normalize
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Prediction
        prediction = cnn_model.predict(
            image_array,
            verbose=0
        )[0][0]

        # CNN class mapping:
        # green = 0
        # urban = 1

        if prediction >= 0.5:

            predicted_class = "urban"
            confidence = float(prediction)

        else:

            predicted_class = "green"
            confidence = float(1 - prediction)

        return {

            "success": True,

            "filename": file.filename,

            "prediction": predicted_class,

            "confidence": round(
                confidence * 100,
                2
            ),

            "message": (
                "Urban area detected"
                if predicted_class == "urban"
                else "Green area detected"
            )
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ==========================================
# ANN GREEN COVER PREDICTION
# ==========================================

@app.post("/predict-green-cover")
def predict_green_cover(

    year: int,
    ndvi_mean: float,
    ndvi_min: float,
    ndvi_max: float,
    builtup_percentage: float,
    water_percentage: float,
    temperature: float,
    rainfall: float

):

    try:

        # Feature order used during ANN training
        features = np.array([[
            year,
            ndvi_mean,
            ndvi_min,
            ndvi_max,
            builtup_percentage,
            water_percentage,
            temperature,
            rainfall
        ]])

        # Scale features
        scaled_features = ann_scaler.transform(
            features
        )

        # Prediction
        prediction = ann_model.predict(
            scaled_features,
            verbose=0
        )[0][0]

        # Keep percentage between 0 and 100
        prediction = np.clip(
            prediction,
            0,
            100
        )

        return {

            "success": True,

            "predicted_green_cover_percentage":
                round(float(prediction), 2)

        }

    except Exception as e:

        return {

            "success": False,
            "error": str(e)

        }