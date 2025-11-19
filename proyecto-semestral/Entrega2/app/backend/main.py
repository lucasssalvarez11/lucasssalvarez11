from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from pathlib import Path
from typing import Optional

app = FastAPI(
    title="SodAI Drinks Backend",
    description="API de predicciones para SodAI Drinks 🥤",
    version="1.0.0",
)

# ----------------------------------------------------
# RUTAS POR DEFECTO A LOS MODELOS (usando entrega2/airflow)
# ----------------------------------------------------
# main.py está en: entrega2/app/backend/main.py
# parents[0] = backend
# parents[1] = app
# parents[2] = entrega2   ← raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

AIRFLOW_MODELS_DIR = PROJECT_ROOT / "airflow" / "data" / "models"

DEFAULT_MODEL_PATH = AIRFLOW_MODELS_DIR / "xgb_model.pkl"
DEFAULT_PREPROCESSOR_PATH = AIRFLOW_MODELS_DIR / "preprocessor.pkl"

# Permitimos override vía variables de entorno si quieres cambiarlo en Docker
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", str(DEFAULT_PREPROCESSOR_PATH))

print(f"🔍 Buscando modelo en: {MODEL_PATH}")
print(f"🔍 Buscando preprocesador en: {PREPROCESSOR_PATH}")

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_cols = model_bundle.get("feature_cols", None)
    print("✅ Modelo cargado correctamente")
except Exception as e:
    model = None
    feature_cols = None
    print(f"⚠️ No se pudo cargar el modelo desde {MODEL_PATH}: {e}")

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    num_features = list(preprocessor.transformers_[0][2])
    cat_features = list(preprocessor.transformers_[1][2])
    INPUT_COLS = num_features + cat_features
    print("✅ Preprocesador cargado correctamente")
except Exception as e:
    preprocessor = None
    INPUT_COLS = []
    print(f"⚠️ No se pudo cargar el preprocesador desde {PREPROCESSOR_PATH}: {e}")

