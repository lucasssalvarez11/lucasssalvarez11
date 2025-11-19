from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import os

from recommender import SimpleRecommender

app = FastAPI(
    title="SodAI Drinks – Recommender API",
    description="Sistema de recomendación de productos para clientes de SodAI Drinks 🥤",
    version="1.0.0",
)

# ----------------------------------------------------
# Cargar recomendador al iniciar
# ----------------------------------------------------

def _default_data_paths():
    """
    Paths por defecto para correr SIN Docker.

    Estructura asumida:
      entrega2/
        airflow/
          data/
            raw/
              transacciones.parquet
              productos.parquet
        app/
          recsys/
            backend/
              main.py
    """
    project_root = Path(__file__).resolve().parents[3]  # entrega2
    raw_dir = project_root / "airflow" / "data" / "raw"
    trans_path = raw_dir / "transacciones.parquet"
    prod_path = raw_dir / "productos.parquet"
    return trans_path, prod_path


# Permitir override por variables de entorno (útil en Docker)
TRANSACCIONES_PATH = os.getenv("TRANSACCIONES_PATH")
PRODUCTOS_PATH = os.getenv("PRODUCTOS_PATH")

if not TRANSACCIONES_PATH or not PRODUCTOS_PATH:
    try:
        t_default, p_default = _default_data_paths()
        TRANSACCIONES_PATH = TRANSACCIONES_PATH or str(t_default)
        PRODUCTOS_PATH = PRODUCTOS_PATH or str(p_default)
    except Exception as e:
        print(f"⚠️ No se pudieron construir paths por defecto: {e}")

print(f"🔍 TRANSACCIONES_PATH = {TRANSACCIONES_PATH}")
print(f"🔍 PRODUCTOS_PATH     = {PRODUCTOS_PATH}")

recommender = None
try:
    recommender = SimpleRecommender(
        transacciones_path=TRANSACCIONES_PATH,
        productos_path=PRODUCTOS_PATH,
    )
    print("✅ Recomendador cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando el recomendador: {e}")


# ----------------------------------------------------
# Esquemas de respuesta
# ----------------------------------------------------

class RecommendationItem(BaseModel):
    product_id: int
    brand: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    segment: Optional[str] = None
    package: Optional[str] = None
    size: Optional[float] = None


class RecommendationResponse(BaseModel):
    customer_id: int
    k: int
    recommendations: List[RecommendationItem]


# ----------------------------------------------------
# Rutas
# ----------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "recommender_loaded": recommender is not None,
        "transacciones_path": TRANSACCIONES_PATH,
        "productos_path": PRODUCTOS_PATH,
    }


@app.get("/recommend", response_model=RecommendationResponse)
def recommend(customer_id: int, k: int = 5):
    if recommender is None:
        raise HTTPException(status_code=500, detail="Recomendador no cargado (revisa paths de datos)")

    try:
        recs = recommender.recommend(customer_id=customer_id, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando recomendaciones: {e}")

    # Adaptar dict→modelo pydantic
    items = [RecommendationItem(**r) for r in recs]

    return RecommendationResponse(
        customer_id=customer_id,
        k=k,
        recommendations=items,
    )
