
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import os

from qa_engine import DataQABot

app = FastAPI(
    title="SodAI Drinks – Data Chatbot API",
    description="Chatbot conversacional para responder preguntas sobre el dataset de SodAI Drinks 🥤",
    version="1.0.0",
)

# ----------------------------------------------------
# Helpers para paths por defecto
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
              clientes.parquet
              productos.parquet
        app/
          llm/
            backend/
              main.py
    """
    project_root = Path(__file__).resolve().parents[3]  # entrega2
    raw_dir = project_root / "airflow" / "data" / "raw"
    trans_path = raw_dir / "transacciones.parquet"
    clientes_path = raw_dir / "clientes.parquet"
    productos_path = raw_dir / "productos.parquet"
    return trans_path, clientes_path, productos_path


# Permitir override por variables de entorno (útil en Docker)
TRANSACCIONES_PATH = os.getenv("TRANSACCIONES_PATH")
CLIENTES_PATH = os.getenv("CLIENTES_PATH")
PRODUCTOS_PATH = os.getenv("PRODUCTOS_PATH")

if not (TRANSACCIONES_PATH and CLIENTES_PATH and PRODUCTOS_PATH):
    try:
        t_default, c_default, p_default = _default_data_paths()
        TRANSACCIONES_PATH = TRANSACCIONES_PATH or str(t_default)
        CLIENTES_PATH = CLIENTES_PATH or str(c_default)
        PRODUCTOS_PATH = PRODUCTOS_PATH or str(p_default)
    except Exception as e:
        print(f" No se pudieron construir paths por defecto: {e}")

print(f" TRANSACCIONES_PATH = {TRANSACCIONES_PATH}")
print(f" CLIENTES_PATH      = {CLIENTES_PATH}")
print(f" PRODUCTOS_PATH     = {PRODUCTOS_PATH}")

bot = None
try:
    bot = DataQABot(
        transacciones_path=TRANSACCIONES_PATH,
        clientes_path=CLIENTES_PATH,
        productos_path=PRODUCTOS_PATH,
    )
    print(" Chatbot de datos cargado correctamente")
except Exception as e:
    print(f" Error cargando el chatbot de datos: {e}")


# ----------------------------------------------------
# Esquemas Pydantic
# ----------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


# ----------------------------------------------------
# Rutas
# ----------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "bot_loaded": bot is not None,
        "transacciones_path": TRANSACCIONES_PATH,
        "clientes_path": CLIENTES_PATH,
        "productos_path": PRODUCTOS_PATH,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if bot is None:
        return ChatResponse(
            answer="El chatbot no está inicializado correctamente. Revisa la configuración de los datos."
        )

    question = req.message
    answer = bot.answer(question)
    return ChatResponse(answer=answer)

