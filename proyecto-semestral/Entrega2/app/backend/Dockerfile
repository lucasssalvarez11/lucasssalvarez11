FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el código del backend
COPY main.py /app/main.py

# OJO: los modelos se montan como volumen (ver docker-compose.yml)
# en /app/models, y se puede sobreescribir MODEL_PATH/PREPROCESSOR_PATH
# vía variables de entorno.

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
