FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# En Docker montaremos /app/data con los datos de airflow
ENV TRANSACCIONES_PATH=/app/data/raw/transacciones.parquet
ENV CLIENTES_PATH=/app/data/raw/clientes.parquet
ENV PRODUCTOS_PATH=/app/data/raw/productos.parquet

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
