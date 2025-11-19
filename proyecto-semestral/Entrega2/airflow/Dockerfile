FROM apache/airflow:2.9.3-python3.11

# Cambiamos al usuario airflow (ya viene creado en la imagen)
USER airflow

# Copiamos requirements de ML al contenedor
COPY requirements.txt /requirements.txt

# Instalamos dependencias adicionales (pandas, sklearn, xgboost, optuna, etc.)
RUN pip install --no-cache-dir -r /requirements.txt

# No copiamos los DAGs ni la data aquí para poder montarlos como volumen
# desde docker-compose (así puedes editar código sin rebuild)
