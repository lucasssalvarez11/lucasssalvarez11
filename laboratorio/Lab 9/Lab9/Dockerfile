# ---- Imagen base recomendada ----
    FROM python:3.10-slim

    # ---- Variables de entorno básicas ----
    ENV AIRFLOW_HOME=/opt/airflow \
        PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PYTHONUNBUFFERED=1 \
        AIRFLOW__CORE__LOAD_EXAMPLES=False
    
    # ---- Paquetes del sistema (incluye curl) ----
    RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR ${AIRFLOW_HOME}
    
    # ---- Instalar Airflow con constraints ----
    ARG AIRFLOW_VERSION=2.9.2
    RUN python -m pip install --no-cache-dir \
        "apache-airflow==${AIRFLOW_VERSION}" \
        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.10.txt"
    
    # ---- Dependencias de tu pipeline (ML + UI) ----
    RUN python -m pip install --no-cache-dir \
        pandas scikit-learn joblib gradio
    
    # ---- Estructura esperada por Airflow ----
    RUN mkdir -p dags plugins logs artifacts
    # Copia tus scripts del DAG y funciones
    COPY dags/ ${AIRFLOW_HOME}/dags/
    
    # ---- Exponer webserver ----
    EXPOSE 8080
    
    # ---- Comando de arranque (webserver + scheduler + usuario admin) ----
    CMD ["airflow", "standalone"]
    