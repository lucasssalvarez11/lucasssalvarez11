from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator

from helper_functions import (
    ensure_dirs,
    run_extraction,
    run_preprocessing,
    run_drift_detection,
    run_optuna_tuning,
    train_model,
)

# ====================================================
# WRAPPERS / CALLABLES PARA LAS TASKS
# ====================================================

def drift_branch_callable(**kwargs):
    """
    1) Corre la detección de drift.
    2) Imprime el resultado.
    3) Devuelve el task_id al que debe ir el DAG:
       - "train_xgboost_with_optuna" si hay drift
       - "skip_training" si NO hay drift
    """
    result = run_drift_detection(threshold=0.1)
    print(f"[DRIFT] Resultado: {result}")

    if result.get("drift_detected", False):
        print("[DRIFT] Se detectó drift, se realizará reentrenamiento.")
        return "train_xgboost_with_optuna"
    else:
        print("[DRIFT] No se detecta drift, se omite el reentrenamiento.")
        return "skip_training"


def train_with_optuna_callable():
    """
    Ejecuta Optuna para encontrar los mejores hiperparámetros
    y luego entrena el modelo XGBoost final con esos parámetros.
    """
    print("[TRAIN] Iniciando tuning con Optuna...")
    best_params = run_optuna_tuning(n_trials=30)
    print(f"[TRAIN] Mejores hiperparámetros encontrados: {best_params}")

    print("[TRAIN] Entrenando modelo XGBoost con mejores hiperparámetros...")
    metrics = train_model(best_params=best_params)
    print(f"[TRAIN] Métricas finales en test: {metrics}")


# ====================================================
# DEFINICIÓN DEL DAG
# ====================================================

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="weekly_ml_pipeline",
    default_args=default_args,
    description="Pipeline semanal de ML: extracción, preprocesamiento, drift, Optuna + XGBoost",
    schedule_interval="@weekly",   # o None si quieres solo lanzamiento manual
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "xgboost", "optuna", "drift"],
) as dag:

    start = EmptyOperator(task_id="start")

    ensure_dirs_task = PythonOperator(
        task_id="ensure_dirs",
        python_callable=ensure_dirs,
    )

    extract_data_task = PythonOperator(
        task_id="extract_data",
        python_callable=run_extraction,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocessing,
    )

    # SI HAY DRIFT >> RETRAIN
    drift_branch = BranchPythonOperator(
        task_id="check_drift_and_branch",
        python_callable=drift_branch_callable,
    )

    train_task = PythonOperator(
        task_id="train_xgboost_with_optuna",
        python_callable=train_with_optuna_callable,
    )

    skip_training = EmptyOperator(
        task_id="skip_training",
    )

    end = EmptyOperator(task_id="end")

    # Encadenar tareas
    start >> ensure_dirs_task >> extract_data_task >> preprocess_task >> drift_branch

    # Branch: según el resultado de drift, el DAG sigue por un camino u otro
    drift_branch >> train_task
    drift_branch >> skip_training

    # Ambos caminos se unen al final
    [train_task, skip_training] >> end

