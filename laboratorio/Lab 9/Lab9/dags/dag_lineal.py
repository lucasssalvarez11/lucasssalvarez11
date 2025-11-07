# dags/hiring_dag.py
from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Importa las funciones definidas anteriormente en dags/hiring_functions.py
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

# -----------------------
# Configuración del DAG
# -----------------------
with DAG(
    dag_id="hiring_lineal",
    start_date=datetime(2024, 10, 1),
    schedule=None,           # ejecución manual
    catchup=False,           # sin backfill
    params={                 # valor por defecto; puedes reemplazarlo
        "data_url": "https://REEMPLAZA-ESTE-ENLACE/data_1.csv"
    },
    tags=["hiring", "random-forest", "gradio"],
) as dag:

    # 0) Placeholder de inicio
    start = EmptyOperator(task_id="start_pipeline")

    # 1) Crear carpeta de ejecución + subcarpetas raw/splits/models
    #    create_folders devuelve la ruta del run en XCom (p.ej., /opt/airflow/artifacts/2025-11-04)
    create_run_folders = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={
            # Directorio base de artefactos (Variable opcional con fallback)
            "base_dir": '{{ var.value.hiring_base_dir | default("/opt/airflow/artifacts") }}',
        },
    )

    # 2) Descargar data_1.csv en la carpeta raw del run
    #    Usa curl y guarda en <run_dir>/raw/data_1.csv
    download_data = BashOperator(
        task_id="download_data",
        bash_command=(
            "mkdir -p '{{ ti.xcom_pull(task_ids=\"create_folders\") }}/raw' && "
            "curl -L -o '{{ ti.xcom_pull(task_ids=\"create_folders\") }}/raw/data_1.csv' "
            "'{{ dag_run.conf.get(\"data_url\", params.data_url) }}'"
        ),
    )

    # 3) Hold-out (split_data)
    do_split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={
            "base_dir": '{{ var.value.hiring_base_dir | default("/opt/airflow/artifacts") }}',
            "test_size": 0.20,
            "random_state": 42,
        },
    )

    # 4) Preprocesamiento + entrenamiento (RandomForest) y guardado de pipeline.joblib
    train_model = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        op_kwargs={
            "base_dir": '{{ var.value.hiring_base_dir | default("/opt/airflow/artifacts") }}',
            # Puedes pasar rf_params si quieres, ej:
            # "rf_params": {"n_estimators": 400, "random_state": 42, "class_weight": "balanced"}
        },
    )

    # 5) Interfaz Gradio (subir JSON y predecir)
    launch_gradio = PythonOperator(
        task_id="gradio_interface",
        python_callable=gradio_interface,
        op_kwargs={
            "base_dir": '{{ var.value.hiring_base_dir | default("/opt/airflow/artifacts") }}',
        },
    )

    # Orquestación
    start >> create_run_folders >> download_data >> do_split >> train_model >> launch_gradio
