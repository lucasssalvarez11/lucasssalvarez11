# dags/hiring_dynamic_adapted.py
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import os

# Funciones del pipeline dinámico
from hiring_dynamic_functions import (
    create_folders,      # debe devolver por XCom la ruta del run (igual que en el primer DAG)
    load_and_merge,      # firma esperada: load_and_merge(base_dir=<ruta del run>)
    split_data,          # firma esperada: split_data(base_dir=<ruta del run>, test_size=?, random_state=?)
    train_model,         # firma esperada: train_model(base_dir=<ruta del run>, model=..., model_name=...)
    evaluate_models      # firma esperada: evaluate_models(base_dir=<ruta del run>)
)

# Modelos a entrenar
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

default_args = {
    "owner": "matias",
    "start_date": datetime(2024, 10, 1),
    "retries": 0,
}

with DAG(
    dag_id="hiring_dynamic",
    default_args=default_args,
    # puedes mantener schedule_interval si prefieres; en Airflow 2.x se recomienda 'schedule'
    schedule_interval="0 15 5 * *",  # día 5 de cada mes a las 15:00
    catchup=True,
    params={
        # URLs por defecto (puedes sobreescribir en dag_run.conf)
        "data_url_1": "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv",
        "data_url_2": "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv",
        # base_dir por defecto (puedes definir Variable 'hiring_base_dir' también)
        "base_dir": "/opt/airflow/artifacts",
        # parámetros de split como en el primer DAG
        "test_size": 0.20,
        "random_state": 42,
    },
    tags=["lab9", "mlops", "dynamic"],
) as dag:

    # 0) Inicio
    start = EmptyOperator(task_id="inicio_pipeline")

    # 1) Crear carpeta de ejecución (como el primer DAG)
    #    create_folders debe crear <base_dir>/<YYYY-MM-DD>/ y devolver esa ruta por XCom
    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={
            "base_dir": '{{ dag_run.conf.get("base_dir", var.value.hiring_base_dir | default(params.base_dir)) }}',
        },
    )

    # 2) Lógica de branching por fecha (igual que el segundo DAG, pero retornando task_ids compatibles)
    def choose_branch(**kwargs):
        execution_date = datetime.strptime(kwargs["ds"], "%Y-%m-%d")
        cutoff = datetime(2024, 11, 1)
        return "download_data1" if execution_date < cutoff else "download_data1_and_2"

    branching = BranchPythonOperator(
        task_id="branching_download_logic",
        python_callable=choose_branch,
    )

    # 3) Descargas usando la ruta del run desde XCom y URLs desde dag_run.conf/params
    def download_data1(**kwargs):
        ti = kwargs["ti"]
        run_dir = ti.xcom_pull(task_ids="create_folders")  # p.ej. /opt/airflow/artifacts/2025-11-04
        raw_dir = os.path.join(run_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # URL prioriza dag_run.conf, luego params
        url_1 = kwargs["dag_run"].conf.get("data_url_1") if kwargs.get("dag_run") else None
        if not url_1:
            url_1 = kwargs["params"]["data_url_1"]

        os.system(f"curl -s -L -o {os.path.join(raw_dir, 'data_1.csv')} '{url_1}'")
        print("data_1.csv descargado correctamente en:", raw_dir)

    def download_data1_and_2(**kwargs):
        ti = kwargs["ti"]
        run_dir = ti.xcom_pull(task_ids="create_folders")
        raw_dir = os.path.join(run_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # URLs priorizan dag_run.conf, luego params
        url_1 = kwargs["dag_run"].conf.get("data_url_1") if kwargs.get("dag_run") else None
        url_2 = kwargs["dag_run"].conf.get("data_url_2") if kwargs.get("dag_run") else None
        if not url_1:
            url_1 = kwargs["params"]["data_url_1"]
        if not url_2:
            url_2 = kwargs["params"]["data_url_2"]

        dl_list = [("data_1.csv", url_1), ("data_2.csv", url_2)]
        for fname, url in dl_list:
            os.system(f"curl -s -L -o {os.path.join(raw_dir, fname)} '{url}'")
            print(f"{fname} descargado correctamente en: {raw_dir}")

    download_data1_task = PythonOperator(
        task_id="download_data1",
        python_callable=download_data1,
    )

    download_data1_and_2_task = PythonOperator(
        task_id="download_data1_and_2",
        python_callable=download_data1_and_2,
    )

    # 4) Carga y merge usando base_dir = run_dir (nombre de parámetro alineado al primer DAG)
    def run_load_and_merge(**kwargs):
        ti = kwargs["ti"]
        run_dir = ti.xcom_pull(task_ids="create_folders")
        # Adaptado: pasar base_dir (no base_path)
        load_and_merge(base_dir=run_dir)

    load_and_merge_task = PythonOperator(
        task_id="load_and_merge",
        python_callable=run_load_and_merge,
        trigger_rule=TriggerRule.ONE_SUCCESS,  # ejecuta si descargó 1 o 2 archivos
    )

    # 5) Split con mismos nombres de parámetros (base_dir, test_size, random_state)
    def run_split(**kwargs):
        ti = kwargs["ti"]
        run_dir = ti.xcom_pull(task_ids="create_folders")
        test_size = kwargs["dag_run"].conf.get("test_size") if kwargs.get("dag_run") else None
        random_state = kwargs["dag_run"].conf.get("random_state") if kwargs.get("dag_run") else None
        if test_size is None:
            test_size = kwargs["params"]["test_size"]
        if random_state is None:
            random_state = kwargs["params"]["random_state"]
        split_data(base_dir=run_dir, test_size=float(test_size), random_state=int(random_state))

    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=run_split,
    )

    # 6) Entrenamiento de 3 modelos (pasando base_dir como en el primer DAG)
    def train_rf(**kwargs):
        ti = kwargs["ti"]
        run_dir = ti.xcom_pull(task_ids="create_folders")
        model = RandomForestClassifier(n_estimators=150, random_state=6)
        train_model(base_dir=run_dir, model=model, model_name="RandomForest")

    def train_gb(**kwargs):
        ti = kwargs["ti"]
        run_dir = ti.xcom_pull(task_ids="create_folders")
        model = GradientBoostingClassifier(random_state=6)
        train_model(base_dir=run_dir, model=model, model_name="GradientBoosting")

    def train_lr(**kwargs):
        ti = kwargs["ti"]
        run_dir = ti.xcom_pull(task_ids="create_folders")
        model = LogisticRegression(max_iter=500)
        train_model(base_dir=run_dir, model=model, model_name="LogisticRegression")

    train_rf_task = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_rf,
    )

    train_gb_task = PythonOperator(
        task_id="train_gradient_boosting",
        python_callable=train_gb,
    )

    train_lr_task = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_lr,
    )

    # 7) Evaluación comparativa (pasando base_dir)
    def run_evaluate(**kwargs):
        ti = kwargs["ti"]
        run_dir = ti.xcom_pull(task_ids="create_folders")
        evaluate_models(base_dir=run_dir)

    evaluate_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=run_evaluate,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Orquestación
    start >> create_folders_task >> branching
    branching >> [download_data1_task, download_data1_and_2_task] >> load_and_merge_task
    load_and_merge_task >> split_data_task >> [train_rf_task, train_gb_task, train_lr_task] >> evaluate_task
