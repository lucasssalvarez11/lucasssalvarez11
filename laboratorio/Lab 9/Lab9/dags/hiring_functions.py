# dags/hiring_functions.py
from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def create_folders(base_dir: Optional[str] = None, **kwargs) -> str:
    """
    Crea una carpeta con nombre = fecha de ejecución y dentro:
        raw/, splits/, models/
    - En Airflow, toma la fecha desde kwargs['ds'] (YYYY-MM-DD).
    - En ejecución local, usa la fecha de hoy.

    Parámetros
    ----------
    base_dir : str | None
        Directorio base donde se creará la carpeta del run. Si None, usa cwd.
    **kwargs :
        Debe contener 'ds' cuando se ejecuta desde un DAG de Airflow.

    Returns
    -------
    str
        Ruta absoluta de la carpeta creada para este run.
    """
    # 1) Resolver fecha de ejecución
    run_date = kwargs.get("ds")  # p.ej. '2025-11-04' en Airflow
    if not run_date:
        run_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    # 2) Resolver directorio base
    root = Path(base_dir) if base_dir else Path.cwd()

    # 3) Crear carpeta del run y subcarpetas
    run_dir = root / run_date
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "splits").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    logger.info(f"Carpetas creadas en: {run_dir.resolve()}")
    return str(run_dir.resolve())


##############################################################

from pathlib import Path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def split_data(base_dir: str | None = None, test_size: float = 0.20, random_state: int = 42, **kwargs) -> dict:
    """
    Lee raw/data_1.csv (dentro de la carpeta del run) y realiza un hold-out estratificado
    en 'HiringDecision' (20% test por defecto). Guarda train.csv y test.csv en splits/.

    Parámetros
    ----------
    base_dir : str | None
        Directorio base donde se creó la carpeta del run (misma lógica que create_folders).
        Si None, usa cwd.
    test_size : float
        Proporción para el conjunto de prueba (default 0.20).
    random_state : int
        Semilla para reproducibilidad (default 42).
    **kwargs :
        Debe contener 'ds' al ejecutarse desde Airflow (YYYY-MM-DD).

    Returns
    -------
    dict
        Rutas de los archivos generados: {"train_path": "...", "test_path": "..."}
    """
    # 1) Resolver fecha de ejecución
    run_date = kwargs.get("ds")
    if not run_date:
        run_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    # 2) Resolver directorios
    root = Path(base_dir) if base_dir else Path.cwd()
    run_dir = root / run_date
    raw_fp = run_dir / "raw" / "data_1.csv"
    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 3) Validaciones
    if not raw_fp.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {raw_fp.resolve()}")

    df = pd.read_csv(raw_fp)
    target = "HiringDecision"
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en data_1.csv. Columnas: {list(df.columns)}")

    # 4) Hold-out estratificado
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target]
    )

    # 5) Guardado
    train_fp = splits_dir / "train.csv"
    test_fp = splits_dir / "test.csv"
    train_df.to_csv(train_fp, index=False)
    test_df.to_csv(test_fp, index=False)

    logger.info(f"Split listo. Train: {train_df.shape}, Test: {test_df.shape}")
    logger.info(f"Archivos guardados en:\n- {train_fp.resolve()}\n- {test_fp.resolve()}")

    return {"train_path": str(train_fp.resolve()), "test_path": str(test_fp.resolve())}

#############################################################################

from pathlib import Path
import pandas as pd
import joblib

from typing import Optional, Dict, Any, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def preprocess_and_train(
    base_dir: Optional[str] = None,
    rf_params: Optional[Dict[str, Any]] = None,
    positive_label: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    - Lee splits/train.csv y splits/test.csv (según fecha de ejecución).
    - Preprocesa con ColumnTransformer.
    - Entrena RandomForest y guarda pipeline en models/pipeline.joblib.
    - Imprime accuracy (test) y f1-score de la clase positiva.
    """
    # 1) Fecha de ejecución y rutas
    run_date = kwargs.get("ds") or pd.Timestamp.today().strftime("%Y-%m-%d")
    root = Path(base_dir) if base_dir else Path.cwd()
    run_dir = root / run_date
    splits_dir = run_dir / "splits"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_fp = splits_dir / "train.csv"
    test_fp  = splits_dir / "test.csv"
    if not train_fp.exists() or not test_fp.exists():
        raise FileNotFoundError("No existen los splits. Ejecuta split_data() antes.")

    # 2) Cargar datos
    target = "HiringDecision"
    train_df = pd.read_csv(train_fp)
    test_df  = pd.read_csv(test_fp)
    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target}' en los splits.")

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].astype(int)
    X_test  = test_df.drop(columns=[target])
    y_test  = test_df[target].astype(int)

    # 3) Columnas (enunciado)
    numeric_features: List[str] = [
        "Age","ExperienceYears","PreviousCompanies","DistanceFromCompany",
        "InterviewScore","SkillScore","PersonalityScore",
    ]
    categorical_features: List[str] = [
        "Gender","EducationLevel","RecruitmentStrategy",
    ]
    numeric_features = [c for c in numeric_features if c in X_train.columns]
    categorical_features = [c for c in categorical_features if c in X_train.columns]

    # 4) Preprocesamiento (imputación; escalado opcional para num; OHE para cat)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # 5) Modelo RF (baseline robusto)
    if rf_params is None:
        rf_params = dict(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight=None,  # usa "balanced" si hay desbalance fuerte
        )
    clf = RandomForestClassifier(**rf_params)

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", clf),
    ])

    # 6) Entrenar y evaluar
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, pos_label=positive_label)

    print(f"Accuracy (test): {acc:.4f}")
    print(f"F1-score clase positiva={positive_label} (test): {f1_pos:.4f}")

    # 7) Guardar pipeline
    model_path = models_dir / "pipeline.joblib"
    joblib.dump(pipe, model_path)

    return {"accuracy": acc, "f1_positive": f1_pos, "model_path": str(model_path.resolve())}


def gradio_interface(base_dir: Optional[str] = None, **kwargs):
    """
    Interfaz Gradio que carga el modelo desde ./<ds>/models/pipeline.joblib
    y predice la contratación. No define helpers externos.
    """
    import gradio as gr

    run_date = kwargs.get("ds") or pd.Timestamp.today().strftime("%Y-%m-%d")
    root = Path(base_dir) if base_dir else Path.cwd()
    model_path = root / run_date / "models" / "pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}. Entrena primero con preprocess_and_train().")

    pipe: Pipeline = joblib.load(model_path)

    # Columnas esperadas (se filtran a las realmente vistas por el preprocesador)
    input_cols = [
        "Age","Gender","EducationLevel","ExperienceYears","PreviousCompanies",
        "DistanceFromCompany","InterviewScore","SkillScore","PersonalityScore",
        "RecruitmentStrategy",
    ]
    preprocess = pipe.named_steps["preprocess"]
    trained_in = list(getattr(preprocess, "feature_names_in_", [])) or input_cols
    trained_cols = [c for c in input_cols if c in trained_in]

    def _predict(Age, Gender, EducationLevel, ExperienceYears, PreviousCompanies,
                 DistanceFromCompany, InterviewScore, SkillScore, PersonalityScore,
                 RecruitmentStrategy):
        row = {
            "Age": Age, "Gender": Gender, "EducationLevel": EducationLevel,
            "ExperienceYears": ExperienceYears, "PreviousCompanies": PreviousCompanies,
            "DistanceFromCompany": DistanceFromCompany, "InterviewScore": InterviewScore,
            "SkillScore": SkillScore, "PersonalityScore": PersonalityScore,
            "RecruitmentStrategy": RecruitmentStrategy,
        }
        X = pd.DataFrame([{k: row[k] for k in trained_cols}])
        pred = int(pipe.predict(X)[0])
        proba = float(pipe.predict_proba(X)[0][1]) if hasattr(pipe, "predict_proba") else None
        label = "Contratado (1)" if pred == 1 else "No contratado (0)"
        return f"{label} | P(1)={proba:.3f}" if proba is not None else label

    demo = gr.Interface(
        fn=_predict,
        inputs=[
            gr.Number(label="Age", precision=0, value=30),
            gr.Dropdown([0,1], value=0, label="Gender (0=Male,1=Female)"),
            gr.Dropdown([1,2,3,4], value=3, label="EducationLevel (1..4)"),
            gr.Number(label="ExperienceYears", precision=0, value=3),
            gr.Number(label="PreviousCompanies", precision=0, value=1),
            gr.Number(label="DistanceFromCompany (km)", precision=2, value=5.0),
            gr.Number(label="InterviewScore (0-100)", precision=0, value=75),
            gr.Number(label="SkillScore (0-100)", precision=0, value=80),
            gr.Number(label="PersonalityScore (0-100)", precision=0, value=70),
            gr.Dropdown([1,2,3], value=2, label="RecruitmentStrategy (1..3)"),
        ],
        outputs=gr.Textbox(label="Resultado"),
        title="Hiring Decision Predictor",
        description="Modelo Random Forest entrenado con pipeline de preprocesamiento."
    )
    demo.launch()


