# dags/hiring_dynamic_functions.py  (versión adaptada a la convención del primer código)
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, Any

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------
# Logger consistente con el primer módulo
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _resolve_run_dirs(base_dir: Optional[str], ds: Optional[str]) -> Dict[str, Path]:
    """
    Utilidad interna: resuelve root/run_dir y subcarpetas usando base_dir + ds (YYYY-MM-DD).
    Si ds es None, usa fecha de hoy.
    """
    run_date = ds or datetime.now().date().isoformat()
    root = Path(base_dir) if base_dir else Path.cwd()
    run_dir = root / run_date
    return {
        "root": root,
        "run_dir": run_dir,
        "raw": run_dir / "raw",
        "pre": run_dir / "preprocessed",
        "splits": run_dir / "splits",
        "models": run_dir / "models",
    }


# ---------------------------------------------------------------------
def create_folders(base_dir: Optional[str] = None, **kwargs) -> str:
    """
    Crea carpeta del run con nombre = fecha de ejecución (ds) y subcarpetas:
    raw/, preprocessed/, splits/, models/.

    Parámetros (alineados al primer código)
    ---------------------------------------
    base_dir : str | None
        Directorio base donde se creará la carpeta del run. Si None, usa cwd.
    **kwargs :
        Puede contener 'ds' (YYYY-MM-DD) si viene de Airflow.

    Returns
    -------
    str : ruta absoluta del run_dir creado.
    """
    ds = kwargs.get("ds")
    dirs = _resolve_run_dirs(base_dir, ds)
    for d in [dirs["raw"], dirs["pre"], dirs["splits"], dirs["models"]]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Carpetas creadas en: {dirs['run_dir'].resolve()}")
    return str(dirs["run_dir"].resolve())


def load_and_merge(base_dir: Optional[str] = None, **kwargs) -> str:
    """
    Lee raw/data_1.csv y opcionalmente raw/data_2.csv,
    concatena y guarda en preprocessed/merged_data.csv.

    Parámetros (alineados)
    ----------------------
    base_dir : str | None
    **kwargs : puede contener 'ds'

    Returns
    -------
    str : ruta del archivo merged_data.csv
    """
    ds = kwargs.get("ds")
    dirs = _resolve_run_dirs(base_dir, ds)

    files = [dirs["raw"] / "data_1.csv"]
    data_2 = dirs["raw"] / "data_2.csv"
    if data_2.exists():
        files.append(data_2)

    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"No se encontró el archivo requerido: {f.resolve()}")

    dfs = [pd.read_csv(f) for f in files]
    df_merged = pd.concat(dfs, ignore_index=True)

    merged_fp = dirs["pre"] / "merged_data.csv"
    merged_fp.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(merged_fp, index=False)
    logger.info(f"Datos combinados guardados en: {merged_fp.resolve()}")
    return str(merged_fp.resolve())


def split_data(
    base_dir: Optional[str] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    **kwargs
) -> Dict[str, str]:
    """
    Lee preprocessed/merged_data.csv, aplica hold-out (estratificado en HiringDecision),
    y guarda X_train/X_test/y_train/y_test en splits/.

    Parámetros (alineados)
    ----------------------
    base_dir : str | None
    test_size : float (default 0.20)
    random_state : int (default 42)
    **kwargs : puede contener 'ds'

    Returns
    -------
    dict : rutas de archivos generados
    """
    ds = kwargs.get("ds")
    dirs = _resolve_run_dirs(base_dir, ds)

    merged_fp = dirs["pre"] / "merged_data.csv"
    if not merged_fp.exists():
        raise FileNotFoundError(
            f"No existe {merged_fp}. Ejecuta primero load_and_merge()."
        )

    df = pd.read_csv(merged_fp)
    target = "HiringDecision"
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en merged_data.csv.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    split_dir = dirs["splits"]
    split_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "X_train": split_dir / "X_train.csv",
        "X_test": split_dir / "X_test.csv",
        "y_train": split_dir / "y_train.csv",
        "y_test": split_dir / "y_test.csv",
    }

    X_train.to_csv(paths["X_train"], index=False)
    X_test.to_csv(paths["X_test"], index=False)
    y_train.to_csv(paths["y_train"], index=False)
    y_test.to_csv(paths["y_test"], index=False)

    logger.info(f"Datos divididos y guardados en {split_dir.resolve()}")
    return {k: str(v.resolve()) for k, v in paths.items()}


def train_model(
    base_dir: Optional[str] = None,
    model: Optional[object] = None,
    model_name: str = "model",
    **kwargs
) -> str:
    """
    Entrena un modelo recibido usando un pipeline con preprocesamiento.
    Guarda el pipeline en models/<model_name>.joblib.

    Parámetros (alineados)
    ----------------------
    base_dir : str | None
    model : estimador sklearn (si None, usa RandomForestClassifier())
    model_name : str
    **kwargs : puede contener 'ds'

    Returns
    -------
    str : ruta del .joblib guardado
    """
    ds = kwargs.get("ds")
    dirs = _resolve_run_dirs(base_dir, ds)

    X_train_fp = dirs["splits"] / "X_train.csv"
    y_train_fp = dirs["splits"] / "y_train.csv"
    if not X_train_fp.exists() or not y_train_fp.exists():
        raise FileNotFoundError("No existen los splits. Ejecuta split_data() antes.")

    X_train = pd.read_csv(X_train_fp)
    y_train = pd.read_csv(y_train_fp).values.ravel()

    # Columnas (mismas del segundo código; compatibles con el primero)
    numeric_features = [
        "Age", "ExperienceYears", "PreviousCompanies",
        "DistanceFromCompany", "InterviewScore", "SkillScore", "PersonalityScore"
    ]
    categorical_features = ["Gender", "EducationLevel", "RecruitmentStrategy"]

    # Filtrar por columnas presentes (robustez)
    numeric_features = [c for c in numeric_features if c in X_train.columns]
    categorical_features = [c for c in categorical_features if c in X_train.columns]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    if model is None:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    save_fp = dirs["models"] / f"{model_name}.joblib"
    dirs["models"].mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_fp)
    logger.info(f"Modelo '{model_name}' entrenado y guardado en {save_fp.resolve()}")
    return str(save_fp.resolve())


def evaluate_models(base_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Evalúa todos los modelos en models/ contra splits/X_test,y_test,
    selecciona el de mejor accuracy y lo guarda como models/best_model.joblib.

    Parámetros (alineados)
    ----------------------
    base_dir : str | None
    **kwargs : puede contener 'ds'

    Returns
    -------
    dict : {"best_model": <nombre>, "best_accuracy": <float>, "best_path": <ruta>}
    """
    ds = kwargs.get("ds")
    dirs = _resolve_run_dirs(base_dir, ds)

    X_test_fp = dirs["splits"] / "X_test.csv"
    y_test_fp = dirs["splits"] / "y_test.csv"
    if not X_test_fp.exists() or not y_test_fp.exists():
        raise FileNotFoundError("No existen los splits. Ejecuta split_data() antes.")

    X_test = pd.read_csv(X_test_fp)
    y_test = pd.read_csv(y_test_fp).values.ravel()

    results: Dict[str, float] = {}
    if not dirs["models"].exists():
        raise ValueError("La carpeta 'models' no existe o no contiene modelos entrenados.")

    for file in os.listdir(dirs["models"]):
        if file.endswith(".joblib") and file != "best_model.joblib":
            model_file = dirs["models"] / file
            model_name = file.replace(".joblib", "")
            pipeline = joblib.load(model_file)

            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[model_name] = acc
            logger.info(f"Modelo: {model_name} | Accuracy: {acc:.4f}")

    if not results:
        raise ValueError("No se encontraron modelos entrenados en la carpeta 'models'.")

    best_model_name = max(results, key=results.get)
    best_acc = results[best_model_name]

    best_model_path = dirs["models"] / f"{best_model_name}.joblib"
    best_copy_path = dirs["models"] / "best_model.joblib"
    joblib.dump(joblib.load(best_model_path), best_copy_path)

    logger.info(f"Mejor modelo: {best_model_name} | Accuracy: {best_acc:.4f}")
    logger.info(f"Guardado como {best_copy_path.resolve()}")

    return {
        "best_model": best_model_name,
        "best_accuracy": float(best_acc),
        "best_path": str(best_copy_path.resolve())
    }
