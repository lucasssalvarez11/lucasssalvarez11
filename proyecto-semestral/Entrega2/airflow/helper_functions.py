import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

import optuna
import xgboost as xgb

from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

import joblib


# ====================================================
# RUTAS Y CONFIG BÁSICA
# ====================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # carpeta raíz del proyecto (donde está /data)
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"

TARGET_COL = "target"
ID_COLS = ["customer_id", "product_id", "Año", "Semana"]


def ensure_dirs() -> None:
    """Crea las carpetas base si no existen."""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, PREDICTIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ====================================================
# 1. EXTRACCIÓN DE DATOS
# ====================================================

def run_extraction() -> None:
    """
    Lee los 3 parquet desde data/raw/, renombra items->payment en transacciones
    y los deja listos en data/processed/.
    """
    ensure_dirs()

    trans_path = RAW_DIR / "transacciones.parquet"
    clientes_path = RAW_DIR / "clientes.parquet"
    productos_path = RAW_DIR / "productos.parquet"

    if not trans_path.exists():
        raise FileNotFoundError(f"No se encontró {trans_path}")
    if not clientes_path.exists():
        raise FileNotFoundError(f"No se encontró {clientes_path}")
    if not productos_path.exists():
        raise FileNotFoundError(f"No se encontró {productos_path}")

    df_trans = pd.read_parquet(trans_path)
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)

    # renombrar items -> payment
    if "items" in df_trans.columns and "payment" not in df_trans.columns:
        df_trans = df_trans.rename(columns={"items": "payment"})

    print(f"Transacciones (raw): {df_trans.shape}")
    print(f"Clientes (raw): {df_clientes.shape}")
    print(f"Productos (raw): {df_productos.shape}")

    # Guardar "limpios" en processed para que el preprocesamiento los use
    df_trans.to_parquet(PROCESSED_DIR / "transacciones.parquet", index=False)
    df_clientes.to_parquet(PROCESSED_DIR / "clientes.parquet", index=False)
    df_productos.to_parquet(PROCESSED_DIR / "productos.parquet", index=False)

    print("Extracción completada y datos guardados en data/processed/.")


# ====================================================
# 2. CONSTRUCCIÓN DF BASE (cliente-producto-semana)
# ====================================================

def _load_processed_raws() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los parquet de data/processed/ generados por run_extraction()."""
    trans_path = PROCESSED_DIR / "transacciones.parquet"
    clientes_path = PROCESSED_DIR / "clientes.parquet"
    productos_path = PROCESSED_DIR / "productos.parquet"

    df_trans = pd.read_parquet(trans_path)
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)

    # asegurar que exista payment
    if "items" in df_trans.columns and "payment" not in df_trans.columns:
        df_trans = df_trans.rename(columns={"items": "payment"})

    return df_trans, df_clientes, df_productos


def build_base_df(
    df_trans: pd.DataFrame,
    df_clientes: pd.DataFrame,
    df_productos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye un df base con:
      - customer_id, product_id, Año, Semana
      - TARGET_COL = 1 si hay compra (payment>0) en esa semana, 0 si no
      - Merge con features de clientes y productos
    """

    print("Construyendo DataFrame base...")

    df_trans = df_trans.copy()
    df_clientes = df_clientes.copy()
    df_productos = df_productos.copy()

    # Categóricas como string
    if "customer_type" in df_clientes.columns:
        df_clientes["customer_type"] = df_clientes["customer_type"].astype("string")

    for col in ["brand", "category", "sub_category", "segment", "package"]:
        if col in df_productos.columns:
            df_productos[col] = df_productos[col].astype("string")

    # Fecha
    df_trans["purchase_date"] = pd.to_datetime(df_trans["purchase_date"])
    df_trans["payment"] = pd.to_numeric(df_trans["payment"], errors="coerce").fillna(0)
    df_trans = df_trans[df_trans["payment"] >= 0]

    # Agregar por orden (por si hay duplicados)
    df_trans = (
        df_trans
        .groupby(["order_id", "product_id", "customer_id", "purchase_date"], as_index=False)
        ["payment"]
        .sum()
    )

    # Año y semana ISO
    iso = df_trans["purchase_date"].dt.isocalendar()
    df_trans["Año"] = iso.year
    df_trans["Semana"] = iso.week

    # Target = 1 si hubo compra en esa semana
    df_trans[TARGET_COL] = 1

    # Agregamos a nivel cliente-producto-Año-Semana por si hay varias órdenes
    df_trans_week = (
        df_trans
        .groupby(["customer_id", "product_id", "Año", "Semana"], as_index=False)
        .agg(
            total_payment=("payment", "sum"),
            n_orders=("order_id", "nunique"),
            **{TARGET_COL: (TARGET_COL, "max")}
        )
    )

    # Todas las combinaciones cliente-producto-semana
    clientes = df_trans_week["customer_id"].unique()
    productos = df_trans_week["product_id"].unique()
    semanas = df_trans_week[["Año", "Semana"]].drop_duplicates()

    df_clientes_unq = pd.DataFrame({"customer_id": clientes})
    df_productos_unq = pd.DataFrame({"product_id": productos})

    todas_comb = (
        df_clientes_unq
        .merge(df_productos_unq, how="cross")
        .merge(semanas, how="cross")
    )

    df = todas_comb.merge(
        df_trans_week[["customer_id", "product_id", "Año", "Semana", "total_payment", "n_orders", TARGET_COL]],
        on=["customer_id", "product_id", "Año", "Semana"],
        how="left",
    )

    # Donde no hubo compra, payment=0, n_orders=0, target=0
    df["total_payment"] = df["total_payment"].fillna(0)
    df["n_orders"] = df["n_orders"].fillna(0).astype(int)
    df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

    # Merge con clientes y productos
    df = df.merge(df_clientes, on="customer_id", how="left")
    df = df.merge(df_productos, on="product_id", how="left")

    df = df.drop_duplicates()

    print(f"DF base construido: {df.shape}")
    return df


# ====================================================
# 3. PREPROCESAMIENTO: DUMMIES + STANDARD SCALER
# ====================================================

def preprocess_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    - Imputa missing
    - OneHotEncoder a categóricas
    - StandardScaler a numéricas
    - Devuelve df_final con ID_COLS + TARGET_COL + features transformadas
    - Devuelve también el preprocessor (ColumnTransformer)
    """

    print("Iniciando preprocesamiento (dummies + StandardScaler)...")
    df = df.copy()

    for col in ID_COLS + [TARGET_COL]:
        if col not in df.columns:
            raise KeyError(f"Falta columna obligatoria '{col}' en df")

    # Numéricas: todas las numéricas excepto target
    numeric_features = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c not in ID_COLS + [TARGET_COL]
    ]

    # Categóricas: object o string
    categorical_features = [
        c for c in df.columns
        if (df[c].dtype == "object" or "string" in str(df[c].dtype))
        and c not in ID_COLS + [TARGET_COL]
    ]

    print(f"Features numéricas: {numeric_features}")
    print(f"Features categóricas: {categorical_features}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    X = df[numeric_features + categorical_features]

    print("Ajustando preprocessor (fit_transform)...")
    X_trans = preprocessor.fit_transform(X)

    # nombres de columnas
    num_out_cols = numeric_features
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_out_cols = list(cat_encoder.get_feature_names_out(categorical_features))

    feature_cols = num_out_cols + cat_out_cols

    df_features = pd.DataFrame(X_trans, columns=feature_cols, index=df.index)

    df_final = pd.concat(
        [df[ID_COLS + [TARGET_COL]].reset_index(drop=True),
         df_features.reset_index(drop=True)],
        axis=1,
    )

    out_path = PROCESSED_DIR / "df_preprocesado.parquet"
    df_final.to_parquet(out_path, index=False)
    print(f"df_preprocesado guardado en: {out_path}")
    print(f"Shape df_preprocesado: {df_final.shape}")

    # guardamos también el preprocessor
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    return df_final, preprocessor


def run_preprocessing() -> pd.DataFrame:
    """
    Función de alto nivel que:
      1) Carga data procesada de run_extraction()
      2) Construye df base
      3) Aplica preprocesamiento (dummies + scaler)
      4) Hace split temporal 70/15/15 (train/val/test) y guarda a parquet
      5) Devuelve df_preprocesado
    """
    ensure_dirs()
    df_trans, df_clientes, df_productos = _load_processed_raws()
    df_base = build_base_df(df_trans, df_clientes, df_productos)
    df_pre, _ = preprocess_df(df_base)

    temporal_split(df_pre, train_frac=0.70, val_frac=0.15)
    return df_pre


# ====================================================
# 4. SPLIT TEMPORAL 70 / 15 / 15
# ====================================================

def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> None:
    """
    Split temporal respetando Año/Semana.
    - Ordena por Año, Semana
    - 70% primeras filas -> train
    - 15% siguientes -> val
    - 15% finales -> test
    Guarda en:
      - data/processed/train.parquet
      - data/processed/val.parquet
      - data/processed/test.parquet
    """
    df = df.copy()

    # Crear una fecha de referencia (lunes de esa semana) para ordenar bien
    df["fecha_semana"] = pd.to_datetime(
        df["Año"].astype(str) + "-W" + df["Semana"].astype(str) + "-1",
        format="%G-W%V-%u",
    )

    df_sorted = df.sort_values("fecha_semana").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:train_end].drop(columns=["fecha_semana"])
    val_df = df_sorted.iloc[train_end:val_end].drop(columns=["fecha_semana"])
    test_df = df_sorted.iloc[val_end:].drop(columns=["fecha_semana"])

    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print(f"Split temporal completado:")
    print(f"  train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")


# ====================================================
# 5. OPTUNA + XGBOOST
# ====================================================

def _load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    return train, val, test


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]


def run_optuna_tuning(n_trials: int = 30) -> Dict[str, Any]:
    """
    Optimiza hiperparámetros de XGBoost usando train/val.
    Devuelve best_params.
    """
    print(f"Iniciando Optuna con {n_trials} trials...")
    train_df, val_df, _ = _load_splits()

    feature_cols = _get_feature_cols(train_df)

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro")
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Mejor F1-macro: {study.best_value:.4f}")
    print(f"Mejores parámetros: {study.best_params}")

    return study.best_params


def train_model(best_params: Dict[str, Any] | None = None) -> Dict[str, float]:
    """
    Entrena XGBoost con los mejores hiperparámetros.
    - Si best_params es None, llama a run_optuna_tuning.
    - Entrega métricas sobre el set de test.
    - Guarda modelo entrenado en data/models/xgb_model.pkl
    """
    train_df, val_df, test_df = _load_splits()
    feature_cols = _get_feature_cols(train_df)

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values

    if best_params is None:
        best_params = run_optuna_tuning(n_trials=30)

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }

    params = {**base_params, **best_params}

    print("Entrenando modelo XGBoost final...")
    model = xgb.XGBClassifier(**params)

    # Entrenamos con train+val para usar más datos
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    model.fit(X_train_full, y_train_full, verbose=False)

    # Evaluar en test
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)

    print(f"Métricas test -> F1-macro: {f1:.4f}, Accuracy: {acc:.4f}")

    metrics = {"f1_macro_test": float(f1), "accuracy_test": float(acc)}

    # Guardar modelo
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgb_model.pkl"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"Modelo XGBoost guardado en: {model_path}")

    return metrics


# ====================================================
# 6. DETECCIÓN DE DRIFT
# ====================================================

def detect_drift(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Detecta drift entre dos datasets:
      - KS test para numéricas
      - Jensen-Shannon para categóricas
    Retorna:
      - drift_detected (bool)
      - avg_score (float)
    """
    metrics: List[float] = []

    common_cols = [c for c in df_old.columns if c in df_new.columns]

    for col in common_cols:
        s_old = df_old[col].dropna()
        s_new = df_new[col].dropna()

        if s_old.empty or s_new.empty:
            continue

        # numéricas -> KS
        if np.issubdtype(s_old.dtype, np.number) and s_old.nunique() >= 10 and s_new.nunique() >= 10:
            try:
                p = ks_2samp(s_old, s_new).pvalue
                metrics.append(p)   # p-value como "similaridad"
            except Exception:
                continue
        else:
            # categóricas -> Jensen-Shannon
            old_counts = s_old.value_counts(normalize=True)
            new_counts = s_new.value_counts(normalize=True)

            all_cats = sorted(set(old_counts.index) | set(new_counts.index))
            p = np.array([old_counts.get(cat, 0) for cat in all_cats])
            q = np.array([new_counts.get(cat, 0) for cat in all_cats])

            js = jensenshannon(p, q)
            metrics.append(1 - js)  # 1-js como "similaridad"

    avg_score = float(np.mean(metrics)) if metrics else 1.0
    drift = avg_score < threshold

    print(f"Score promedio={avg_score:.4f} -> Drift={drift}")

    return {"drift_detected": drift, "avg_score": avg_score}


def run_drift_detection(threshold: float = 0.1) -> Dict[str, Any]:
    """
    Versión concreta para el pipeline:
      - Usa train.parquet como referencia
      - Usa test.parquet como "nuevo"
    """
    train_df, _, test_df = _load_splits()
    # Quitamos ID_COLS y TARGET_COL para medir drift solo en features
    cols_to_use = [c for c in train_df.columns if c not in ID_COLS + [TARGET_COL]]

    return detect_drift(
        df_old=train_df[cols_to_use],
        df_new=test_df[cols_to_use],
        threshold=threshold,
    )
