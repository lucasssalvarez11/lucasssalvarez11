# SodAI Drinks 🥤 – Pipeline productivo (Airflow)

Este directorio contiene el **pipeline productivo** para SodAI Drinks, orquestado con **Airflow**.  
El objetivo es automatizar el flujo completo:

1. Extracción y preparación de datos
2. Construcción de dataset cliente–producto–semana
3. Preprocesamiento (dummies, escalado, features adicionales)
4. Detección de drift de datos
5. Reentrenamiento condicional del modelo (XGBoost + Optuna)
6. Persistencia de artefactos (modelo y preprocesador) para ser usados por la aplicación (`app/backend`)

---

## 1. Estructura de carpetas

```text
airflow/
  dags/
    helper_functions.py   # Lógica de negocio y funciones auxiliares de ML
    pipeline.py           # Definición del DAG de Airflow
  data/
    raw/                  # Datos de entrada (parquet).
    processed/            # Datasets procesados y splits.
    models/               # Artefactos del modelo.
    predictions/          # (Opcional) Salidas de predicción.

## 2. Orquestación del pipeline con Airflow

Este script define un **DAG de Airflow** llamado `weekly_ml_pipeline` que ejecuta de forma orquestada el pipeline de ML:

1. Preparación de directorios.
2. Extracción de datos.
3. Preprocesamiento y generación de splits.
4. Detección de drift.
5. Reentrenamiento condicional del modelo XGBoost usando Optuna (solo si hay drift).

El DAG está agendado para correrse de forma **semanal** (`schedule_interval="@weekly"`) y no hace *catchup* de fechas pasadas.

### Funciones principales (callables)

- `drift_branch_callable(**kwargs)`
  - Ejecuta `run_drift_detection(threshold=0.1)`.
  - Imprime el resultado de la detección de drift.
  - Devuelve el `task_id` al que debe ir el flujo:
    - `"train_xgboost_with_optuna"` si **se detecta drift**.
    - `"skip_training"` si **no hay drift**.
  - Esta función se usa en un `BranchPythonOperator` para decidir dinámicamente el camino del DAG.

- `train_with_optuna_callable()`
  - Lanza la optimización de hiperparámetros con `run_optuna_tuning(n_trials=30)`.
  - Usa los mejores parámetros encontrados para entrenar el modelo final llamando a `train_model(best_params=best_params)`.
  - Imprime las métricas finales del modelo en el set de test.

Ambas funciones consumen las funciones definidas en `helper_functions.py`:
`ensure_dirs`, `run_extraction`, `run_preprocessing`, `run_drift_detection`, `run_optuna_tuning`, `train_model`.

### Estructura del DAG y tareas

El DAG se define con:

- `dag_id="weekly_ml_pipeline"`
- `schedule_interval="@weekly"`
- `start_date=datetime(2024, 1, 1)`
- `tags=["ml", "xgboost", "optuna", "drift"]`

Las tareas son:

- `start` (`EmptyOperator`): nodo inicial del flujo.
- `ensure_dirs` (`PythonOperator`):
  - Llama a `ensure_dirs()` para asegurarse de que existan las carpetas base (`data/`, `raw`, `processed`, `models`, etc.).
- `extract_data` (`PythonOperator`):
  - Ejecuta `run_extraction()` para leer los datos crudos, hacer ajustes básicos y guardarlos en `data/processed/`.
- `preprocess_data` (`PythonOperator`):
  - Llama a `run_preprocessing()` para construir el dataset base, aplicar preprocesamiento y generar los splits temporales `train/val/test`.
- `check_drift_and_branch` (`BranchPythonOperator`):
  - Ejecuta `drift_branch_callable()` y, según el resultado, bifurca el flujo hacia:
    - `train_xgboost_with_optuna` (si hay drift).
    - `skip_training` (si no hay drift).
- `train_xgboost_with_optuna` (`PythonOperator`):
  - Ejecuta `train_with_optuna_callable()`, que corre Optuna + entrenamiento final del modelo XGBoost.
- `skip_training` (`EmptyOperator`):
  - Nodo “dummy” que representa el camino donde se decide **no reentrenar** el modelo (no hay drift relevante).
- `end` (`EmptyOperator`):
  - Nodo final al que convergen ambos caminos (con o sin reentrenamiento).

### Flujo de dependencias

El flujo general del DAG es:

```text
start
  ↓
ensure_dirs
  ↓
extract_data
  ↓
preprocess_data
  ↓
check_drift_and_branch
        ↙               ↘
train_xgboost_with_optuna   skip_training
        ↘               ↙
             end


## 2. Pipeline de modelado (extracción, features, entrenamiento y drift)

Este script implementa el **pipeline completo** para un problema de clasificación binaria de compra (`target`) a nivel `cliente-producto-semana`.  
Incluye:

- Lectura y preparación de datos crudos (`.parquet`).
- Construcción del dataset base (nivel cliente–producto–Año–Semana).
- Preprocesamiento (imputación, dummies y escalado).
- Split temporal train/val/test.
- Optimización de hiperparámetros y entrenamiento de un modelo **XGBoost**.
- Detección de **data drift** entre distintos datasets.

La estructura de carpetas asumida es:

- `data/raw/` → datos crudos:
  - `transacciones.parquet`
  - `clientes.parquet`
  - `productos.parquet`
- `data/processed/` → datos procesados e intermedios.
- `data/models/` → modelos y objetos serializados.
- `data/predictions/` → (reservado para salidas de predicción).

### Flujo general del pipeline

1. **Extracción de datos crudos**
   - `run_extraction()`
     - Lee los tres archivos de origen desde `data/raw/`.
     - En la tabla de transacciones, renombra la columna `items` a `payment` si corresponde.
     - Guarda las tres tablas “limpias” en `data/processed/` para el siguiente paso.

2. **Construcción del DataFrame base (cliente–producto–semana)**
   - `_load_processed_raws()`
     - Carga las versiones procesadas de `transacciones`, `clientes` y `productos` desde `data/processed/`.
   - `build_base_df(df_trans, df_clientes, df_productos)`
     - Limpia tipos de datos (por ejemplo, columnas categóricas como `string`).
     - Convierte `purchase_date` a fecha y asegura que `payment` sea numérica.
     - Agrega transacciones por orden para evitar duplicados.
     - Calcula **Año** y **Semana ISO** a partir de la fecha.
     - Define la variable objetivo `target = 1` si hubo compra (payment > 0) en esa combinación `cliente-producto-Año-Semana`, y `0` en caso contrario.
     - Genera **todas las combinaciones posibles** de `cliente-producto-Año-Semana` observadas en los datos y completa con:
       - `total_payment`, `n_orders` y `target`.
     - Hace *join* con las tablas de clientes y productos para incorporar sus atributos como features.

3. **Preprocesamiento de datos (dummies + escalado)**
   - `preprocess_df(df)`
     - Revisa que existan las columnas de ID (`customer_id`, `product_id`, `Año`, `Semana`) y `target`.
     - Separa:
       - **Features numéricas**: columnas numéricas distintas de los IDs y del `target`.
       - **Features categóricas**: columnas de tipo `object`/`string` distintas de IDs y `target`.
     - Aplica un `ColumnTransformer` con dos pipelines:
       - Numéricas: `SimpleImputer(strategy="median")` + `StandardScaler()`.
       - Categóricas: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.
     - Devuelve un `df_final` que contiene:
       - Columnas de ID + `target`.
       - Todas las features transformadas (incluyendo dummies).
     - Guarda el DataFrame preprocesado en `data/processed/df_preprocesado.parquet`.
     - Serializa el `preprocessor` (ColumnTransformer) en `data/models/preprocessor.pkl`.

   - `run_preprocessing()`
     - Función “de alto nivel” que ejecuta:
       1. Carga de datos procesados (`_load_processed_raws`).
       2. Construcción del DF base (`build_base_df`).
       3. Preprocesamiento (`preprocess_df`).
       4. **Split temporal** 70/15/15 (train/val/test) vía `temporal_split`.
     - Devuelve el DataFrame preprocesado completo.

4. **Split temporal (train / val / test)**
   - `temporal_split(df, train_frac=0.7, val_frac=0.15)`
     - Crea una fecha de referencia a partir de `Año` y `Semana` (lunes de cada semana).
     - Ordena el dataset por esta fecha.
     - Separa en:
       - 70% filas iniciales → `train.parquet`
       - 15% siguientes → `val.parquet`
       - 15% finales → `test.parquet`
     - Guarda estos 3 conjuntos en `data/processed/`.

   - `_load_splits()`
     - Carga `train.parquet`, `val.parquet` y `test.parquet` desde `data/processed/`.

   - `_get_feature_cols(df)`
     - Devuelve la lista de columnas de features (es decir, todas menos las columnas de ID y la columna objetivo `target`).

5. **Optimización de hiperparámetros y entrenamiento de XGBoost**
   - `run_optuna_tuning(n_trials=30)`
     - Carga los splits `train` y `val`.
     - Define un espacio de búsqueda de hiperparámetros para `xgboost.XGBClassifier` (profundidad, learning rate, n_estimators, subsample, regularizaciones, etc.).
     - Usa **Optuna** para maximizar el **F1-macro** en el set de validación.
     - Devuelve el diccionario `best_params` con los mejores hiperparámetros encontrados.

   - `train_model(best_params: Dict[str, Any] | None = None)`
     - Carga train, val y test.
     - Si `best_params` es `None`, llama internamente a `run_optuna_tuning()`.
     - Combina `train` + `val` para entrenar el modelo final con más datos.
     - Entrena un `XGBClassifier` con los parámetros base + `best_params`.
     - Evalúa en el set de test:
       - `F1-macro`
       - `Accuracy`
     - Guarda el modelo entrenado en `data/models/xgb_model.pkl` junto con la lista de columnas de features.
     - Retorna un diccionario con las métricas de test.

6. **Detección de data drift**
   - `detect_drift(df_old, df_new, threshold=0.1)`
     - Compara la distribución de las columnas comunes entre dos datasets:
       - Para columnas numéricas (con suficiente cardinalidad) aplica **Kolmogorov–Smirnov (KS)** y usa el *p-value* como métrica de similitud.
       - Para columnas categóricas calcula la distancia **Jensen–Shannon** entre las distribuciones de frecuencias y utiliza `1 - js` como similitud.
     - Calcula un `avg_score` promedio de similitud entre todas las columnas.
     - Define que hay drift (`drift_detected = True`) si `avg_score` es menor que el umbral (`threshold`).
     - Devuelve:
       - `{"drift_detected": bool, "avg_score": float}`.

   - `run_drift_detection(threshold=0.1)`
     - Usa por defecto:
       - `train.parquet` como dataset de referencia (`df_old`).
       - `test.parquet` como “nuevo” dataset (`df_new`).
     - Excluye las columnas de ID y `target` (solo analiza features).
     - Llama a `detect_drift` y devuelve el resultado.

### Funciones utilitarias

- `ensure_dirs()`
  - Se asegura de que existan las rutas base (`data/`, `data/raw/`, `data/processed/`, `data/models/`, `data/predictions/`), creándolas si no existen.


