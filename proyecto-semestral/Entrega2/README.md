# Conclusiones y reflexiones sobre el enfoque MLOps

## 1. Impacto de las herramientas de tracking y despliegue

La incorporación de herramientas propias de MLOps cambió la forma en que abordamos el proyecto. En vez de pensar solo en “entrenar un modelo que funcione bien”, tuvimos que pensar en **ciclos completos**: desde la llegada de nuevos datos, hasta cómo versionar modelos, registrar métricas y exponer predicciones.

El uso de un enfoque de tracking (por ejemplo, mediante MLflow) nos ayudó a:

- Tener un registro claro de **qué modelo se entrenó con qué hiperparámetros**.
- Comparar fácilmente corridas distintas sin depender de “notebooks sueltos”.
- Guardar artefactos entrenados (modelo, preprocesador) de forma más ordenada, lo que facilitó conectarlos después con la API de FastAPI.

Aunque en un proyecto de ramo no siempre se sienten todas las ventajas, el simple hecho de estructurar el código pensando en tracking y despliegue nos obligó a separar mejor las responsabilidades: funciones auxiliares, pipeline de entrenamiento, API de predicción y frontends. Eso hace que el proyecto sea **más mantenible y más cercano a un flujo real de producción**.

## 2. Desafíos con Gradio y FastAPI

La parte de despliegue con **FastAPI + Gradio** fue probablemente una de las más desafiantes, porque implica cambiar el mindset desde “modelo en notebook” a “modelo como servicio”.

En **FastAPI**, los principales aprendizajes y desafíos fueron:

- Diseñar endpoints limpios (`/predict`, `/recommend`, `/chat`) con **esquemas Pydantic**, lo que obliga a definir bien qué datos entran y qué datos salen.
- Manejar la **carga del modelo y el preprocesador** al iniciar la aplicación, en vez de cargarlos ad hoc en cada request.
- Trabajar con rutas relativas a los artefactos generados por Airflow (por ejemplo, `xgb_model.pkl` y `preprocessor.pkl`), lo que obliga a pensar en rutas de proyecto y no solo en “mi máquina local”.

En **Gradio**, lo más interesante fue:

- Lo rápido que se puede construir una **interfaz utilizable** (formularios, tablas, chatbot) con muy pocas líneas de código.
- Aprender a comunicar el frontend con el backend usando `requests` y variables de entorno como `BACKEND_URL`.
- Adaptar la UX para que alguien que no vio el código pueda entender qué ingresar y cómo interpretar la salida (por ejemplo, explicaciones en la interfaz sobre qué significa la probabilidad de compra o las recomendaciones).

En resumen, Gradio/FastAPI nos obligaron a pensar el modelo como un **producto que alguien va a usar**, no solo como un experimento.

## 3. Rol de Airflow en la robustez y escalabilidad

Airflow fue el eje para darle **robustez y estructura** al pipeline. Pasamos de una secuencia manual de pasos (cargar datos, entrenar, evaluar, guardar modelo) a un **DAG claro** con tareas separadas:

- Extracción de datos
- Preprocesamiento y generación del dataset cliente–producto–semana
- Detección de drift
- Reentrenamiento condicional con Optuna + XGBoost
- Exportación de modelos para consumo por la API

Los principales aportes de Airflow fueron:

- **Orquestación explícita**: ver gráficamente el flujo de tareas ayuda a entender dependencias, y obliga a que cada etapa sea una unidad autocontenida.
- **Programación periódica** (@weekly): nos acerca a la idea de un sistema que corre de forma regular con datos nuevos, en vez de un entrenamiento “una sola vez”.
- **Branching condicional (drift → reentrenar)**: esto hace que el pipeline no solo sea una secuencia fija, sino que tome decisiones en función del estado de los datos.
- Posibilidad de **reintentos, logs centralizados y monitoreo básico** usando la UI de Airflow.

Aunque nuestro proyecto corre a una escala pequeña, el diseño con Airflow nos deja más preparados para escalar a entornos donde hay más datos, más modelos o múltiples pipelines corriendo en paralelo.

## 4. Oportunidades de mejora y trabajo futuro

Si pensáramos en una versión futura del flujo, hay varias áreas donde se podría mejorar o profundizar:

1. **Más automatización en la integración con datos reales**  
   En esta entrega asumimos que los archivos `.parquet` “aparecen mágicamente” en el directorio. Una versión más robusta integraría:
   - Conectores a bases de datos o sistemas de almacenamiento (S3, data warehouse, etc.).
   - Validaciones de esquema y calidad de datos antes de seguir con el pipeline (por ejemplo, usando algo tipo Great Expectations).

2. **Monitoreo continuo del modelo en producción**  
   Hoy tenemos una detección de drift a nivel de datos, pero se podría:
   - Monitorear métricas de desempeño con etiquetas retrasadas (cuando se sepa si la recomendación o la predicción fue correcta).
   - Registrar estadísticas de uso de la API (cuántas predicciones, para qué clientes/productos, etc.).
   - Agregar alertas cuando el modelo empiece a degradarse.

3. **Métricas adicionales y más cercanas al negocio**  
   Más allá de F1 o accuracy, se podrían agregar:
   - Métricas de negocio (por ejemplo, uplift esperado, valor esperado de las recomendaciones).
   - Métricas de cobertura (qué porcentaje de clientes recibe recomendaciones razonables).
   - Métricas específicas para el sistema de recomendación y el chatbot (por ejemplo, interacción promedio, preguntas más frecuentes).

4. **Mejor modularización y empaquetado**  
   El código ya está separado en módulos (Airflow, app, recsys, llm), pero una evolución natural sería:
   - Convertir algunos componentes en **paquetes reutilizables** (por ejemplo, el preprocesador, el loader de datos o el motor de recomendaciones).
   - Unificar configuraciones mediante archivos de `config` o variables de entorno bien documentadas.



En conjunto, el proyecto nos permitió experimentar un ciclo MLOps completo: desde la construcción de un pipeline reproducible con Airflow, hasta su exposición a usuarios vía APIs y frontends. La principal conclusión es que **el valor de un modelo no es solo su desempeño, sino la infraestructura y los procesos que lo rodean**, y que MLOps es justamente el marco que permite que ese valor sea sostenible en el tiempo.

