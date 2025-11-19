import os
import requests
import gradio as gr

# En Docker, backend se llama "backend" (nombre del servicio en docker-compose)
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


def call_backend(
    customer_id,
    product_id,
    year,
    week,
    region_id,
    customer_type,
    brand,
    category,
    sub_category,
    segment,
    package,
    size,
    num_deliver_per_week,
    num_visit_per_week,
):
    """Llama al backend FastAPI y devuelve un texto para mostrar en Gradio."""
    payload = {
        "customer_id": int(customer_id),
        "product_id": int(product_id),
        "year": int(year),
        "week": int(week),
        "region_id": int(region_id) if region_id is not None else None,
        "customer_type": customer_type or None,
        "brand": brand or None,
        "category": category or None,
        "sub_category": sub_category or None,
        "segment": segment or None,
        "package": package or None,
        "size": float(size) if size is not None else None,
        "num_deliver_per_week": float(num_deliver_per_week) if num_deliver_per_week is not None else None,
        "num_visit_per_week": float(num_visit_per_week) if num_visit_per_week is not None else None,
    }

    try:
        resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    except Exception as e:
        return f"❌ Error al conectar con el backend: {e}"

    if resp.status_code != 200:
        return f"❌ Error del backend ({resp.status_code}): {resp.text}"

    data = resp.json()
    proba = data.get("probability", 0.0)
    pred = data.get("will_buy_next_week", 0)

    texto_pred = (
        "✅ Se espera que el cliente compre el producto la próxima semana."
        if pred == 1
        else "⚠️ Es poco probable que el cliente compre el producto la próxima semana."
    )

    resultado = (
        f"Probabilidad estimada de compra: **{proba:.2%}**\n\n"
        f"Predicción binaria: **{pred}**\n\n"
        f"{texto_pred}"
    )
    return resultado


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # SodAI Drinks 🥤 – Panel de Predicciones

        Esta interfaz permite consultar la **probabilidad de que un cliente compre un producto la próxima semana**.

        ### Cómo usar la aplicación:
        1. Ingresa los datos del cliente y del producto.
        2. Especifica el año y la semana ISO para los que quieres evaluar la probabilidad.
        3. Presiona **"Predecir"**.
        4. Verás la probabilidad estimada y una interpretación simple (comprará / no comprará).

        > Nota: Los resultados se basan en el modelo entrenado previamente y en los patrones históricos de compra.
        """
    )

    with gr.Row():
        with gr.Column():
            customer_id = gr.Number(label="ID Cliente (customer_id)", value=1, precision=0)
            product_id = gr.Number(label="ID Producto (product_id)", value=1, precision=0)
            year = gr.Number(label="Año (ISO)", value=2024, precision=0)
            week = gr.Number(label="Semana (ISO)", value=1, precision=0)

            region_id = gr.Number(label="Region ID (opcional)", value=None, precision=0)
            customer_type = gr.Textbox(
                label="Tipo de cliente (customer_type)",
                placeholder="TIENDA DE CONVENIENCIA, etc.",
            )

        with gr.Column():
            brand = gr.Textbox(label="Marca (brand)")
            category = gr.Textbox(label="Categoría (category)")
            sub_category = gr.Textbox(label="Subcategoría (sub_category)")
            segment = gr.Textbox(label="Segmento (segment)")
            package = gr.Textbox(label="Envase (package)")
            size = gr.Number(label="Tamaño (size, litros)", value=None)

            num_deliver_per_week = gr.Number(
                label="Entregas por semana (num_deliver_per_week)",
                value=None,
            )
            num_visit_per_week = gr.Number(
                label="Visitas por semana (num_visit_per_week)",
                value=None,
            )

    boton = gr.Button("Predecir")
    salida = gr.Markdown(label="Resultado de la predicción")

    boton.click(
        fn=call_backend,
        inputs=[
            customer_id,
            product_id,
            year,
            week,
            region_id,
            customer_type,
            brand,
            category,
            sub_category,
            segment,
            package,
            size,
            num_deliver_per_week,
            num_visit_per_week,
        ],
        outputs=salida,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
