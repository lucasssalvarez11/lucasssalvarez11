import os
import requests
import pandas as pd
import gradio as gr

# En Docker, el backend se llamará "backend"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")


def get_recommendations(customer_id, k):
    """
    Llama al backend /recommend y devuelve texto + tabla de recomendaciones.
    """
    try:
        customer_id = int(customer_id)
    except (TypeError, ValueError):
        return "❌ Debes ingresar un ID de cliente válido (entero).", pd.DataFrame()

    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5

    try:
        resp = requests.get(
            f"{BACKEND_URL}/recommend",
            params={"customer_id": customer_id, "k": k},
            timeout=10,
        )
    except Exception as e:
        return f"❌ Error al conectar con el backend: {e}", pd.DataFrame()

    if resp.status_code != 200:
        return f"❌ Error del backend ({resp.status_code}): {resp.text}", pd.DataFrame()

    data = resp.json()
    recs = data.get("recommendations", [])

    if not recs:
        return "⚠️ No se encontraron recomendaciones para este cliente.", pd.DataFrame()

    df = pd.DataFrame(recs)
    texto = (
        f"Se muestran hasta {len(df)} productos recomendados para el cliente **{customer_id}**.\n\n"
        "Las recomendaciones se basan en el historial de compras del cliente; si no tiene historial, "
        "se muestran los productos más populares globalmente."
    )

    return texto, df


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # SodAI Drinks 🥤 – Sistema de Recomendación

        Esta interfaz genera **5 recomendaciones de productos** (por defecto) para cualquier cliente de SodAI Drinks.

        ### Cómo usar:
        1. Ingresa el `ID de cliente` que quieres analizar.
        2. (Opcional) Ajusta cuántas recomendaciones quieres ver.
        3. Haz clic en **"Recomendar"**.
        4. Revisa la tabla con los productos sugeridos.

        Las recomendaciones se basan en el historial de compras:
        - Si el cliente es conocido, se priorizan los productos que más ha comprado.
        - Si el cliente no tiene historial, se usan los productos más populares globalmente.
        """
    )

    with gr.Row():
        customer_id_in = gr.Number(label="ID Cliente", value=1, precision=0)
        k_in = gr.Number(label="N° de recomendaciones", value=5, precision=0)

    boton = gr.Button("Recomendar")

    texto_out = gr.Markdown()
    tabla_out = gr.Dataframe(label="Productos recomendados")

    boton.click(
        fn=get_recommendations,
        inputs=[customer_id_in, k_in],
        outputs=[texto_out, tabla_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
