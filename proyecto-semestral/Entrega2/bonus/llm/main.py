import os
import requests
import gradio as gr

# En Docker, usaremos el nombre del servicio: llm-backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8002")


def send_message(history, message):
    """
    Envía el mensaje al backend /chat y devuelve el historial actualizado.
    """
    if not message:
        return history

    # Agregar mensaje del usuario al historial
    history = history + [(message, None)]

    try:
        resp = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": message},
            timeout=10,
        )
    except Exception as e:
        answer = f"❌ Error al conectar con el backend: {e}"
        history[-1] = (message, answer)
        return history

    if resp.status_code != 200:
        answer = f"❌ Error del backend ({resp.status_code}): {resp.text}"
        history[-1] = (message, answer)
        return history

    data = resp.json()
    answer = data.get("answer", "")

    history[-1] = (message, answer)
    return history


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # SodAI Drinks 🥤 – Chatbot de Datos

        Este chatbot puede responder preguntas simples sobre el **dataset de SodAI Drinks**, por ejemplo:

        - ¿Cuántos clientes únicos hay en el dataset?
        - ¿Cuántos productos únicos se encuentran en los datos?
        - ¿Cuántas transacciones ha realizado el cliente 123?

        Escribe tu pregunta en el cuadro inferior y presiona **Enter** o el botón de enviar.
        """
    )

    chatbox = gr.Chatbot(label="Chat de SodAI Drinks")
    msg = gr.Textbox(
        label="Escribe tu pregunta",
        placeholder="Ej: ¿Cuántos clientes únicos hay en el dataset?",
    )
    clear = gr.Button("Limpiar chat")

    msg.submit(send_message, [chatbox, msg], [chatbox])
    msg.submit(lambda: "", None, msg)  # limpiar input después de enviar

    clear.click(lambda: [], None, chatbox)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
