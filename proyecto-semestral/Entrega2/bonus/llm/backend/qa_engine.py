import re
from pathlib import Path
from typing import Optional, Dict

import pandas as pd


class DataQABot:
    """
    Chatbot sencillo basado en reglas para responder
    preguntas sobre los datos de SodAI Drinks.

    Responde cosas como:
    - ¿Cuántos clientes únicos hay en el dataset?
    - ¿Cuántos productos únicos hay?
    - ¿Cuántas transacciones ha realizado el cliente X?
    """

    def __init__(self, transacciones_path: str, clientes_path: str, productos_path: str):
        transacciones_path = Path(transacciones_path)
        clientes_path = Path(clientes_path)
        productos_path = Path(productos_path)

        if not transacciones_path.exists():
            raise FileNotFoundError(f"No se encontró transacciones: {transacciones_path}")
        if not clientes_path.exists():
            raise FileNotFoundError(f"No se encontró clientes: {clientes_path}")
        if not productos_path.exists():
            raise FileNotFoundError(f"No se encontró productos: {productos_path}")

        print(f"📥 Cargando transacciones desde: {transacciones_path}")
        self.df_trans = pd.read_parquet(transacciones_path)
        print(f"Transacciones: {self.df_trans.shape}")

        print(f"📥 Cargando clientes desde: {clientes_path}")
        self.df_clientes = pd.read_parquet(clientes_path)
        print(f"Clientes: {self.df_clientes.shape}")

        print(f"📥 Cargando productos desde: {productos_path}")
        self.df_productos = pd.read_parquet(productos_path)
        print(f"Productos: {self.df_productos.shape}")

        # Pre-cálculos
        self.n_clientes_unicos = self.df_clientes["customer_id"].nunique()
        self.n_productos_unicos = self.df_productos["product_id"].nunique()
        self.n_transacciones = len(self.df_trans)

        # Transacciones por cliente
        self.tx_por_cliente: Dict[int, int] = (
            self.df_trans.groupby("customer_id")
            .size()
            .rename("n_transacciones")
            .to_dict()
        )

    def _extraer_id_cliente(self, question: str) -> Optional[int]:
        """
        Intenta extraer un ID de cliente desde la pregunta.
        Ejemplos:
          - "cliente 123"
          - "cliente 45?"
        """
        nums = re.findall(r"\d+", question)
        if not nums:
            return None
        try:
            return int(nums[0])
        except ValueError:
            return None

    def answer(self, question: str) -> str:
        """
        Devuelve una respuesta en texto a la pregunta dada.
        Lógica simple basada en palabras clave.
        """
        if not question:
            return "No entendí la pregunta. Intenta preguntarme algo sobre clientes, productos o transacciones."

        q = question.lower()

        # 1) Clientes únicos
        if ("cliente" in q or "clientes" in q) and ("unico" in q or "único" in q or "distinto" in q or "diferente" in q):
            return f"En el dataset hay **{self.n_clientes_unicos} clientes únicos**."

        # 2) Productos únicos
        if ("producto" in q or "productos" in q) and ("unico" in q or "único" in q or "distinto" in q or "diferente" in q):
            return f"En el dataset hay **{self.n_productos_unicos} productos únicos**."

        # 3) Total de transacciones
        if ("transaccion" in q or "transacción" in q or "transacciones" in q or "compras" in q) and (
            "total" in q or "hay" in q or "cuantas" in q or "cuántas" in q
        ):
            return f"En total, el dataset contiene **{self.n_transacciones} transacciones** registradas."

        # 4) Transacciones por cliente específico
        if "cliente" in q and ("transaccion" in q or "transacciones" in q or "compra" in q or "compras" in q):
            cliente_id = self._extraer_id_cliente(q)
            if cliente_id is None:
                return (
                    "Puedes preguntarme, por ejemplo: "
                    "`¿Cuántas transacciones ha realizado el cliente 123?`"
                )

            n_tx = self.tx_por_cliente.get(cliente_id, 0)
            if n_tx == 0:
                return f"El cliente **{cliente_id}** no registra transacciones en el dataset."
            else:
                return f"El cliente **{cliente_id}** ha realizado **{n_tx} transacciones** en el dataset."

        # 5) Otras preguntas frecuentes (ayuda)
        if "ayuda" in q or "puedes hacer" in q or "que sabes" in q:
            return (
                "Puedo responder preguntas simples sobre el dataset, por ejemplo:\n\n"
                "- ¿Cuántos clientes únicos hay en el dataset?\n"
                "- ¿Cuántas transacciones ha realizado el cliente 123?\n"
                "- ¿Cuántos productos únicos se encuentran en los datos?\n"
            )

        # 6) Fallback genérico
        return (
            "Soy un chatbot enfocado en estadísticas simples del dataset de SodAI Drinks. "
            "Puedes preguntarme, por ejemplo:\n\n"
            "- ¿Cuántos clientes únicos hay en el dataset?\n"
            "- ¿Cuántas transacciones ha realizado el cliente 123?\n"
            "- ¿Cuántos productos únicos se encuentran en los datos?\n"
        )
