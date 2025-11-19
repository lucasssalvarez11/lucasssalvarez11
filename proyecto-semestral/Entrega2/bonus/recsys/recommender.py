import pandas as pd
from pathlib import Path
from typing import List, Dict


class SimpleRecommender:
    """
    Recomendador MUY simple basado en historial:

    - Si el cliente existe:
        recomienda los productos que más ha comprado.
    - Si no existe:
        recomienda los productos más populares globalmente.
    """

    def __init__(self, transacciones_path: str, productos_path: str):
        transacciones_path = Path(transacciones_path)
        productos_path = Path(productos_path)

        if not transacciones_path.exists():
            raise FileNotFoundError(f"No se encontró transacciones: {transacciones_path}")
        if not productos_path.exists():
            raise FileNotFoundError(f"No se encontró productos: {productos_path}")

        print(f"📥 Cargando transacciones desde: {transacciones_path}")
        self.df_trans = pd.read_parquet(transacciones_path)
        print(f"Transacciones: {self.df_trans.shape}")

        print(f"📥 Cargando productos desde: {productos_path}")
        self.df_prod = pd.read_parquet(productos_path)
        print(f"Productos: {self.df_prod.shape}")

        # Popularidad global por product_id (cantidad de transacciones)
        self.global_popularity = (
            self.df_trans.groupby("product_id")
            .size()
            .sort_values(ascending=False)
        )

        # Frecuencia por cliente-producto
        self.user_product_counts = (
            self.df_trans.groupby(["customer_id", "product_id"])
            .size()
            .rename("count")
            .reset_index()
        )

        # Metadatos de producto (por si quieres mostrar info más bonita)
        cols_meta = [
            "product_id",
            "brand",
            "category",
            "sub_category",
            "segment",
            "package",
            "size",
        ]
        self.product_meta = self.df_prod[[c for c in cols_meta if c in self.df_prod.columns]].drop_duplicates()

    def recommend(self, customer_id: int, k: int = 5) -> List[Dict]:
        """
        Devuelve una lista de hasta k productos recomendados
        para el cliente dado.
        """
        k = max(1, int(k))

        # Filtrar historial del cliente
        user_hist = self.user_product_counts[self.user_product_counts["customer_id"] == customer_id]

        if user_hist.empty:
            # Cliente nuevo / desconocido → top global
            top_product_ids = self.global_popularity.head(k).index.tolist()
        else:
            # Cliente conocido → productos que más ha comprado
            top_product_ids = (
                user_hist.sort_values("count", ascending=False)["product_id"]
                .head(k)
                .tolist()
            )

        # Recuperar metadatos
        recs = self.product_meta[self.product_meta["product_id"].isin(top_product_ids)].copy()

        # Ordenar según el ranking calculado
        recs["product_id"] = pd.Categorical(recs["product_id"], top_product_ids, ordered=True)
        recs = recs.sort_values("product_id")

        # Asegurar que devolvemos solo columnas razonables
        return recs.to_dict(orient="records")
