"""Búsqueda y votación de vecinos de la sesión 6, sin cargar datos al importar."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


class Vecinos:
    """Vecinos euclídeos reutilizables; voto uniforme o ponderado por kernel."""

    def __init__(self, X, y, consulta=None, max_k=21, estandarizar=False, *, semilla=12345):
        # Conserva clases, índices y semilla; aprende las escalas solo con entrenamiento.
        self.semilla = semilla
        self.dimension = np.asarray(X).shape[1]
        self.clases, self.y = np.unique(y, return_inverse=True)
        self.indice = X.index.copy() if consulta is None else consulta.index.copy()
        self.escala = np.asarray(X).std(axis=0, ddof=1) if estandarizar else np.ones(self.dimension)
        self.escala[self.escala == 0] = 1
        self.X = np.asarray(X, dtype=float) / self.escala
        self.consulta = None if consulta is None else np.asarray(consulta, dtype=float) / self.escala

        # Una caché acotada y un vecino extra; kneighbors(None) excluye solo la propia fila.
        self.max_k = min(max_k, len(X) - 1)
        self.buscador = NearestNeighbors(n_neighbors=self.max_k + 1, algorithm="auto", n_jobs=1).fit(self.X)
        with threadpool_limits(limits=1):
            self.distancias, self.indices = self.buscador.kneighbors(self.consulta)

    def probabilidades(self, k=1, kernel="uniforme"):
        if not 1 <= k <= self.max_k:
            raise ValueError(f"k debe estar entre 1 y {self.max_k}.")
        indices = self.indices

        # Pesos originales: radio uniforme, rango óptimo o distancia relativa.
        if kernel == "uniforme":
            radio = self.distancias[:, k-1] * np.sqrt(1 + 1e-4)
            pesos = (self.distancias <= radio[:, None]).astype(float)
        else:
            indices = indices[:, :k]
            if kernel == "óptimo":
                rango = np.arange(1, k+1, dtype=float)
                exponente = 1 + 2 / self.dimension
                pesos_rango = (1 + self.dimension/2 - self.dimension *
                               (rango**exponente - (rango-1)**exponente) / (2*k**(2/self.dimension))) / k
                pesos = np.broadcast_to(pesos_rango, indices.shape)
            else:
                ancho = np.maximum(self.distancias[:, k], 1e-6)
                u = np.clip(self.distancias[:, :k] / ancho[:, None], 1e-6, 1-1e-6)
                if kernel == "triangular":
                    pesos = 1-u
                elif kernel == "epanechnikov":
                    pesos = 0.75*(1-u*u)
                else:
                    raise ValueError("Kernel desconocido.")

        votos = np.column_stack([(pesos * (self.y[indices] == j)).sum(axis=1)
                                 for j in range(len(self.clases))])
        if kernel == "uniforme":
            # Recupera todos los empates que desbordan la caché, sin eliminar duplicados ajenos.
            desbordadas = np.flatnonzero(pesos[:, -1] > 0)
            for fila in desbordadas:
                punto = self.X[fila] if self.consulta is None else self.consulta[fila]
                cercanos = self.buscador.radius_neighbors(punto.reshape(1, -1),
                           radius=np.nextafter(radio[fila], np.inf), return_distance=False)[0]
                if self.consulta is None:
                    cercanos = cercanos[cercanos != fila]
                votos[fila] = np.bincount(self.y[cercanos], minlength=len(self.clases))
        self.ultimos_votos = votos
        return pd.DataFrame(votos / votos.sum(axis=1, keepdims=True),
                            columns=self.clases, index=self.indice)

    def predecir(self, k=1, kernel="uniforme", minimo=0):
        probabilidades = self.probabilidades(k, kernel)
        valores = probabilidades.to_numpy()
        ganadora = valores.argmax(axis=1)
        if kernel == "uniforme":
            rng = np.random.default_rng(self.semilla)
            for fila in np.flatnonzero((valores == valores.max(axis=1, keepdims=True)).sum(axis=1) > 1):
                ganadora[fila] = rng.choice(np.flatnonzero(valores[fila] == valores[fila].max()))
        predichos = pd.Series(self.clases[ganadora], index=self.indice, name="Predicción")

        # El mínimo conserva el incremento original por vecinos empatados adicionales.
        if minimo and kernel == "uniforme":
            extras = self.ultimos_votos.sum(axis=1) - k
            rechazo = self.ultimos_votos.max(axis=1) < minimo + extras
            predichos = predichos.where(~rechazo)
        return predichos


def barrido_vecinos(ajuste, reales, valores_k, kernel="uniforme"):
    """Cambia únicamente el voto: reutiliza la búsqueda de ajuste."""
    return pd.Series({k: np.mean(ajuste.predecir(k, kernel).to_numpy() == np.asarray(reales))
                      for k in valores_k}, name="Proporción de aciertos").rename_axis("k").to_frame()


def to_dummy(x, ordinal=False, orden=None):
    """Códigos ordinales desde 1 o dummies con la última categoría de referencia."""
    serie = pd.Series(x)
    if ordinal:
        if orden is None:
            raise ValueError("Indica el orden de las categorías.")
        codigos = pd.Series(pd.Categorical(serie, categories=orden, ordered=True).codes,
                            index=serie.index, name=serie.name)
        return (codigos + 1).where(codigos >= 0)
    categorias = sorted(serie.dropna().unique())
    return pd.DataFrame({f"{categoria}.dummy": (serie == categoria).astype(int)
                         for categoria in categorias[:-1]}, index=serie.index)