"""Criterios descriptivos de selección de S6, sin datos ni ajustes globales."""

import numpy as np
import pandas as pd


def gini_medias(x):
    """Desigualdad de medias no negativas; no es impureza ni importancia causal."""
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("El índice requiere medias no negativas.")
    return np.abs(x[:, None] - x[None, :]).sum() / (2*len(x)*x.sum()) if x.sum() else 0.0


def r2_multiples(correlaciones):
    """R² de cada variable explicada por todas las demás, con intercepto."""
    matriz = np.asarray(correlaciones, dtype=float)
    if len(matriz) == 1:
        return pd.Series([0.0], index=correlaciones.index)
    if np.linalg.matrix_rank(matriz) == len(matriz):
        valores = 1 - 1 / np.diag(np.linalg.inv(matriz))
    else:
        valores = []
        for j in range(len(matriz)):
            otras = np.arange(len(matriz)) != j
            r = matriz[j, otras]
            valores.append(r @ np.linalg.pinv(matriz[np.ix_(otras, otras)]) @ r)
    return pd.Series(np.clip(valores, 0, 1), index=correlaciones.index, name="R²")


def reducir_por_r2(X, limite=0.5):
    """Retirada original, incluida la última iteración por debajo del límite."""
    correlaciones = X.corr()
    historial = []
    maximo = limite
    while maximo >= limite and len(correlaciones) > 1:
        valores = r2_multiples(correlaciones)
        variable = valores.idxmax()
        maximo = valores[variable]
        correlaciones = correlaciones.drop(index=variable, columns=variable)
        historial.append({"Iteración": len(historial)+1, "Retirada": variable,
                          "R² máximo": maximo, "Variables restantes": len(correlaciones)})
    return list(correlaciones.columns), pd.DataFrame(historial)