"""Tablas y resúmenes de evaluación de S6; no entrena ni selecciona modelos."""

import numpy as np
import pandas as pd


def tabla_confusion(reales, predichos, clases=None, incluir_rechazos=False):
    """Predicción en filas, realidad en columnas; rechazos en una fila separada."""
    reales = np.asarray(reales)
    predichos = np.asarray(predichos)
    clases = np.unique(reales) if clases is None else np.asarray(clases)
    tabla = pd.crosstab(pd.Series(predichos, name="Predicción"),
                        pd.Series(reales, name="Real"))
    tabla = tabla.reindex(index=clases, columns=clases, fill_value=0)
    if incluir_rechazos:
        tabla.loc["Sin decisión"] = [np.sum(pd.isna(predichos) & (reales == c)) for c in clases]
    return tabla.rename_axis(index="Predicción", columns="Real")


def evaluar(reales, predichos, clases=None):
    """Acierto entre clasificadas, cobertura y acierto sobre todas las filas."""
    reales, predichos = np.asarray(reales), np.asarray(predichos)
    aceptadas = ~pd.isna(predichos)
    aciertos = aceptadas & (reales == predichos)
    return pd.Series({"Acierto entre clasificadas": aciertos.sum() / aceptadas.sum() if aceptadas.any() else np.nan,
                      "Cobertura": aceptadas.mean(),
                      "Acierto sobre todas las observaciones": aciertos.mean()})


def filas_comparacion_variantes(variantes, reales, clase_minoritaria):
    """Resume etiquetas ya predichas en el orden recibido, sin llamar a fit."""
    minoritarias = reales.eq(clase_minoritaria)
    filas = []
    for nombre, modelo, etiquetas in variantes:
        assert etiquetas.index.equals(reales.index), "Productos desalineados."
        assert etiquetas.isin(modelo.clases).all(), "Etiqueta desconocida."
        aciertos = etiquetas.eq(reales)
        filas.append({
            "Variante": nombre,
            "Entradas": len(modelo.variables),
            f"Aciertos / {len(reales)}": int(aciertos.sum()),
            "Accuracy (%)": 100 * aciertos.mean(),
            f"Recuperados {clase_minoritaria} / {int(minoritarias.sum())}": int(aciertos[minoritarias].sum()),
            f"Recall {clase_minoritaria} (%)": 100 * aciertos[minoritarias].mean(),
        })
    return filas