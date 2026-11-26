"""Métricas y matriz de confusión para la clasificación de S7."""  # Resume el propósito de este módulo auxiliar.

from types import SimpleNamespace  # Permite devolver varias métricas agrupadas en un solo objeto.

import numpy as np  # Proporciona arrays y operaciones numéricas.
import pandas as pd  # Proporciona tablas, series y recuentos por clase.


def confusion(reales, predichas, clases):  # Define una matriz de confusión con un orden de clases explícito.
    """Devuelve la matriz con predicciones en filas y clases reales en columnas."""  # Documenta orientación y propósito.
    return (  # Devuelve una tabla procesada en varios pasos encadenados.
        pd.crosstab(  # Cuenta cuántas veces aparece cada pareja predicción-realidad.
            pd.Series(np.asarray(predichas), name="Predicción"),  # Convierte las predicciones en una serie con nombre.
            pd.Series(np.asarray(reales), name="Real"),  # Convierte las clases observadas en una serie.
        )  # Cierra la construcción de la tabla de recuentos.
        .reindex(index=clases, columns=clases, fill_value=0)  # Mantiene todas las clases, incluso si alguna falta.
        .rename_axis(index="Predicción", columns="Real")  # Deja explícita la orientación de filas y columnas.
    )  # Cierra y devuelve la matriz terminada.


def evaluar_clasificador(reales, predichas, clases):  # Define el resumen de acierto y matriz para un clasificador.
    """Resume acierto global, matriz de confusión y acierto por clase."""  # Explica qué valores devolverá.
    tabla = confusion(reales, predichas, clases)  # Construye primero la matriz reutilizable.
    acierto = np.mean(np.asarray(reales) == np.asarray(predichas))  # Cuenta la fracción de etiquetas iguales.
    por_clase = pd.Series(  # Prepara una serie indexada por las etiquetas de clase.
        np.divide(  # Divide los aciertos por el total real de cada clase.
            np.diag(tabla),  # Extrae la diagonal: número de aciertos dentro de cada clase.
            tabla.sum(axis=0),  # Cuenta los casos reales de cada clase para usarlos como denominador.
            out=np.full(len(clases), np.nan),  # Usa NaN si no hay casos reales de una clase.
            where=tabla.sum(axis=0) > 0,  # Divide únicamente las clases con al menos un caso real.
        ),  # Calcula aciertos por clase como diagonal / total real.
        index=clases,  # Etiqueta cada resultado con el nombre de la clase.
        name="Acierto por clase",  # Nombra la serie para facilitar su interpretación.
    )  # Completa la serie de aciertos por clase.
    return SimpleNamespace(confusion=tabla, acierto=acierto, por_clase=por_clase)  # Agrupa y devuelve los tres resultados.
