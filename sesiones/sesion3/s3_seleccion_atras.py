"""Selección hacia atrás de grupos de términos usando AIC."""

import pandas as pd
import statsmodels.api as sm
from threadpoolctl import threadpool_limits


def seleccion_atras(respuesta, matriz, grupos):
    """Elimina grupos de predictores mientras la eliminación reduzca el AIC."""
    activas = list(grupos)

    def ajustar(variables):
        columnas = ["const"] + [columna for variable in variables for columna in grupos[variable]]
        return sm.OLS(respuesta, matriz[columnas], missing="raise").fit()

    with threadpool_limits(limits=1):
        modelo = ajustar(activas)
        pasos = [{"Paso": 0, "Eliminada": "—", "AIC": modelo.aic, "Términos": len(activas)}]
        while activas:
            candidatos = []
            for variable in activas:
                restantes = [nombre for nombre in activas if nombre != variable]
                candidato = ajustar(restantes)
                candidatos.append((candidato.aic, variable, candidato))
            aic, retirada, candidato = min(candidatos, key=lambda resultado: resultado[0])
            if aic >= modelo.aic - 1e-7:
                break
            activas.remove(retirada)
            modelo = candidato
            pasos.append({"Paso": len(pasos), "Eliminada": retirada, "AIC": aic, "Términos": len(activas)})
    return modelo, activas, pd.DataFrame(pasos)
