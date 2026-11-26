"""Ajuste, selección y diagnóstico de la regresión logística de la sesión 4.

Las funciones reciben sus datos explícitamente: no leen variables del notebook
ni ajustan modelos al importar este módulo.
"""

from numbers import Integral

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.stats import chi2
from threadpoolctl import threadpool_limits


REFERENCIAS = {
    "status": "no checking account",
    "credit.hist": "no credits taken/all credits paid back duly",
    "purpose": "others",
    "savings": "unknown/no savings account",
    "employed.time": "unemployed",
    "status.sex": "male:single",
    "debtors": "none",
    "property": "unknown/no property",
    "installment2": "none",
    "housing": "for free",
    "job": "unskilled - resident",
    "telephone": "none",
    "foreign": "no",
}


def datos_binarios(observado, probabilidades):
    """Comprueba vectores emparejados: respuesta 0/1 y probabilidades finitas."""
    real = np.asarray(observado, dtype=float)
    predicho = np.asarray(probabilidades, dtype=float)
    if real.ndim != 1 or predicho.ndim != 1 or len(real) != len(predicho):
        raise ValueError("Empareja una respuesta y una probabilidad por observación.")
    if not len(real) or not np.isin(real, [0, 1]).all():
        raise ValueError("La respuesta debe ser 0 o 1, sin ausentes y no vacía.")
    if not np.isfinite(predicho).all() or ((predicho < 0) | (predicho > 1)).any():
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1.")
    return real.astype(int), predicho


def ajustar_modelo(respuesta, matriz, grupos, variables):
    """Ajusta los grupos indicados sobre el diseño y las filas suministrados."""
    if pd.isna(respuesta).any() or matriz.isna().any().any():
        raise ValueError("Resuelve los ausentes antes de ajustar o seleccionar.")
    columnas = ["const"] + [columna for variable in variables for columna in grupos[variable]]
    with threadpool_limits(limits=1):
        modelo = sm.GLM(
            respuesta, matriz[columnas], family=sm.families.Binomial(), missing="raise"
        ).fit(maxiter=100)
    if not modelo.converged or not np.isfinite(modelo.llf):
        raise RuntimeError("El modelo no converge; revisa la especificación y la separación.")
    return modelo


def seleccion_atras(respuesta, matriz, grupos, variables):
    """Retira grupos completos únicamente mientras reduzcan el AIC."""
    activas = list(variables)
    modelo = ajustar_modelo(respuesta, matriz, grupos, activas)
    historial = [{"Paso": 0, "Eliminada": "—", "AIC": modelo.aic, "Variables": len(activas)}]
    while activas:
        candidatos = []
        for variable in activas:
            restantes = [nombre for nombre in activas if nombre != variable]
            candidato = ajustar_modelo(respuesta, matriz, grupos, restantes)
            candidatos.append((candidato.aic, variable, candidato))
        aic, retirada, candidato = min(candidatos, key=lambda resultado: resultado[0])
        if aic >= modelo.aic - 1e-7:
            break
        activas.remove(retirada)
        modelo = candidato
        historial.append({"Paso": len(historial), "Eliminada": retirada,
                          "AIC": aic, "Variables": len(activas)})
    return modelo, activas, pd.DataFrame(historial)


def tabla_por_cuantiles(observado, predicho, grupos=10):
    """Cuenta pagos observados y esperados en grupos por cuantiles."""
    real, probabilidades = datos_binarios(observado, predicho)
    if isinstance(grupos, bool) or not isinstance(grupos, Integral) or grupos < 1:
        raise ValueError("Indica un número entero positivo de grupos.")
    tabla = pd.DataFrame({"observado": real, "predicho": probabilidades})
    tabla["intervalo"] = pd.qcut(tabla["predicho"], q=grupos, duplicates="drop")
    return tabla.groupby("intervalo", observed=True).agg(
        n=("observado", "size"), obs=("observado", "sum"), exp=("predicho", "sum"),
    )


def hosmer_lemeshow(observado, predicho, grupos=10):
    """Contraste aproximado por grupos efectivos; no prueba calibración perfecta."""
    tabla = tabla_por_cuantiles(observado, predicho, grupos)
    if len(tabla) <= 2:
        raise ValueError("Se necesitan más de dos grupos de probabilidad distintos.")
    no_obs = tabla["n"] - tabla["obs"]
    no_exp = tabla["n"] - tabla["exp"]
    if (tabla["exp"] <= 0).any() or (no_exp <= 0).any():
        raise ValueError("Hay grupos con frecuencia esperada cero.")
    estadistico = (((tabla["obs"] - tabla["exp"]) ** 2 / tabla["exp"])
                   + ((no_obs - no_exp) ** 2 / no_exp)).sum()
    gl = len(tabla) - 2
    return pd.Series({"Chi-cuadrado": estadistico, "Grados de libertad": gl,
                      "p-valor": chi2.sf(estadistico, gl)})


def intervalos_perfil(modelo, nivel=0.95):
    """IC de coeficientes mediante perfil de verosimilitud, antes de exponenciar."""
    if not 0 < nivel < 1:
        raise ValueError("El nivel de confianza debe estar entre 0 y 1.")
    matriz = np.asarray(modelo.model.exog, dtype=float)
    respuesta = np.asarray(modelo.model.endog, dtype=float)
    beta = np.asarray(modelo.params)
    errores = np.asarray(modelo.bse)
    nombres = modelo.model.exog_names
    corte = chi2.ppf(nivel, df=1)
    limites = []
    with threadpool_limits(limits=1):
        for j, centro in enumerate(beta):
            otras = np.arange(len(beta)) != j

            def diferencia(valor):
                # El offset fija un coeficiente; los restantes se vuelven a estimar.
                ajuste = sm.GLM(
                    respuesta, matriz[:, otras], offset=matriz[:, j] * valor,
                    family=sm.families.Binomial(),
                ).fit(maxiter=100, tol=1e-9)
                if not ajuste.converged or not np.isfinite(ajuste.llf):
                    raise RuntimeError(f"El perfil no converge para {nombres[j]}.")
                return 2 * (modelo.llf - ajuste.llf) - corte

            extremos = []
            for signo in [-1, 1]:
                paso = max(2 * errores[j], 1e-8)
                extremo = centro + signo * paso
                for _ in range(30):
                    if diferencia(extremo) >= 0:
                        break
                    paso *= 2
                    extremo = centro + signo * paso
                else:
                    raise RuntimeError("No se encontró un límite finito para el perfil.")
                inferior, superior = sorted([centro, extremo])
                extremos.append(brentq(diferencia, inferior, superior, xtol=1e-10))
            limites.append(extremos)
    alfa = 1 - nivel
    columnas = [f"{100 * alfa / 2:g} %", f"{100 * (1 - alfa / 2):g} %"]
    return pd.DataFrame(limites, index=nombres, columns=columnas)


def metricas_corte(observado, probabilidades, corte):
    """Métricas con pago=1; una clase ausente tiene tasa indefinida, no cero."""
    real, probabilidades = datos_binarios(observado, probabilidades)
    if not np.isfinite(corte) or not 0 <= corte <= 1:
        raise ValueError("El corte debe estar entre 0 y 1.")
    real = real.astype(bool)
    predicho = probabilidades > corte
    return {
        "Umbral": corte,
        "Sensibilidad": (predicho & real).sum() / real.sum() if real.any() else np.nan,
        "Especificidad": (~predicho & ~real).sum() / (~real).sum() if (~real).any() else np.nan,
        "Exactitud": (predicho == real).mean(),
    }