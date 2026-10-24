"""Log-verosimilitud perfilada para estimar el parámetro Box–Cox."""

import numpy as np
from scipy.stats import boxcox


def logverosimilitud_boxcox(valor_lambda, modelo):
    """Calcula la log-verosimilitud perfilada para una respuesta y un modelo OLS."""
    respuesta = np.asarray(modelo.model.endog, dtype=float)
    if (respuesta <= 0).any():
        raise ValueError("Box–Cox requiere una respuesta estrictamente positiva.")
    transformada = boxcox(respuesta, valor_lambda)
    coeficientes = modelo.model.pinv_wexog @ transformada
    residuos = transformada - modelo.model.exog @ coeficientes
    n = len(respuesta)
    return -n / 2 * np.log(np.sum(residuos ** 2) / n) + (valor_lambda - 1) * np.log(respuesta).sum()
