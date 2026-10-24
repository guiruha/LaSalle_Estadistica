"""Gráficos diagnósticos para un modelo lineal OLS."""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


def diagnosticos(modelo, titulo):
    """Representa residuos, normalidad, escala y apalancamiento en cuatro paneles."""
    influencia = modelo.get_influence()
    ajustados = np.asarray(modelo.fittedvalues)
    residuos = np.asarray(modelo.resid)
    estandarizados = influencia.resid_studentized_internal
    leverage = influencia.hat_matrix_diag
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    axes[0, 0].scatter(ajustados, residuos, s=30, color="#9fc3e4",
                       edgecolor="#666666", linewidth=0.4, alpha=0.65)
    curva = lowess(residuos, ajustados, frac=2 / 3, it=3)
    axes[0, 0].plot(curva[:, 0], curva[:, 1], color="#800000", linewidth=1.8)
    axes[0, 0].axhline(0, color="#666666", linestyle="--")
    axes[0, 0].set(title="Residuos frente a ajustados", xlabel="Ajustados", ylabel="Residuos")

    sm.ProbPlot(estandarizados).qqplot(line="45", ax=axes[0, 1], marker="o", markersize=4, alpha=0.65)
    for linea in axes[0, 1].lines:
        if linea.get_marker() == "o":
            linea.set(markerfacecolor="#9fc3e4", markeredgecolor="#666666", markeredgewidth=0.6)
        else:
            linea.set(color="#800000", linewidth=1.8)
    axes[0, 1].set(title="Q–Q de residuos estandarizados", xlabel="Cuantiles normales",
                   ylabel="Residuos estandarizados")

    escala = np.sqrt(np.abs(estandarizados))
    axes[1, 0].scatter(ajustados, escala, s=30, color="#9fc3e4",
                       edgecolor="#666666", linewidth=0.4, alpha=0.65)
    curva = lowess(escala, ajustados, frac=2 / 3, it=3)
    axes[1, 0].plot(curva[:, 0], curva[:, 1], color="#800000", linewidth=1.8)
    axes[1, 0].set(title="Escala y localización", xlabel="Ajustados",
                   ylabel="Raíz de |residuo estandarizado|")

    axes[1, 1].scatter(leverage, estandarizados, s=30, color="#9fc3e4",
                       edgecolor="#666666", linewidth=0.4, alpha=0.65)
    axes[1, 1].axhline(0, color="#666666", linestyle="--")
    h = np.linspace(max(leverage.min(), 1e-5), leverage.max() * 1.05, 200)
    for distancia in [0.5, 1]:
        limite = np.sqrt(distancia * len(modelo.params) * (1 - h) / h)
        axes[1, 1].plot(h, limite, "--", color="#800000", alpha=0.65, label=f"Cook = {distancia}")
        axes[1, 1].plot(h, -limite, "--", color="#800000", alpha=0.65)
    axes[1, 1].set(title="Residuos y apalancamiento", xlabel="Apalancamiento",
                   ylabel="Residuo estandarizado",
                   ylim=(estandarizados.min() - 1, estandarizados.max() + 1))
    axes[1, 1].legend(fontsize=10)
    for eje in axes.flat:
        eje.title.set_fontsize(18)
        eje.xaxis.label.set_fontsize(16)
        eje.yaxis.label.set_fontsize(16)
        eje.tick_params(labelsize=14)
    fig.suptitle(titulo, fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
