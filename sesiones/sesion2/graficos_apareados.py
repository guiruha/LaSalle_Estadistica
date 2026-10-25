"""Gráficos descriptivos para dos mediciones emparejadas por individuo."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def graficos_apareados(
    tabla: pd.DataFrame,
    titulo: str = "Todos los pares",
    columna_izquierda: str = "LftSpan",
    columna_derecha: str = "RtSpan",
    etiqueta_izquierda: str = "Izquierda",
    etiqueta_derecha: str = "Derecha",
    unidades: str = "cm",
) -> tuple[plt.Figure, np.ndarray]:
    """Dibuja cuatro vistas descriptivas de pares completos.

    Los paneles muestran la relación entre medidas, un gráfico de Bland–Altman,
    las diferencias individuales y los perfiles que unen ambas medidas por par.
    Los límites de Bland–Altman son descriptivos, no un intervalo de confianza.
    """
    pares = tabla[[columna_izquierda, columna_derecha]].dropna()
    if pares.empty:
        raise ValueError("No hay pares completos para representar.")

    izquierda = pares[columna_izquierda].to_numpy(dtype=float)
    derecha = pares[columna_derecha].to_numpy(dtype=float)
    diferencias = derecha - izquierda
    promedios = (izquierda + derecha) / 2

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    axes[0, 0].scatter(
        izquierda, derecha, s=55, color="#9fc3e4", edgecolor="black",
        linewidth=0.7, alpha=0.8,
    )
    minimo = min(izquierda.min(), derecha.min())
    maximo = max(izquierda.max(), derecha.max())
    axes[0, 0].plot([minimo, maximo], [minimo, maximo], color="#800000", linewidth=1.8)
    axes[0, 0].set(
        xlabel=f"{etiqueta_izquierda} ({unidades})",
        ylabel=f"{etiqueta_derecha} ({unidades})",
        title="Relación entre las dos medidas",
    )

    media_diferencias = diferencias.mean()
    desviacion_diferencias = diferencias.std(ddof=1) if len(diferencias) > 1 else 0.0
    axes[0, 1].scatter(
        promedios, diferencias, s=55, color="#9fc3e4", edgecolor="black",
        linewidth=0.7, alpha=0.8,
    )
    axes[0, 1].axhline(media_diferencias, color="#800000", linewidth=2, label="Diferencia media")
    for limite in (
        media_diferencias - 1.96 * desviacion_diferencias,
        media_diferencias + 1.96 * desviacion_diferencias,
    ):
        axes[0, 1].axhline(limite, color="#800000", linestyle="--", linewidth=1.5)
    axes[0, 1].set(
        xlabel=f"Promedio de cada par ({unidades})",
        ylabel=f"{etiqueta_derecha} − {etiqueta_izquierda} ({unidades})",
        title="Bland–Altman (límites descriptivos)",
    )

    axes[1, 0].scatter(
        np.arange(1, len(diferencias) + 1), diferencias, s=55,
        color="#9fc3e4", edgecolor="black", linewidth=0.7,
    )
    axes[1, 0].axhline(0, color="#800000", linestyle="--", linewidth=1.8)
    axes[1, 0].set(
        xlabel="Posición del par",
        ylabel=f"{etiqueta_derecha} − {etiqueta_izquierda} ({unidades})",
        title="Diferencia individual",
    )

    perfiles = axes[1, 1]
    leyenda_vista: set[str] = set()
    for valor_izquierda, valor_derecha in zip(izquierda, derecha):
        if valor_derecha < valor_izquierda:
            color, etiqueta, transparencia = "#34699a", f"{etiqueta_derecha} menor", 0.45
        elif valor_derecha > valor_izquierda:
            color, etiqueta, transparencia = "#9fc3e4", f"{etiqueta_derecha} mayor", 0.65
        else:
            color, etiqueta, transparencia = "#b0b0b0", "Iguales", 0.3
        perfiles.plot(
            [0, 1], [valor_izquierda, valor_derecha], color=color,
            alpha=transparencia, linewidth=1.0, marker="o", markersize=3.5,
            markeredgecolor="#444444", markeredgewidth=0.4,
            label=etiqueta if etiqueta not in leyenda_vista else None,
        )
        leyenda_vista.add(etiqueta)
    perfiles.plot(
        [0, 1], [izquierda.mean(), derecha.mean()], color="#800000",
        linewidth=3, marker="o", markersize=7, label="Media de cada lado", zorder=5,
    )
    perfiles.set(
        xticks=[0, 1], xticklabels=[etiqueta_izquierda, etiqueta_derecha],
        ylabel=f"Medida ({unidades})", title="Perfiles por individuo",
    )
    perfiles.set_xlim(-0.1, 1.1)
    perfiles.legend(loc="best", fontsize=9)

    for eje in axes.flat:
        eje.title.set_fontsize(15)
        eje.xaxis.label.set_fontsize(12)
        eje.yaxis.label.set_fontsize(12)
        eje.tick_params(labelsize=10)
    fig.suptitle(titulo, fontsize=18)
    fig.tight_layout()
    plt.show()
    return fig, axes
