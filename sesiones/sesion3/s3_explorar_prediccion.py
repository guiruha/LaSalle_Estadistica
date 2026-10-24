"""Vista HTML de una predicción individual en el conjunto de test."""

from base64 import b64encode
from io import BytesIO

import matplotlib.pyplot as plt
from IPython.display import HTML as HTMLDisplay


def explorar_prediccion(modelo, X_test, predicciones, test, columnas_originales, observacion):
    """Muestra la predicción, sus intervalos y los datos originales de un caso."""
    intervalo = modelo.get_prediction(
        X_test.loc[[observacion], modelo.model.exog_names]
    ).summary_frame(alpha=0.05).iloc[0]
    predicho = predicciones.loc[observacion]
    observado = test.loc[observacion, "Strength"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(predicciones, test["Strength"], s=40, color="#9fc3e4", alpha=0.65,
               edgecolors="#666666", linewidths=0.4, label="Observaciones de test", zorder=2)
    minimo = min(predicciones.min(), test["Strength"].min(), intervalo["obs_ci_lower"])
    maximo = max(predicciones.max(), test["Strength"].max(), intervalo["obs_ci_upper"])
    margen = (maximo - minimo) * 0.05
    ax.plot([minimo, maximo], [minimo, maximo], color="#800000", linewidth=1.8,
            label="Predicción = observación", zorder=1)
    ax.vlines(predicho, intervalo["obs_ci_lower"], intervalo["obs_ci_upper"],
              color="#9fc3e4", linewidth=11, alpha=0.95,
              label="IP individual (95 %)", zorder=3)
    ax.vlines(predicho, intervalo["mean_ci_lower"], intervalo["mean_ci_upper"],
              color="#34699a", linewidth=5, label="IC de la media (95 %)", zorder=4)
    ax.scatter(predicho, observado, s=150, color="#800000", edgecolors="white",
               linewidths=2, label=f"Observación {observacion}", zorder=5)
    ax.annotate(f"{observacion}", (predicho, observado), xytext=(10, 10),
                textcoords="offset points", fontsize=12, fontweight="bold", color="#800000",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
    ax.set(xlim=(minimo - margen, maximo + margen),
           ylim=(minimo - margen, maximo + margen))
    ax.set_xlabel("Resistencia predicha (MPa)", fontsize=16)
    ax.set_ylabel("Resistencia observada (MPa)", fontsize=16)
    ax.set_title(f"Predicción en test · observación {observacion}", fontsize=20, pad=18)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=14)
    ax.grid(color="#b0b0b0", alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    fig.tight_layout()

    tabla = test.loc[[observacion], columnas_originales].assign(
        Prediccion=predicho, Residuo=observado - predicho,
        IC_media_inf=intervalo["mean_ci_lower"], IC_media_sup=intervalo["mean_ci_upper"],
        IP_individual_inf=intervalo["obs_ci_lower"], IP_individual_sup=intervalo["obs_ci_upper"],
    ).T
    imagen = BytesIO()
    fig.savefig(imagen, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    tabla_html = tabla.to_html(header=False, border=0, float_format=lambda valor: f"{valor:.2f}")
    return HTMLDisplay(
        "<style>.prediccion-detalle table {border-collapse: collapse; width: 100%; "
        "color: #333; font-size: 13px;}"
        ".prediccion-detalle th, .prediccion-detalle td {padding: 5px 8px; "
        "border-bottom: 1px solid #ddd; text-align: right;}"
        ".prediccion-detalle th {text-align: left; color: #34699a; font-weight: 600;}"
        ".prediccion-detalle tr:nth-child(even) {background: #f3f7fb;}</style>"
        "<div style='display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start;'>"
        "<div style='flex: 2 1 520px; min-width: 0;'>"
        f"<img src='data:image/png;base64,{b64encode(imagen.getvalue()).decode()}' "
        f"alt='Predicciones en test para la observación {observacion}' "
        "style='display: block; width: 100%; height: auto;'></div>"
        "<div class='prediccion-detalle' style='flex: 1 1 260px; min-width: 0;'>"
        f"<h4 style='margin: 12px 0; color: #800000;'>Observación {observacion}</h4>"
        f"{tabla_html}</div></div>"
    )
