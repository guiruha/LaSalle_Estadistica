"""Gráficos y vistas HTML de la sesión 4, sin datos globales del notebook."""

from base64 import b64encode
from html import escape
from io import BytesIO
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML as HTMLDisplay, display
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.graphics.mosaicplot import mosaic

from s4_modelos import datos_binarios, metricas_corte, tabla_por_cuantiles


COLOR_DATOS = "#9fc3e4"
COLOR_CLAVE = "#800000"
COLOR_BORDE = "#666666"


def configurar_estilo():
    """Aplica el formato original únicamente cuando el cuaderno lo solicita."""
    plt.rcParams.update({"figure.figsize": (11, 6), "figure.dpi": 110,
                         "axes.titlesize": 18, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12})
    pd.set_option("display.max_columns", 25)


def imagen_widget(fig, descripcion):
    """Convierte una figura en HTML y la cierra para no acumular imágenes."""
    imagen = BytesIO()
    fig.savefig(imagen, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return (f"<img src='data:image/png;base64,{b64encode(imagen.getvalue()).decode()}' "
            f"alt='{escape(descripcion, quote=True)}' style='max-width: 100%; height: auto;'>")


def mostrar_html(contenido):
    """Muestra una vista estática cuando el editor no representa controles."""
    display(HTMLDisplay(contenido))


def mosaico_pago(datos, variable):
    """Muestra categorías dentro de cada respuesta y los conteos conjuntos."""
    fig, ax = plt.subplots(figsize=(13, 7))
    _, rectangulos = mosaic(
        datos, ["y", variable], ax=ax,
        properties=lambda clave: {
            "facecolor": COLOR_CLAVE if clave[0] == "no" else COLOR_DATOS,
            "edgecolor": COLOR_BORDE,
        },
        labelizer=lambda _: "", title=f"Composición por pago: {variable}", axes_label=False,
    )
    for clave, (x, y_rect, ancho, alto) in rectangulos.items():
        if alto >= 0.08:
            ax.text(x + ancho / 2, y_rect + alto / 2, textwrap.fill(str(clave[1]), width=22),
                    ha="center", va="center", fontsize=10)
    centros = []
    for clase in ["no", "yes"]:
        x, _, ancho, _ = next(rect for clave, rect in rectangulos.items() if clave[0] == clase)
        centros.append(x + ancho / 2)
    ax.set_xticks(centros)
    ax.set_xticklabels(["No paga", "Paga"])
    ax.set_yticks([])
    ax.set(xlabel="Pago observado", ylabel=f"Categorías de {variable}")
    fig.tight_layout()
    plt.show()
    display(pd.crosstab(datos[variable], datos["y"]))


def densidad_pago(datos, variable, suavizado, vista_widget=False):
    """Estimación kernel exploratoria de P(pago | una entrada numérica)."""
    x = datos[variable].to_numpy(dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Se necesitan al menos dos valores numéricos finitos.")
    if not np.isfinite(suavizado) or suavizado <= 0:
        raise ValueError("El suavizado debe ser positivo y finito.")
    # Ancho común y frecuencia de clase: conserva el estimador del cuaderno.
    escala = min(x.std(ddof=1), (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.34)
    if escala <= 0:
        escala = x.std(ddof=1) or 1.0
    ancho = 0.9 * escala * len(x) ** (-0.2) * suavizado
    rejilla = np.linspace(x.min() - 3 * ancho, x.max() + 3 * ancho, 512)
    densidades = []
    for clase in ["no", "yes"]:
        valores = x[(datos["y"] == clase).to_numpy()]
        kernels = norm.pdf((rejilla[:, None] - valores[None, :]) / ancho) / ancho
        densidades.append(kernels.sum(axis=1) / len(x))
    densidades = np.asarray(densidades)
    total = densidades.sum(axis=0)
    if (total <= 0).any():
        raise ValueError("El suavizado no permite estimar la probabilidad en toda la rejilla.")
    probabilidades = densidades / total
    fig, ax = plt.subplots()
    ax.stackplot(rejilla, probabilidades, labels=["No paga", "Paga"],
                 colors=[COLOR_CLAVE, COLOR_DATOS], edgecolor=COLOR_BORDE, linewidth=0.5)
    ax.set(xlabel=variable, ylabel="Probabilidad condicional", title=f"Pago según {variable}", ylim=(0, 1))
    ax.legend(loc="upper right")
    fig.tight_layout()
    if vista_widget:
        return imagen_widget(fig, f"Probabilidad condicional de pago según {variable}")
    plt.show()


def comparar_odds(probabilidad_inicial, odds_ratio):
    """Ejemplo ficticio; no ajusta un modelo ni predice créditos."""
    if not 0 < probabilidad_inicial < 1 or not np.isfinite(odds_ratio) or odds_ratio <= 0:
        raise ValueError("Usa una probabilidad entre 0 y 1 y un OR positivo finito.")
    odds_iniciales = probabilidad_inicial / (1 - probabilidad_inicial)
    odds_finales = odds_iniciales * odds_ratio
    probabilidad_final = odds_finales / (1 + odds_finales)
    return ("<div style='padding: 12px; border-left: 4px solid #800000; font-size: 16px;'>"
            f"Probabilidad inicial: <b>{probabilidad_inicial:.1%}</b> → odds: <b>{odds_iniciales:.3f}</b>"
            f"<br>OR: <b>{odds_ratio:.2f}</b> → odds finales: <b>{odds_finales:.3f}</b>"
            f" → probabilidad final: <b>{probabilidad_final:.1%}</b>"
            f"<br>Cambio: <b>{100 * (probabilidad_final - probabilidad_inicial):+.1f}</b> puntos porcentuales."
            "</div>")


def curvas_logisticas():
    """Representa las tres pendientes del ejemplo ficticio con intercepto cero."""
    x = np.linspace(-5, 5, 250)
    fig, ax = plt.subplots(figsize=(11, 6))
    for pendiente, color, etiqueta in [
        (1.0, COLOR_CLAVE, "Efecto positivo"),
        (0.0, COLOR_BORDE, "Sin efecto"),
        (-1.0, COLOR_DATOS, "Efecto negativo"),
    ]:
        probabilidad = 1 / (1 + np.exp(-pendiente * x))
        ax.plot(x, probabilidad, color=color, linewidth=2.5, label=etiqueta)
    ax.set(xlabel="Variable x (escala ficticia)", ylabel="Probabilidad de pago",
           title="Efecto de un predictor en la probabilidad", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    plt.show()


def curva_logistica(intercepto, pendiente):
    """Devuelve una vista HTML de una curva didáctica, no de un ajuste real."""
    rejilla = np.linspace(-5, 5, 250)
    logit = intercepto + pendiente * rejilla
    probas = 1 / (1 + np.exp(-logit))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(rejilla, probas, color=COLOR_CLAVE, linewidth=2.5, label="Probabilidad")
    ax.axhline(0.5, color=COLOR_BORDE, linestyle="--", alpha=0.65)
    ax.set(xlabel="Variable x (escala ficticia)", ylabel="Probabilidad de pago",
           title="Logit lineal → probabilidad logística", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    p_en_cero = 1 / (1 + np.exp(-intercepto))
    return (imagen_widget(fig, "Curva logística didáctica")
            + f"<p>En x = 0: p = <b>{p_en_cero:.1%}</b>. "
            + f"Por cada unidad adicional de x los odds se multiplican por <b>{np.exp(pendiente):.2f}</b> "
            + "(no es un cambio fijo de probabilidad).</p>")


def pagos_por_intervalo(observado, probabilidades, grupos):
    """Dibuja pagos observados y esperados, sin recalcular probabilidades."""
    tabla = tabla_por_cuantiles(observado, probabilidades, grupos)
    posiciones = np.arange(1, len(tabla) + 1)
    fig, ax = plt.subplots()
    ancho = 0.36
    ax.bar(posiciones - ancho / 2, tabla["obs"], width=ancho, color=COLOR_DATOS,
           edgecolor=COLOR_BORDE, label="Pagos observados")
    ax.bar(posiciones + ancho / 2, tabla["exp"], width=ancho, color=COLOR_CLAVE,
           edgecolor=COLOR_BORDE, label="Pagos esperados")
    ax.set(xlabel="Grupos de probabilidad (de menor a mayor)", ylabel="Número de pagos",
           title="Pagos observados y esperados", xticks=posiciones)
    ax.legend()
    fig.tight_layout()
    plt.show()


def cajas_probabilidad(probabilidades, respuesta, titulo):
    """Compara las probabilidades de pago por clase observada no/yes."""
    fig, ax = plt.subplots(figsize=(11, 6))
    grupos = [np.asarray(probabilidades)[np.asarray(respuesta) == clase] for clase in ["no", "yes"]]
    cajas = ax.boxplot(grupos, patch_artist=True, medianprops={"color": COLOR_CLAVE, "linewidth": 2},
                       boxprops={"edgecolor": COLOR_BORDE}, whiskerprops={"color": COLOR_BORDE},
                       capprops={"color": COLOR_BORDE}, flierprops={"markeredgecolor": COLOR_BORDE})
    for caja in cajas["boxes"]:
        caja.set_facecolor(COLOR_DATOS)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No paga", "Paga"])
    ax.set(xlabel="Pago observado", ylabel="Probabilidad predicha de pago",
           ylim=(0, 1), title=f"Probabilidades de pago — {titulo}")
    fig.tight_layout()
    plt.show()


def _figura_roc(observado, probabilidades, titulo, etiqueta="Modelo"):
    real, predicho = datos_binarios(observado, probabilidades)
    fpr, tpr, _ = roc_curve(real, predicho)
    area = roc_auc_score(real, predicho)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color=COLOR_CLAVE, linewidth=2.5, label=f"{etiqueta} (AUC = {area:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=COLOR_BORDE, label="Azar")
    ax.set(xlabel="1 − Especificidad", ylabel="Sensibilidad", title=titulo,
           xlim=(0, 1), ylim=(0, 1))
    return fig, ax


def dibujar_roc(observado, probabilidades, titulo):
    """Muestra la discriminación de las puntuaciones suministradas."""
    fig, ax = _figura_roc(observado, probabilidades, titulo)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()


def calibracion_modelo(observado, probabilidades, intervalos=10, titulo="entrenamiento"):
    """Dibuja y devuelve calibración por intervalos de anchura fija."""
    real, predicho = datos_binarios(observado, probabilidades)
    tabla = pd.DataFrame({"observado": real, "predicho": predicho})
    tabla["intervalo"] = pd.cut(tabla["predicho"], bins=np.linspace(0, 1, intervalos + 1),
                                 include_lowest=True)
    resumen = tabla.groupby("intervalo", observed=True).agg(
        n=("observado", "size"), predicho=("predicho", "mean"), observado=("observado", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(resumen["predicho"], resumen["observado"], "o-", color=COLOR_CLAVE, linewidth=2,
            markerfacecolor=COLOR_DATOS, markeredgecolor=COLOR_BORDE, markersize=9, label="Modelo")
    ax.plot([0, 1], [0, 1], "--", color=COLOR_BORDE, label="Calibración perfecta")
    ax.set(xlabel="Probabilidad predicha", ylabel="Proporción observada de pagos",
           title=f"Calibración — {titulo}", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    plt.show()
    display(resumen.rename(columns={"predicho": "Probabilidad media", "observado": "Proporción de pagos"}).round(3))
    return resumen


def explorar_corte_entrenamiento(observado, probabilidades, umbral_fijado, corte):
    """Vista HTML de un corte explorado, sin alterar el umbral suministrado."""
    metricas = metricas_corte(observado, probabilidades, corte)
    decisiones = np.where(np.asarray(probabilidades) > corte, "si", "no")
    resultados = np.where(np.asarray(observado) == 1, "yes", "no")
    matriz = pd.crosstab(pd.Series(decisiones), pd.Series(resultados)).reindex(
        index=["no", "si"], columns=["no", "yes"], fill_value=0)
    fig, ax = _figura_roc(observado, probabilidades, "Umbrales en entrenamiento (no en test)", "Entrenamiento")
    ax.scatter(1 - metricas["Especificidad"], metricas["Sensibilidad"],
               s=150, color=COLOR_DATOS, edgecolor=COLOR_BORDE, zorder=3,
               label=f"Corte explorado = {corte:.2f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return (imagen_widget(fig, "Curva ROC de entrenamiento con el corte explorado")
            + "<p><b>Filas: decisión; columnas: pago observado.</b> Conteos en entrenamiento.</p>"
            + matriz.to_html(border=0)
            + f"<p>Sensibilidad: <b>{metricas['Sensibilidad']:.1%}</b> · "
            + f"Especificidad: <b>{metricas['Especificidad']:.1%}</b> · "
            + f"Exactitud: <b>{metricas['Exactitud']:.1%}</b></p>"
            + f"<p>Regla de test, sin cambios: <b>{umbral_fijado:.2f}</b>.</p>")


def evaluar_umbral_fijado(observado, probabilidades, corte):
    """Relaciona los conteos y porcentajes con un único corte de evaluación."""
    metricas = metricas_corte(observado, probabilidades, corte)
    decisiones = np.where(np.asarray(probabilidades) > corte, "si", "no")
    resultados = np.where(np.asarray(observado) == 1, "yes", "no")
    matriz = pd.crosstab(pd.Series(decisiones), pd.Series(resultados)).reindex(
        index=["no", "si"], columns=["no", "yes"], fill_value=0)
    porcentajes = matriz.div(matriz.sum(axis=1).replace(0, np.nan), axis=0).mul(100).round(1)
    print("Matriz de confusión: filas = decisión; columnas = pago real")
    display(matriz)
    print("Porcentajes por fila (una fila vacía no tiene porcentaje definido)")
    display(porcentajes)
    display(pd.Series(metricas).round(3))
    fig, ax = _figura_roc(observado, probabilidades, "Decisión preespecificada — test")
    ax.scatter(1 - metricas["Especificidad"], metricas["Sensibilidad"], color=COLOR_CLAVE,
               edgecolor=COLOR_BORDE, s=130, zorder=3, label=f"Corte fijado = {corte:.2f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()