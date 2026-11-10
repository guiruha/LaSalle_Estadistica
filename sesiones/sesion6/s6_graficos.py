"""Vistas estáticas de S6: todos los datos y ajustes se reciben como argumentos."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display

from s6_evaluacion import evaluar, tabla_confusion


def dibujar_barrido(resultados, titulo, referencia=None):
    """Acierto observado por k, con referencia opcional; no selecciona un modelo."""
    valores = resultados.iloc[:, 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(valores.index, valores, marker="o", linewidth=2, color="steelblue", label="KNN")
    if referencia is not None:
        ax.axhline(referencia, color="darkorange", linestyle="--", linewidth=1.5,
                   label=f"Referencia mayoritaria ({referencia:.1%})")
    mejor_k = valores.idxmax()
    mejor_valor = valores.loc[mejor_k]
    ax.scatter([mejor_k], [mejor_valor], color="firebrick", zorder=3)
    ax.annotate(f"{mejor_valor:.1%}", (mejor_k, mejor_valor), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=9)
    ax.set(xlabel="Número de vecinos (k)", ylabel="Acierto", title=titulo,
           xticks=valores.index, ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def dibujar_confusion(tabla, titulo):
    """Muestra una matriz de confusión anotada: predicciones en filas, realidad en columnas."""
    valores = tabla.to_numpy()
    ancho = max(5, 0.8 * len(tabla.columns))
    alto = max(4, 0.65 * len(tabla.index))
    fig, ax = plt.subplots(figsize=(ancho, alto))
    imagen = ax.imshow(valores, cmap="Blues", aspect="auto")
    ax.set(xticks=np.arange(len(tabla.columns)), xticklabels=tabla.columns,
           yticks=np.arange(len(tabla.index)), yticklabels=tabla.index,
           xlabel="Clase real", ylabel="Clase predicha", title=titulo)
    ax.tick_params(axis="x", labelrotation=35)
    plt.setp(ax.get_xticklabels(), ha="right")
    limite = valores.max() / 2 if valores.size else 0
    for fila in range(valores.shape[0]):
        for columna in range(valores.shape[1]):
            etiqueta = f"{int(valores[fila, columna]):,}".replace(",", "\u202f")
            ax.text(columna, fila, etiqueta, ha="center", va="center",
                    color="white" if valores[fila, columna] > limite else "black", fontsize=9)
    fig.colorbar(imagen, ax=ax, label="Número de registros")
    fig.tight_layout()
    plt.show()


def explorar_knn(k, cliente, *, ajuste, X_train, y_train, X_test, reales):
    """Métricas completas y detalle de un cliente, reutilizando la caché."""
    prediccion = ajuste.predecir(k)
    display(tabla_confusion(reales, prediccion), evaluar(reales, prediccion))
    fila = int(cliente)
    display(pd.Series({"Índice del cliente": X_test.index[fila],
                       "Clase real": reales.iloc[fila], "Predicción": prediccion.iloc[fila]}))
    posiciones = ajuste.indices[fila, :k]
    detalle = pd.DataFrame({"Índice de entrenamiento": X_train.index[posiciones],
                            "Distancia": ajuste.distancias[fila, :k],
                            "churn": y_train.iloc[posiciones].to_numpy()})
    display(detalle)
    display(ajuste.probabilidades(k).iloc[[fila]])


def explorar_cp1(k, *, ajuste, reales, entrenamiento, clases_train, semilla, colores):
    """Evalúa toda la consulta y dibuja una muestra reproducible de entrenamiento."""
    prediccion = ajuste.predecir(k)
    display(tabla_confusion(reales, prediccion), evaluar(reales, prediccion))
    seleccion = np.random.default_rng(semilla).choice(len(entrenamiento),
                size=min(1000, len(entrenamiento)), replace=False)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    rng = np.random.default_rng(semilla)
    for j, clase in enumerate(np.unique(clases_train)):
        posiciones = seleccion[clases_train.iloc[seleccion].to_numpy() == clase]
        jitter = rng.uniform(-0.08, 0.08, size=len(posiciones))
        ax.scatter(entrenamiento.iloc[posiciones, 0], clase + jitter,
                   s=18, alpha=0.3, label=f"churn {clase} · n mostrado={len(posiciones)}",
                   color=colores[j % len(colores)], edgecolors="none")
    ax.set(xlabel="Primera componente principal (CP1)", ylabel="Clase real",
           yticks=np.unique(clases_train), yticklabels=[str(c) for c in np.unique(clases_train)],
           ylim=(min(clases_train) - 0.35, max(clases_train) + 0.35),
           title=f"Distribución de entrenamiento en CP1 · k={k}")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    plt.show()


def explorar_kernel_knn(datos, kernel, k, *, originales, cp1, reales):
    """Alterna geometrías ya calculadas, sin volver a buscar vecinos."""
    ajuste = originales if datos == "Originales" else cp1
    prediccion = ajuste.predecir(k=k, kernel=kernel)
    display(tabla_confusion(reales, prediccion), evaluar(reales, prediccion))
    display(ajuste.probabilidades(k=k, kernel=kernel).head().rename_axis(columns="Clase"))


def explorar_tabla_nb(variable, *, modelo):
    """Parámetros de entrenamiento; las barras no son intervalos de confianza."""
    display(modelo.tablas[variable])
    fig, ax = plt.subplots()
    medias, sd = modelo.medias[variable], modelo.desviaciones[variable]
    ax.bar(np.arange(len(medias)), medias, yerr=sd, capsize=3)
    ax.set(xticks=np.arange(len(medias)), xticklabels=medias.index,
           xlabel="Clase", ylabel="Media ± desviación típica", title=variable)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    plt.show()


def explorar_probabilidades(inicio, cantidad, *, probabilidades, mapa_colores):
    """Posteriores de un tramo por posición, conservando índices y clases."""
    tabla = probabilidades.iloc[inicio:inicio+cantidad]
    fig, ax = plt.subplots(figsize=(9, max(4, len(tabla)*0.12)))
    mapa = ax.imshow(tabla.to_numpy(), aspect="auto", vmin=0, vmax=1,
                     cmap=mapa_colores, interpolation="nearest")
    ax.set(xticks=np.arange(len(tabla.columns)), xticklabels=tabla.columns,
           xlabel="Clase", ylabel="Producto (orden reservado)",
           title=f"Probabilidades · filas {inicio} a {inicio+len(tabla)-1}")
    posiciones = np.linspace(0, len(tabla)-1, min(10, len(tabla)), dtype=int)
    ax.set_yticks(posiciones)
    ax.set_yticklabels(tabla.index[posiciones])
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.colorbar(mapa, ax=ax, label="Probabilidad posterior")
    fig.tight_layout()
    plt.show()
    display(tabla.head())


def panel_medias(variables, titulo, *, modelo, importancia, color):
    """Hasta nueve medias y desviaciones por clase, leídas del ajuste base recibido."""
    fig, ejes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    for ax, variable in zip(ejes.flat, variables):
        medias = modelo.medias[variable]
        desviaciones = modelo.desviaciones[variable]
        posiciones = np.arange(len(medias))
        ax.bar(posiciones, medias, yerr=desviaciones, capsize=2, color=color,
               alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.set(title=f"{variable} · Gini {importancia[variable]:.3f}",
               xticks=posiciones, xticklabels=[str(clase).replace("Class_", "") for clase in medias.index],
               xlabel="Clase", ylabel="Media ± desviación típica")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    for ax in list(ejes.flat)[len(variables):]:
        ax.set_visible(False)
    fig.suptitle(titulo)
    plt.show()


def explorar_densidad(variable, cerca_de_cero, clase, *, modelo, colores):
    """Densidades ya ajustadas; los controles solo cambian la vista."""
    clases = modelo.clases if clase == "Todas" else [clase]
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, etiqueta in enumerate(clases):
        densidad = modelo.densidades[variable][etiqueta]
        ax.plot(densidad.x, densidad.y, label=etiqueta, color=colores[j % len(colores)])
    ax.axvline(0, color="black", linewidth=0.8)
    if cerca_de_cero:
        ax.set_xlim(-1, 1)
        alturas = [np.interp(np.linspace(-1, 1, 200), modelo.densidades[variable][c].x,
                            modelo.densidades[variable][c].y).max() for c in clases]
        ax.set_ylim(0, max(alturas)*1.08 if max(alturas) > 0 else 1)
    ax.set(xlabel=variable, ylabel="Densidad", title="Distribución condicional por clase")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    plt.show()
    display(pd.Series({c: modelo.densidades[variable][c].ancho for c in clases}, name="Ancho de banda"))