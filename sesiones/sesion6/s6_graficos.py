"""Vistas estáticas de S6: todos los datos y ajustes se reciben como argumentos."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from s6_evaluacion import evaluar, tabla_confusion


def dibujar_barrido(resultados, titulo):
    """Acierto observado para cada k; no realiza selección independiente."""
    fig, ax = plt.subplots()
    ax.plot(resultados.index, resultados.iloc[:, 0], "o-")
    ax.set(xlabel="Número de vecinos (k)", ylabel="Proporción de aciertos", title=titulo,
           xticks=resultados.index)
    ax.grid(alpha=0.2)
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
    fig, ax = plt.subplots(figsize=(9, 3))
    for j, clase in enumerate(np.unique(clases_train)):
        posiciones = seleccion[clases_train.iloc[seleccion].to_numpy() == clase]
        ax.scatter(entrenamiento.iloc[posiciones, 0], np.full(len(posiciones), clase),
                   s=15, alpha=0.35, label=f"churn {clase}", color=colores[j % len(colores)])
    ax.set(xlabel="CP1", ylabel="Clase", yticks=np.unique(clases_train), title=f"Entrenamiento · k={k}")
    ax.legend()
    plt.show()


def explorar_rechazo(k, minimo, *, ajuste, reales):
    """Conserva la regla de votos mínimos y muestra también los rechazos."""
    if minimo > k:
        display("El mínimo de votos debe ser menor o igual que k.")
        return
    prediccion = ajuste.predecir(k=k, minimo=minimo)
    display(tabla_confusion(reales, prediccion, incluir_rechazos=True), evaluar(reales, prediccion))


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
    """Hasta nueve medias por clase, leídas del ajuste base recibido."""
    fig, ejes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for ax, variable in zip(ejes.flat, variables):
        medias = modelo.medias[variable]
        ax.bar(np.arange(len(medias)), medias, color=color)
        ax.set(title=f"{variable} · Gini {importancia[variable]:.3f}",
               xticks=np.arange(len(medias)), xticklabels=range(1, len(medias)+1),
               xlabel="Clase", ylabel="Media")
    for ax in list(ejes.flat)[len(variables):]:
        ax.set_visible(False)
    fig.suptitle(titulo)
    plt.show()


def explorar_medias(pagina, *, predictores, panel):
    """Recorre las características en su orden original, de nueve en nueve."""
    comienzo = (pagina-1)*9
    variables = list(predictores[comienzo:comienzo+9])
    panel(variables, f"Medias por clase · página {pagina}")


def explorar_gini(umbral, *, importancia, X_train, y_train, X_test, reales, modelo_clase):
    """Explora un filtro real, ajustando un modelo local sin modificar el base."""
    seleccion = importancia.index[importancia > umbral].tolist()
    if not seleccion:
        display("Ninguna variable supera este umbral.")
        return
    modelo = modelo_clase().fit(X_train[seleccion], y_train)
    prediccion = modelo.predict(X_test)
    display(pd.Series({"Variables seleccionadas": len(seleccion),
                       "Proporción de aciertos": np.mean(prediccion.to_numpy() == reales.to_numpy())}))
    display(importancia.loc[seleccion].sort_values(ascending=False).to_frame())


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