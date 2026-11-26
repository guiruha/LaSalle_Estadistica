"""Gráficos de la sesión 5 con datos explícitos, sin estado del notebook."""

from base64 import b64encode
from html import escape
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML as HTMLDisplay, display
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle, Wedge
from pandas.plotting import scatter_matrix
from scipy.cluster.hierarchy import cophenet, cut_tree, dendrogram, linkage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from s5_modelos import (
    ajustar_kmeans, clara, escalar, generar_atipicos, hcpc as _ajustar_hcpc,
    kernel_kmeans, pam, renumerar, resumen_particion,
)


COLORES_GRUPO = sns.color_palette("colorblind", 10)
MAPA_GRUPOS = ListedColormap(COLORES_GRUPO)
COLOR_BASE, COLOR_ACENTO = COLORES_GRUPO[:2]


def configurar_estilo():
    """Aplica el formato original cuando se solicita, no al importar."""
    sns.set_theme(style="white", context="notebook", palette="colorblind", rc={
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 100, "lines.linewidth": 2, "lines.markersize": 6,
        "grid.color": "#DCE4E8", "grid.alpha": 0.8,
    })
    plt.rcParams["figure.figsize"] = (8, 4)
    pd.set_option("display.max_columns", 30)


def estilo_ejes(ax, cuadricula=False):
    """Activa una cuadrícula discreta solo cuando ayuda a leer magnitudes."""
    ax.set_axisbelow(True)
    if cuadricula:
        ax.grid(axis="y")
    return ax


def imagen_control(fig, descripcion):
    """Convierte una figura a HTML y la cierra para no acumular salidas."""
    archivo = BytesIO()
    fig.savefig(archivo, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return (f"<img src='data:image/png;base64,{b64encode(archivo.getvalue()).decode()}' "
            f"alt='{escape(descripcion, quote=True)}' style='max-width:100%; height:auto;'>")


def mostrar_html(contenido):
    """Alternativa estática a los controles del cuaderno."""
    display(HTMLDisplay(contenido))


def representar_clusters(datos, etiquetas, titulo="Agrupación", envolventes=True,
                         estandarizar=True, mostrar=True):
    """Proyecta etiquetas ya calculadas: no reajusta los grupos."""
    tabla = pd.DataFrame(datos)
    entrada = escalar(tabla) if estandarizar else tabla
    pca = PCA(n_components=2)
    coordenadas = pca.fit_transform(entrada)
    fig, ax = plt.subplots(figsize=(8, 5))
    for grupo in np.unique(etiquetas):
        puntos = coordenadas[np.asarray(etiquetas) == grupo]
        color = COLORES_GRUPO[int(grupo) % 10]
        ax.scatter(puntos[:, 0], puntos[:, 1], s=22, color=color, label=f"Grupo {int(grupo)+1}")
        if envolventes and len(puntos) >= 3 and np.linalg.matrix_rank(puntos - puntos.mean(axis=0)) == 2:
            borde = ConvexHull(puntos).vertices
            ax.fill(puntos[borde, 0], puntos[borde, 1], color=color, alpha=0.15)
    porcentajes = 100 * pca.explained_variance_ratio_
    ax.set(xlabel=f"CP1 ({porcentajes[0]:.1f} %)", ylabel=f"CP2 ({porcentajes[1]:.1f} %)", title=titulo)
    ax.legend(frameon=False)
    fig.tight_layout()
    if not mostrar:
        return imagen_control(fig, titulo)
    plt.show()


def pares_coloreados(tabla, etiquetas):
    """Compara las medidas recibidas, coloreadas por la partición indicada."""
    colores = [COLORES_GRUPO[int(grupo) % 10] for grupo in etiquetas]
    scatter_matrix(tabla, c=colores, figsize=(9, 9), diagonal="hist", s=15, alpha=0.7)
    plt.show()


def mapa_iris(datos, datos_escalados, estandarizado):
    """Mapa de medidas originales o escaladas, sin modificar ninguna tabla."""
    tabla = datos_escalados if estandarizado else datos
    grafico = sns.clustermap(
        tabla, method="complete", cmap="vlag", figsize=(9, 7), yticklabels=False,
        cbar_kws={"label": "Valor estandarizado" if estandarizado else "Valor (cm)"},
    )
    titulo = "Iris estandarizado" if estandarizado else "Iris sin escalar"
    grafico.figure.suptitle(titulo, y=1.02)
    mostrar_html(imagen_control(grafico.figure, titulo))


def dibujar_dendrograma(arbol, etiquetas):
    """Dibuja el referente de enlace completo con las etiquetas originales."""
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(arbol, labels=list(etiquetas), leaf_rotation=90, leaf_font_size=5,
               color_threshold=0, above_threshold_color=COLOR_BASE, ax=ax)
    ax.set(xlabel="Observación", ylabel="Distancia (cm)", title="Clustering jerárquico: enlace completo")
    fig.tight_layout()
    plt.show()


def explorar_arbol(datos, distancia="euclidean", enlace="complete", estandarizar=False):
    """Devuelve otro árbol en HTML, sin cambiar la jerarquía de referencia."""
    if enlace == "ward" and distancia != "euclidean":
        raise ValueError("Para interpretar Ward como incremento de inercia usa distancia euclídea.")
    tabla = escalar(datos) if estandarizar else datos
    disimilitudes = pdist(tabla, metric=distancia)
    arbol = linkage(disimilitudes, method=enlace)
    calidad = cophenet(arbol, disimilitudes)[0]
    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(arbol, no_labels=True, ax=ax, color_threshold=0, above_threshold_color=COLOR_BASE)
    ax.set(title=f"{enlace.title()} · {distancia} · "
                 f"{'estandarizado' if estandarizar else 'sin escalar'}",
           xlabel="Flores ordenadas por el árbol", ylabel="Altura de la fusión")
    fig.tight_layout()
    return imagen_control(fig, "Dendrograma comparativo de Iris") + (
        f"<p>Correlación cofenética con su propia distancia: <b>{calidad:.3f}</b>. "
        "Un valor alto no demuestra que exista un número verdadero de grupos.</p>"
    )


def particion_jerarquica(datos, arbol, k):
    """Representa un corte local y devuelve tamaños, sin sustituir el referente."""
    grupos = renumerar(cut_tree(arbol, n_clusters=k).ravel())
    fig, ax = plt.subplots(figsize=(14, 5))
    dibujado = dendrogram(arbol, no_labels=True, link_color_func=lambda _: "#5B6770", ax=ax)
    orden = np.asarray(dibujado["leaves"])
    altura = (arbol[-k, 2] + arbol[-k+1, 2]) / 2
    for grupo in np.unique(grupos):
        posiciones = np.flatnonzero(grupos[orden] == grupo)
        color = COLORES_GRUPO[int(grupo) % 10]
        ax.add_patch(Rectangle((10 * posiciones.min(), 0), 10 * len(posiciones), altura,
                               fill=False, edgecolor=color, linewidth=2))
    ax.set(xlabel="Observaciones ordenadas por el árbol", ylabel="Distancia (cm)", title=f"Partición en {k} grupos")
    fig.tight_layout()
    mostrar_html(imagen_control(fig, f"Dendrograma dividido en {k} grupos"))
    resumen = resumen_particion(datos, grupos)
    print(f"Reducción relativa de dispersión B/T (no acierto): {resumen.betweenss / resumen.totss:.1%}")
    return pd.Series(np.bincount(grupos), index=np.arange(1, k+1), name="Tamaño del grupo")


def hcpc(datos_escalados, k, consolidar):
    """Representa y devuelve ACP completa, corte y consolidación opcional."""
    resultado = _ajustar_hcpc(datos_escalados, k, consolidar)
    grafico = representar_clusters(datos_escalados, resultado.labels_,
                                   titulo="Clustering sobre componentes principales",
                                   estandarizar=False, mostrar=False)
    mostrar_html(grafico)
    return resultado


def pares_especies(datos, grupos, especies):
    """Contraste visual externo: color por grupo, símbolo por especie."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    grupos = np.asarray(grupos)
    for i, nombre_y in enumerate(datos.columns):
        for j, nombre_x in enumerate(datos.columns):
            ax = axes[i, j]
            if i == j:
                ax.hist(datos[nombre_x], bins=12, color=COLOR_BASE, alpha=0.75)
            else:
                for especie, marcador in enumerate(["o", "s", "^"]):
                    mascara = np.asarray(especies) == especie
                    ax.scatter(datos.loc[mascara, nombre_x], datos.loc[mascara, nombre_y],
                               c=grupos[mascara], cmap=MAPA_GRUPOS, vmin=-0.5, vmax=9.5,
                               marker=marcador, s=12)
            if i == 3:
                ax.set_xlabel(nombre_x)
            if j == 0:
                ax.set_ylabel(nombre_y)
    fig.tight_layout()
    plt.show()


def tanglegram(izquierdo, derecho):
    """Conecta la misma observación entre dos árboles, no sus grupos."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1, 2]})
    a = dendrogram(izquierdo, orientation="left", no_labels=True, ax=axes[0],
                   color_threshold=0, above_threshold_color=COLOR_BASE)
    b = dendrogram(derecho, orientation="right", no_labels=True, ax=axes[2],
                   color_threshold=0, above_threshold_color=COLOR_BASE)
    posiciones_a = {observacion: 5 + 10 * i for i, observacion in enumerate(a["leaves"])}
    posiciones_b = {observacion: 5 + 10 * i for i, observacion in enumerate(b["leaves"])}
    for observacion in posiciones_a:
        axes[1].plot([0, 1], [posiciones_a[observacion], posiciones_b[observacion]],
                     color=COLOR_ACENTO, alpha=0.3, linewidth=0.6)
    for ax in axes:
        ax.set_ylim(0, 10 * len(a["leaves"]))
    axes[1].set(xlim=(0, 1))
    axes[1].axis("off")
    axes[0].set_title("Enlace completo")
    axes[2].set_title("Enlace promedio")
    fig.tight_layout()
    plt.show()


def dibujar_correlaciones(correlaciones):
    """Representa coeficientes entre árboles; no recalcula particiones."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(len(correlaciones)):
        for j in range(i+1):
            valor = correlaciones.iloc[i, j]
            ax.add_patch(Circle((j, i), 0.42, edgecolor="#DCE4E8", facecolor="white"))
            ax.add_patch(Wedge((j, i), 0.42, 90, 90 + 360 * abs(valor),
                               facecolor=COLOR_BASE if valor >= 0 else COLOR_ACENTO))
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=9)
    ax.set(xticks=range(len(correlaciones)), yticks=range(len(correlaciones)),
           xlim=(-0.6, len(correlaciones)-0.4), ylim=(len(correlaciones)-0.4, -0.6),
           title="Correlaciones entre alturas cofenéticas de árboles")
    ax.set_xticklabels(correlaciones.columns, rotation=45)
    ax.set_yticklabels(correlaciones.index)
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()


def dibujar_arboles_vino(arboles):
    """Muestra los cuatro árboles ya calculados, incluidos los avisos de Ward."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (nombre, (arbol, _)) in zip(axes.flat, arboles.items()):
        dendrogram(arbol, no_labels=True, ax=ax, color_threshold=0, above_threshold_color=COLOR_BASE)
        advertencia = " · Ward no euclídeo" if nombre == "cityblock / ward" else ""
        ax.set(title=nombre + advertencia, xlabel="Vinos ordenados", ylabel="Altura de la fusión")
    fig.tight_layout()
    plt.show()


def iris_3d(datos, grupos, especies, nombres_especies, angulo):
    """Vista externa de etiquetas ya calculadas; el ángulo no cambia los grupos."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    for especie, marcador in enumerate(["o", "s", "^"]):
        mascara = np.asarray(especies) == especie
        ax.scatter(datos.loc[mascara, "Sepal.Length"], datos.loc[mascara, "Sepal.Width"],
                   datos.loc[mascara, "Petal.Length"], c=np.asarray(grupos)[mascara], cmap=MAPA_GRUPOS,
                   vmin=-0.5, vmax=9.5, marker=marcador, label=nombres_especies[especie])
    for fila, grupo in zip(datos.to_numpy(), grupos):
        ax.plot([fila[0], fila[0]], [fila[1], fila[1]], [0, fila[2]],
                color=COLORES_GRUPO[int(grupo) % 10], alpha=0.15, linewidth=0.5)
    ax.set(xlabel="Sepal.Length", ylabel="Sepal.Width", zlabel="Petal.Length")
    ax.view_init(elev=20, azim=angulo)
    ax.legend(frameon=False)
    fig.tight_layout()
    mostrar_html(imagen_control(fig, f"Iris en tres dimensiones, ángulo {angulo}°"))


def explorar_kmeans(datos, semilla, k, inicios, algoritmo):
    """Ajuste local para comparar opciones, sin alterar los modelos de referencia."""
    modelo = ajustar_kmeans(datos, k, n_init=inicios, algoritmo=algoritmo, random_state=semilla)
    mostrar_html(representar_clusters(datos, modelo.labels_, titulo=f"K-means: {k} grupos", mostrar=False))
    resumen = resumen_particion(datos, modelo.labels_)
    display(pd.Series({"Inercia intra": modelo.inertia_, "Variabilidad explicada": resumen.betweenss / resumen.totss,
                       "Silhouette": silhouette_score(datos, modelo.labels_)}))


def explorar_kernel(datos_originales, datos_escalados, semilla, gamma):
    """Reajusta solo el experimento local del control, sin sustituir el kernel fijo."""
    resultado = kernel_kmeans(datos_escalados, k=2, gamma=gamma, random_state=semilla)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(datos_originales["x"], datos_originales["y"], c=resultado.labels_,
               cmap=MAPA_GRUPOS, vmin=-0.5, vmax=9.5, s=10)
    titulo = f"K-means con kernel gaussiano: gamma = {gamma:.2f}"
    ax.set(xlabel="x", ylabel="y", title=titulo)
    ax.set_aspect("equal")
    fig.tight_layout()
    mostrar_html(imagen_control(fig, titulo))


def dibujar_atipicos(tabla, modelos, mostrar=True):
    """Ejes logarítmicos para ver las mismas particiones de datos originales."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (nombre, grupos) in zip(axes.flat, modelos.items()):
        if grupos is None:
            ax.scatter(tabla["x1"], tabla["x2"], color=COLOR_BASE, s=14)
        else:
            ax.scatter(tabla["x1"], tabla["x2"], c=grupos,
                       cmap=MAPA_GRUPOS, vmin=-0.5, vmax=9.5, s=14)
        ax.set(xscale="log", yscale="log", xlabel="x1", ylabel="x2", title=nombre)
    fig.tight_layout()
    if not mostrar:
        return imagen_control(fig, "Comparación de datos, k-means, PAM y CLARA con valores atípicos")
    plt.show()


def explorar_atipicos(semilla, numero):
    """Simula y ajusta objetos locales, conservando los referentes del notebook."""
    tabla = generar_atipicos(nout=numero, random_state=semilla)
    km = ajustar_kmeans(tabla, k=2, n_init=1, random_state=semilla)
    med = pam(tabla, k=2)
    muestra = clara(tabla, k=2, random_state=semilla)
    mostrar_html(dibujar_atipicos(tabla, {"Datos": None, "K-means": km.labels_,
                                        "PAM": med.labels_, "CLARA": muestra.labels_}, mostrar=False))


def dibujar_criterios_mayorista(resumen):
    """Compara WSS y silhouette de modelos ya ajustados en la misma escala."""
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    resumen["Inercia intra"].plot(ax=ejes[0], marker="o", color=COLOR_BASE)
    resumen["Silhouette media"].plot(ax=ejes[1], marker="o", color=COLOR_ACENTO)
    for eje, titulo in zip(ejes, ["Inercia (busca el codo)", "Silhouette (mayor = mejor)"]):
        eje.set(xlabel="Número de grupos", title=titulo)
        estilo_ejes(eje, cuadricula=True)
    fig.tight_layout()
    plt.show()


def dibujar_perfiles_mayorista(datos, grupos):
    """Cajas en log1p del gasto, sin alterar las medianas de los datos originales."""
    etiquetas = np.asarray(grupos)
    fig, ejes = plt.subplots(2, 3, figsize=(14, 8))
    for eje, nombre in zip(ejes.flat, datos.columns):
        cajas = [np.log1p(datos.loc[etiquetas == g, nombre]) for g in np.unique(etiquetas)]
        grafico = eje.boxplot(cajas, patch_artist=True,
                              medianprops={"color": "#273443", "linewidth": 1.8},
                              flierprops={"marker": ".", "markersize": 3, "alpha": 0.35})
        for g, caja in enumerate(grafico["boxes"]):
            caja.set(facecolor=COLORES_GRUPO[g], edgecolor=COLORES_GRUPO[g], alpha=0.5)
        eje.set(title=nombre, xlabel="Grupo", ylabel="log(1 + gasto)")
        estilo_ejes(eje, cuadricula=True)
    fig.tight_layout()
    plt.show()