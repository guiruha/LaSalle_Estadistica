"""Los 26 indicadores internos y sus reglas originales de la sesión 5.

Conserva las fórmulas históricas, incluida T sin centrar, DB con dispersión
RMS y la corrección dimensional de Duda. Los indicadores tienen distintas
unidades y sus recomendaciones cuentan clústeres, no aciertos. Importar el
módulo no carga datos, evalúa particiones ni modifica opciones globales.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import silhouette_score

from s5_modelos import ajustar_kmeans, renumerar, resumen_particion


def indice_dunn(distancias, etiquetas):
    """Razón adimensional de separación mínima y diámetro interno máximo.

    Recibe una matriz numérica (n, n) de distancias y un array 1D de
    identificadores de clúster. Devuelve un escalar numérico; un diámetro
    nulo devuelve infinito. Un valor alto no es exactitud de clasificación.
    """
    # Razón adimensional: menor separación entre grupos / mayor diámetro dentro de ellos.
    D = np.asarray(distancias)
    grupos = np.unique(etiquetas)
    diametro = max(D[np.ix_(etiquetas == g, etiquetas == g)].max() for g in grupos)
    separacion = min(D[np.ix_(etiquetas == g, etiquetas == h)].min()
                     for i, g in enumerate(grupos) for h in grupos[i+1:])
    return separacion / diametro if diametro > 0 else np.inf


def evaluar_indices(datos, metodo="ward", min_k=2, max_k=15, random_state=12345):
    """26 criterios didácticos: 24 reglas de elección y dos diagnósticos gráficos.

    Recibe datos numéricos (n, p), convertidos a float sin estandarizar.
    ``metodo`` elige el enlace jerárquico o ``kmeans``; min_k y max_k delimitan
    números de clústeres, no clases conocidas. Devuelve un SimpleNamespace
    con tablas numéricas ``All_index`` y ``Critical``, recomendaciones
    ``Best_n`` de dtype pandas Int64, ``Votes`` y arrays de etiquetas enteras
    en ``Best_partition`` y ``partitions``. Las unidades dependen del índice;
    Hubert y D-index no votan. Conserva T sin centrar, DB RMS y las reglas de
    diferencias y selección originales: ningún voto mide aciertos.
    """
    # 1. Prepara una sola matriz de distancias y todas las particiones necesarias.
    a = np.asarray(datos, dtype=float)
    n, p = a.shape
    D = squareform(pdist(a))
    superiores = np.triu_indices(n, 1)
    distancias = D[superiores]
    ordenadas = np.sort(distancias)
    particiones = {1: np.zeros(n, dtype=int)}
    arbol = None if metodo == "kmeans" else linkage(pdist(a), method=metodo)
    for k in range(2, max_k + 2):
        particiones[k] = (ajustar_kmeans(a, k, n_init=1, random_state=random_state).labels_
                          if metodo == "kmeans" else renumerar(cut_tree(arbol, n_clusters=k).ravel()))
    # T es aquí una MATRIZ histórica sin centrar; no confundir con el totss escalar.
    total_centrado = np.sum((a - a.mean(axis=0)) ** 2)
    T = a.T @ a
    s = np.sqrt(np.maximum(np.linalg.eigvalsh(T / (n - 1))[::-1], 1e-15))
    determinante_T = np.linalg.slogdet(T)[1]
    cache = {}
    # 2. Reutiliza centroides, residuos, pares internos/externos y densidades por partición.
    for k, etiquetas in particiones.items():
        r = resumen_particion(a, etiquetas)
        etiquetas = r.labels_
        residuos = a - r.cluster_centers_[etiquetas]
        W = residuos.T @ residuos
        B = T - W
        mismas = etiquetas[superiores[0]] == etiquetas[superiores[1]]
        dentro, fuera = distancias[mismas], distancias[~mismas]
        var_grupos = np.array([a[etiquetas == j].var(axis=0) for j in range(k)])
        norma_var = np.linalg.norm(var_grupos, axis=1)
        scatt = norma_var.mean() / np.linalg.norm(a.var(axis=0))
        radio = np.sqrt(norma_var.sum()) / k
        centros_D = squareform(pdist(r.cluster_centers_))
        dis = ((pdist(r.cluster_centers_).max() / pdist(r.cluster_centers_).min())
               * np.sum(1 / centros_D.sum(axis=1))) if k > 1 else np.nan
        cache[k] = dict(r=r, labels=etiquetas, W=W, B=B, dentro=dentro, fuera=fuera,
                        scatt=scatt, radio=radio, dis=dis, centros_D=centros_D)

    filas, criticos = [], []
    for k in range(min_k, max_k + 1):
        c = cache[k]
        r, etiquetas, W, B = c["r"], c["labels"], c["W"], c["B"]
        previo, siguiente = cache[k-1]["r"], cache[k+1]["r"]
        dentro, fuera = c["dentro"], c["fuera"]
        intra = r.inertia_
        _, logdetW = np.linalg.slogdet(W)
        # 3. DB conserva la versión RMS; C-index usa las sumas de pares extremos.
        dispersion = np.sqrt(r.withinss / r.size)
        ratios = (dispersion[:, None] + dispersion[None, :]) / np.where(c["centros_D"] > 0, c["centros_D"], np.inf)
        m = len(dentro)
        minimo, maximo = ordenadas[:m].sum(), ordenadas[-m:].sum()
        # CCC conserva la referencia histórica de T sin centrar: no es B_k/T del codo.
        u = s / (np.prod(s) / k) ** (1 / p)
        p1 = min(int((u >= 1).sum()), k - 1)
        if 0 < p1 < p:
            u = s / (np.prod(s[:p1]) / k) ** (1 / p1)
            suma = np.sum(1 / (n + u[:p1])) + np.sum(u[p1:] ** 2 / (n + u[p1:]))
            dimension = p1
        else:
            suma, dimension = np.sum(1 / (n + u)), p
        esperado_R2 = 1 - suma / np.sum(u ** 2) * (n - k) ** 2 / n * (1 + 4 / n)
        R2 = 1 - np.trace(W) / np.trace(T)
        ccc = (np.log((1 - esperado_R2) / (1 - R2)) * np.sqrt(n * dimension / 2)
               / (0.001 + esperado_R2) ** 1.2)

        # 4. División LOCAL en dos del grupo más disperso, no contraste global sobre k.
        padre = int(r.withinss.argmax())
        local = a[etiquetas == padre]
        division = ajustar_kmeans(local, 2, n_init=1, random_state=random_state)
        wl = division.inertia_
        wm = r.withinss[padre]
        nm = len(local)
        duda = wl / wm
        # Corrección mínima: el término 8/(pi²*p) también incluye la dimensión p.
        critica_duda = 1 - 2 / (np.pi * p) - 3.2 * np.sqrt(2 * (1 - 8 / (np.pi ** 2 * p)) / (nm * p))
        pseudo = (wm - wl) / (wl / (nm - 2))
        beale = ((wm - wl) / wl) / ((nm - 1) / (nm - 2) * (2 ** (2 / p) - 1))
        criticos.append({"k": k, "Duda": critica_duda,
                         "PseudoT2": (1 - critica_duda) / critica_duda * (nm - 2),
                         "Beale": stats.f.sf(beale, p, (nm - 2) * p)})

        # 5. Densidades alrededor de centros y puntos medios; Hubert es solo diagnóstico.
        dist_centro = np.linalg.norm(a - r.cluster_centers_[etiquetas], axis=1)
        densidades = np.bincount(etiquetas, weights=(dist_centro <= c["radio"]), minlength=k)
        dens_bw = 0.0
        for i in range(k):
            for j in range(i+1, k):
                candidatos = a[(etiquetas == i) | (etiquetas == j)]
                centro = (r.cluster_centers_[i] + r.cluster_centers_[j]) / 2
                cantidad = (np.linalg.norm(candidatos - centro, axis=1) <= c["radio"]).sum()
                denominador = max(densidades[i], densidades[j])
                dens_bw += cantidad / denominador if denominador else 0
        Q = cdist(r.cluster_centers_[etiquetas], r.cluster_centers_[etiquetas])
        varP = np.linalg.norm(D.var(axis=0))
        varQ = np.linalg.norm(Q.var(axis=0))
        hubert = np.sum((D - D.mean())[superiores] * (Q - Q.mean())[superiores]) / (len(distancias) * varP * varQ)
        proxima = cache[k+1]
        denominador_frey = proxima["dentro"].mean() - dentro.mean()
        kl_num = (k-1) ** (2/p) * previo.inertia_ - k ** (2/p) * intra
        kl_den = k ** (2/p) * intra - (k+1) ** (2/p) * siguiente.inertia_
        # 6. Reúne criterios. solve(W, B) calcula W⁻¹B, no multiplicación elemento a elemento.
        filas.append({
            "k": k, "KL": abs(kl_num / kl_den),
            "CH": ((total_centrado - intra) / (k-1)) / (intra / (n-k)),
            "Hartigan": (intra / siguiente.inertia_ - 1) * (n-k), "CCC": ccc,
            "Scott": n * (determinante_T - logdetW), "Marriott": k ** 2 * np.linalg.det(W),
            "TraceCovW": np.var(W, axis=0, ddof=1).sum(), "TraceW": np.trace(W),
            "Friedman": np.trace(np.linalg.solve(W, B)), "Rubin": np.trace(T) / np.trace(W),
            "Cindex": (dentro.sum() - minimo) / (maximo - minimo), "DB": ratios.max(axis=1).mean(),
            "Silhouette": silhouette_score(D, etiquetas, metric="precomputed"),
            "Duda": duda, "PseudoT2": pseudo, "Beale": beale,
            "Ratkowsky": np.sqrt(1 - np.sum((a - r.cluster_centers_[etiquetas]) ** 2, axis=0)
                                    / np.sum((a - a.mean(axis=0)) ** 2, axis=0)).mean() / np.sqrt(k),
            "Ball": intra / k,
            "PointBiserial": (fuera.mean() - dentro.mean()) * np.sqrt(len(fuera)*len(dentro))
                              / len(distancias) / distancias.std(ddof=1),
            "Frey": (proxima["fuera"].mean() - fuera.mean()) / denominador_frey,
            "McClain": (dentro.mean()/len(dentro)) / (fuera.mean()/len(fuera)),
            "Dunn": indice_dunn(D, etiquetas), "Hubert": hubert,
            "SDindex": cache[max_k]["dis"] * c["scatt"] + c["dis"],
            "Dindex": dist_centro.mean(), "SDbw": c["scatt"] + 2 * dens_bw / (k * (k-1)),
        })
    indices = pd.DataFrame(filas).set_index("k")
    criticos = pd.DataFrame(criticos).set_index("k")
    # 7. Aplica máximos, mínimos, diferencias y reglas locales; no todos se minimizan.
    recomendaciones = {}
    maximos = ["KL", "CH", "CCC", "Silhouette", "Ratkowsky", "PointBiserial", "Dunn"]
    minimos = ["Cindex", "DB", "McClain", "SDindex", "SDbw"]
    for nombre in maximos:
        recomendaciones[nombre] = indices[nombre].idxmax()
    for nombre in minimos:
        recomendaciones[nombre] = indices[nombre].idxmin()
    for nombre in ["Hartigan", "Scott", "TraceCovW", "Friedman", "Ball"]:
        recomendaciones[nombre] = (indices[nombre] - indices[nombre].shift(-1)).abs().idxmax()
    for nombre, maximizar in [("Marriott", True), ("TraceW", True), ("Rubin", False)]:
        diferencias = indices[nombre].shift(1) - 2 * indices[nombre] + indices[nombre].shift(-1)
        recomendaciones[nombre] = diferencias.idxmax() if maximizar else diferencias.idxmin()
    condiciones = {"Duda": indices["Duda"] >= criticos["Duda"],
                   "PseudoT2": indices["PseudoT2"] <= criticos["PseudoT2"],
                   "Beale": criticos["Beale"] >= 0.1, "Frey": indices["Frey"] < 1}
    for nombre, condicion in condiciones.items():
        candidatos = indices.index[condicion]
        recomendaciones[nombre] = (max(min_k, int(candidatos[0]) - 1) if nombre == "Frey" else candidatos[0]) if len(candidatos) else np.nan
    # 8. Cuenta solo recomendaciones válidas y devuelve una partición orientativa, no verdadera.
    best = pd.Series(recomendaciones, name="k recomendado").astype("Int64")
    votos = best.dropna().value_counts().sort_index()
    ganador = int(votos.idxmax())
    return SimpleNamespace(All_index=indices, Best_n=best.to_frame(), Votes=votos,
                           Best_partition=particiones[ganador], Critical=criticos, partitions=particiones)


def comparar_sistemas(datos, arbol, distancias, valores_k, random_state=12345):
    """Compara los cuatro ajustes originales del mayorista en la misma escala.

    Devuelve todas las etiquetas y la tabla de índices, sin elegir un sistema
    por su rendimiento ni consultar clases conocidas.
    """
    resultados = {}
    filas = []
    for k in valores_k:
        grupos_arbol = renumerar(cut_tree(arbol, n_clusters=k).ravel())
        centros_arbol = np.array([np.asarray(datos)[grupos_arbol == j].mean(axis=0)
                                  for j in range(k)])
        sistemas = {
            "Lloyd (1)": ajustar_kmeans(datos, k, n_init=1, random_state=random_state, algoritmo="lloyd"),
            "Hartigan (1)": ajustar_kmeans(datos, k, n_init=1, random_state=random_state),
            "Hartigan (10)": ajustar_kmeans(datos, k, n_init=10, random_state=random_state),
            "Jerárquico + Hartigan": ajustar_kmeans(datos, k, init=centros_arbol, random_state=random_state),
        }
        for nombre, modelo in sistemas.items():
            etiquetas = modelo.labels_
            resultados[(k, nombre)] = etiquetas
            filas.append({"k": k, "Sistema": nombre,
                          "Dunn": indice_dunn(distancias, etiquetas),
                          "Silhouette": silhouette_score(distancias, etiquetas, metric="precomputed"),
                          "Inercia": modelo.inertia_})
    return resultados, pd.DataFrame(filas).set_index(["k", "Sistema"])