"""Modelos y simulación de clustering de la sesión 5, sin estado del notebook.

Los datos y las semillas se reciben explícitamente. Importar este módulo no
carga datos, ajusta modelos ni configura gráficos. Los identificadores de
clúster no son clases predichas; los costes conservan la escala de cada método
y no representan porcentajes de aciertos.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def renumerar(etiquetas):
    """Devuelve un array 1D de enteros con grupos numerados desde cero.

    Recibe una secuencia 1D de identificadores y conserva la pertenencia y el
    orden de las observaciones. Los números identifican clústeres, no clases.
    """
    # Cambia identificadores, no miembros de los grupos: 0, 1, ... por orden de aparición.
    equivalencias = {valor: i for i, valor in enumerate(dict.fromkeys(np.asarray(etiquetas).tolist()))}
    return np.array([equivalencias[valor] for valor in etiquetas], dtype=int)


def escalar(tabla):
    """Estandariza un DataFrame numérico por columnas con ``ddof=1``.

    Conserva índice y columnas; los valores resultantes son adimensionales.
    Requiere datos sin ausentes ni columnas constantes. No ajusta clústeres.
    """
    # Estandarización por columna: media 0 y desviación típica muestral 1.
    return (tabla - tabla.mean()) / tabla.std(ddof=1)


def resumen_particion(datos, etiquetas):
    """Resume una matriz numérica (n, p) y sus etiquetas 1D sin reagrupar.

    Convierte los datos a float y devuelve un SimpleNamespace: ``labels_`` y
    ``size`` son arrays enteros; ``cluster_centers_`` y ``withinss`` son arrays
    float. ``inertia_``, ``totss`` y ``betweenss`` son sumas de cuadrados en la
    escala recibida, no errores de clasificación ni porcentajes de aciertos.
    """
    # 1. Obtiene los centroides de la partición dada, sin modificarla.
    a = np.asarray(datos, dtype=float)
    grupos = renumerar(etiquetas)
    centros = np.array([a[grupos == i].mean(axis=0) for i in np.unique(grupos)])
    # 2. Suma distancias cuadráticas al centro propio: WSS por grupo y total.
    errores = np.sum((a - centros[grupos]) ** 2, axis=1)
    withinss = np.bincount(grupos, weights=errores)
    totss = np.sum((a - a.mean(axis=0)) ** 2)
    # 3. Devuelve T = W + B y tamaños; no son errores de clasificación.
    return SimpleNamespace(labels_=grupos, cluster_centers_=centros, withinss=withinss,
                           inertia_=withinss.sum(), totss=totss,
                           betweenss=totss - withinss.sum(), size=np.bincount(grupos))


def hartigan_wong(datos, k, n_init=1, random_state=12345, max_iter=100, init=None):
    """Transferencias óptimas y rápidas para minimizar la suma de cuadrados.

    Recibe datos numéricos (n, p), k grupos y, opcionalmente, centros (k, p).
    Trabaja en float, sin estandarizar; conserva inicios, orden y tolerancias
    de la sesión. Devuelve el resumen de la mejor partición y ``n_iter_``
    entero. La WSS está en unidades cuadráticas, no mide aciertos de clase.
    """
    # 1. Prepara datos e inicios reproducibles; exige k centros distintos.
    a = np.asarray(datos, dtype=float)
    rng = np.random.default_rng(random_state)
    unicos = np.unique(a, axis=0)
    if not 1 <= k <= len(unicos):
        raise ValueError("k debe estar entre 1 y el número de observaciones distintas.")
    mejor = None
    for _ in range(n_init if init is None else 1):
        centros = unicos[rng.choice(len(unicos), k, replace=False)].copy() if init is None else np.asarray(init, dtype=float).copy()
        etiquetas = cdist(a, centros, metric="sqeuclidean").argmin(axis=1)
        tamanos = np.bincount(etiquetas, minlength=k).astype(float)
        if (tamanos == 0).any():
            raise ValueError("La inicialización produce un grupo vacío.")
        centros = np.array([a[etiquetas == j].mean(axis=0) for j in range(k)])
        alternativos = etiquetas.copy()

        def transferir(i, destino):
            # Actualiza dos medias y tamaños sin recalcular toda la partición.
            origen = etiquetas[i]
            centros[origen] = (tamanos[origen] * centros[origen] - a[i]) / (tamanos[origen] - 1)
            centros[destino] = (tamanos[destino] * centros[destino] + a[i]) / (tamanos[destino] + 1)
            tamanos[origen] -= 1
            tamanos[destino] += 1
            etiquetas[i] = destino
            alternativos[i] = origen

        # 2. Alterna transferencias hasta no mejorar o llegar al límite de iteraciones.
        for iteracion in range(max_iter):
            cambios = 0
            # Etapa óptima: examina todos los destinos y evita vaciar el grupo de origen.
            for i, punto in enumerate(a):
                origen = etiquetas[i]
                if tamanos[origen] <= 1 or k == 1:
                    continue
                distancias = np.sum((centros - punto) ** 2, axis=1)
                retirada = tamanos[origen] / (tamanos[origen] - 1) * distancias[origen]
                costes = tamanos / (tamanos + 1) * distancias
                costes[origen] = np.inf
                destino = int(costes.argmin())
                alternativos[i] = destino
                if costes[destino] < retirada - 1e-12:
                    transferir(i, destino)
                    cambios += 1
            if cambios == 0:
                break
            # Etapa rápida: prueba el último par candidato, con el mismo criterio de coste.
            for _ in range(max_iter):
                rapidos = 0
                for i, punto in enumerate(a):
                    origen, destino = etiquetas[i], alternativos[i]
                    if origen == destino or tamanos[origen] <= 1:
                        continue
                    retirada = tamanos[origen] / (tamanos[origen] - 1) * np.sum((punto - centros[origen]) ** 2)
                    entrada = tamanos[destino] / (tamanos[destino] + 1) * np.sum((punto - centros[destino]) ** 2)
                    if entrada < retirada - 1e-12:
                        transferir(i, destino)
                        rapidos += 1
                if rapidos == 0:
                    break
        # 3. Conserva el inicio con menor WSS para este k y esta misma escala.
        resultado = resumen_particion(a, etiquetas)
        resultado.n_iter_ = iteracion + 1
        if mejor is None or resultado.inertia_ < mejor.inertia_:
            mejor = resultado
    return mejor


def ajustar_kmeans(datos, k, n_init=10, random_state=12345, algoritmo="hartigan-wong", init=None):
    """Ajusta Hartigan–Wong o Lloyd a los datos numéricos recibidos.

    ``k`` cuenta clústeres; ``init`` puede contener centros explícitos (k, p).
    Devuelve un SimpleNamespace para Hartigan–Wong o un KMeans para Lloyd,
    con etiquetas enteras y WSS en la escala del ajuste. No estandariza ni
    evalúa exactitud; los centros explícitos implican un único inicio.
    """
    # Esta función NO estandariza: elige el algoritmo y limita hilos a uno.
    with threadpool_limits(limits=1):
        if algoritmo == "hartigan-wong":
            return hartigan_wong(datos, k, n_init, random_state, init=init)
        if algoritmo != "lloyd":
            raise ValueError("Algoritmo desconocido.")
        # Con centros explícitos se realiza un único inicio, no n_init sorteos.
        inicial = "random" if init is None else np.asarray(init)
        return KMeans(n_clusters=k, n_init=n_init if init is None else 1, init=inicial,
                      algorithm="lloyd", random_state=random_state, max_iter=300).fit(datos)


def hcpc(datos, k, consolidar, random_state=12345):
    """ACP completa, enlace completo y consolidación opcional, sin gráficos.

    Recibe la matriz numérica ya estandarizada del notebook; no la vuelve a
    escalar. Devuelve un SimpleNamespace con ``labels_`` entero, ``tree_``
    float y ``scores_`` float con todas las componentes. El árbol conserva
    distancias en la escala recibida. Consolidar parte de los centroides del
    corte y reduce su criterio WSS, no certifica clases ni aciertos.
    """
    # 1. ACP completa sobre los datos recibidos: rota, pero no reduce la dimensión.
    coordenadas = PCA().fit_transform(datos)
    arbol = linkage(pdist(coordenadas), method="complete")
    inicial = renumerar(cut_tree(arbol, n_clusters=k).ravel())
    # 2. Los centroides del corte son el inicio explícito de la consolidación.
    centros = np.array([coordenadas[inicial == j].mean(axis=0) for j in range(k)])
    grupos = ajustar_kmeans(coordenadas, k, init=centros, random_state=random_state).labels_ if consolidar else inicial
    # 3. Devuelve solo el modelo; todas las componentes intervienen en el ajuste.
    return SimpleNamespace(labels_=grupos, tree_=arbol, scores_=coordenadas)


def hopkins(datos, n=None, random_state=12345):
    """Calcula H0 adimensional con distancias euclídeas sin elevar a p.

    Recibe datos numéricos (filas por observación), convertidos a float.
    ``n`` es el tamaño muestral, no el número de clústeres; por defecto usa
    todas las filas menos una. Valores bajos sugieren concentración frente
    a la caja uniforme, no exactitud, significación ni un k óptimo.
    """
    # 1. Muestra filas reales sin reemplazo; n es tamaño muestral, NO número de grupos.
    a = np.asarray(datos, dtype=float)
    m = len(a) - 1 if n is None else int(n)
    if not 1 <= m < len(a):
        raise ValueError("El tamaño de muestra debe estar entre 1 y n-1.")
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(a), m, replace=False)
    # 2. Vecino real más próximo: excluye solo la propia fila; conserva posibles duplicados.
    reales = cdist(a[indices], a)
    reales[np.arange(m), indices] = np.inf
    w = reales.min(axis=1)
    # 3. Referencia uniforme independiente por variable dentro de su caja de mínimos/máximos.
    uniformes = rng.uniform(a.min(axis=0), a.max(axis=0), size=(m, a.shape[1]))
    u = cdist(uniformes, a).min(axis=1)
    # 4. Convención de este cuaderno: valores bajos sugieren concentración frente a la caja.
    return w.sum() / (w.sum() + u.sum())


def pam(datos, k, distancias=None, max_iter=100):
    """PAM con BUILD y SWAP sobre una matriz numérica (n, p).

    Convierte los datos a float. ``distancias`` admite una matriz numérica
    (n, n) de disimilitudes; por defecto usa distancia euclídea sin cuadrar.
    Devuelve un SimpleNamespace con etiquetas e índices de medoides enteros
    y centros float que son filas reales. ``inertia_`` y ``history_`` suman
    distancias en sus unidades originales, no WSS ni aciertos de clase.
    """
    # Preparación: una fila por observación; D contiene disimilitudes, no cuadrados.
    a = np.asarray(datos, dtype=float)
    D = squareform(pdist(a)) if distancias is None else np.asarray(distancias)
    if not 1 <= k < len(a):
        raise ValueError("k debe ser menor que el número de observaciones.")
    # BUILD: añadir observaciones reales que reduzcan la distancia total.
    medoides = [int(D.sum(axis=0).argmin())]
    while len(medoides) < k:
        cercana = D[:, medoides].min(axis=1)
        costes = np.minimum(D, cercana[:, None]).sum(axis=0)
        costes[medoides] = np.inf
        medoides.append(int(costes.argmin()))
    # SWAP: buscar el mejor intercambio de un medoide por otra observación.
    historial = []
    for iteracion in range(max_iter):
        actuales = D[:, medoides]
        asignacion = actuales.argmin(axis=1)
        primera = actuales.min(axis=1)
        segunda = np.partition(actuales, 1, axis=1)[:, 1] if k > 1 else np.full(len(a), np.inf)
        coste_actual = primera.sum()
        historial.append(coste_actual)
        mejor = (coste_actual, None, None)
        # Si retiramos el medoide j, sus miembros recurren al segundo más próximo.
        for j in range(k):
            base = np.where(asignacion == j, segunda, primera)
            costes = np.minimum(D, base[:, None]).sum(axis=0)
            costes[medoides] = np.inf
            candidato = int(costes.argmin())
            if costes[candidato] < mejor[0] - 1e-9:
                mejor = (costes[candidato], j, candidato)
        if mejor[1] is None:
            break
        medoides[mejor[1]] = mejor[2]
    # Resultado: asignar al medoide más cercano; inertia_ es coste PAM, no WSS.
    etiquetas = D[:, medoides].argmin(axis=1)
    return SimpleNamespace(labels_=etiquetas, medoid_indices_=np.array(medoides),
                           cluster_centers_=a[medoides], inertia_=D[:, medoides].min(axis=1).sum(),
                           history_=np.array(historial), n_iter_=iteracion + 1)


def kernel_kmeans(datos, k=2, gamma=2.0, random_state=12345, max_iter=100):
    """K-means con kernel gaussiano e inicialización espectral conservada.

    Recibe datos numéricos (n, p), convertidos a float sin estandarizar.
    ``gamma`` actúa sobre sus distancias cuadráticas. Devuelve etiquetas
    enteras, ``history_`` float y ``n_iter_`` entero en un SimpleNamespace.
    Los costes corresponden al espacio implícito del kernel, no a WSS en
    las unidades originales ni a porcentajes de clasificación correcta.
    """
    # Similitudes gaussianas: gamma actúa sobre las distancias de los datos recibidos.
    a = np.asarray(datos, dtype=float)
    K = np.exp(-gamma * squareform(pdist(a, metric="sqeuclidean")))
    # Inicialización espectral: normalizar afinidades y usar sus k autovectores superiores.
    pesos = np.sqrt(K.sum(axis=1))
    normalizada = K / np.outer(pesos, pesos)
    with threadpool_limits(limits=1):
        _, vectores = eigh(normalizada, subset_by_index=[len(a) - k, len(a) - 1])
        vectores = vectores / np.linalg.norm(vectores, axis=1, keepdims=True)
        etiquetas = KMeans(k, n_init=10, random_state=random_state).fit_predict(vectores)
    # Actualizaciones kernel: distancia cuadrática al centro implícito de cada grupo.
    historial = []
    for iteracion in range(max_iter):
        distancias = np.empty((len(a), k))
        for grupo in range(k):
            miembros = np.flatnonzero(etiquetas == grupo)
            if not len(miembros):
                raise ValueError("El kernel ha producido un grupo vacío; cambia gamma.")
            distancias[:, grupo] = np.diag(K) - 2 * K[:, miembros].mean(axis=1) + K[np.ix_(miembros, miembros)].mean()
        # Reasignar sin aceptar un grupo vacío; detenerse si las etiquetas no cambian.
        nuevas = distancias.argmin(axis=1)
        historial.append(float(distancias[np.arange(len(a)), etiquetas].sum()))
        if np.array_equal(nuevas, etiquetas):
            break
        if len(np.unique(nuevas)) < k:
            break
        etiquetas = nuevas
    # El historial mide el objetivo en el espacio del kernel, no la WSS original.
    return SimpleNamespace(labels_=etiquetas, history_=np.array(historial), n_iter_=iteracion + 1)


def clara(datos, k, samples=5, sampsize=None, random_state=12345):
    """Aplica PAM a muestras y evalúa los medoides en todos los datos.

    Recibe una matriz numérica (n, p), convertida a float; ``k`` cuenta
    clústeres y ``sampsize`` filas por muestra. Devuelve un SimpleNamespace
    con etiquetas e índices de medoides enteros y centros float. ``inertia_``
    es suma de distancias euclídeas, no WSS ni exactitud de clasificación.
    """
    # Preparación: muestras reproducibles; por defecto, 40 + 2k observaciones.
    a = np.asarray(datos, dtype=float)
    rng = np.random.default_rng(random_state)
    tamano = min(len(a), 40 + 2 * k) if sampsize is None else min(len(a), sampsize)
    if tamano <= k:
        raise ValueError("El tamaño de muestra debe ser mayor que k.")
    mejor = None
    for _ in range(samples):
        # Conservar los mejores medoides previos y completar la muestra sin reemplazo.
        conservar = np.array([], dtype=int) if mejor is None else mejor.medoid_indices_
        candidatos = np.setdiff1d(np.arange(len(a)), conservar)
        indices = np.r_[conservar, rng.choice(candidatos, tamano - len(conservar), replace=False)]
        # PAM local: traducir los índices de la muestra a filas del conjunto completo.
        local = pam(a[indices], k)
        medoides = indices[local.medoid_indices_]
        # Evaluación global: elegir por suma de distancias a medoides, no por WSS.
        distancias = cdist(a, a[medoides])
        coste = distancias.min(axis=1).sum()
        if mejor is None or coste < mejor.inertia_:
            mejor = SimpleNamespace(labels_=distancias.argmin(axis=1), medoid_indices_=medoides,
                                     cluster_centers_=a[medoides], inertia_=coste)
    return mejor


def generar_atipicos(nout=3, random_state=12345):
    """Devuelve un DataFrame float de 1000 + nout filas y columnas x1, x2.

    Las coordenadas tienen unidades simuladas: dos grupos base y atípicos
    lejanos, sin etiquetas de clase. Conserva el orden original de sorteos;
    cambiar ``nout`` también desplaza los sorteos de la base de x2. No
    construye una muestra base distinta para después añadirle atípicos.
    """
    # Regenerar la simulación con la semilla explícita para cada número de atípicos.
    rng = np.random.default_rng(random_state)
    n = 1000
    # Dos grupos se separan por x1; los atípicos se alejan en ambos ejes.
    x1 = np.r_[rng.normal(10, 1, n//2), rng.normal(100, 1, n//2), rng.normal(10**5, 1, nout)]
    x2 = np.r_[rng.normal(10, 1, n), rng.normal(10**5, 1, nout)]
    return pd.DataFrame({"x1": x1, "x2": x2})