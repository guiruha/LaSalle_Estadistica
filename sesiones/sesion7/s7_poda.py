"""Árbol de entropía y secuencia de poda por coste-complejidad para S7."""  # Resume el contenido de este módulo.

import numpy as np  # Proporciona arreglos y operaciones numéricas.
import pandas as pd  # Organiza las trayectorias y resultados en tablas.
from types import SimpleNamespace  # Permite guardar atributos de cada nodo de forma sencilla.
from sklearn.tree import DecisionTreeClassifier  # Proporciona el árbol de entropía base.
from sklearn.model_selection import KFold  # Divide entrenamiento en pliegues reproducibles.

from s7_arboles import ArbolNumerico  # Reutiliza el formato de nodos y predicción de S7.


def arbol_entropia(X, y, seed=12345):  # Ajusta el árbol base y lo copia al formato compartido de S7.
    """Copia un árbol de entropía de scikit-learn a la representación didáctica."""  # Describe el modelo producido.
    ajuste = DecisionTreeClassifier(  # Configura el algoritmo de árbol de decisión de scikit-learn.
        criterion="entropy",  # Usa entropía para medir mezcla de clases en los nodos.
        min_samples_split=10,  # Exige al menos diez filas para intentar una división.
        min_samples_leaf=5,  # Exige al menos cinco filas en cada hoja resultante.
        max_leaf_nodes=250,  # Limita el árbol base a un máximo de 250 hojas.
        random_state=seed,  # Fija la semilla para repetir el ajuste.
    ).fit(X, y)  # Ajusta el árbol con predictores X y etiquetas y.
    modelo = ArbolNumerico()  # Crea el contenedor compatible con las funciones de S7.
    modelo.variables, modelo.clases, modelo.nodos = list(X.columns), ajuste.classes_, []  # Guarda nombres, etiquetas y espacio de nodos.
    t = ajuste.tree_  # Recupera la estructura interna del árbol de scikit-learn.
    umbral = 0.0005 * t.weighted_n_node_samples[0] * t.impurity[0]  # Fija el filtro auxiliar en 0,05 % del valor de impureza de la raíz.

    def copiar(i, profundidad):  # Copia recursivamente un nodo de scikit-learn y sus descendientes.
        recuentos = t.value[i, 0].astype(float)  # Lee los recuentos de clase del nodo y los convierte a decimales.
        recuentos = np.rint(recuentos / recuentos.sum() * t.weighted_n_node_samples[i])  # Recupera recuentos absolutos desde las proporciones.
        nodo = SimpleNamespace(  # Crea la representación didáctica del nodo actual.
            profundidad=profundidad,  # Guarda la distancia del nodo respecto a la raíz.
            recuentos=recuentos,  # Conserva cuántos ejemplos de cada clase llegan aquí.
            variable=None,  # Inicializa sin predictor porque todavía no se copia la división.
            corte=np.nan,  # Marca que aún no hay un umbral de división.
            logp=None,  # Este árbol no calcula p-valores de inferencia condicional.
            izquierda=None,  # Inicializa la referencia al hijo izquierdo como inexistente.
            derecha=None,  # Inicializa la referencia al hijo derecho como inexistente.
        )  # Construye el objeto que representa un nodo.
        indice = len(modelo.nodos)  # Usa la posición siguiente como identificador del nodo.
        modelo.nodos.append(nodo)  # Añade el nodo al árbol antes de copiar a sus descendientes.
        if (  # Comprueba que exista una división y que su mejora supere el mínimo elegido.
            t.children_left[i] != t.children_right[i]  # Comprueba que scikit-learn no marcó el nodo como hoja.
            and t.weighted_n_node_samples[i] * t.impurity[i] >= umbral  # Comprueba que la impureza supera el umbral mínimo.
        ):  # Solo conserva una división si se cumplen ambas condiciones.
            nodo.variable, nodo.corte = int(t.feature[i]), float(t.threshold[i])  # Copia predictor y corte de la división.
            nodo.izquierda = copiar(t.children_left[i], profundidad + 1)  # Copia recursivamente el subárbol izquierdo.
            nodo.derecha = copiar(t.children_right[i], profundidad + 1)  # Copia recursivamente el subárbol derecho.
        return indice  # Devuelve el identificador de este nodo al nodo padre.

    copiar(0, 0)  # Empieza a copiar desde la raíz, identificador 0 y profundidad 0.
    return modelo  # Devuelve el árbol en el formato usado por el notebook.


def trayectoria_poda(modelo):  # Construye la secuencia de árboles al aumentar la penalización de poda.
    """Secuencia de coste-complejidad con pérdida de clasificación 0–1."""  # Explica el criterio de la secuencia.
    podados = set()  # Inicializa el conjunto de nodos ya convertidos en hojas.
    filas = []  # Acumula una fila de resultados por cada subárbol.
    alpha = -np.inf  # Marca el punto inicial antes de cualquier penalización.
    while True:  # Repite el cálculo hasta que solo quede un árbol sin ramas que podar.
        candidatos = {}  # Guarda la penalización necesaria para podar cada nodo interno.

        def recorrer(i):  # Calcula errores, hojas y costes de podar desde el nodo recibido.
            nodo = modelo.nodos[i]  # Recupera el nodo que se está visitando.
            error_nodo = nodo.recuentos.sum() - nodo.recuentos.max()  # Calcula errores si todo el nodo fuera una sola hoja.
            if nodo.izquierda is None or i in podados:  # Comprueba si ya es hoja o se podó previamente.
                return error_nodo, 1  # Devuelve su error y que representa una hoja.
            e1, h1 = recorrer(nodo.izquierda)  # Calcula recursivamente error y hojas del subárbol izquierdo.
            e2, h2 = recorrer(nodo.derecha)  # Calcula recursivamente error y hojas del subárbol derecho.
            candidatos[i] = (error_nodo - e1 - e2) / (h1 + h2 - 1)  # Mide incremento de error por hoja eliminada.
            return e1 + e2, h1 + h2  # Devuelve error y hojas de ambos subárboles combinados.

        errores, hojas = recorrer(0)  # Resume el árbol actual empezando desde la raíz.
        filas.append(  # Añade una nueva fila que describe el subárbol actual.
            {"Hojas": hojas, "alpha": alpha, "Errores train": errores, "podados": frozenset(podados)}  # Guarda tamaño, penalización, errores y nodos podados.
        )  # Cierra la adición del estado actual a la secuencia.
        if not candidatos:  # Comprueba si ya no quedan nodos internos que puedan podarse.
            break  # Termina el bucle cuando la raíz también es una hoja.
        alpha = max(0.0, alpha, min(candidatos.values()))  # Avanza a la siguiente penalización de coste-complejidad.
        podados.update(i for i, a in candidatos.items() if a <= alpha + 1e-10)  # Poda todos los candidatos con la penalización mínima.
    return pd.DataFrame(filas)  # Devuelve la trayectoria como tabla para inspección.


def podar_a_hojas(trayectoria, numero=10):  # Elige de la secuencia el árbol con el tamaño solicitado.
    """Elige la última subestructura de la secuencia con al menos ese número de hojas."""  # Explica qué modelo selecciona.
    posibles = trayectoria.loc[trayectoria["Hojas"] >= numero]  # Filtra árboles con el número pedido o más hojas.
    return (posibles.iloc[-1] if len(posibles) else trayectoria.iloc[0])["podados"]  # Devuelve los nodos a podar del tamaño elegido.


def validar_poda(X, y, trayectoria, cv=10, seed=12345):  # Compara podas mediante pliegues de entrenamiento.
    """Evalúa cada penalización por validación cruzada sobre entrenamiento."""  # Describe el procedimiento de comparación.
    errores = np.zeros((cv, len(trayectoria)))  # Reserva una matriz para guardar errores por pliegue y árbol.
    particiones = KFold(n_splits=cv, shuffle=True, random_state=seed)  # Prepara divisiones aleatorias reproducibles.
    for vuelta, (entrenar, validar) in enumerate(particiones.split(X)):  # Recorre los pliegues y sus índices de entrenamiento/validación.
        modelo = arbol_entropia(X.iloc[entrenar], y.iloc[entrenar], seed=seed)  # Ajusta cada pliegue con sus filas de entrenamiento.
        camino = trayectoria_poda(modelo)  # Calcula los tamaños de árbol que se podrán comparar en ese pliegue.
        for j, alpha in enumerate(trayectoria["alpha"]):  # Evalúa cada penalización de la trayectoria en este pliegue.
            disponibles = (  # Escoge el subárbol de este pliegue correspondiente a la penalización actual.
                camino.loc[camino["alpha"] <= alpha + 1e-10]  # Busca el subárbol compatible con la penalización actual.
                if np.isfinite(alpha)  # Usa este filtro cuando alpha es un número finito.
                else camino.iloc[:1]  # Para el árbol inicial, conserva solo el primer estado sin podar.
            )  # Guarda la trayectoria disponible para ese candidato.
            prediccion = modelo.predict(  # Predice las observaciones que quedaron fuera del ajuste de este pliegue.
                X.iloc[validar], podados=disponibles.iloc[-1]["podados"]  # Predice las filas de validación con esa poda.
            )  # Completa las predicciones del pliegue.
            errores[vuelta, j] = np.mean(prediccion.to_numpy() != y.iloc[validar].to_numpy())  # Guarda la fracción de clases incorrectas.
    return pd.DataFrame(  # Organiza los errores agregados en una tabla final.
        {  # Define las columnas de la tabla de validación cruzada.
            "Hojas": trayectoria["Hojas"],  # Conserva el tamaño de árbol asociado a cada penalización.
            "alpha": trayectoria["alpha"],  # Conserva la penalización que define la poda.
            "Error CV": errores.mean(axis=0),  # Calcula el error medio de cada árbol entre pliegues.
            "Desviación CV": errores.std(axis=0, ddof=1),  # Resume cuánto varía el error entre pliegues.
        }  # Define las columnas del resumen de validación.
    )  # Devuelve el resumen como tabla.
