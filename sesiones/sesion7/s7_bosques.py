"""Diagnósticos OOB, importancia y ajuste de mtry para random forest en S7."""  # Resume las herramientas de este módulo.

from types import SimpleNamespace  # Agrupa varias tablas relacionadas en un objeto.

import numpy as np  # Proporciona arreglos, muestreo y operaciones numéricas.
import pandas as pd  # Organiza votos y resultados en tablas.
from sklearn.ensemble import RandomForestClassifier  # Crea bosques de árboles de clasificación.

from s7_evaluacion import confusion  # Reutiliza la matriz de confusión común de la sesión.


def indices_oob(arbol, n):  # Identifica filas que no participaron en el bootstrap de un árbol.
    """Índices no extraídos en la muestra bootstrap de un árbol."""  # Explica qué filas devolverá.
    muestra = np.random.RandomState(arbol.random_state).randint(0, n, n)  # Recrea las n selecciones bootstrap de este árbol.
    return np.flatnonzero(np.bincount(muestra, minlength=n) == 0)  # Devuelve filas cuya frecuencia bootstrap fue cero.


def diagnostico_oob(bosque, X, y):  # Acumula predicciones fuera de bolsa y calcula sus errores.
    """Error OOB acumulado, cobertura, errores por clase y matriz de confusión."""  # Describe todos los diagnósticos que calcula.
    valores = np.asarray(X, dtype=np.float32)  # Convierte las características a una matriz compacta para los árboles.
    codigos = pd.Categorical(y, categories=bosque.classes_).codes  # Codifica cada etiqueta real como el índice de su clase.
    votos = np.zeros((len(X), len(bosque.classes_)), dtype=int)  # Reserva un contador por fila y por clase.
    historial = []  # Guardará una fila de errores tras añadir cada árbol.
    for arbol in bosque.estimators_:  # Recorre los árboles del bosque uno por uno.
        fuera = indices_oob(arbol, len(X))  # Encuentra las filas que este árbol no usó para entrenar.
        prediccion = arbol.predict(valores[fuera]).astype(int)  # Predice las clases de esas filas excluidas.
        np.add.at(votos, (fuera, prediccion), 1)  # Suma un voto a la clase predicha de cada fila OOB.
        disponibles = votos.sum(axis=1) > 0  # Marca filas que ya recibieron al menos un voto fuera de bolsa.
        ganadora = votos.argmax(axis=1)  # Elige para cada fila la clase con más votos acumulados.
        error = ganadora != codigos  # Marca True donde la clase ganadora no coincide con la etiqueta real.
        fila = {"OOB": error[disponibles].mean()}  # Calcula el error global solo entre filas con votos disponibles.
        for j, clase in enumerate(bosque.classes_):  # Recorre las clases para calcular sus errores por separado.
            seleccion = disponibles & (codigos == j)  # Selecciona filas OOB disponibles que pertenecen a esta clase.
            fila[clase] = error[seleccion].mean() if seleccion.any() else np.nan  # Calcula error o NaN si no hay casos.
        fila["Cobertura"] = disponibles.mean()  # Calcula qué fracción de todas las filas ya tiene votos OOB.
        historial.append(fila)  # Guarda el resultado después de incorporar este árbol.
    disponibles = votos.sum(axis=1) > 0  # Vuelve a identificar las filas con al menos un voto al terminar.
    predicciones = pd.Series(  # Crea una serie con las etiquetas OOB y sus índices originales.
        bosque.classes_[votos[disponibles].argmax(axis=1)], index=X.index[disponibles]  # Recupera etiquetas y conserva índices originales.
    )  # Construye una serie de predicciones OOB disponibles.
    tabla = confusion(np.asarray(y)[disponibles], predicciones, bosque.classes_).T  # Forma matriz y pone clases reales en filas.
    tabla = tabla.rename_axis(index="Real", columns="Predicción")  # Etiqueta claramente filas y columnas.
    tabla["Error de clase"] = 1 - np.diag(tabla.to_numpy()) / tabla.sum(axis=1)  # Añade uno menos el acierto de cada clase.
    curva = pd.DataFrame(historial, index=np.arange(1, len(historial) + 1)).rename_axis("Árboles")  # Indexa resultados por número de árboles.
    return SimpleNamespace(  # Agrupa todas las salidas del diagnóstico en un objeto.
        curva=curva,  # Incluye la evolución de errores y cobertura.
        votos=pd.DataFrame(votos, index=X.index, columns=bosque.classes_),  # Expone el número de votos por fila y clase.
        confusion=tabla,  # Incluye la matriz OOB con error por clase.
        predicciones=predicciones,  # Conserva las etiquetas predichas para las filas con votos.
    )  # Devuelve todas las salidas agrupadas.


def importancia_oob(bosque, X, y, seed=12345):  # Calcula importancia por permutación y disminución Gini.
    """Importancia por permutación OOB (global y por clase) y disminución Gini."""  # Explica las dos familias de importancia.
    valores = np.asarray(X, dtype=np.float32)  # Prepara las características en un array numérico.
    codigos = pd.Categorical(y, categories=bosque.classes_).codes  # Traduce las clases reales a códigos enteros.
    n_arboles, n_variables, n_clases = len(bosque.estimators_), X.shape[1], len(bosque.classes_)  # Guarda dimensiones del cálculo.
    descensos = np.zeros((n_arboles, n_variables, n_clases + 1))  # Reserva pérdidas por árbol, variable y clase más global.
    gini = np.zeros((n_arboles, n_variables))  # Reserva importancias Gini por árbol y predictor.
    rng = np.random.default_rng(seed)  # Crea un generador reproducible para permutar columnas.
    for i, arbol in enumerate(bosque.estimators_):  # Recorre cada árbol junto a su posición.
        fuera = indices_oob(arbol, len(X))  # Selecciona filas excluidas de la muestra bootstrap del árbol.
        entrada, reales = valores[fuera].copy(), codigos[fuera]  # Copia entradas OOB y conserva sus clases reales.
        base = arbol.predict(entrada).astype(int) == reales  # Mide qué filas clasifica bien antes de permutar.
        bases = np.r_[[base[reales == c].mean() for c in range(n_clases)], base.mean()]  # Guarda aciertos por clase y global.
        for j in range(n_variables):  # Permuta una característica cada vez.
            original = entrada[:, j].copy()  # Guarda temporalmente los valores originales de esa característica.
            entrada[:, j] = rng.permutation(original)  # Desordena valores para romper su relación con la clase.
            aciertos = arbol.predict(entrada).astype(int) == reales  # Vuelve a medir aciertos tras la permutación.
            resultado = np.r_[  # Combina resultados por clase y el resultado global.
                [aciertos[reales == c].mean() for c in range(n_clases)], aciertos.mean()  # Calcula aciertos por clase y global.
            ]  # Combina todas las medidas en un único vector.
            descensos[i, j] = bases - resultado  # Guarda la pérdida de acierto causada por desordenar esta variable.
            entrada[:, j] = original  # Restaura la columna para evaluar el siguiente predictor por separado.
        gini[i] = (  # Empieza a guardar la importancia Gini acumulada de este árbol.
            arbol.tree_.compute_feature_importances(normalize=False)  # Obtiene disminución Gini sin normalizar.
            * arbol.tree_.weighted_n_node_samples[0]  # La expresa en escala acumulada de observaciones del árbol.
        )  # Guarda la importancia Gini de este árbol.
    medias = descensos.mean(axis=0)  # Promedia pérdidas entre árboles para cada variable y clase.
    error_estandar = descensos.std(axis=0, ddof=0) / np.sqrt(n_arboles)  # Estima el error estándar entre árboles.
    escaladas = medias / np.where(error_estandar > 1e-15, error_estandar, 1)  # Escala el descenso por su error estándar evitando dividir por cero.
    nombres = list(bosque.classes_) + ["MeanDecreaseAccuracy"]  # Nombra columnas por clase y por la medida global.
    tabla = pd.DataFrame(escaladas, index=X.columns, columns=nombres)  # Construye la tabla de importancia escalada.
    tabla["MeanDecreaseGini"] = gini.mean(axis=0)  # Añade el promedio de importancia Gini de cada predictor.
    return SimpleNamespace(  # Agrupa la tabla principal y sus componentes estadísticos.
        tabla=tabla,  # Devuelve importancia escalada y Gini juntas.
        descensos=pd.DataFrame(medias, index=X.columns, columns=nombres),  # Incluye pérdidas medias sin escalar.
        error_estandar=pd.DataFrame(error_estandar, index=X.columns, columns=nombres),  # Incluye error estándar de cada pérdida.
    )  # Agrupa los resúmenes para su inspección.


def tunear_mtry(X, y, n_arboles=50, mejora_minima=0.05, seed=12345):  # Explora valores de mtry comparando sus errores OOB.
    """Búsqueda exploratoria OOB en ambas direcciones desde sqrt(p)."""  # Documenta la búsqueda aproximada.
    inicio = int(np.sqrt(X.shape[1]))  # Empieza con la raíz entera del número de características.
    resultados = {}  # Guardará el error OOB asociado a cada valor de mtry evaluado.

    def evaluar(mtry):  # Ajusta un bosque para mtry y devuelve su error OOB final.
        modelo = RandomForestClassifier(  # Construye el bosque correspondiente al valor de mtry actual.
            n_estimators=n_arboles,  # Usa el mismo número de árboles en cada candidato.
            max_features=mtry,  # Cambia cuántas características se consideran por división.
            bootstrap=True,  # Muestrea filas con reemplazo para cada árbol.
            random_state=seed,  # Hace reproducible el bosque candidato.
            n_jobs=1,  # Evita paralelismo excesivo en esta exploración.
        ).fit(X, y)  # Ajusta el candidato con las filas que recibió la búsqueda.
        diagnostico = diagnostico_oob(modelo, X, y)  # Calcula errores OOB del candidato.
        error = float(diagnostico.curva["OOB"].iloc[-1])  # Extrae el error tras añadir todos sus árboles.
        resultados[mtry] = error  # Guarda el resultado para la tabla final.
        return error  # Devuelve el error para comparar con la referencia.

    error_referencia = evaluar(inicio)  # Calcula el resultado inicial para medir mejoras posteriores.
    for direccion in [-1, 1]:  # Explora primero valores menores y después valores mayores.
        actual = inicio  # Reinicia la búsqueda desde el valor inicial en cada dirección.
        while error_referencia > 0:  # Continúa mientras exista margen para mejorar un error positivo.
            nuevo = max(1, int(np.ceil(actual / 2))) if direccion < 0 else min(X.shape[1], actual * 2)  # Divide o duplica mtry sin salir de sus límites.
            if nuevo == actual:  # Comprueba si ya alcanzamos el límite de valores posibles.
                break  # Detiene esta dirección cuando no hay un candidato nuevo.
            error = evaluar(nuevo)  # Ajusta el siguiente bosque y mide su error OOB.
            mejora = 1 - error / error_referencia  # Calcula la reducción relativa respecto al mejor valor de referencia.
            if mejora > mejora_minima:  # Comprueba si la mejora supera el 5 % requerido.
                error_referencia = error  # Actualiza la referencia si el nuevo resultado mejora suficientemente.
            actual = nuevo  # Avanza la búsqueda desde el candidato recién probado.
            if mejora < mejora_minima:  # Comprueba si la mejora es demasiado pequeña.
                break  # Detiene la búsqueda en esta dirección.
    return pd.Series(resultados, name="Error OOB").sort_index().rename_axis("mtry").to_frame()  # Devuelve errores en una tabla ordenada.
