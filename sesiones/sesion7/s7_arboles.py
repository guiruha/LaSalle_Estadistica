"""Árboles de inferencia condicional para predictores numéricos."""  # Describe el modelo implementado en este archivo.

from types import SimpleNamespace  # Permite guardar datos relacionados de cada nodo como atributos.

import numpy as np  # Proporciona arreglos y operaciones matemáticas.
import pandas as pd  # Devuelve tablas etiquetadas para predicciones y resúmenes.
from scipy import stats  # Proporciona la distribución chi-cuadrado para los contrastes.
from scipy.special import gammaln  # Calcula logaritmos de la función gamma con estabilidad numérica.
from threadpoolctl import threadpool_limits  # Limita hilos de cálculo numérico durante el ajuste.


def log_cola_chi2(estadistico, grados):  # Define el cálculo de p-valores chi-cuadrado en escala logarítmica.
    """Logaritmo de la cola chi-cuadrado, con una aproximación en extremos."""  # Define el resultado estadístico calculado.
    valores = np.asarray(estadistico, dtype=float)  # Asegura que la entrada sea un array de números decimales.
    logp = stats.chi2.logsf(valores, grados)  # Calcula logaritmos de probabilidades de cola sin restas inestables.
    extremos = ~np.isfinite(logp) & (valores > 0)  # Detecta estadísticos tan grandes que la probabilidad no se representa.
    if extremos.any():  # Aplica una aproximación solo cuando existen valores extremos.
        x, a = valores[extremos] / 2, grados / 2  # Reescribe los argumentos para la aproximación asintótica.
        termino, suma = np.ones(len(x)), np.ones(len(x))  # Inicializa el primer término y la suma de la serie.
        for j in range(1, 12):  # Añade once correcciones para mejorar la aproximación.
            termino *= (a - j) / x  # Calcula el siguiente término de la expansión.
            suma += termino  # Acumula el término recién calculado.
        logp[extremos] = -x + (a - 1) * np.log(x) - gammaln(a) + np.log(suma)  # Sustituye probabilidades extremas por su logaritmo aproximado.
    return logp  # Devuelve un log-p para cada estadístico de entrada.


class ArbolNumerico:  # Agrupa métodos de predicción y consulta compartidos por los árboles.
    """Predicción y vistas comunes a árboles binarios con predictores numéricos."""  # Explica las herramientas compartidas.

    def hojas(self, podados=None, profundidad=None):  # Define cómo listar las hojas visibles del árbol.
        podados = set() if podados is None else set(podados)  # Prepara una búsqueda rápida de nodos podados.
        resultado = []  # Guardará los identificadores de hojas que quedan visibles.

        def visitar(i):  # Recorre el árbol desde el nodo cuyo índice recibe.
            nodo = self.nodos[i]  # Recupera la información del nodo actual.
            if nodo.izquierda is None or i in podados or (  # Agrupa también la condición opcional de profundidad.
                profundidad is not None and nodo.profundidad >= profundidad  # Comprueba si se alcanzó el límite de dibujo.
            ):  # Cualquiera de estas condiciones convierte el nodo en hoja visible.
                resultado.append(i)  # Conserva el identificador de esta hoja.
            else:  # Si no es terminal, sigue recorriendo las dos ramas.
                visitar(nodo.izquierda)  # Explora primero el hijo izquierdo.
                visitar(nodo.derecha)  # Explora después el hijo derecho.

        visitar(0)  # Comienza el recorrido desde la raíz.
        return resultado  # Devuelve todos los identificadores de hojas visibles.

    def predict_proba(self, X, podados=None, profundidad=None):  # Define predicciones en forma de probabilidades por clase.
        valores = X.loc[:, self.variables].to_numpy(dtype=float)  # Ordena columnas y las convierte a números para comparar con cortes.
        resultado = np.zeros((len(X), len(self.clases)))  # Reserva una probabilidad por fila y por clase.
        podados = set() if podados is None else set(podados)  # Prepara los nodos que se deben tratar como hojas.

        def visitar(i, filas):  # Envía las filas de entrada por las ramas apropiadas.
            if not len(filas):  # Comprueba si alguna fila llegó a esta rama.
                return  # No hay nada que predecir en una rama vacía.
            nodo = self.nodos[i]  # Recupera la regla y recuentos del nodo actual.
            if nodo.izquierda is None or i in podados or (  # Comprueba cuándo detener el recorrido de predicción.
                profundidad is not None and nodo.profundidad >= profundidad  # Comprueba si la vista debe detenerse a esta altura.
            ):  # Si es terminal, se asignan las frecuencias del nodo como probabilidades.
                resultado[filas] = nodo.recuentos / nodo.recuentos.sum()  # Convierte el recuento por clase en proporciones.
            else:  # Si tiene hijos, reparte las filas según la regla aprendida.
                lado = valores[filas, nodo.variable] <= nodo.corte  # Marca filas que cumplen la condición del nodo.
                visitar(nodo.izquierda, filas[lado])  # Envía las filas verdaderas al hijo izquierdo.
                visitar(nodo.derecha, filas[~lado])  # Envía las restantes al hijo derecho.

        visitar(0, np.arange(len(X)))  # Inicia el recorrido con todas las filas desde la raíz.
        return pd.DataFrame(resultado, columns=self.clases, index=X.index)  # Devuelve probabilidades con índices y clases etiquetados.

    def predict(self, X, podados=None, profundidad=None):  # Define la predicción de etiqueta más probable.
        return self.predict_proba(X, podados, profundidad).idxmax(axis=1).rename("Predicción")  # Elige la clase de mayor probabilidad para cada fila.

    def tabla_nodos(self):  # Define una tabla legible con una fila por nodo.
        return pd.DataFrame(  # Construye una tabla con las filas que se preparan a continuación.
            [  # Empieza una lista donde cada elemento representa un nodo.
                {  # Abre el diccionario con los datos del nodo actual.
                    "Nodo": i,  # Identifica el número de nodo dentro del árbol.
                    "Profundidad": n.profundidad,  # Indica cuántas divisiones hay desde la raíz.
                    "n": int(n.recuentos.sum()),  # Cuenta observaciones que llegan a este nodo.
                    "Variable": self.variables[n.variable] if n.variable is not None else None,  # Muestra el predictor usado o None si es hoja.
                    "Corte": n.corte,  # Muestra el valor límite de la regla de división.
                    "Clase": self.clases[n.recuentos.argmax()],  # Identifica la clase con más observaciones.
                    "log10(p)": n.logp / np.log(10) if n.logp is not None else np.nan,  # Expresa el p-valor en base 10 cuando existe.
                    "Terminal": n.izquierda is None,  # Indica si el nodo no tiene hijos y es terminal.
                }  # Crea una fila con la información de este nodo.
                for i, n in enumerate(self.nodos)  # Recorre todos los nodos guardando su número y objeto.
            ]  # Termina la lista de filas que formarán la tabla.
        ).set_index("Nodo")  # Convierte la tabla y usa el identificador como índice.


class ArbolCondicional(ArbolNumerico):  # Implementa los contrastes y divisiones del árbol condicional.
    """Inferencia condicional para predictores numéricos y respuesta nominal."""  # Resume el tipo de datos admitido.

    def __init__(  # Configura los límites y criterios que controlarán el crecimiento del árbol.
        self,  # Recibe la instancia del árbol que se va a configurar.
        max_depth=None,  # Limita opcionalmente la profundidad máxima.
        min_samples_split=20,  # Fija el mínimo de observaciones para intentar dividir.
        min_samples_leaf=7,  # Fija el mínimo de observaciones de cada hoja.
        alpha=0.05,  # Define el nivel de significación de los contrastes.
        min_probability=0.01,  # Define el tamaño proporcional mínimo de cada grupo.
        splitstat="maximum",  # Cierra la lista de opciones y fija el criterio de corte predeterminado.
    ):  # Cierra la lista de parámetros de configuración.
        self.max_depth, self.min_samples_split = max_depth, min_samples_split  # Guarda el límite de profundidad y el mínimo para intentar dividir.
        self.min_samples_leaf, self.alpha = min_samples_leaf, alpha  # Guarda el tamaño mínimo de hoja y el nivel de significación.
        self.min_probability, self.splitstat = min_probability, splitstat  # Guarda tamaño proporcional mínimo y estadístico de corte.

    def fit(self, X, y):  # Ajusta el árbol a predictores X y etiquetas conocidas y.
        self.variables = list(X.columns)  # Guarda los nombres y el orden de las características.
        self.clases, codigos = np.unique(y, return_inverse=True)  # Ordena etiquetas y asigna un código entero a cada una.
        valores = X.to_numpy(dtype=float)  # Convierte predictores a una matriz numérica para acelerar los cálculos.
        if not np.isfinite(valores).all():  # Comprueba que no haya ausentes ni valores infinitos.
            raise ValueError("Esta implementación requiere predictores numéricos sin ausentes.")  # Explica los datos que debe recibir el método.
        self.nodos = []  # Inicializa la lista donde se irán guardando los nodos del árbol.

        def construir(filas, profundidad):  # Crea recursivamente un nodo y, si procede, sus dos hijos.
            A, respuesta = valores[filas], codigos[filas]  # Extrae predictores y etiquetas de las filas de esta rama.
            n = len(filas)  # Cuenta cuántas observaciones llegan al nodo.
            recuentos = np.bincount(respuesta, minlength=len(self.clases)).astype(float)  # Cuenta filas de cada clase, incluyendo ceros.
            nodo = SimpleNamespace(  # Crea el registro que almacena la información de esta rama.
                profundidad=profundidad,  # Registra el nivel actual del árbol.
                recuentos=recuentos,  # Conserva cuántas filas de cada clase hay en el nodo.
                variable=None,  # Se completará si se encuentra un predictor para dividir.
                corte=np.nan,  # Se completará con el límite del predictor elegido.
                logp=None,  # Se completará con el p-valor ajustado del nodo.
                izquierda=None,  # Inicializa la referencia al futuro hijo izquierdo.
                derecha=None,  # Inicializa la referencia al futuro hijo derecho.
            )  # Crea el objeto que almacena la información del nodo.
            indice = len(self.nodos)  # Asigna el siguiente índice libre al nodo.
            self.nodos.append(nodo)  # Guarda el nodo antes de crear sus hijos.
            presentes = recuentos > 0  # Marca cuáles clases aparecen en este nodo.
            if (  # Agrupa las condiciones que deciden si este nodo debe seguir creciendo.
                presentes.sum() < 2  # Se detiene si ya hay una sola clase: no hay nada que separar.
                or n < self.min_samples_split  # Se detiene si el nodo tiene muy pocas observaciones.
                or (self.max_depth is not None and profundidad >= self.max_depth)  # Se detiene al alcanzar la profundidad máxima configurada.
            ):  # Si alguna regla de parada se cumple, el nodo queda como hoja.
                return indice  # Devuelve su índice sin intentar crear hijos.
            centradas = A - A.mean(axis=0)  # Resta la media de cada característica para centrar los valores.
            ss_total = np.einsum("ij,ij->j", centradas, centradas)  # Calcula variación total por predictor.
            candidatas = ss_total > 1e-12  # Descarta características constantes o casi constantes.
            if not candidatas.any():  # Comprueba si no queda ningún predictor variable.
                return indice  # Deja el nodo como hoja si no hay forma de separarlo.
            sumas_clase = np.vstack(  # Apila una suma por clase para comparar sus medias.
                [centradas[respuesta == c].sum(axis=0) for c in np.flatnonzero(presentes)]  # Suma valores centrados por clase y característica.
            )  # Apila una fila por cada clase que está presente.
            ss_entre = (sumas_clase**2 / recuentos[presentes, None]).sum(axis=0)  # Resume cuánto varían las medias entre clases.
            estadisticos = np.divide(  # Divide la variación entre clases por la variación total.
                (n - 1) * ss_entre,  # Usa la variación entre clases para formar el numerador del estadístico.
                ss_total,  # Divide por la variación total de cada predictor.
                out=np.zeros(len(self.variables)),  # Escribe cero en las variables que no se pueden evaluar.
                where=candidatas,  # Calcula solo para predictores no constantes.
            )  # Obtiene un estadístico de asociación por característica.
            logp = log_cola_chi2(estadisticos, presentes.sum() - 1)  # Convierte estadísticos en probabilidades de cola logarítmicas.
            # Ajuste familiar 1-(1-p)^m, evaluado con estabilidad numérica.
            ajustados = logp.copy()  # Copia los p-valores para corregirlos sin modificar los originales.
            pequenos = logp < -20  # Detecta p-valores muy pequeños para usar una aproximación estable.
            ajustados[pequenos] += np.log(candidatas.sum())  # Aplica la corrección por número de contrastes a esos valores.
            with np.errstate(divide="ignore", invalid="ignore"):  # Controla avisos de operaciones límite conocidas.
                ajustados[~pequenos] = np.log(  # Calcula los contrastes ajustados para p-valores no extremos.
                    -np.expm1(candidatas.sum() * np.log1p(-np.exp(logp[~pequenos])))  # Calcula corrección familiar sin perder precisión numérica.
                )  # Guarda el logaritmo de los p-valores ajustados que no son extremos.
            ajustados[~candidatas] = 0  # Evita que variables no candidatas sean elegidas como significativas.
            nodo.logp = float(ajustados.min())  # Guarda el p-valor ajustado más pequeño del nodo.
            if nodo.logp > np.log(self.alpha):  # Comprueba si la menor evidencia supera el nivel permitido.
                return indice  # Detiene el crecimiento si ningún predictor es significativo.
            minimo = max(self.min_samples_leaf, int(np.floor(n * self.min_probability)) + 1)  # Calcula el tamaño mínimo de cada lado del corte.
            for variable in np.argsort(ajustados, kind="stable")[:2]:  # Prueba los dos predictores con menor p-valor ajustado.
                if ajustados[variable] > np.log(self.alpha):  # Comprueba si este predictor no alcanza significación.
                    continue  # Salta a la siguiente variable candidata.
                orden = np.argsort(A[:, variable], kind="stable")  # Ordena filas según los valores de este predictor.
                x = A[orden, variable]  # Guarda los valores ordenados para buscar cortes posibles.
                posiciones = np.arange(1, n)  # Enumera las posiciones entre observaciones consecutivas.
                validas = (  # Combina las reglas que determinan si cada punto de corte es admisible.
                    (x[:-1] < x[1:])  # Solo permite cortes entre valores distintos.
                    & (posiciones >= minimo)  # Exige suficientes observaciones a la izquierda.
                    & (n - posiciones >= minimo)  # Exige suficientes observaciones a la derecha.
                )  # Marca los cortes que respetan todas las condiciones.
                if not validas.any():  # Comprueba si quedó al menos un corte válido.
                    continue  # Prueba el siguiente predictor si no hay cortes posibles.
                izquierda = np.eye(len(self.clases))[respuesta[orden]].cumsum(axis=0)[:-1, presentes]  # Cuenta acumulativamente las clases a cada posible corte.
                proporcion = recuentos[presentes] / n  # Calcula la proporción de cada clase en el nodo.
                diferencias = izquierda - posiciones[:, None] * proporcion  # Compara recuentos observados con los esperados sin separación.
                if self.splitstat == "maximum":  # Usa la mayor diferencia estandarizada entre clases.
                    varianza = (  # Calcula la variabilidad esperada de cada diferencia bajo la hipótesis nula.
                        posiciones[:, None]  # Número de filas a la izquierda de cada posible corte.
                        * (n - posiciones[:, None])  # Número de filas a la derecha.
                        / (n - 1)  # Ajuste por tamaño total del nodo.
                        * proporcion  # Aporta la proporción de cada clase.
                        * (1 - proporcion)  # Completa la varianza binomial de cada clase.
                    )  # Estima la varianza de las diferencias bajo ausencia de asociación.
                    puntuacion = (diferencias**2 / varianza).max(axis=1)  # Escoge la mayor separación estandarizada en cada corte.
                elif self.splitstat == "quadratic":  # Usa una medida cuadrática conjunta de separación.
                    puntuacion = (n - 1) * (diferencias**2 / proporcion).sum(axis=1) / (  # Forma el estadístico cuadrático para cada posible división.
                        posiciones * (n - posiciones)  # Ajusta por tamaños de los grupos de ambos lados del corte.
                    )  # Devuelve una puntuación conjunta para cada corte posible.
                else:  # Se ejecuta si se solicitó un nombre de estadístico no reconocido.
                    raise ValueError("splitstat debe ser maximum o quadratic.")  # Informa los valores permitidos.
                puntuacion[~validas] = -np.inf  # Impide seleccionar posiciones que no cumplen los mínimos.
                posicion = int(np.argmax(puntuacion))  # Elige la posición con mayor separación de clases.
                nodo.variable, nodo.corte = int(variable), float(x[posicion])  # Guarda predictor y valor de corte en el nodo.
                nodo.logp = float(ajustados[variable])  # Guarda la evidencia asociada al predictor elegido.
                division = A[:, variable] <= nodo.corte  # Separa las filas originales según la regla elegida.
                nodo.izquierda = construir(filas[division], profundidad + 1)  # Construye el subárbol izquierdo recursivamente.
                nodo.derecha = construir(filas[~division], profundidad + 1)  # Construye el subárbol derecho recursivamente.
                break  # Detiene el ensayo de variables tras encontrar una división válida.
            return indice  # Devuelve el nodo actual después de construir sus ramas o dejarlo como hoja.

        with threadpool_limits(limits=1):  # Limita el uso de hilos en operaciones numéricas internas.
            construir(np.arange(len(X)), 0)  # Empieza el árbol con todas las filas en la raíz y profundidad cero.
        return self  # Devuelve el propio modelo ya ajustado para permitir encadenar llamadas.
