"""Factorías y resúmenes para los modelos SVM de S7."""  # Indica que este archivo contiene herramientas reutilizables para SVM.

import pandas as pd  # Organiza los resúmenes en tablas y series.
from sklearn.pipeline import Pipeline  # Encadena transformación y modelo para que se ajusten juntos.
from sklearn.preprocessing import StandardScaler  # Estandariza cada predictor usando media y desviación.
from sklearn.svm import SVC  # Proporciona el clasificador de máquinas de vectores soporte.


def construir_svm(n_predictores, kernel="rbf", C=1):  # Crea un pipeline reutilizable para un modelo SVM.
    """Crea un pipeline que ajusta el escalado junto con el clasificador."""  # Explica el resultado de la función.
    return Pipeline(  # Devuelve los pasos encadenados para ajuste y predicción.
        [  # Empieza la lista ordenada de transformaciones y modelo.
            ("escala", StandardScaler()),  # Aprende escalas de entrenamiento y las aplica antes de predecir.
            (  # Agrupa nombre y estimador del paso de clasificación.
                "svm",  # Asigna el nombre svm al segundo paso del pipeline.
                SVC(  # Construye el clasificador con los parámetros indicados.
                    C=C,  # Controla la penalización de errores de entrenamiento.
                    kernel=kernel,  # Elige la forma de la frontera, por ejemplo radial o lineal.
                    gamma=1 / n_predictores,  # Fija gamma en función del número de características.
                    degree=3,  # Usa grado 3 cuando se selecciona el kernel polinómico.
                    coef0=0,  # Fija el término independiente usado por algunos kernels.
                    tol=0.001,  # Define la tolerancia para detener la optimización.
                    cache_size=256,  # Reserva hasta 256 MB de caché para los cálculos SVM.
                ),  # Construye el clasificador con los parámetros anteriores.
            ),  # Completa el segundo paso del pipeline.
        ]  # Define el orden: primero escalar y después clasificar.
    )  # Devuelve el pipeline listo para ajustar.


def resumen_busqueda(busqueda):  # Convierte los resultados de GridSearchCV en una tabla sencilla.
    """Tabla de error medio, desviación entre particiones y orden de la búsqueda."""  # Resume el contenido devuelto.
    tabla = pd.DataFrame(busqueda.cv_results_)  # Convierte el registro de GridSearchCV en una tabla.
    resultado = pd.DataFrame(  # Crea una tabla breve con métricas por candidato.
        {  # Define las columnas que aparecerán en el resumen.
            "Parámetros": tabla["params"],  # Guarda el conjunto de hiperparámetros de cada candidato.
            "Error CV": 1 - tabla["mean_test_score"],  # Convierte acierto medio en error medio.
            "Desviación CV": tabla["std_test_score"]  # Empieza desde la variabilidad observada entre pliegues.
            * (busqueda.n_splits_ / (busqueda.n_splits_ - 1)) ** 0.5,  # Ajusta la desviación entre pliegues.
            "Orden": tabla["rank_test_score"],  # Añade el puesto del candidato según su puntuación.
        }  # Define las columnas del resumen.
    )  # Construye el DataFrame reducido.
    return resultado.sort_values("Orden", kind="stable")  # Ordena de mejor a peor conservando empates.


def resumen_svm(modelo):  # Extrae configuración y vectores soporte del modelo ajustado.
    """Resume configuración y cantidad de vectores soporte del ajuste."""  # Describe el resumen mostrado al estudiante.
    ajuste = modelo.named_steps["svm"]  # Recupera el clasificador que está dentro del pipeline.
    return pd.Series(  # Devuelve los valores del modelo en formato etiquetado.
        {  # Define las etiquetas y valores del resumen.
            "Kernel": ajuste.kernel,  # Informa la forma de frontera utilizada.
            "C": ajuste.C,  # Informa la penalización configurada.
            "gamma": ajuste.gamma,  # Informa el parámetro de alcance del kernel.
            "Grado": ajuste.degree,  # Informa el grado, relevante para el kernel polinómico.
            "Vectores soporte": len(ajuste.support_),  # Cuenta observaciones que definen la frontera.
            "Soporte clase False": ajuste.n_support_[0],  # Cuenta vectores soporte de la primera clase.
            "Soporte clase True": ajuste.n_support_[1],  # Cuenta vectores soporte de la segunda clase.
        }  # Define los valores del resumen.
    )  # Devuelve los resultados como una serie etiquetada.
