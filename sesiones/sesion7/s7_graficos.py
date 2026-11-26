"""Gráficos reutilizables para árboles, clases y función de decisión SVM en S7."""  # Describe el contenido de este archivo auxiliar.

import numpy as np  # Proporciona arreglos, coordenadas y selecciones aleatorias.
import pandas as pd  # Organiza los puntos de la cuadrícula como tabla de predictores.
import matplotlib.pyplot as plt  # Crea y configura los gráficos.
from IPython.display import display  # Presenta tablas y figuras desde el widget interactivo.
from matplotlib.patches import FancyBboxPatch, Patch  # Dibuja nodos redondeados y marcas de color para la leyenda.


def barras_clases(valores, titulo, ylabel="Proporción de aciertos"):  # Dibuja y presenta una barra por clase.
    """Dibuja una barra por clase para una serie indexada por clase."""  # Explica el tipo de gráfico producido.
    fig, ax = plt.subplots()  # Crea una figura y un área de dibujo.
    ax.bar(valores.index.astype(str), valores)  # Dibuja una barra con el valor de cada clase.
    ax.set(ylim=(0, 1), ylabel=ylabel, title=titulo)  # Fija escala vertical y rótulos del gráfico.
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")  # Inclina nombres largos para que se lean sin solaparse.
    fig.tight_layout()  # Ajusta los márgenes de la figura.
    plt.show()  # Presenta el gráfico debajo de la celda.


def dibujar_arbol(modelo, profundidad=3, extendido=False, podados=None, titulo="Árbol de clasificación", escala_texto=1.0):  # Organiza y dibuja los nodos del árbol.
    """Colorea los nodos por su clase mayoritaria y muestra frecuencias si se solicita."""  # Explica la vista que crea.
    podados = set() if podados is None else set(podados)  # Convierte los nodos a podar en conjunto para buscarlos rápido.
    terminales = modelo.hojas(podados, profundidad)  # Encuentra las hojas visibles teniendo en cuenta poda y profundidad.
    x_hoja = {n: i * 1.7 for i, n in enumerate(terminales)}  # Asigna a cada hoja una posición horizontal separada.
    posiciones = {}  # Guardará las coordenadas de cada nodo del dibujo.
    conexiones = []  # Guardará las ramas que unen cada padre con sus hijos.

    def situar(i):  # Calcula la posición horizontal de un nodo mediante sus descendientes.
        nodo = modelo.nodos[i]  # Recupera el objeto que contiene la regla y los recuentos del nodo.
        if i in x_hoja:  # Comprueba si este nodo debe dibujarse como una hoja visible.
            x = x_hoja[i]  # Usa la posición horizontal ya asignada a esa hoja.
        else:  # Si no es hoja, primero determina dónde van sus descendientes.
            izq, der = situar(nodo.izquierda), situar(nodo.derecha)  # Sitúa recursivamente hijo izquierdo y derecho.
            x = (izq + der) / 2  # Centra el nodo padre entre sus dos hijos.
            conexiones.extend([(i, nodo.izquierda, "≤"), (i, nodo.derecha, ">")])  # Registra ramas y condiciones de cada lado.
        posiciones[i] = (x, -nodo.profundidad * 1.6)  # Guarda x y una coordenada vertical según la profundidad.
        return x  # Devuelve la posición horizontal para que el padre pueda centrarse.

    situar(0)  # Empieza a ubicar el dibujo desde la raíz, que tiene índice 0.
    fig, ax = plt.subplots(figsize=(max(9, len(terminales) * 1.55), max(4, (profundidad + 1) * 1.5)))  # Mantiene el dibujo compacto para que el notebook no lo reduzca demasiado.
    colores = plt.get_cmap("tab10")  # Elige una paleta con colores distintos para las clases.
    for padre, hijo, signo in conexiones:  # Dibuja primero todas las ramas del árbol.
        xp, yp = posiciones[padre]  # Recupera la coordenada del nodo padre.
        xh, yh = posiciones[hijo]  # Recupera la coordenada del nodo hijo.
        ax.plot([xp, xh], [yp - 0.3, yh + 0.3], color="0.6", zorder=0)  # Traza una línea gris entre padre e hijo.
        ax.text((xp + xh) / 2, (yp + yh) / 2, signo, fontsize=10 * escala_texto, ha="center", backgroundcolor="white")  # Aumenta las condiciones de las ramas según la profundidad.
    for i, (x, y) in posiciones.items():  # Recorre los nodos y sus coordenadas para dibujarlos.
        nodo = modelo.nodos[i]  # Recupera datos del nodo actual.
        clase = int(nodo.recuentos.argmax())  # Encuentra el índice de la clase con más observaciones.
        terminal = i in x_hoja  # Comprueba si el nodo es una hoja en esta representación.
        color = colores(clase)  # Asocia cada nodo con el color de su clase mayoritaria.
        caja = FancyBboxPatch(  # Crea la caja visual que rodeará el texto del nodo.
            (x - 0.72, y - 0.36), 1.44, 0.72, boxstyle="round,pad=0.04",  # Define la posición, tamaño y esquinas de la caja.
            facecolor=color, alpha=0.28 if terminal else 0.14, edgecolor="0.35",  # Distingue hojas e internos con la intensidad del color de clase.
        )  # Construye el rectángulo redondeado.
        ax.add_patch(caja)  # Añade la caja al gráfico.
        total = int(nodo.recuentos.sum())  # Cuenta cuántas observaciones llegan al nodo.
        if terminal:  # Las hojas muestran la clase asignada y su proporción.
            texto = f"Nodo {i} · n={total}\n{modelo.clases[clase]} ({nodo.recuentos[clase] / total:.0%})"  # Prepara etiqueta y porcentaje mayoritario.
            ax.text(x, y + 0.1 if extendido else y, texto, ha="center", va="center", fontsize=9 * escala_texto)  # Coloca el texto legible dentro de la hoja.
            if extendido:  # Solo dibuja barras cuando se solicita la vista extendida.
                proporciones = nodo.recuentos / total  # Convierte recuentos por clase en proporciones.
                ancho = 1.18 / len(proporciones)  # Divide el ancho disponible entre las clases.
                ax.bar(  # Dibuja las frecuencias de las clases dentro de una hoja.
                    x - 0.59 + np.arange(len(proporciones)) * ancho,  # Coloca una barra horizontal por clase.
                    proporciones * 0.23,  # Escala la altura para encajarla dentro de la hoja.
                    bottom=y - 0.29,  # Sitúa la base de las barras cerca del borde inferior.
                    width=ancho * 0.9,  # Deja pequeños espacios entre barras.
                    color=[colores(j) for j in range(len(proporciones))],  # Usa un color distinto para cada clase.
                    align="edge",  # Alinea cada barra desde su borde izquierdo.
                )  # Dibuja las proporciones de las clases en la hoja.
        else:  # Los nodos internos muestran la regla que divide los datos.
            texto = f"Nodo {i} · n={total}\n{modelo.variables[nodo.variable]} ≤ {nodo.corte:g}"  # Escribe variable, corte y tamaño.
            if nodo.logp is not None:  # Comprueba si este árbol almacena un p-valor en el nodo.
                texto += f"\nlog₁₀(p)={nodo.logp / np.log(10):.1f}"  # Añade el p-valor en escala logarítmica.
            ax.text(x, y, texto, ha="center", va="center", fontsize=9 * escala_texto)  # Centra el texto legible dentro de la caja.
    ys = [p[1] for p in posiciones.values()]  # Recoge las coordenadas verticales para ajustar los límites.
    ax.set(xlim=(-1, max(x_hoja.values()) + 1), ylim=(min(ys) - 0.65, 0.8), title=titulo)  # Define límites y título.
    ax.axis("off")  # Oculta ejes porque las coordenadas solo organizan el dibujo.
    leyenda = [  # Prepara una marca de color por clase para explicar el relleno de los nodos.
        Patch(facecolor=colores(j), edgecolor="0.35", alpha=0.5, label=str(clase).replace("_", " "))  # Presenta las clases sin guion bajo.
        for j, clase in enumerate(modelo.clases)  # Recorre las etiquetas guardadas en el modelo.
    ]  # Reúne las marcas que formarán la leyenda.
    titulo_leyenda = "Clase mayoritaria del nodo · barras: proporción por clase" if extendido else "Clase mayoritaria del nodo"  # Explica ambos usos del color.
    fig.legend(  # Coloca una leyenda enmarcada y centrada bajo el árbol.
        handles=leyenda, title=titulo_leyenda, loc="lower center", bbox_to_anchor=(0.5, 0.01),  # Separa la leyenda del área del árbol.
        ncol=min(5, len(leyenda)), fontsize=8 * escala_texto, title_fontsize=8 * escala_texto, frameon=True, fancybox=True,  # Distribuye clases en filas compactas y escala las etiquetas.
        framealpha=0.95, edgecolor="#cbd5e1", columnspacing=1.2, handlelength=1.2,  # Mejora contraste y espaciado de las etiquetas.
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))  # Reserva espacio suficiente para la leyenda inferior.
    plt.show()  # Presenta el dibujo del árbol.


def inspeccionar_nodo(modelo, nodo):  # Muestra el camino y la composición de clases de un nodo elegido.
    ruta = []  # Guardará, en orden, las reglas seguidas desde la raíz.

    def buscar_ruta(actual):  # Busca recursivamente el camino desde la raíz hasta el nodo seleccionado.
        if actual == nodo:  # Comprueba si se llegó al destino elegido.
            return True  # Conserva la ruta acumulada hasta este nodo.
        padre = modelo.nodos[actual]  # Recupera las reglas del nodo actual.
        if padre.izquierda is None:  # Si es una hoja, no puede contener el nodo buscado más abajo.
            return False  # Indica que esta rama no conduce al destino.
        variable = modelo.variables[padre.variable]  # Obtiene el nombre del predictor usado para dividir.
        for hijo, operador in ((padre.izquierda, "≤"), (padre.derecha, ">")):  # Recorre las ramas izquierda y derecha.
            condicion = f"{variable} {operador} {padre.corte:g}"  # Construye la condición de la rama actual.
            ruta.append({"Paso": len(ruta) + 1, "Nodo padre": actual,  # Registra el orden y el nodo desde el que se divide.
                        "Regla seguida": condicion, "Nodo alcanzado": hijo})  # Anota la condición y el hijo correspondiente.
            if buscar_ruta(hijo):  # Continúa buscando el nodo seleccionado por esta rama.
                return True  # Deja intacta la secuencia correcta.
            ruta.pop()  # Retira la condición si esta rama no conduce al nodo seleccionado.
        return False  # Indica que el nodo no está en este subárbol.

    if not buscar_ruta(0):  # Comprueba que el número seleccionado corresponde a un nodo del modelo.
        raise ValueError(f"El nodo {nodo} no pertenece al árbol seleccionado.")  # Explica el error de selección.
    if ruta:  # La raíz es el único nodo que no tiene reglas previas.
        display(pd.DataFrame(ruta).set_index("Paso"))  # Muestra todas las condiciones seguidas hasta el nodo.
    else:  # El nodo seleccionado es la propia raíz.
        display(pd.DataFrame({"Recorrido": ["Raíz: no hay reglas previas."]}))  # Explicita que el camino está vacío.

    detalle = modelo.nodos[nodo]  # Recupera del modelo el objeto con la información de ese nodo.
    display(modelo.tabla_nodos().loc[[nodo]])  # Muestra una sola fila de la tabla de nodos.
    frecuencias = detalle.recuentos.astype(int)  # Cuenta cuántos productos de entrenamiento de cada clase llegaron al nodo.
    proporciones = detalle.recuentos / detalle.recuentos.sum()  # Calcula la fracción de cada clase dentro del nodo.
    clases = modelo.clases  # Obtiene las clases en el mismo orden que sus recuentos.
    display(pd.DataFrame({"Frecuencia": frecuencias, "Proporción": proporciones}, index=clases))  # Muestra los recuentos y las proporciones exactas.

    etiquetas = [str(clase).replace("_", " ") for clase in clases]  # Prepara nombres legibles para el eje vertical.
    colores = plt.get_cmap("tab10")(np.arange(len(clases)))  # Mantiene la correspondencia de colores usada en el árbol.
    fig, ax = plt.subplots(figsize=(9, 4.5))  # Da espacio suficiente para las clases y sus etiquetas.
    barras = ax.barh(etiquetas, proporciones, color=colores, edgecolor="white")  # Compara visualmente la proporción de las clases.
    ax.invert_yaxis()  # Coloca la primera clase arriba, en el mismo orden que la tabla.
    ax.set_xlim(0, 1.4)  # Reserva espacio a la derecha para anotar porcentaje y frecuencia.
    ax.set_xlabel("Proporción dentro del nodo")  # Aclara que las barras suman uno dentro del nodo seleccionado.
    ax.set_title(f"Nodo {nodo}: composición de clases (n={int(frecuencias.sum())})")  # Identifica el nodo y el total de observaciones.
    marcas = np.arange(0, 1.01, 0.2)  # Define las marcas del eje en incrementos de veinte puntos porcentuales.
    ax.set_xticks(marcas, labels=[f"{valor:.0%}" for valor in marcas])  # Presenta el eje horizontal en porcentajes.
    ax.grid(axis="x", alpha=0.2)  # Facilita comparar las longitudes de las barras.
    ax.set_axisbelow(True)  # Mantiene la cuadrícula detrás de las barras.
    for barra, proporcion, frecuencia in zip(barras, proporciones, frecuencias):  # Añade el porcentaje y el recuento al final de cada barra.
        ax.text(proporcion + 0.02, barra.get_y() + barra.get_height() / 2,  # Sitúa la etiqueta junto al extremo de la barra.
                f"{proporcion:.0%} · n={frecuencia}", va="center", fontsize=8)  # Muestra proporción y número de productos.
    fig.tight_layout()  # Ajusta márgenes para que nombres y etiquetas queden visibles.
    plt.show()  # Muestra la composición de clases del nodo.


def dibujar_decision_svm(
    # La siguiente línea enumera el modelo, los datos, los predictores, los ejes y opciones del dibujo.
    # Los argumentos de datos y ejes se pasan en líneas separadas para facilitar su lectura.
    modelo, X_train, X_test, y_test, predictores, eje_x, eje_y, seed=12345, etiqueta="SVM"  # Recibe modelo, datos, nombres de ejes y semilla.
):  # Dibuja un corte bidimensional de la frontera del modelo SVM.
    """Representa un corte 2D de la decisión fijando el resto de variables en medianas."""  # Explica la limitación del gráfico.
    limites = []  # Guardará el rango visible de cada característica elegida.
    for variable in [eje_x, eje_y]:  # Calcula un rango para el eje horizontal y otro para el vertical.
        a, b = X_train[variable].quantile([0.01, 0.99])  # Usa percentiles para evitar que valores extremos dominen el dibujo.
        if a == b:  # Comprueba si ambos percentiles coinciden.
            a, b = X_train[variable].min(), X_train[variable].max()  # En ese caso prueba el mínimo y el máximo.
        if a == b:  # Comprueba si la característica es constante incluso en todo entrenamiento.
            a, b = a - 1, b + 1  # Crea un intervalo artificial pequeño para que el eje tenga anchura.
        margen = (b - a) * 0.05  # Calcula un margen del cinco por ciento del rango.
        limites.append((a - margen, b + margen))  # Guarda los límites ampliados para esa característica.
    xx, yy = np.meshgrid(np.linspace(*limites[0], 70), np.linspace(*limites[1], 70))  # Crea puntos regulares en el plano.
    rejilla = pd.DataFrame(np.tile(X_train.median().to_numpy(), (xx.size, 1)), columns=predictores)  # Fija todos los predictores en medianas.
    rejilla[eje_x], rejilla[eje_y] = xx.ravel(), yy.ravel()  # Sustituye solo las dos columnas que aparecen en los ejes.
    superficie = modelo.decision_function(rejilla).reshape(xx.shape)  # Calcula el signo de decisión en cada punto del plano.
    fig, ax = plt.subplots(figsize=(9, 5))  # Crea la figura para dibujar superficie y observaciones.
    ax.contourf(  # Rellena el plano con un color distinto para cada clase predicha.
        xx, yy, (superficie > 0).astype(int), levels=[-0.5, 0.5, 1.5],  # Convierte el signo de decisión en dos regiones de color.
        colors=["#d7e9fb", "#fbdcca"], alpha=0.8,  # Asigna un color a cada clase y una transparencia moderada.
    )  # Dibuja el fondo coloreado según la clase que predice el modelo.
    if superficie.min() < 0 < superficie.max():  # Comprueba que la frontera de cero atraviesa el área visible.
        ax.contour(xx, yy, superficie, levels=[0], colors="black", linewidths=1)  # Traza la frontera de decisión en negro.
    muestra = np.random.default_rng(seed).choice(len(X_test), size=min(600, len(X_test)), replace=False)  # Elige hasta 600 filas reproducibles.
    for clase, color in [(False, "steelblue"), (True, "darkorange")]:  # Asigna un color a cada clase real binaria.
        seleccion = muestra[y_test.iloc[muestra].to_numpy() == clase]  # Conserva puntos elegidos que pertenecen a la clase actual.
        ax.scatter(  # Añade al gráfico los productos de test de la clase actual.
            X_test.iloc[seleccion][eje_x], X_test.iloc[seleccion][eje_y],  # Usa sus valores en los dos ejes elegidos.
            s=16, alpha=0.5, c=color, label=f"Clase real: {clase}",  # Ajusta tamaño, transparencia, color y leyenda.
        )  # Añade los productos de test como puntos sobre el fondo.
    ax.set(  # Define títulos y límites del gráfico de decisión.
        xlim=limites[0], ylim=limites[1], xlabel=eje_x, ylabel=eje_y,  # Fija límites y nombres de los ejes.
        title=f"{etiqueta} · corte con las otras variables en su mediana",  # Advierte que es un corte del espacio de predictores.
    )  # Configura el aspecto informativo del gráfico.
    ax.legend()  # Distingue los puntos según su clase real.
    fig.tight_layout()  # Ajusta los márgenes alrededor del gráfico.
    plt.show()  # Muestra el corte bidimensional de la decisión.
