"""Controles de S5: funciones y datos fijados explícitamente por el cuaderno."""

import ipywidgets as widgets


def cabecera_widget(titulo):
    return widgets.HTML(
        f"<div style='border-left:4px solid #800000; padding:8px 12px; "
        f"font-size:18px; font-weight:600;'>{titulo}</div>")


def fila_controles(controles):
    for control in controles:
        if isinstance(control, widgets.Checkbox):
            control.layout.width = "auto"
        else:
            control.style.description_width = "75px"
            control.layout.width = "300px"
    return widgets.HBox(controles, layout=widgets.Layout(
        align_items="center", flex_flow="row wrap", grid_gap="12px"))


def panel_interactivo(titulo, control):
    """Presenta controles y salida manteniendo sus observadores y resultado."""
    *entradas, salida = control.children
    salida.layout.width = "100%"
    return widgets.VBox([cabecera_widget(titulo), fila_controles(entradas), salida],
                        layout=widgets.Layout(width="100%", grid_gap="12px"))


def _control_interactivo(funcion, **controles):
    """Evita que ipywidgets inspeccione directamente objetos functools.partial."""
    def actualizar(**valores):
        return funcion(**valores)

    return widgets.interactive(actualizar, **controles)


def crear_control_mapa(dibujar):
    return _control_interactivo(dibujar,
        estandarizado=widgets.Checkbox(value=False, description="Estandarizar columnas"))


def crear_control_arbol(explorar):
    """Impide Ward no euclídeo; cada panel conserva sus propios controles."""
    distancia = widgets.Dropdown(options=[("Euclídea", "euclidean"), ("Manhattan", "cityblock")],
                                 value="euclidean", description="Distancia")
    enlace = widgets.Dropdown(options=[("Completo", "complete"), ("Promedio", "average"),
                                      ("Simple", "single"), ("Ward", "ward")],
                              value="complete", description="Enlace")
    escala = widgets.Checkbox(value=False, description="Estandarizar")
    visor = widgets.HTML(layout=widgets.Layout(width="100%"))

    def actualizar(cambio=None):
        if cambio is not None and cambio["name"] != "value":
            return
        if distancia.value != "euclidean" and enlace.value == "ward":
            enlace.value = "complete"
            return
        visor.value = explorar(distancia.value, enlace.value, escala.value)

    for control in (distancia, enlace, escala):
        control.observe(actualizar, names="value")
    actualizar()
    return widgets.VBox([cabecera_widget("Explora distancias y enlaces"),
                         fila_controles([distancia, enlace, escala]), visor],
                        layout=widgets.Layout(width="100%", grid_gap="12px"))


def crear_control_jerarquia(particion):
    return _control_interactivo(particion,
        k=widgets.IntSlider(value=3, min=2, max=8, description="Grupos", continuous_update=False))


def crear_control_hcpc(ajustar):
    """El resultado del control se consulta en .result, no en una variable global."""
    return _control_interactivo(ajustar,
        k=widgets.IntSlider(value=3, min=2, max=8, description="Grupos", continuous_update=False),
        consolidar=widgets.Checkbox(value=True, description="Consolidar con k-means"))


def crear_control_3d(dibujar):
    return _control_interactivo(dibujar,
        angulo=widgets.IntSlider(value=60, min=0, max=180, step=5,
                                 description="Ángulo", continuous_update=False))


def crear_control_kmeans(explorar):
    return _control_interactivo(explorar,
        k=widgets.IntSlider(value=3, min=2, max=8, description="Grupos", continuous_update=False),
        inicios=widgets.SelectionSlider(options=[1, 5, 10], value=10,
                                         description="Inicios", continuous_update=False),
        algoritmo=widgets.Dropdown(options=[("Hartigan–Wong", "hartigan-wong"), ("Lloyd", "lloyd")],
                                    description="Algoritmo"))


def crear_control_kernel(explorar):
    return _control_interactivo(explorar,
        gamma=widgets.FloatSlider(value=2.0, min=0.5, max=4, step=0.25,
                                   description="Gamma", continuous_update=False))


def crear_control_atipicos(explorar):
    return _control_interactivo(explorar,
        numero=widgets.IntSlider(value=3, min=0, max=8, description="Atípicos", continuous_update=False))