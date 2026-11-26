"""Construcción de controles opcionales para la sesión 4.

Cada panel conserva sus controles y callbacks en un ámbito local. Importar el
módulo no crea widgets ni consulta datos de entrenamiento o test.
"""

from ipywidgets import Dropdown, FloatSlider, HTML, HBox, Layout, VBox

from s4_graficos import comparar_odds, curva_logistica


def _crear_panel(titulo, controles, contenido):
    """Conecta valores y vista HTML sin depender de selectores globales."""
    visor = HTML(layout=Layout(width="100%"))

    def actualizar(cambio=None):
        if cambio is not None and cambio["name"] != "value":
            return
        visor.value = contenido(*(control.value for control in controles))

    for control in controles:
        control.observe(actualizar, names="value")
    actualizar()
    cabecera = HTML(
        "<div style='border-left: 4px solid #800000; padding: 8px 12px; "
        f"font-size: 18px; font-weight: 600;'>{titulo}</div>"
    )
    fila = HBox(controles, layout=Layout(align_items="center", flex_flow="row wrap"))
    return VBox([cabecera, fila, visor], layout=Layout(width="100%", grid_gap="12px"))


def crear_control_densidad(dibujar_densidad, variables):
    """Explora la misma tabla asociada a dibujar_densidad, sin modificarla."""
    variables = list(variables)
    if not variables:
        raise ValueError("Se necesita al menos una variable numérica.")
    selector = Dropdown(
        options=variables, value="age" if "age" in variables else variables[0],
        description="Variable", style={"description_width": "95px"}, layout=Layout(width="300px"),
    )
    suavizado = FloatSlider(
        value=1.0, min=0.3, max=3.0, step=0.1, description="Suavizado", continuous_update=False,
        style={"description_width": "95px"}, layout=Layout(width="350px"),
    )

    def contenido(variable, factor):
        return dibujar_densidad(variable, factor, vista_widget=True)

    return _crear_panel("Explora la densidad condicional", [selector, suavizado], contenido)


def crear_control_odds():
    """Probabilidad inicial y OR del ejemplo ficticio; no ajusta un modelo."""
    probabilidad = FloatSlider(
        value=0.2, min=0.05, max=0.95, step=0.05, description="P inicial", continuous_update=False,
        style={"description_width": "85px"}, layout=Layout(width="350px"),
    )
    ratio = FloatSlider(
        value=2.0, min=0.25, max=5.0, step=0.25, description="OR", continuous_update=False,
        style={"description_width": "45px"}, layout=Layout(width="350px"),
    )
    return _crear_panel("Probabilidades y odds: ejemplo ficticio", [probabilidad, ratio], comparar_odds)


def crear_control_logistica():
    """Intercepto y pendiente de una curva ficticia, sin datos de crédito."""
    intercepto = FloatSlider(
        value=0, min=-3, max=3, step=0.25, description="Intercepto", continuous_update=False,
        style={"description_width": "90px"}, layout=Layout(width="350px"),
    )
    pendiente = FloatSlider(
        value=1, min=-2, max=2, step=0.25, description="Pendiente", continuous_update=False,
        style={"description_width": "90px"}, layout=Layout(width="350px"),
    )
    return _crear_panel("Explora la función logística", [intercepto, pendiente], curva_logistica)


def crear_control_corte(explorar_corte, umbral=0.5):
    """Mueve un corte de exploración; no escribe la regla de evaluación."""
    selector = FloatSlider(
        value=umbral, min=0.05, max=0.95, step=0.05, description="Corte", continuous_update=False,
        style={"description_width": "85px"}, layout=Layout(width="450px"),
    )
    return _crear_panel("Explora los cortes SOLO en entrenamiento", [selector], explorar_corte)