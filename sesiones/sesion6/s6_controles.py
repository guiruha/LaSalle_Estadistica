"""Factorías de widgets de S6; importar no crea controles ni ejecuta callbacks."""

import ipywidgets as widgets


def crear_control_knn(explorar, n_clientes):
    """El wrapper expone solo los argumentos interactivos de la llamada parcial."""
    def actualizar(k, cliente):
        return explorar(k=k, cliente=cliente)

    return widgets.interactive(actualizar,
        k=widgets.IntSlider(value=1, min=1, max=21, step=2, description="Vecinos", continuous_update=False),
        cliente=widgets.IntSlider(value=0, min=0, max=n_clientes-1, description="Cliente", continuous_update=False))


def crear_control_kernel_knn(explorar):
    def actualizar(datos, kernel, k):
        return explorar(datos=datos, kernel=kernel, k=k)

    return widgets.interactive(actualizar,
        datos=widgets.Dropdown(options=["Originales", "CP1"], description="Datos"),
        kernel=widgets.Dropdown(options=["óptimo", "triangular", "epanechnikov"], description="Kernel"),
        k=widgets.IntSlider(value=1, min=1, max=21, description="Vecinos", continuous_update=False))


def crear_control_tabla_nb(explorar, variables):
    def actualizar(variable):
        return explorar(variable=variable)

    return widgets.interactive(actualizar,
        variable=widgets.Dropdown(options=list(variables), description="Variable"))


def crear_control_probabilidades(explorar, n_productos):
    def actualizar(inicio, cantidad):
        return explorar(inicio=inicio, cantidad=cantidad)

    return widgets.interactive(actualizar,
        inicio=widgets.IntSlider(value=0, min=0, max=n_productos-1, description="Inicio", continuous_update=False),
        cantidad=widgets.IntSlider(value=50, min=10, max=100, step=10, description="Filas", continuous_update=False))


def crear_control_densidad(explorar, variables, clases):
    def actualizar(variable, cerca_de_cero, clase):
        return explorar(variable=variable, cerca_de_cero=cerca_de_cero, clase=clase)

    return widgets.interactive(actualizar,
        variable=widgets.Dropdown(options=list(variables), value="feat_45", description="Variable"),
        cerca_de_cero=widgets.Checkbox(value=True, description="Detalle alrededor de 0"),
        clase=widgets.Dropdown(options=["Todas"] + list(clases), description="Clase"))