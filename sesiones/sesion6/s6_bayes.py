"""Naive Bayes y densidades univariantes de S6; importar no ajusta modelos."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import fftconvolve
from scipy.special import logsumexp


class BayesGaussiano:
    """Naive Bayes numérico con medias y desviaciones muestrales por clase."""

    def __init__(self, laplace=0, umbral=0.001):
        # Laplace permanece inerte para las densidades numéricas de este recorrido.
        self.laplace, self.umbral = laplace, umbral

    def fit(self, X, y):
        self.variables = list(X.columns)
        self.clases = np.unique(y)
        self.recuentos = pd.Series(y).value_counts().reindex(self.clases)
        self.prior = self.recuentos / self.recuentos.sum()
        grupos = [np.asarray(X, dtype=float)[np.asarray(y) == clase] for clase in self.clases]
        self.medias = pd.DataFrame([grupo.mean(axis=0) for grupo in grupos],
                                  index=self.clases, columns=self.variables)
        self.desviaciones = pd.DataFrame([grupo.std(axis=0, ddof=1) for grupo in grupos],
                                        index=self.clases, columns=self.variables)
        self.tablas = {v: pd.DataFrame({"Media": self.medias[v], "Desviación": self.desviaciones[v]})
                       for v in self.variables}
        return self

    def predict_proba(self, X):
        valores = X.loc[:, self.variables].to_numpy(dtype=float)
        logp = np.tile(np.log(self.prior.to_numpy()), (len(X), 1))
        for j, clase in enumerate(self.clases):
            sd = self.desviaciones.loc[clase].to_numpy().copy()
            sd[sd <= 0] = self.umbral
            densidad = stats.norm.pdf(valores, loc=self.medias.loc[clase].to_numpy(), scale=sd)
            # No se recortan las densidades positivas, por pequeñas que sean.
            densidad[densidad <= 0] = self.umbral
            densidad[np.isnan(valores)] = 1
            logp[:, j] += np.log(densidad).sum(axis=1)
        probabilidades = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
        return pd.DataFrame(probabilidades, index=X.index, columns=self.clases)

    def predict(self, X):
        return self.predict_proba(X).idxmax(axis=1).rename("Predicción")


def densidad_kernel(x, puntos=512):
    """KDE gaussiana: ancho original, binning lineal, FFT e interpolación."""
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    escala = min(sd, iqr)
    if escala <= 0:
        escala = sd if sd > 0 else (abs(x[0]) if x[0] != 0 else 1)
    ancho = 0.9 * escala * len(x)**(-0.2)
    desde, hasta = x.min()-3*ancho, x.max()+3*ancho
    grilla = np.linspace(desde-4*ancho, hasta+4*ancho, puntos)
    paso = grilla[1]-grilla[0]
    coordenadas = (x-grilla[0]) / paso
    inferiores = np.floor(coordenadas).astype(int)
    fraccion = coordenadas-inferiores
    masas = np.bincount(inferiores, weights=1-fraccion, minlength=puntos)
    masas += np.bincount(inferiores+1, weights=fraccion, minlength=puntos)
    desplazamientos = np.arange(-(puntos-1), puntos)*paso
    nucleo = stats.norm.pdf(desplazamientos, scale=ancho)
    densidad = np.maximum(fftconvolve(masas/len(x), nucleo, mode="same"), 0)
    salida = np.linspace(desde, hasta, puntos)
    return SimpleNamespace(x=salida, y=np.interp(salida, grilla, densidad), ancho=ancho)


class BayesKernel:
    """Una densidad univariante por predictor y clase, con priors de entrenamiento."""

    def __init__(self, puntos=512, umbral=0.001):
        self.puntos, self.umbral = puntos, umbral

    def fit(self, X, y):
        self.variables = list(X.columns)
        self.clases = np.unique(y)
        recuentos = pd.Series(y).value_counts().reindex(self.clases)
        self.prior = recuentos / recuentos.sum()
        self.densidades = {v: {} for v in self.variables}
        for clase in self.clases:
            grupo = X.loc[np.asarray(y) == clase]
            for v in self.variables:
                self.densidades[v][clase] = densidad_kernel(grupo[v], self.puntos)
        return self

    def predict_proba(self, X):
        logp = np.tile(np.log(self.prior.to_numpy()), (len(X), 1))
        for variable in self.variables:
            valores = X[variable].to_numpy(dtype=float)
            for j, clase in enumerate(self.clases):
                d = self.densidades[variable][clase]
                # np.interp mantiene los valores extremos fuera de la rejilla.
                densidad = np.interp(valores, d.x, d.y)
                densidad[densidad <= 0] = self.umbral
                densidad[np.isnan(valores)] = 1
                logp[:, j] += np.log(densidad)
        return pd.DataFrame(np.exp(logp-logsumexp(logp, axis=1, keepdims=True)),
                            index=X.index, columns=self.clases)

    def predict(self, X):
        return self.predict_proba(X).idxmax(axis=1).rename("Predicción")