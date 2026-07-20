"""
preprocessing.py
================
Transformaciones espectrales y escalado.

Soporta la contribucion (i): evaluar el efecto del preprocesado y la fusion.
Metodos: raw, snv, sg0, sg1, sg2, sg1_snv, msc.

Escaladores ajustados SOLO en train (evita fuga de informacion) y reutilizables
en val/test e inferencia.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

import config


# --------------------------------------------------------------------------
# Transformaciones espectrales (fila a fila = muestra a muestra)
# --------------------------------------------------------------------------
def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: centra y escala cada espectro por su media/desv."""
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-8
    return (X - m) / s


def msc(X: np.ndarray, ref: np.ndarray | None = None):
    """Multiplicative Scatter Correction. Devuelve (X_corr, ref) para reusar ref."""
    if ref is None:
        ref = X.mean(axis=0)
    Xc = np.empty_like(X)
    for i in range(X.shape[0]):
        a, b = np.polyfit(ref, X[i], 1)
        Xc[i] = (X[i] - b) / (a + 1e-8)
    return Xc, ref


def savgol(X: np.ndarray, deriv: int = 0) -> np.ndarray:
    w = min(config.SG_WINDOW, X.shape[1] - (1 - X.shape[1] % 2))
    if w % 2 == 0:
        w += 1
    w = max(w, config.SG_POLYORDER + 2)
    return savgol_filter(X, window_length=w, polyorder=config.SG_POLYORDER,
                         deriv=deriv, axis=1)


def apply_preprocessing(X_train, X_val, X_test, method: str):
    """Aplica una transformacion espectral. MSC usa la referencia de train."""
    if method == "raw":
        return X_train, X_val, X_test
    if method == "snv":
        return snv(X_train), snv(X_val), snv(X_test)
    if method == "sg0":
        return savgol(X_train, 0), savgol(X_val, 0), savgol(X_test, 0)
    if method == "sg1":
        return savgol(X_train, 1), savgol(X_val, 1), savgol(X_test, 1)
    if method == "sg2":
        return savgol(X_train, 2), savgol(X_val, 2), savgol(X_test, 2)
    if method == "sg1_snv":
        return snv(savgol(X_train, 1)), snv(savgol(X_val, 1)), snv(savgol(X_test, 1))
    if method == "msc":
        Xtr, ref = msc(X_train)
        Xv, _ = msc(X_val, ref)
        Xte, _ = msc(X_test, ref)
        return Xtr, Xv, Xte
    raise ValueError(f"Metodo de preprocesado desconocido: {method}")


# --------------------------------------------------------------------------
# Escaladores (ajustados en train)
# --------------------------------------------------------------------------
class Scaler:
    """Escalador simple zscore / minmax / none, invertible (para el objetivo)."""

    def __init__(self, kind: str = "zscore", axis: int = 0):
        self.kind = kind
        self.axis = axis
        self.a = None  # media o min
        self.b = None  # desv o (max-min)

    def fit(self, X):
        if self.kind == "zscore":
            self.a = X.mean(axis=self.axis, keepdims=True)
            self.b = X.std(axis=self.axis, keepdims=True) + 1e-8
        elif self.kind == "minmax":
            self.a = X.min(axis=self.axis, keepdims=True)
            self.b = (X.max(axis=self.axis, keepdims=True) - self.a) + 1e-8
        else:  # none
            self.a = 0.0
            self.b = 1.0
        return self

    def transform(self, X):
        return (X - self.a) / self.b

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.b + self.a
