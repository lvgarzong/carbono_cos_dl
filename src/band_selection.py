"""
band_selection.py
=================
Seleccion de longitudes de onda mas relevantes (contribucion ii) mediante
VIP (Variable Importance in Projection) de un modelo PLS.

VIP > 1  -> banda relevante (criterio estandar en quimiometria).
Permite:
  - reducir ruido y sobreajuste entrenando solo con las bandas informativas,
  - identificar regiones espectrales clave para el COS en citricos.

Se ajusta SOLO con datos de entrenamiento (sin fuga).
"""
from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression


def pls_vip(X: np.ndarray, y: np.ndarray, n_components: int = 15) -> np.ndarray:
    """Calcula el score VIP por banda a partir de un PLS ajustado en (X, y)."""
    n_components = min(n_components, X.shape[1], X.shape[0] - 1)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    t = pls.x_scores_           # (n, A)
    w = pls.x_weights_          # (p, A)
    q = pls.y_loadings_         # (1, A)
    p_, h = w.shape
    ss = np.sum((t ** 2) * (q[0] ** 2), axis=0)   # varianza explicada por comp.
    total = np.sum(ss) + 1e-12
    vip = np.sqrt(p_ * np.sum((w ** 2) * ss, axis=1) / total)
    return vip


def select_bands(X_train, y_train, n_components: int = 15,
                 threshold: float = 1.0, top_k: int | None = None):
    """Devuelve los indices de bandas seleccionadas.

    Si top_k se especifica, toma las top_k por VIP; si no, usa VIP > threshold.
    """
    vip = pls_vip(X_train, y_train, n_components)
    if top_k is not None:
        idx = np.argsort(vip)[::-1][:top_k]
        idx = np.sort(idx)
    else:
        idx = np.where(vip >= threshold)[0]
        if idx.size == 0:                      # fallback: top 25%
            idx = np.argsort(vip)[::-1][: max(10, X_train.shape[1] // 4)]
            idx = np.sort(idx)
    return idx, vip
