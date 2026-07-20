"""
feature_importance.py
=====================
Importancia de longitudes de onda (contribucion ii).

Dos metodos complementarios:
  1. Saliencia por gradiente: |d(prediccion)/d(banda)| promediada sobre test.
     Rapido, indica sensibilidad local del modelo a cada banda.
  2. Importancia por permutacion: cuanto sube el RMSE al permutar cada banda
     (o bloque de bandas). Independiente del modelo, mas robusto.

Permite extraer las longitudes de onda mas relevantes para estimar COS en
cultivos de citricos en Colombia y compararlas con las del paper (p.ej. picos
de OC alrededor de 1340-1380 nm y 1860-1900 nm).
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf


def gradient_saliency(model, X):
    """|d y_pred / d x| promediada sobre las muestras. X: (n, L, 1)."""
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X)
        pred = model(X, training=False)
    grads = tape.gradient(pred, X)
    sal = tf.reduce_mean(tf.abs(grads), axis=0).numpy().ravel()
    return sal / (sal.max() + 1e-12)


def permutation_importance(model, X, y_true, tscaler, block: int = 10, n_repeats: int = 3):
    """Aumento del RMSE al permutar bloques de 'block' bandas. X: (n, L, 1)."""
    rng = np.random.default_rng(0)

    def rmse(Xin):
        p = tscaler.inverse_transform(model.predict(Xin, verbose=0)).ravel()
        return float(np.sqrt(np.mean((y_true - p) ** 2)))

    base = rmse(X)
    L = X.shape[1]
    imp = np.zeros(L)
    for start in range(0, L, block):
        end = min(start + block, L)
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, start:end, :] = Xp[perm][:, start:end, :]
            scores.append(rmse(Xp) - base)
        imp[start:end] = np.mean(scores)
    imp = np.clip(imp, 0, None)
    return imp / (imp.max() + 1e-12)


def top_wavelengths(importance, wavelengths, top_n: int = 10, min_dist_nm: float = 20):
    """Devuelve las top_n longitudes de onda mas importantes, separadas entre si."""
    order = np.argsort(importance)[::-1]
    chosen = []
    for idx in order:
        wl = float(wavelengths[idx])
        if all(abs(wl - c[0]) >= min_dist_nm for c in chosen):
            chosen.append((round(wl, 1), float(importance[idx])))
        if len(chosen) >= top_n:
            break
    return chosen
