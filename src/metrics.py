"""
metrics.py
==========
Metricas de regresion para COS, en unidades reales (% de carbono).

Se reemplaza tensorflow_addons (descontinuado) por implementaciones puras:
  R2, RMSE, MAE, BIAS, RPD, RPIQ.

RPD y RPIQ son estandar en espectroscopia de suelos:
  RPD  = std(observado) / RMSE       (>2.0 bueno, >2.5 excelente)
  RPIQ = IQR(observado) / RMSE       (robusto a distribuciones no normales)
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula todas las metricas en unidades reales. Entrada en % de COS."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    rmse = float(np.sqrt(np.mean(err ** 2)))
    q75, q25 = np.percentile(y_true, [75, 25])
    iqr = q75 - q25
    return {
        "r2": float(1 - ss_res / ss_tot),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "rpd": float(y_true.std() / rmse) if rmse > 0 else np.inf,
        "rpiq": float(iqr / rmse) if rmse > 0 else np.inf,
    }


class RSquare(tf.keras.metrics.Metric):
    """Metrica R2 para monitorear durante el entrenamiento (en espacio escalado).

    Reemplazo directo de tfa.metrics.RSquare, compatible con Keras 3.
    """

    def __init__(self, name="r_square", **kwargs):
        super().__init__(name=name, **kwargs)
        self.res = self.add_weight(name="res", initializer="zeros")
        self.tot = self.add_weight(name="tot", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_y2 = self.add_weight(name="sum_y2", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        self.res.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.sum_y.assign_add(tf.reduce_sum(y_true))
        self.sum_y2.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        mean_y = self.sum_y / (self.count + 1e-8)
        tot = self.sum_y2 - self.count * tf.square(mean_y)
        return 1.0 - self.res / (tot + 1e-8)

    def reset_state(self):
        for v in self.variables:
            v.assign(0.0)
