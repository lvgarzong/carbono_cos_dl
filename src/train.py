"""
train.py
========
Entrenamiento de UN experimento (modelo x rango x preprocesado x semilla).

Devuelve un dict con metricas de test (en % de COS reales), historia de
entrenamiento y predicciones, y guarda artefactos en results/<exp>/:
  - model.keras        (modelo entrenado)
  - history.csv        (curvas de entrenamiento)
  - predictions.csv    (y real vs y predicho en test)
  - metrics.json       (metricas finales)
  - log.txt            (log del experimento)
"""
from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
from src import models as M
from src.metrics import regression_metrics, RSquare
from src.preprocessing import apply_preprocessing, Scaler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _stratified_split(X, y, ids, seed):
    """Split train/val/test estratificado por quantiles de COS y POR MUESTRA.

    El split se hace sobre las muestras unicas (ids) y luego se expande a todas
    las filas (escaneos) de cada muestra. Asi, en modo 'replicas' ningun escaneo
    de una misma muestra cae en dos particiones distintas (evita fuga de datos).
    Cuando hay una sola fila por muestra (median/mean) se comporta como antes.
    """
    ids = np.asarray(ids)
    uniq = np.unique(ids)
    # y representativo por muestra (media de sus escaneos).
    y_by_sample = np.array([y[ids == s].mean() for s in uniq])
    bins = np.quantile(y_by_sample, np.linspace(0, 1, config.STRATIFY_BINS + 1))
    strata = np.clip(np.digitize(y_by_sample, bins[1:-1]), 0, config.STRATIFY_BINS - 1)

    s_tr, s_tmp, st_tr, st_tmp = train_test_split(
        uniq, strata, test_size=config.TEST_RATIO + config.VAL_RATIO,
        random_state=seed, stratify=strata)
    rel = config.TEST_RATIO / (config.TEST_RATIO + config.VAL_RATIO)
    s_va, s_te = train_test_split(
        s_tmp, test_size=rel, random_state=seed, stratify=st_tmp)

    def rows(sample_ids):
        mask = np.isin(ids, sample_ids)
        return np.where(mask)[0]

    return rows(s_tr), rows(s_va), rows(s_te)


def _make_optimizer():
    if config.OPTIMIZER == "adam":
        return tf.keras.optimizers.Adam(config.LEARNING_RATE)
    return tf.keras.optimizers.Nadam(config.LEARNING_RATE, epsilon=1e-7)


def train_one(dataset, model_name: str, preprocess: str, seed: int,
              exp_dir: Path, logger, epochs: int | None = None) -> dict:
    """Entrena un modelo y devuelve metricas + artefactos."""
    t0 = time.time()
    epochs = epochs or config.EPOCHS
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    X, y, ids = dataset.X, dataset.y, dataset.ids
    tr, va, te = _stratified_split(X, y, ids, seed)

    # Preprocesado espectral (ajustado conceptualmente en train).
    Xtr, Xva, Xte = apply_preprocessing(X[tr], X[va], X[te], preprocess)

    # Escalado de features por banda (fit en train).
    fscaler = Scaler(config.FEATURE_SCALING, axis=0).fit(Xtr)
    Xtr, Xva, Xte = fscaler.transform(Xtr), fscaler.transform(Xva), fscaler.transform(Xte)

    # Escalado del objetivo (fit en train) -> se invierte para reportar en %.
    tscaler = Scaler(config.TARGET_SCALING, axis=0).fit(y[tr].reshape(-1, 1))
    ytr = tscaler.transform(y[tr].reshape(-1, 1)).ravel()
    yva = tscaler.transform(y[va].reshape(-1, 1)).ravel()

    Xtr = Xtr[..., None].astype(np.float32)
    Xva = Xva[..., None].astype(np.float32)
    Xte = Xte[..., None].astype(np.float32)

    model = M.get_model(model_name, input_len=X.shape[1])
    model.compile(optimizer=_make_optimizer(), loss="mse",
                  metrics=[RSquare(), "mae"])
    n_params = model.count_params()
    logger.info("[%s|%s|seed=%d] params=%d, train=%d val=%d test=%d, bandas=%d",
                model_name, preprocess, seed, n_params, len(tr), len(va), len(te), X.shape[1])

    ckpt = exp_dir / "model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(exp_dir / "history.csv")),
    ]

    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva),
        epochs=epochs, batch_size=config.BATCH_SIZE,
        verbose=0, shuffle=True, callbacks=callbacks)

    # Prediccion en test e inversion a unidades reales (%).
    pred_scan = tscaler.inverse_transform(model.predict(Xte, verbose=0)).ravel()
    # Agrega por muestra (promedio de escaneos) -> una prediccion por muestra.
    te_ids = ids[te]
    uniq = np.unique(te_ids)
    yte_pred = np.array([pred_scan[te_ids == s].mean() for s in uniq])
    yte_true = np.array([y[te][te_ids == s].mean() for s in uniq])
    met = regression_metrics(yte_true, yte_pred)

    # Guardado de artefactos (por muestra).
    pd.DataFrame({"id": uniq, "y_true": yte_true, "y_pred": yte_pred}
                 ).to_csv(exp_dir / "predictions.csv", index=False)
    record = {
        "model": model_name, "range": dataset.range_name,
        "preprocess": preprocess, "seed": seed,
        "n_params": int(n_params), "n_bands": int(X.shape[1]),
        "n_train": int(len(tr)), "n_test": int(len(uniq)),
        "epochs_run": int(len(hist.history["loss"])),
        "time_s": round(time.time() - t0, 1),
        **met,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(record, f, indent=2)

    logger.info("[%s|%s|seed=%d] R2=%.3f RMSE=%.3f RPD=%.2f (%.1fs)",
                model_name, preprocess, seed, met["r2"], met["rmse"],
                met["rpd"], record["time_s"])
    return record
