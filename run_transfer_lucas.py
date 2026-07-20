"""
run_transfer_lucas.py
=====================
TRANSFER LEARNING desde la libreria espectral LUCAS hacia tus 820 muestras de
citricos. Es la via mas prometedora hacia R2 >= 0.80: se preentrena un modelo
con las ~14.000 muestras de LUCAS (mucha informacion) y luego se afina con tus
datos locales.

------------------------------------------------------------------------------
 IMPORTANTE: este script NO incluye los datos de LUCAS (son varios GB).
 Debes descargarlos primero:
   1. LUCAS 2009/2015 TOPSOIL - propiedades + espectros Vis-NIR:
      https://esdac.jrc.ec.europa.eu/projects/lucas  (registro gratuito)
   2. Necesitas un archivo con:
        - espectros de absorbancia (400-2500 nm) -> N filas x ~4200 columnas
        - la propiedad OC (organic carbon, g/kg)
   3. Guardalo como CSV/parquet y apunta --lucas-spectra y --lucas-oc abajo,
      o adapta load_lucas() a tu formato.
------------------------------------------------------------------------------

Estrategia tecnica (clave para que el transfer funcione):
  * El modelo espera la MISMA dimension de entrada en ambas fases. Por eso se
    re-muestrean (interpolan) los espectros de LUCAS sobre la rejilla de
    longitudes de onda de TUS datos (p.ej. NIR 917-2449 nm, 486 bandas).
  * Se trabaja en ABSORBANCia en ambos dominios: A = log10(1/Reflectancia).
    LUCAS ya es absorbancia; tu reflectancia se convierte a absorbancia.
  * Preentreno: modelo completo sobre LUCAS-OC.
  * Afinamiento: se congela el cuerpo convolucional, se reinicia la cabeza
    densa y se entrena con tus datos; luego (opcional) se descongela todo con
    learning rate bajo.

Uso (una vez tengas LUCAS):
    # 1) formatea LUCAS (auto-detecta columnas, convierte a absorbancia):
    python prepare_lucas.py --spectra LUCAS_Spectra.csv --properties LUCAS_props.csv
    # 2) preentrena en LUCAS y afina en tus citricos:
    python run_transfer_lucas.py --lucas lucas_prepared.npz --model LucasResNet16 --range NIR
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from src.logging_utils import get_logger
from src.data_loader import load_dataset
from src.preprocessing import Scaler
from src.train import _stratified_split, set_seed, _make_optimizer
from src.metrics import regression_metrics, RSquare
from src import models as M


def refl_to_absorbance(R: np.ndarray) -> np.ndarray:
    R = np.clip(R, 1e-4, None)
    return np.log10(1.0 / R)


def load_lucas(npz_path, target_wavelengths, logger=None):
    """Carga el .npz generado por prepare_lucas.py y lo re-muestrea a la rejilla
    'target_wavelengths' (las longitudes de onda de TUS datos)."""
    d = np.load(npz_path)
    A, src_wl, y = d["A"], d["wl"], d["oc"]
    # Re-muestreo (interpolacion) a la rejilla destino.
    A_rs = np.vstack([np.interp(target_wavelengths, src_wl, row) for row in A])
    mask = np.isfinite(y)
    if logger:
        logger.info("LUCAS: %d espectros (%.0f-%.0f nm) re-muestreados a %d bandas; OC validos=%d",
                    A_rs.shape[0], src_wl.min(), src_wl.max(),
                    len(target_wavelengths), int(mask.sum()))
    return A_rs[mask].astype(np.float32), y[mask].astype(np.float32)


def _freeze_body(model, freeze=True):
    """Congela todo menos las ultimas capas densas (cabeza)."""
    dense_seen = 0
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            dense_seen += 1
            layer.trainable = True
        else:
            layer.trainable = not freeze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lucas", required=True,
                    help="ruta a lucas_prepared.npz (generado por prepare_lucas.py)")
    ap.add_argument("--model", default="LucasResNet16")
    ap.add_argument("--range", default="NIR")
    ap.add_argument("--pretrain-epochs", type=int, default=100)
    ap.add_argument("--finetune-epochs", type=int, default=150)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    logger = get_logger("transfer", log_dir=config.LOGS_DIR)
    set_seed(args.split_seed)

    # ---- Datos locales (citricos) en absorbancia ----
    ds = load_dataset(args.range, agg="median", logger=logger)
    tr, va, te = _stratified_split(ds.X, ds.y, ds.ids, args.split_seed)
    Xloc = refl_to_absorbance(ds.X)

    # ---- LUCAS re-muestreado a la rejilla local ----
    Xl, yl = load_lucas(args.lucas, ds.wavelengths, logger)

    # Escaladores (ajustados en LUCAS para el preentreno).
    fsc = Scaler(config.FEATURE_SCALING, axis=0).fit(Xl)
    tsc = Scaler(config.TARGET_SCALING, axis=0).fit(yl.reshape(-1, 1))
    Xl_s = fsc.transform(Xl)[..., None].astype(np.float32)
    yl_s = tsc.transform(yl.reshape(-1, 1)).ravel()

    # ===== Fase 1: preentreno en LUCAS =====
    model = M.get_model(args.model, input_len=ds.X.shape[1])
    model.compile(optimizer=_make_optimizer(), loss="mse", metrics=[RSquare()])
    logger.info("Preentrenando %s en LUCAS (%d muestras)...", args.model, len(yl))
    model.fit(Xl_s, yl_s, validation_split=0.1, epochs=args.pretrain_epochs,
              batch_size=64, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=20, restore_best_weights=True)])

    # ===== Fase 2: afinamiento en citricos =====
    # Reescala con estadisticas locales de train.
    fsc_l = Scaler(config.FEATURE_SCALING, axis=0).fit(Xloc[tr])
    tsc_l = Scaler(config.TARGET_SCALING, axis=0).fit(ds.y[tr].reshape(-1, 1))
    Xtr = fsc_l.transform(Xloc[tr])[..., None].astype(np.float32)
    Xva = fsc_l.transform(Xloc[va])[..., None].astype(np.float32)
    Xte = fsc_l.transform(Xloc[te])[..., None].astype(np.float32)
    ytr = tsc_l.transform(ds.y[tr].reshape(-1, 1)).ravel()
    yva = tsc_l.transform(ds.y[va].reshape(-1, 1)).ravel()

    # 2a: congelar cuerpo, entrenar cabeza.
    _freeze_body(model, freeze=True)
    model.compile(optimizer=tf.keras.optimizers.Nadam(config.LEARNING_RATE), loss="mse")
    logger.info("Afinando cabeza (cuerpo congelado)...")
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.finetune_epochs // 2,
              batch_size=config.BATCH_SIZE, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=20, restore_best_weights=True)])

    # 2b: descongelar todo, lr bajo.
    _freeze_body(model, freeze=False)
    model.compile(optimizer=tf.keras.optimizers.Nadam(config.LEARNING_RATE / 10), loss="mse")
    logger.info("Afinamiento completo (lr bajo)...")
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.finetune_epochs,
              batch_size=config.BATCH_SIZE, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=30, restore_best_weights=True)])

    # ---- Evaluacion por muestra ----
    te_ids = ds.ids[te]
    uniq = np.unique(te_ids)
    pred_scan = tsc_l.inverse_transform(model.predict(Xte, verbose=0)).ravel()
    y_pred = np.array([pred_scan[te_ids == s].mean() for s in uniq])
    y_true = np.array([ds.y[te][te_ids == s].mean() for s in uniq])
    met = regression_metrics(y_true, y_pred)
    logger.info("=" * 50)
    logger.info("TRANSFER %s (%s) -> R2=%.3f RMSE=%.3f RPD=%.2f",
                args.model, args.range, met["r2"], met["rmse"], met["rpd"])

    out = config.RESULTS_DIR / f"transfer_{args.range}_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": uniq, "y_true": y_true, "y_pred": y_pred}
                 ).to_csv(out / "predictions.csv", index=False)
    model.save(out / "model_transfer.keras")
    logger.info("Guardado -> %s", out)


if __name__ == "__main__":
    main()
