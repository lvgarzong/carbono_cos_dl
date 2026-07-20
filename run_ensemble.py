"""
run_ensemble.py
===============
Exprime el maximo de los datos actuales combinando tres estrategias:

  1. ENSAMBLE: entrena varios modelos / semillas de inicializacion sobre el
     MISMO split (split fijo) y promedia sus predicciones por muestra.
  2. SELECCION DE BANDAS (opcional, --select-bands): entrena solo con las bandas
     mas relevantes segun VIP de PLS (menos ruido, menos sobreajuste).
  3. MODELO COMPACTO: incluye CompactCNN (pocos parametros, regularizado),
     que suele generalizar mejor que VGG/ResNet con pocas muestras.

A diferencia de la rejilla, aqui el split de test es FIJO (split_seed) y solo
varia la semilla de inicializacion -> los miembros son comparables y promediables.

Uso:
    python run_ensemble.py
    python run_ensemble.py --range NIR --preprocess raw --select-bands --top-k 150
    python run_ensemble.py --models LucasVGG16 CompactCNN --init-seeds 0 1 2 3 4
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from src.logging_utils import get_logger
from src.data_loader import load_dataset
from src.preprocessing import apply_preprocessing, Scaler
from src.train import _stratified_split, set_seed, _make_optimizer
from src.metrics import regression_metrics, RSquare
from src import models as M
from src.band_selection import select_bands


def _train_member(model_name, Xtr, ytr_s, Xva, yva_s, init_seed, epochs, logger):
    set_seed(init_seed)
    model = M.get_model(model_name, input_len=Xtr.shape[1])
    model.compile(optimizer=_make_optimizer(), loss="mse", metrics=[RSquare()])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=config.EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=config.REDUCE_LR_PATIENCE,
                                             min_lr=1e-6),
    ]
    h = model.fit(Xtr, ytr_s, validation_data=(Xva, yva_s),
                  epochs=epochs, batch_size=config.BATCH_SIZE,
                  verbose=0, shuffle=True, callbacks=cbs)
    return model, len(h.history["loss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", default="NIR")
    ap.add_argument("--preprocess", default="raw")
    ap.add_argument("--aggregate", default="median", choices=["median", "mean"])
    ap.add_argument("--models", nargs="+",
                    default=["LucasVGG16", "LucasResNet16", "CompactCNN"])
    ap.add_argument("--init-seeds", nargs="+", type=int, default=[0, 42, 707])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--select-bands", action="store_true")
    ap.add_argument("--top-k", type=int, default=150,
                    help="numero de bandas a conservar si --select-bands")
    args = ap.parse_args()

    logger = get_logger("ensemble", log_dir=config.LOGS_DIR)
    t0 = time.time()
    ds = load_dataset(args.range, agg=args.aggregate, logger=logger)
    tr, va, te = _stratified_split(ds.X, ds.y, ds.ids, args.split_seed)

    Xtr, Xva, Xte = apply_preprocessing(ds.X[tr], ds.X[va], ds.X[te], args.preprocess)

    band_info = "todas"
    if args.select_bands:
        idx, vip = select_bands(Xtr, ds.y[tr], top_k=args.top_k)
        Xtr, Xva, Xte = Xtr[:, idx], Xva[:, idx], Xte[:, idx]
        band_info = f"{len(idx)} bandas VIP (de {ds.X.shape[1]})"
        logger.info("Seleccion de bandas: %s", band_info)

    fsc = Scaler(config.FEATURE_SCALING, axis=0).fit(Xtr)
    Xtr, Xva, Xte = (fsc.transform(Xtr)[..., None].astype(np.float32),
                     fsc.transform(Xva)[..., None].astype(np.float32),
                     fsc.transform(Xte)[..., None].astype(np.float32))
    tsc = Scaler(config.TARGET_SCALING, axis=0).fit(ds.y[tr].reshape(-1, 1))
    ytr_s = tsc.transform(ds.y[tr].reshape(-1, 1)).ravel()
    yva_s = tsc.transform(ds.y[va].reshape(-1, 1)).ravel()

    # y_true por muestra (test)
    te_ids = ds.ids[te]
    uniq = np.unique(te_ids)
    y_true = np.array([ds.y[te][te_ids == s].mean() for s in uniq])

    logger.info("Ensamble en %s/%s/%s | %s | miembros=%d modelos x %d semillas",
                args.range, args.preprocess, args.aggregate, band_info,
                len(args.models), len(args.init_seeds))

    member_preds = []   # cada uno: prediccion por muestra
    rows = []
    for mname in args.models:
        for s in args.init_seeds:
            model, ep = _train_member(mname, Xtr, ytr_s, Xva, yva_s, s, args.epochs, logger)
            pred_scan = tsc.inverse_transform(model.predict(Xte, verbose=0)).ravel()
            pred = np.array([pred_scan[te_ids == u].mean() for u in uniq])
            met = regression_metrics(y_true, pred)
            member_preds.append(pred)
            rows.append({"member": f"{mname}_seed{s}", "r2": met["r2"],
                         "rmse": met["rmse"], "rpd": met["rpd"], "epochs": ep})
            logger.info("  miembro %s_seed%d -> R2=%.3f RMSE=%.3f (%d ep)",
                        mname, s, met["r2"], met["rmse"], ep)

    # Ensamble = promedio de predicciones de todos los miembros.
    ens_pred = np.mean(member_preds, axis=0)
    ens = regression_metrics(y_true, ens_pred)
    logger.info("=" * 50)
    logger.info("ENSAMBLE (%d miembros) -> R2=%.3f RMSE=%.3f RPD=%.2f RPIQ=%.2f",
                len(member_preds), ens["r2"], ens["rmse"], ens["rpd"], ens["rpiq"])

    # Tambien ensamble solo del mejor modelo (sus semillas).
    best_model = max(args.models,
                     key=lambda m: np.mean([r["r2"] for r in rows
                                            if r["member"].startswith(m)]))
    bidx = [i for i, r in enumerate(rows) if r["member"].startswith(best_model)]
    best_ens_pred = np.mean([member_preds[i] for i in bidx], axis=0)
    best_ens = regression_metrics(y_true, best_ens_pred)
    logger.info("ENSAMBLE solo %s -> R2=%.3f RMSE=%.3f", best_model,
                best_ens["r2"], best_ens["rmse"])

    # Guardado
    suffix = "_bandsel" if args.select_bands else ""
    out_dir = config.RESULTS_DIR / f"ensemble_{args.range}_{args.preprocess}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": uniq, "y_true": y_true, "y_pred_ensemble": ens_pred,
                  f"y_pred_{best_model}_ens": best_ens_pred}
                 ).to_csv(out_dir / "predictions.csv", index=False)
    summary = {
        "config": vars(args), "bands": band_info,
        "members": rows,
        "ensemble_all": ens, "best_model": best_model,
        "ensemble_best_model": best_ens,
        "n_test_samples": int(len(uniq)),
        "time_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Anexa filas al historico para que aparezcan en el reporte.
    master = config.RESULTS_DIR / "all_results.csv"
    hist = pd.read_csv(master).to_dict("records") if master.exists() else []
    for tag, m in [("ENSEMBLE_all", ens), (f"ENSEMBLE_{best_model}", best_ens)]:
        hist.append({"model": tag, "range": args.range, "preprocess": args.preprocess,
                     "aggregate": "ensemble", "seed": args.split_seed, "status": "ok",
                     "n_bands": Xtr.shape[1], "n_test": len(uniq), "epochs_run": 0,
                     "exp_name": f"{args.range}__{tag}__{args.preprocess}"
                                 + ("_bandsel" if args.select_bands else ""),
                     **m})
    pd.DataFrame(hist).drop_duplicates(subset="exp_name", keep="last").to_csv(master, index=False)
    logger.info("Resumen -> %s | filas de ensamble anexadas a all_results.csv", out_dir)


if __name__ == "__main__":
    main()
