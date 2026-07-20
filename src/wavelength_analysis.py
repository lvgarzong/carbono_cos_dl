"""
wavelength_analysis.py
======================
Calcula la importancia de longitudes de onda del MEJOR experimento y la guarda
en results/<exp>/wavelength_importance.npz para que el reporte la dibuje.

Combina saliencia por gradiente + importancia por permutacion (promedio) y
extrae el top-N de bandas (contribucion ii).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from src.data_loader import load_dataset
from src.preprocessing import apply_preprocessing, Scaler
from src.train import _stratified_split
from src.feature_importance import gradient_saliency, permutation_importance, top_wavelengths


def _best_experiment():
    df = pd.read_csv(config.RESULTS_DIR / "all_results.csv")
    ok = df[df.status == "ok"]
    row = ok.sort_values("r2", ascending=False).iloc[0]
    return row


def analyze_best(logger=None, top_n=10):
    row = _best_experiment()
    exp_dir = config.RESULTS_DIR / row["exp_name"]
    model_path = exp_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    agg = row.get("aggregate")
    if not isinstance(agg, str):
        agg = "median"
    ds = load_dataset(row["range"], agg=agg, logger=logger)
    tr, va, te = _stratified_split(ds.X, ds.y, ds.ids, int(row["seed"]))

    Xtr, _, Xte = apply_preprocessing(ds.X[tr], ds.X[va], ds.X[te], row["preprocess"])
    fscaler = Scaler(config.FEATURE_SCALING, axis=0).fit(Xtr)
    Xte = fscaler.transform(Xte)[..., None].astype(np.float32)
    tscaler = Scaler(config.TARGET_SCALING, axis=0).fit(ds.y[tr].reshape(-1, 1))

    model = tf.keras.models.load_model(model_path, compile=False)

    sal = gradient_saliency(model, Xte)
    perm = permutation_importance(model, Xte, ds.y[te], tscaler,
                                  block=max(1, ds.X.shape[1] // 200))
    importance = 0.5 * sal + 0.5 * perm
    importance = importance / (importance.max() + 1e-12)
    top = top_wavelengths(importance, ds.wavelengths, top_n=top_n)

    np.savez_compressed(exp_dir / "wavelength_importance.npz",
                        wavelengths=ds.wavelengths, importance=importance,
                        saliency=sal, permutation=perm,
                        top=np.array(top, dtype=object))
    with open(exp_dir / "top_wavelengths.json", "w", encoding="utf-8") as f:
        json.dump({"range": row["range"], "model": row["model"],
                   "top_wavelengths_nm": top}, f, indent=2, ensure_ascii=False)
    if logger:
        logger.info("Top %d bandas (%s): %s", top_n, row["range"],
                    ", ".join(f"{w:.0f}nm" for w, _ in top))
    return top


if __name__ == "__main__":
    from src.logging_utils import get_logger
    analyze_best(get_logger("wl"))
