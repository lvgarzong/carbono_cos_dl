"""
run_baselines.py
================
Baselines clasicos de quimiometria (referencia del paper Zhong et al. 2021):
  - PLSR  (Partial Least Squares Regression)  -> estandar de oro en suelos
  - RandomForest
  - SVR (RBF)

Usa exactamente el mismo split estratificado y preprocesado que los modelos de
deep learning, para una comparacion justa.

Por defecto escribe en su PROPIO archivo (results/classical_results.csv) para no
chocar con un entrenamiento de redes que este escribiendo all_results.csv en
paralelo. Luego, `analyze_results.py` / `report.py` fusionan ambos
automaticamente. Si quieres anexar directo a all_results.csv usa --out.

Uso:
    python run_baselines.py
    python run_baselines.py --ranges VISNIR --preprocess sg1
    python run_baselines.py --out results/all_results.csv   # anexar directo
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

import config
from src.logging_utils import get_logger
from src.data_loader import load_dataset
from src.preprocessing import apply_preprocessing, Scaler
from src.train import _stratified_split
from src.metrics import regression_metrics


def _fit_plsr(Xtr, ytr):
    n = min(40, Xtr.shape[1], Xtr.shape[0] - 1)
    gs = GridSearchCV(PLSRegression(), {"n_components": list(range(2, n, 2))},
                      scoring="r2", cv=5)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_


def _fit_rf(Xtr, ytr):
    m = RandomForestRegressor(n_estimators=300, max_depth=None,
                              n_jobs=-1, random_state=0)
    m.fit(Xtr, ytr)
    return m


def _fit_svr(Xtr, ytr):
    gs = GridSearchCV(SVR(kernel="rbf"),
                      {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.001]},
                      scoring="r2", cv=5)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_


FITTERS = {"PLSR": _fit_plsr, "RF": _fit_rf, "SVR": _fit_svr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", nargs="+", default=config.RANGES)
    ap.add_argument("--preprocess", nargs="+", default=["raw", "sg1"])
    ap.add_argument("--models", nargs="+", default=list(FITTERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "classical_results.csv"),
                    help="CSV de salida (por defecto separado de all_results.csv)")
    args = ap.parse_args()

    logger = get_logger("baselines", log_dir=config.LOGS_DIR)
    master = Path(args.out)
    appending = master.name == "all_results.csv"
    existing = pd.read_csv(master) if (appending and master.exists()) else pd.DataFrame()
    records = existing.to_dict("records") if not existing.empty else []

    for rng in args.ranges:
        ds = load_dataset(rng, logger=logger)
        tr, va, te = _stratified_split(ds.X, ds.y, ds.ids, args.seed)
        for prep in args.preprocess:
            Xtr, _, Xte = apply_preprocessing(ds.X[tr], ds.X[va], ds.X[te], prep)
            sc = Scaler(config.FEATURE_SCALING, axis=0).fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            for name in args.models:
                t0 = time.time()
                try:
                    model = FITTERS[name](Xtr, ds.y[tr])
                    pred = np.asarray(model.predict(Xte)).ravel()
                    met = regression_metrics(ds.y[te], pred)
                    rec = {"model": name, "range": rng, "preprocess": prep,
                           "seed": args.seed, "status": "ok",
                           "n_params": 0, "n_bands": ds.X.shape[1],
                           "n_train": len(tr), "n_test": len(te),
                           "epochs_run": 0, "time_s": round(time.time() - t0, 1),
                           "exp_name": f"{rng}__{name}__{prep}__seed{args.seed}",
                           **met}
                    records.append(rec)
                    logger.info("[%s|%s|%s] R2=%.3f RMSE=%.3f RPD=%.2f",
                                name, rng, prep, met["r2"], met["rmse"], met["rpd"])
                except Exception as e:
                    logger.error("Fallo %s %s %s: %s", name, rng, prep, e)
        pd.DataFrame(records).to_csv(master, index=False)
    logger.info("Baselines anexados a %s", master)


if __name__ == "__main__":
    main()
