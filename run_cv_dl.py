"""
run_cv_dl.py
============
Validacion cruzada estratificada de 5 folds para modelos de aprendizaje profundo,
para una comparacion JUSTA con los metodos quimiometricos (que se evaluaron con CV).
Reporta R2/RMSE/RPD por fold y promedio +- desviacion, en unidades reales de COS.
"""
import argparse, numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import config
from src.data_loader import load_dataset
from src.preprocessing import apply_preprocessing, Scaler
from src.metrics import regression_metrics, RSquare
from src.train import set_seed, _make_optimizer
from src import models as M


def cv_evaluate(model_name, range_name="NIR", prep="raw", n_folds=5,
                epochs=150, seed=0, logger_print=print):
    ds = load_dataset(range_name, agg="median")
    X, y = ds.X, ds.y
    bins = np.quantile(y, np.linspace(0, 1, 6))
    strata = np.clip(np.digitize(y, bins[1:-1]), 0, 4)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    res = []
    for k, (tr, te) in enumerate(skf.split(X, strata), 1):
        set_seed(seed)
        # split interno train/val
        ntr = int(len(tr) * 0.85)
        tri, vai = tr[:ntr], tr[ntr:]
        Xtr, Xva, Xte = apply_preprocessing(X[tri], X[vai], X[te], prep)
        fs = Scaler(config.FEATURE_SCALING, axis=0).fit(Xtr)
        Xtr, Xva, Xte = (fs.transform(Xtr)[..., None], fs.transform(Xva)[..., None],
                         fs.transform(Xte)[..., None])
        ts = Scaler(config.TARGET_SCALING, axis=0).fit(y[tri].reshape(-1, 1))
        ytr = ts.transform(y[tri].reshape(-1, 1)).ravel()
        yva = ts.transform(y[vai].reshape(-1, 1)).ravel()
        model = M.get_model(model_name, input_len=X.shape[1])
        model.compile(optimizer=_make_optimizer(), loss="mse")
        model.fit(Xtr.astype("float32"), ytr, validation_data=(Xva.astype("float32"), yva),
                  epochs=epochs, batch_size=config.BATCH_SIZE, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                             patience=30, restore_best_weights=True)])
        pred = ts.inverse_transform(model.predict(Xte.astype("float32"), verbose=0)).ravel()
        m = regression_metrics(y[te], pred)
        res.append(m)
        logger_print(f"  fold {k}: R2={m['r2']:.3f} RMSE={m['rmse']:.3f}")
    r2 = np.array([r["r2"] for r in res]); rmse = np.array([r["rmse"] for r in res])
    rpd = np.array([r["rpd"] for r in res])
    logger_print(f"== {model_name} | {range_name} | {prep} : "
                 f"R2={r2.mean():.3f}+-{r2.std():.3f}  RMSE={rmse.mean():.3f}  RPD={rpd.mean():.2f}")
    return r2.mean(), r2.std(), rmse.mean(), rpd.mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["LucasVGG16", "MLP", "MLP_improved"])
    ap.add_argument("--range", default="NIR")
    ap.add_argument("--prep", default="sg1")
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    for mdl in args.models:
        print(f"\n### {mdl} ({args.range}, {args.prep}) 5-fold CV")
        cv_evaluate(mdl, args.range, args.prep, epochs=args.epochs)
