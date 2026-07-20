"""
run_explora4.py
===============
(A) MLP optimizado sobre VIS, NIR y VIS-NIR fusionado (efecto de la fusion).
(B) Ensamble de semillas del MLP optimizado (NIR): promedia K modelos por fold.
Todo con validacion cruzada estratificada de 5 folds, predicciones OOF.
"""
import numpy as np, tensorflow as tf, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras import layers, Model, Input, regularizers, optimizers
import config
from src.data_loader import load_dataset
from src.preprocessing import apply_preprocessing, Scaler
from src.metrics import regression_metrics
from src.train import set_seed

CFG = dict(prep="sg1_snv", arch=(256, 128, 64), dropout=0.4, l2=1e-2,
           lr=2e-4, act="elu", pca=50, epochs=200, batch=32)


def build(L):
    inp = Input((L, 1)); x = layers.Flatten()(inp)
    for u in CFG["arch"]:
        x = layers.Dense(u, kernel_regularizer=regularizers.l2(CFG["l2"]))(x)
        x = layers.BatchNormalization()(x); x = layers.Activation(CFG["act"])(x)
        x = layers.Dropout(CFG["dropout"])(x)
    return Model(inp, layers.Dense(1)(x))


def prep_fold(X, tr, va, te, y):
    Xtr, Xva, Xte = apply_preprocessing(X[tr], X[va], X[te], CFG["prep"])
    sc = StandardScaler().fit(Xtr); Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    if CFG["pca"]:
        pca = PCA(n_components=min(CFG["pca"], Xtr.shape[1]), random_state=0).fit(Xtr)
        Xtr, Xva, Xte = pca.transform(Xtr), pca.transform(Xva), pca.transform(Xte)
    ts = Scaler("zscore", axis=0).fit(y[tr].reshape(-1, 1))
    return (Xtr[..., None].astype("float32"), Xva[..., None].astype("float32"), Xte[..., None].astype("float32"),
            ts.transform(y[tr].reshape(-1, 1)).ravel(), ts.transform(y[va].reshape(-1, 1)).ravel(), ts)


def train_one(Xtr, ytr, Xva, yva, seed):
    set_seed(seed); m = build(Xtr.shape[1])
    m.compile(optimizer=optimizers.Nadam(CFG["lr"]), loss="mse")
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=CFG["epochs"], batch_size=CFG["batch"], verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)])
    return m


def cv(X, y, seeds=(0,)):
    bins = np.quantile(y, np.linspace(0, 1, 6)); strata = np.clip(np.digitize(y, bins[1:-1]), 0, 4)
    fs = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, strata))
    oof = np.zeros_like(y)
    for tr, te in fs:
        ntr = int(len(tr) * 0.85); tri, vai = tr[:ntr], tr[ntr:]
        Xtr, Xva, Xte, ytr, yva, ts = prep_fold(X, tri, vai, te, y)
        preds = []
        for s in seeds:
            mdl = train_one(Xtr, ytr, Xva, yva, s)
            preds.append(ts.inverse_transform(mdl.predict(Xte, verbose=0)).ravel())
        oof[te] = np.mean(preds, axis=0)
    return regression_metrics(y, oof)


print("=== (A) MLP optimizado por rango (5-fold CV, OOF) ===")
for rng in ["VIS", "NIR", "VISNIR"]:
    ds = load_dataset(rng, agg="median")
    m = cv(ds.X, ds.y)
    print(f"  {rng:7}: R2={m['r2']:.3f} RMSE={m['rmse']:.3f} RPD={m['rpd']:.2f} RPIQ={m['rpiq']:.2f}")

print("\n=== (B) Ensamble de semillas (NIR) ===")
ds = load_dataset("NIR", agg="median")
for k in [1, 3, 5]:
    m = cv(ds.X, ds.y, seeds=tuple(range(k)))
    print(f"  {k} semilla(s): R2={m['r2']:.3f} RMSE={m['rmse']:.3f} RPD={m['rpd']:.2f} RPIQ={m['rpiq']:.2f}")
print("DONE4")
