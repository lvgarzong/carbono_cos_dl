"""
run_explora2.py
===============
Segunda tanda de busqueda de mejoras del MLP (sobre <=640, NIR, 5-fold CV):
  - BatchNorm si/no
  - tasa de aprendizaje
  - optimizador (Nadam/Adam/AdamW)
  - activacion (relu/elu/selu)
  - regularizacion L2
  - transformacion del objetivo (zscore vs log)
  - entrada: espectro completo vs PCA(30)/PCA(50)
  - tamano de lote
Parte de la mejor configuracion de la primera tanda (editar BASE abajo).
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

# ===== CONFIGURACION BASE (mejor de la 1a tanda; se ajusta tras run_explora) =====
BASE = dict(prep="sg1_snv", arch=(256, 128, 64), dropout=0.4, bn=True, l2=1e-3,
            lr=5e-4, opt="nadam", act="relu", target="zscore", pca=0, batch=32, epochs=200)

ds = load_dataset("NIR", agg="median")
# La 1a tanda mostro que usar TODAS las firmas es mejor que <=640; se usa el conjunto completo.
X0, y0 = ds.X, ds.y
print(f"Busqueda 2 sobre TODAS: {len(y0)} muestras")


def make_opt(name, lr):
    if name == "adam": return optimizers.Adam(lr)
    if name == "adamw": return optimizers.AdamW(lr)
    return optimizers.Nadam(lr)


def build(L, arch, dropout, bn, l2, act):
    inp = Input((L, 1)); x = layers.Flatten()(inp)
    for u in arch:
        x = layers.Dense(u, kernel_regularizer=regularizers.l2(l2) if l2 else None)(x)
        if bn: x = layers.BatchNormalization()(x)
        x = layers.Activation(act)(x); x = layers.Dropout(dropout)(x)
    return Model(inp, layers.Dense(1)(x))


def cv(cfg):
    bins = np.quantile(y0, np.linspace(0, 1, 6))
    strata = np.clip(np.digitize(y0, bins[1:-1]), 0, 4)
    fs = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X0, strata))
    r2s, rpds = [], []
    for tr, te in fs:
        set_seed(0)
        ntr = int(len(tr) * 0.85); tri, vai = tr[:ntr], tr[ntr:]
        Xtr, Xva, Xte = apply_preprocessing(X0[tri], X0[vai], X0[te], cfg["prep"])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        if cfg["pca"]:
            pca = PCA(n_components=min(cfg["pca"], Xtr.shape[1]), random_state=0).fit(Xtr)
            Xtr, Xva, Xte = pca.transform(Xtr), pca.transform(Xva), pca.transform(Xte)
        Xtr, Xva, Xte = Xtr[..., None], Xva[..., None], Xte[..., None]
        if cfg["target"] == "log":
            ytr, yva = np.log(y0[tri]), np.log(y0[vai]); inv = np.exp
        else:
            ts = Scaler("zscore", axis=0).fit(y0[tri].reshape(-1, 1))
            ytr = ts.transform(y0[tri].reshape(-1, 1)).ravel(); yva = ts.transform(y0[vai].reshape(-1, 1)).ravel()
            inv = lambda v: ts.inverse_transform(v)
        mdl = build(Xtr.shape[1], cfg["arch"], cfg["dropout"], cfg["bn"], cfg["l2"], cfg["act"])
        mdl.compile(optimizer=make_opt(cfg["opt"], cfg["lr"]), loss="mse")
        mdl.fit(Xtr.astype("float32"), ytr, validation_data=(Xva.astype("float32"), yva),
                epochs=cfg["epochs"], batch_size=cfg["batch"], verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)])
        p = np.asarray(inv(mdl.predict(Xte.astype("float32"), verbose=0).reshape(-1, 1))).ravel()
        m = regression_metrics(y0[te], p); r2s.append(m["r2"]); rpds.append(m["rpd"])
    return np.mean(r2s), np.std(r2s), np.mean(rpds)


def trial(name, **over):
    cfg = dict(BASE); cfg.update(over)
    r2, sd, rpd = cv(cfg)
    print(f"  {name:28}: R2={r2:.3f}+-{sd:.3f} RPD={rpd:.2f}")
    return r2, over

print(f"BASE: {BASE}")
r0 = trial("BASE", )[0][0] if False else None
b = cv(BASE); print(f"  {'BASE (ref)':28}: R2={b[0]:.3f}+-{b[1]:.3f} RPD={b[2]:.2f}")

print("--- BatchNorm ---")
for bn in [True, False]: trial(f"bn={bn}", bn=bn)
print("--- Learning rate ---")
for lr in [1e-3, 5e-4, 2e-4, 1e-4]: trial(f"lr={lr}", lr=lr)
print("--- Optimizador ---")
for opt in ["nadam", "adam", "adamw"]: trial(f"opt={opt}", opt=opt)
print("--- Activacion ---")
for act in ["relu", "elu", "selu"]: trial(f"act={act}", act=act)
print("--- L2 ---")
for l2 in [1e-2, 1e-3, 1e-4, 0.0]: trial(f"l2={l2}", l2=l2)
print("--- Objetivo ---")
for tg in ["zscore", "log"]: trial(f"target={tg}", target=tg)
print("--- Entrada PCA ---")
for pc in [0, 30, 50]: trial(f"pca={pc}", pca=pc)
print("--- Batch size ---")
for bs in [16, 32, 64]: trial(f"batch={bs}", batch=bs)
print("DONE EXPLORA2")
