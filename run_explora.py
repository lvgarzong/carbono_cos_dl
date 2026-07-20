"""
run_explora.py
==============
(a) Confirma/niega si usar solo las firmas hasta la #640 mejora frente a usar todas.
(b) Busqueda exhaustiva de mejoras del MLP: preprocesamiento, arquitectura,
    dropout y epocas. Todo con validacion cruzada estratificada de 5 folds, NIR.
"""
import numpy as np, tensorflow as tf, warnings, itertools
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from tensorflow.keras import layers, Model, Input, regularizers
import config
from src.data_loader import load_dataset
from src.preprocessing import apply_preprocessing, Scaler
from src.metrics import regression_metrics
from src.train import set_seed, _make_optimizer

ds = load_dataset("NIR", agg="median")
Xall, yall, idall = ds.X, ds.y, ds.ids
m640 = idall <= 640
print(f"Muestras: todas={len(yall)} | hasta 640={int(m640.sum())}")


def folds(y, seed=0):
    bins = np.quantile(y, np.linspace(0, 1, 6))
    strata = np.clip(np.digitize(y, bins[1:-1]), 0, 4)
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros_like(y), strata)), strata


def build_mlp(L, arch=(128, 64, 32), dropout=0.4, bn=True, l2=1e-3):
    inp = Input((L, 1)); x = layers.Flatten()(inp)
    for u in arch:
        x = layers.Dense(u, kernel_regularizer=regularizers.l2(l2))(x)
        if bn: x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x); x = layers.Dropout(dropout)(x)
    return Model(inp, layers.Dense(1)(x))


def cv_mlp(X, y, prep="sg1", arch=(128, 64, 32), dropout=0.4, bn=True, epochs=200):
    fs, _ = folds(y)
    r2s, rmses, rpds = [], [], []
    for tr, te in fs:
        set_seed(0)
        ntr = int(len(tr) * 0.85); tri, vai = tr[:ntr], tr[ntr:]
        Xtr, Xva, Xte = apply_preprocessing(X[tri], X[vai], X[te], prep)
        sc = Scaler(config.FEATURE_SCALING, axis=0).fit(Xtr)
        Xtr, Xva, Xte = sc.transform(Xtr)[..., None], sc.transform(Xva)[..., None], sc.transform(Xte)[..., None]
        ts = Scaler(config.TARGET_SCALING, axis=0).fit(y[tri].reshape(-1, 1))
        ytr = ts.transform(y[tri].reshape(-1, 1)).ravel(); yva = ts.transform(y[vai].reshape(-1, 1)).ravel()
        mdl = build_mlp(X.shape[1], arch, dropout, bn)
        mdl.compile(optimizer=_make_optimizer(), loss="mse")
        mdl.fit(Xtr.astype("float32"), ytr, validation_data=(Xva.astype("float32"), yva),
                epochs=epochs, batch_size=32, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)])
        p = ts.inverse_transform(mdl.predict(Xte.astype("float32"), verbose=0)).ravel()
        m = regression_metrics(y[te], p); r2s.append(m["r2"]); rmses.append(m["rmse"]); rpds.append(m["rpd"])
    return np.mean(r2s), np.std(r2s), np.mean(rmses), np.mean(rpds)


def cv_sklearn(X, y, kind="svr", prep="sg1", ncomp=30):
    fs, _ = folds(y)
    r2s, rpds = [], []
    for tr, te in fs:
        Xtr, _, Xte = apply_preprocessing(X[tr], X[tr][:1], X[te], prep)
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        pca = PCA(n_components=min(ncomp, Xtr.shape[1]), random_state=0).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        yl = np.log(y[tr])
        est = SVR(C=10, gamma="scale", epsilon=0.05) if kind == "svr" else PLSRegression(n_components=min(15, Xtr.shape[1]))
        est.fit(Xtr, yl)
        p = np.exp(np.asarray(est.predict(Xte)).ravel())
        m = regression_metrics(y[te], p); r2s.append(m["r2"]); rpds.append(m["rpd"])
    return np.mean(r2s), np.std(r2s), np.mean(rpds)


print("\n================ (a) 640 vs TODAS (NIR, sg1, 5-fold CV) ================")
for nm, (X, y) in [("TODAS", (Xall, yall)), ("<=640", (Xall[m640], yall[m640]))]:
    r = cv_mlp(X, y, prep="sg1", arch=(128, 64, 32), dropout=0.4, epochs=200)
    print(f"  MLP_improved {nm:6}: R2={r[0]:.3f}+-{r[1]:.3f} RMSE={r[2]:.3f} RPD={r[3]:.2f}")
    s = cv_sklearn(X, y, "svr", "sg1")
    print(f"  SVR-RBF      {nm:6}: R2={s[0]:.3f}+-{s[1]:.3f} RPD={s[2]:.2f}")

# Elegir subconjunto para la busqueda (usar 640 segun pedido del usuario)
Xs, ys = Xall[m640], yall[m640]
print("\n================ (b) BUSQUEDA MLP en <=640 (5-fold CV) ================")
print("--- b1: preprocesamiento (arch 128-64-32, drop 0.4, 200 ep) ---")
best = (-9, None)
for prep in ["raw", "snv", "sg1", "sg2", "sg1_snv"]:
    r = cv_mlp(Xs, ys, prep=prep, arch=(128, 64, 32), dropout=0.4, epochs=200)
    print(f"  prep={prep:8}: R2={r[0]:.3f}+-{r[1]:.3f} RPD={r[3]:.2f}")
    if r[0] > best[0]: best = (r[0], prep)
bp = best[1]; print(f"  -> mejor prep: {bp} (R2={best[0]:.3f})")

print(f"--- b2: arquitectura (prep={bp}, drop 0.4, 200 ep) ---")
besta = (-9, None)
for arch in [(64, 32), (128, 64, 32), (256, 128, 64), (256, 128, 64, 32), (512, 256, 128)]:
    r = cv_mlp(Xs, ys, prep=bp, arch=arch, dropout=0.4, epochs=200)
    print(f"  arch={str(arch):20}: R2={r[0]:.3f}+-{r[1]:.3f} RPD={r[3]:.2f}")
    if r[0] > besta[0]: besta = (r[0], arch)
ba = besta[1]; print(f"  -> mejor arch: {ba} (R2={besta[0]:.3f})")

print(f"--- b3: dropout y epocas (prep={bp}, arch={ba}) ---")
bestc = (-9, None)
for dropout, ep in itertools.product([0.3, 0.4, 0.5], [200, 400]):
    r = cv_mlp(Xs, ys, prep=bp, arch=ba, dropout=dropout, epochs=ep)
    print(f"  drop={dropout} ep={ep}: R2={r[0]:.3f}+-{r[1]:.3f} RPD={r[3]:.2f}")
    if r[0] > bestc[0]: bestc = (r[0], (dropout, ep))
print(f"  -> mejor (drop,ep): {bestc[1]} (R2={bestc[0]:.3f})")

print("\n================ RESUMEN MEJOR MLP ================")
print(f"  Mejor config en <=640: prep={bp}, arch={ba}, drop={bestc[1][0]}, ep={bestc[1][1]} -> R2={bestc[0]:.3f}")
rall = cv_mlp(Xall, yall, prep=bp, arch=ba, dropout=bestc[1][0], epochs=bestc[1][1])
print(f"  Misma config en TODAS: R2={rall[0]:.3f}+-{rall[1]:.3f} RPD={rall[3]:.2f}")
print("DONE EXPLORA")
