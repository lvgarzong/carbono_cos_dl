"""
run_explora3.py
===============
Combina las mejores variantes halladas y selecciona la configuracion final del
MLP; luego calcula predicciones OOF y genera la dispersion del modelo final.
"""
import numpy as np, tensorflow as tf, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras import layers, Model, Input, regularizers, optimizers
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config
from src.data_loader import load_dataset
from src.preprocessing import apply_preprocessing, Scaler
from src.metrics import regression_metrics
from src.train import set_seed

OUT = r"C:\Users\lvgar\Music\COS\Plantilla_Tesis_Trabajo_Final_UNAL_2023__1_\00Figuras"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": .3})
ds = load_dataset("NIR", agg="median")
X0, y0 = ds.X, ds.y

def make_opt(name, lr):
    return {"adam": optimizers.Adam, "adamw": optimizers.AdamW}.get(name, optimizers.Nadam)(lr)

def build(L, arch, dropout, bn, l2, act):
    inp = Input((L, 1)); x = layers.Flatten()(inp)
    for u in arch:
        x = layers.Dense(u, kernel_regularizer=regularizers.l2(l2) if l2 else None)(x)
        if bn: x = layers.BatchNormalization()(x)
        x = layers.Activation(act)(x); x = layers.Dropout(dropout)(x)
    return Model(inp, layers.Dense(1)(x))

def run(cfg, collect_oof=False):
    bins = np.quantile(y0, np.linspace(0, 1, 6)); strata = np.clip(np.digitize(y0, bins[1:-1]), 0, 4)
    fs = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X0, strata))
    r2s, rpds = [], []; oof = np.zeros_like(y0)
    for tr, te in fs:
        set_seed(0); ntr = int(len(tr)*0.85); tri, vai = tr[:ntr], tr[ntr:]
        Xtr, Xva, Xte = apply_preprocessing(X0[tri], X0[vai], X0[te], cfg["prep"])
        sc = StandardScaler().fit(Xtr); Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        if cfg["pca"]:
            pca = PCA(n_components=cfg["pca"], random_state=0).fit(Xtr)
            Xtr, Xva, Xte = pca.transform(Xtr), pca.transform(Xva), pca.transform(Xte)
        Xtr, Xva, Xte = Xtr[..., None], Xva[..., None], Xte[..., None]
        ts = Scaler("zscore", axis=0).fit(y0[tri].reshape(-1, 1))
        ytr = ts.transform(y0[tri].reshape(-1, 1)).ravel(); yva = ts.transform(y0[vai].reshape(-1, 1)).ravel()
        mdl = build(Xtr.shape[1], cfg["arch"], cfg["dropout"], cfg["bn"], cfg["l2"], cfg["act"])
        mdl.compile(optimizer=make_opt(cfg["opt"], cfg["lr"]), loss="mse")
        mdl.fit(Xtr.astype("float32"), ytr, validation_data=(Xva.astype("float32"), yva),
                epochs=cfg["epochs"], batch_size=cfg["batch"], verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)])
        p = ts.inverse_transform(mdl.predict(Xte.astype("float32"), verbose=0)).ravel()
        oof[te] = p; m = regression_metrics(y0[te], p); r2s.append(m["r2"]); rpds.append(m["rpd"])
    return (np.mean(r2s), np.std(r2s), np.mean(rpds), oof) if collect_oof else (np.mean(r2s), np.std(r2s), np.mean(rpds))

B = dict(prep="sg1_snv", arch=(256,128,64), dropout=0.4, bn=True, l2=1e-3, lr=5e-4, opt="nadam", act="relu", target="zscore", pca=0, batch=32, epochs=200)
def C(**o): c=dict(B); c.update(o); return c

combos = {
 "PCA50 (mejor individual)": C(pca=50),
 "PCA50+elu": C(pca=50, act="elu"),
 "PCA50+elu+lr2e-4": C(pca=50, act="elu", lr=2e-4),
 "PCA50+elu+lr2e-4+L2_1e-2": C(pca=50, act="elu", lr=2e-4, l2=1e-2),
 "PCA50+elu+lr2e-4+L2_1e-2+adamw": C(pca=50, act="elu", lr=2e-4, l2=1e-2, opt="adamw"),
 "PCA50+elu+adamw": C(pca=50, act="elu", opt="adamw"),
 "PCA30+elu+lr2e-4+L2_1e-2": C(pca=30, act="elu", lr=2e-4, l2=1e-2),
}
print("=== COMBINACIONES (TODAS, 5-fold CV) ===")
best = (-9, None, None)
for name, cfg in combos.items():
    r = run(cfg); print(f"  {name:34}: R2={r[0]:.3f}+-{r[1]:.3f} RPD={r[2]:.2f}")
    if r[0] > best[0]: best = (r[0], name, cfg)
print(f"\n-> MEJOR: {best[1]} (R2={best[0]:.3f})")

# OOF + figura del modelo final
r2, sd, rpd, oof = run(best[2], collect_oof=True)
m = regression_metrics(y0, oof)
print("OOF final:", {k: round(v,3) for k,v in m.items()})
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(y0, oof, alpha=0.5, s=16, color="#6a51a3", edgecolor="white", lw=0.3)
lim=[min(y0.min(),oof.min()), max(y0.max(),oof.max())]; ax.plot(lim,lim,"r--",lw=1.3,label="1:1")
ax.set_xlabel("COS medido (\\%)"); ax.set_ylabel("COS predicho (\\%)")
ax.set_title(f"Modelo profundo entregado: MLP optimizado\\nOOF 5-fold  R$^2$={m['r2']:.3f}  RPD={m['rpd']:.2f}  RPIQ={m['rpiq']:.2f}")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig_mlp_mejorado_scatter.png", bbox_inches="tight"); plt.close()
print("figura actualizada: fig_mlp_mejorado_scatter.png")
print("DONE3")
