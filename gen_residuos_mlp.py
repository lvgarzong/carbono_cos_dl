"""
gen_residuos_mlp.py
===================
Reproduce las predicciones OOF (5-fold) del modelo profundo entregado
(MLP optimizado, NIR, SG1) y genera fig_residuos.png para el analisis de
residuos del MODELO PROFUNDO ENTREGADO (no de LucasVGG16). Imprime ademas
estadisticas para redactar el texto con precision.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import config
from src.data_loader import load_dataset
from src.preprocessing import apply_preprocessing, Scaler
from src.metrics import regression_metrics
from src.train import set_seed, _make_optimizer
from src import models as M

OUT = r"C:\Users\lvgar\Music\COS\Plantilla_Tesis_Trabajo_Final_UNAL_2023__1_\00Figuras"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": .3})

ds = load_dataset("NIR", agg="median")
X, y = ds.X, ds.y
bins = np.quantile(y, np.linspace(0, 1, 6))
strata = np.clip(np.digitize(y, bins[1:-1]), 0, 4)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
oof = np.zeros_like(y)
for k, (tr, te) in enumerate(skf.split(X, strata), 1):
    set_seed(0)
    ntr = int(len(tr) * 0.85); tri, vai = tr[:ntr], tr[ntr:]
    Xtr, Xva, Xte = apply_preprocessing(X[tri], X[vai], X[te], "sg1")
    fs = Scaler(config.FEATURE_SCALING, axis=0).fit(Xtr)
    Xtr, Xva, Xte = fs.transform(Xtr)[..., None], fs.transform(Xva)[..., None], fs.transform(Xte)[..., None]
    ts = Scaler(config.TARGET_SCALING, axis=0).fit(y[tri].reshape(-1, 1))
    ytr = ts.transform(y[tri].reshape(-1, 1)).ravel(); yva = ts.transform(y[vai].reshape(-1, 1)).ravel()
    model = M.get_model("MLP_improved", input_len=X.shape[1])
    model.compile(optimizer=_make_optimizer(), loss="mse")
    model.fit(Xtr.astype("float32"), ytr, validation_data=(Xva.astype("float32"), yva),
              epochs=150, batch_size=config.BATCH_SIZE, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)])
    oof[te] = ts.inverse_transform(model.predict(Xte.astype("float32"), verbose=0)).ravel()
    print(f"fold {k} listo")

m = regression_metrics(y, oof)
res = y - oof  # residuo = real - predicho
print("OOF MLP_improved:", {k: round(v, 3) for k, v in m.items()})
print(f"sesgo medio (mean residuo) = {res.mean():+.3f}")
q3 = np.quantile(y, 0.75)
print(f"residuo medio en cuartil alto (y>{q3:.2f}) = {res[y > q3].mean():+.3f}  (n={int((y>q3).sum())})")
print(f"residuo medio en cuartil bajo = {res[y < np.quantile(y,0.25)].mean():+.3f}")

# ---- figura de residuos: dispersion vs predicho + histograma ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.scatter(oof, res, alpha=0.5, s=16, color="#6a51a3", edgecolor="white", lw=0.3)
ax.axhline(0, color="red", ls="--", lw=1.2)
ax.set_xlabel("COS predicho (\\%)"); ax.set_ylabel("Residuo = medido $-$ predicho (\\%)")
ax.set_title("Residuos frente al valor predicho")
ax = axes[1]
ax.hist(res, bins=28, color="#9e9ac8", edgecolor="white", lw=0.4)
ax.axvline(0, color="red", ls="--", lw=1.2)
ax.axvline(res.mean(), color="black", ls=":", lw=1.2, label=f"media={res.mean():+.3f}")
ax.set_xlabel("Residuo (\\%)"); ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de residuos")
ax.legend()
fig.suptitle("Análisis de residuos -- MLP optimizado (NIR, SG1+SNV), predicciones OOF de 5-fold", y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_residuos.png", bbox_inches="tight"); plt.close()
print("fig_residuos.png regenerada (MLP optimizado)")
