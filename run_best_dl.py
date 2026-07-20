"""
run_best_dl.py
==============
Entrena el mejor modelo profundo (MLP mejorado, NIR, SG1) con validacion cruzada
de 5 folds, recolecta predicciones fuera de fold (OOF) por muestra y genera:
  - metricas OOF globales (R2, RMSE, MAE, RPD, RPIQ)
  - figura de dispersion medido vs predicho (OOF)
  - figura de comparacion justa DL (CV) vs quimiometrico (CV)
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
print("OOF MLP_improved:", {k: round(v, 3) for k, v in m.items()})

# scatter OOF
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(y, oof, alpha=0.5, s=16, color="#6a51a3", edgecolor="white", lw=0.3)
lim = [min(y.min(), oof.min()), max(y.max(), oof.max())]
ax.plot(lim, lim, "r--", lw=1.3, label="1:1")
ax.set_xlabel("COS medido (\\%)"); ax.set_ylabel("COS predicho (\\%)")
ax.set_title(f"Modelo entregado: MLP mejorado (NIR, SG1)\\nOOF 5-fold  R$^2$={m['r2']:.3f}  RPD={m['rpd']:.2f}")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig_mlp_mejorado_scatter.png", bbox_inches="tight"); plt.close()

# comparacion justa CV
modelos = ["LucasVGG16", "MLP", "MLP mejorado", "SVR-RBF", "Meta-ens."]
r2 = [0.536, 0.592, 0.611, 0.643, 0.658]
err = [0.082, 0.032, 0.050, 0.009, 0.04]
colores = ["#5dade2", "#5dade2", "#2e86c1", "#58d68d", "#28b463"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(modelos, r2, yerr=err, color=colores, edgecolor="black", lw=.4, capsize=4)
ax.axhline(0.68, color="gray", ls="--", lw=1.1, label="Techo estimado (~0.68-0.75)")
ax.set_ylabel("R$^2$ (validación cruzada 5-fold)"); ax.set_ylim(0, 0.85)
ax.set_title("Comparación justa (mismo protocolo CV): profundo vs. quimiométrico")
for i, v in enumerate(r2): ax.text(i, v + err[i] + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig_dl_cv_comparacion.png", bbox_inches="tight"); plt.close()
print("figuras: fig_mlp_mejorado_scatter.png, fig_dl_cv_comparacion.png")
