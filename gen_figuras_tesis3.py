"""Figuras adicionales (3a tanda) para alcanzar la extension requerida."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
OUT = r"C:\Users\lvgar\Music\COS\Plantilla_Tesis_Trabajo_Final_UNAL_2023__1_\00Figuras"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": .3})
CACHE = "cache"
nir = np.load(f"{CACHE}/dataset_NIR_median.npz", allow_pickle=True)
Xn, yn, wn = nir["X"], nir["y"], nir["wavelengths"]
Xpp = savgol_filter(Xn, 25, 3, deriv=1, axis=1)

# 1. PCA 2D scatter coloreado por COS
Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Xpp))
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(Z[:, 0], Z[:, 1], c=yn, cmap="viridis", s=14, alpha=.8)
plt.colorbar(sc, label="COS (\\%)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Proyeccion PCA de los espectros NIR (color = COS)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_pca_scatter.png", bbox_inches="tight"); plt.close()

# 2. Espectros medios por cuartil de COS
q = np.quantile(yn, [0, .25, .5, .75, 1.0])
fig, ax = plt.subplots(figsize=(9, 4))
cols = plt.cm.viridis(np.linspace(0, 1, 4))
for i in range(4):
    m = (yn >= q[i]) & (yn <= q[i+1])
    ax.plot(wn, Xn[m].mean(0), color=cols[i], lw=1.2,
            label=f"Q{i+1} ({q[i]:.1f}-{q[i+1]:.1f}\\%)")
ax.set_xlabel("Longitud de onda (nm)"); ax.set_ylabel("Reflectancia media")
ax.set_title("Espectro NIR medio por cuartil de COS"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_espectros_cuartil.png", bbox_inches="tight"); plt.close()

# 3. Residuos del mejor modelo
p = pd.read_csv("results/NIR__LucasVGG16__raw__seed0/predictions.csv")
res = p.y_true - p.y_pred
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(p.y_pred, res, s=16, alpha=.6, color="#2c7fb8")
ax[0].axhline(0, color="red", lw=1.2)
ax[0].set_xlabel("COS predicho (\\%)"); ax[0].set_ylabel("Residuo (real - predicho)")
ax[0].set_title("Residuos vs. predicho")
ax[1].hist(res, bins=20, color="#d95f02", edgecolor="black", lw=.4, alpha=.85)
ax[1].axvline(0, color="red", lw=1.2)
ax[1].set_xlabel("Residuo (\\%)"); ax[1].set_ylabel("Frecuencia")
ax[1].set_title(f"Distribucion de residuos (sesgo={res.mean():+.3f})")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_residuos.png", bbox_inches="tight"); plt.close()

# 4. Efecto del preprocesamiento (clasicos)
c = pd.read_csv("results/classical_results.csv")
piv = c.pivot_table(index="preprocess", columns="model", values="r2")
fig, ax = plt.subplots(figsize=(7.5, 4))
piv.plot(kind="bar", ax=ax, edgecolor="black", lw=.4)
ax.set_ylabel("R$^2$ (medio sobre rangos)"); ax.set_xlabel("Preprocesamiento")
ax.set_title("Efecto del preprocesamiento (modelos quimiometricos)")
ax.legend(title="Modelo", fontsize=8); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_preprocesamiento_comp.png", bbox_inches="tight"); plt.close()

# 5. Clasico vs DL por rango (mejor de cada familia)
rangos = ["VIS", "NIR", "VISNIR"]
clasico = [c[(c["range"] == r)].r2.max() for r in rangos]
dl = [0.164, 0.450, 0.254]  # mejor DL (LucasVGG16) por rango
x = np.arange(3); w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(x - w/2, clasico, w, label="Mejor quimiométrico", color="#58d68d", edgecolor="black", lw=.4)
ax.bar(x + w/2, dl, w, label="Mejor red profunda (VGG)", color="#5dade2", edgecolor="black", lw=.4)
ax.set_xticks(x); ax.set_xticklabels(rangos); ax.set_ylabel("R$^2$")
ax.set_title("Quimiometría vs. aprendizaje profundo por rango"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig_clasico_vs_dl.png", bbox_inches="tight"); plt.close()

print("Figuras 3a tanda OK:")
for f in ["fig_pca_scatter", "fig_espectros_cuartil", "fig_residuos",
          "fig_preprocesamiento_comp", "fig_clasico_vs_dl"]:
    print("  ", f, os.path.exists(f"{OUT}/{f}.png"))
