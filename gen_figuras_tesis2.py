"""
gen_figuras_tesis2.py
=====================
Figuras adicionales para la expansion de la tesis: diagramas de arquitectura de
las redes, analisis de componentes principales (PCA) y analisis de datos para la
discusion. Se guardan en 00Figuras de la plantilla.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.signal import savgol_filter

OUT = r"C:\Users\lvgar\Music\COS\Plantilla_Tesis_Trabajo_Final_UNAL_2023__1_\00Figuras"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
CACHE = "cache"

# ---------- Fig: diagrama de arquitecturas ----------
def caja(ax, x, y, w, h, txt, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=color, ec="black", lw=0.8))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7.5)

def flecha(ax, x1, y, x2):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=10, lw=0.8, color="#444"))

fig, axes = plt.subplots(5, 1, figsize=(11, 13))
archs = {
    "LucasVGGNet-16 (CNN 1D apilada)": [
        ("Entrada\n(L,1)", "#dfeaf4"), ("2x Conv1D\n6, k3 +Pool", "#9ecae1"),
        ("2x Conv1D\n12 +Pool", "#9ecae1"), ("3x Conv1D\n24 +Pool", "#6baed6"),
        ("3x Conv1D\n48 +Pool", "#6baed6"), ("3x Conv1D\n48 +Pool", "#4292c6"),
        ("Flatten", "#c7e9c0"), ("Dense 200\n+Drop", "#a1d99b"),
        ("Dense 100\n+Drop", "#a1d99b"), ("Salida\nCOS", "#fdae6b")],
    "LucasResNet-16 (bloques residuales)": [
        ("Entrada\n(L,1)", "#dfeaf4"), ("Conv1D 6\nk7 s2 +Pool", "#9ecae1"),
        ("ResBlock\n[6,6,12]", "#bcbddc"), ("ResBlock\n[6,6,12]", "#bcbddc"),
        ("ResBlock\n[12,12,24]", "#9e9ac8"), ("ResBlock\n[12,12,24]", "#9e9ac8"),
        ("Pool +\nFlatten", "#c7e9c0"), ("Dense 200\n+Drop", "#a1d99b"),
        ("Dense 100\n+Drop", "#a1d99b"), ("Salida\nCOS", "#fdae6b")],
    "LSTM 1D (Singh & Kasana)": [
        ("Entrada\n(L,1)", "#dfeaf4"), ("Conv1D 16\ns4 +Pool", "#9ecae1"),
        ("LSTM 128\nseq", "#fdd0a2"), ("LSTM 64", "#fdd0a2"),
        ("Dense 128\n+Drop", "#a1d99b"), ("Dense 64", "#a1d99b"),
        ("Salida\nCOS", "#fdae6b")],
    "SpectralNet (propuesto: multiescala + SE)": [
        ("Entrada\n(L,1)", "#dfeaf4"), ("Multiescala\nk3/7/15 (16)", "#fbb4b9"),
        ("SE +Pool", "#f768a1"), ("Multiescala\n(32) +SE", "#fbb4b9"),
        ("Multiescala\n(64) +SE", "#fbb4b9"), ("Global\nAvgPool", "#c7e9c0"),
        ("Dense 128\n+Drop", "#a1d99b"), ("Salida\nCOS", "#fdae6b")],
    "MLP optimizado (modelo profundo entregado)": [
        ("Entrada\nNIR\nSG1+SNV", "#dfeaf4"), ("PCA\n(50 comp.)", "#c7e9c0"),
        ("Dense 256\nBN+ELU\n+Drop 0.4", "#a1d99b"),
        ("Dense 128\nBN+ELU\n+Drop 0.4", "#a1d99b"),
        ("Dense 64\nBN+ELU\n+Drop 0.4", "#a1d99b"),
        ("Salida\nCOS", "#fdae6b")],
}
for ax, (title, blocks) in zip(axes, archs.items()):
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 1.6); ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    n = len(blocks); w = 9.8 / n - 0.15
    for i, (txt, c) in enumerate(blocks):
        x = 0.2 + i * (w + 0.15)
        caja(ax, x, 0.45, w, 0.7, txt, c)
        if i < n - 1:
            flecha(ax, x + w, 0.8, x + w + 0.15)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_arquitecturas.png", bbox_inches="tight"); plt.close()

# ---------- Fig: PCA varianza explicada (NIR) ----------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
nir = np.load(f"{CACHE}/dataset_NIR_median.npz", allow_pickle=True)
Xn, yn = nir["X"], nir["y"]
Xpp = savgol_filter(Xn, 25, 3, deriv=1, axis=1)
Xs = StandardScaler().fit_transform(Xpp)
pca = PCA(n_components=40).fit(Xs)
ev = pca.explained_variance_ratio_ * 100
cum = np.cumsum(ev)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(range(1, 21), ev[:20], color="#2c7fb8", edgecolor="black", lw=0.4)
ax[0].set_xlabel("Componente principal"); ax[0].set_ylabel("Varianza explicada (\\%)")
ax[0].set_title("Varianza por componente (NIR, SG1)"); ax[0].grid(alpha=0.3)
ax[1].plot(range(1, 41), cum, "o-", color="#d95f02", ms=3)
ax[1].axhline(99, color="gray", ls="--", lw=1, label="99\\%")
ax[1].set_xlabel("N\\textsuperscript{o} de componentes"); ax[1].set_ylabel("Varianza acumulada (\\%)")
ax[1].set_title("Varianza acumulada"); ax[1].legend(); ax[1].grid(alpha=0.3)
n99 = int(np.argmax(cum >= 99) + 1)
ax[1].annotate(f"{n99} comp. -> 99\\%", (n99, 99), (n99+3, 90),
               arrowprops=dict(arrowstyle="->"))
plt.tight_layout(); plt.savefig(f"{OUT}/fig_pca_varianza.png", bbox_inches="tight"); plt.close()

# ---------- Fig: analisis de datos para discusion ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# boxplot por cuartil de COS: dispersion espectral
q = np.quantile(yn, [0, .25, .5, .75, 1.0])
labels = ["Q1\n(bajo)", "Q2", "Q3", "Q4\n(alto)"]
groups = [Xn[(yn >= q[i]) & (yn <= q[i+1])].mean(1) for i in range(4)]
ax[0].boxplot(groups, labels=labels)
ax[0].set_ylabel("Reflectancia media NIR"); ax[0].set_xlabel("Cuartil de COS")
ax[0].set_title("Reflectancia media por nivel de COS"); ax[0].grid(alpha=0.3)
# relacion reflectancia media vs COS (efecto albedo)
ax[1].scatter(yn, Xn.mean(1), s=10, alpha=0.5, color="#2c7fb8")
r = np.corrcoef(yn, Xn.mean(1))[0, 1]
ax[1].set_xlabel("COS (\\%)"); ax[1].set_ylabel("Reflectancia media NIR")
ax[1].set_title(f"Efecto albedo: COS vs reflectancia (r={r:.2f})"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_analisis_datos.png", bbox_inches="tight"); plt.close()

print("Figuras adicionales generadas:")
for f in ["fig_arquitecturas.png", "fig_pca_varianza.png", "fig_analisis_datos.png"]:
    print("  ", f, "OK" if os.path.exists(f"{OUT}/{f}") else "FALTA")
print(f"PCA: {n99} componentes para 99% de varianza")
