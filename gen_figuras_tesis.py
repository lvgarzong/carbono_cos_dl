"""
gen_figuras_tesis.py
====================
Genera las figuras del documento de tesis (PNG) a partir de los datos y
resultados reales, y las guarda en la carpeta 00Figuras de la plantilla UNAL.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

OUT = r"C:\Users\lvgar\Music\COS\Plantilla_Tesis_Trabajo_Final_UNAL_2023__1_\00Figuras"
os.makedirs(OUT, exist_ok=True)
CACHE = "cache"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150})


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)


nir = np.load(f"{CACHE}/dataset_NIR_median.npz", allow_pickle=True)
vis = np.load(f"{CACHE}/dataset_VIS_median.npz", allow_pickle=True)
Xn, yn, wn = nir["X"], nir["y"], nir["wavelengths"]
Xv, yv, wv = vis["X"], vis["y"], vis["wavelengths"]

# ---- Fig 1: distribucion COS ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(yn, bins=30, color="#2c7fb8", edgecolor="black", lw=0.4, alpha=0.85)
ax[0].axvline(yn.mean(), color="red", lw=2, label=f"Media = {yn.mean():.2f}\\%")
ax[0].axvline(np.median(yn), color="orange", lw=2, ls="--", label=f"Mediana = {np.median(yn):.2f}\\%")
ax[0].set_xlabel("COS (\\%)"); ax[0].set_ylabel("Frecuencia")
ax[0].set_title("Distribucion de COS"); ax[0].legend()
ax[1].hist(np.log(yn), bins=30, color="#d95f02", edgecolor="black", lw=0.4, alpha=0.85)
ax[1].set_xlabel("log(COS)"); ax[1].set_ylabel("Frecuencia")
ax[1].set_title("Distribucion de log(COS)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_distribucion_cos.png", bbox_inches="tight"); plt.close()

# ---- Fig 2: espectros medios VIS y NIR ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
mv, sv = Xv.mean(0), Xv.std(0)
mn, sn = Xn.mean(0), Xn.std(0)
ax[0].plot(wv, mv, color="#1b9e77", lw=1)
ax[0].fill_between(wv, mv - sv, mv + sv, color="#1b9e77", alpha=0.25)
ax[0].set_title("Espectro medio VIS"); ax[0].set_xlabel("Longitud de onda (nm)"); ax[0].set_ylabel("Reflectancia")
ax[1].plot(wn, mn, color="#7570b3", lw=1)
ax[1].fill_between(wn, mn - sn, mn + sn, color="#7570b3", alpha=0.25)
ax[1].set_title("Espectro medio NIR"); ax[1].set_xlabel("Longitud de onda (nm)"); ax[1].set_ylabel("Reflectancia")
for a in ax:
    for b in [1400, 1900, 2200]:
        a.axvline(b, color="gray", ls=":", lw=0.8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_espectros_medios.png", bbox_inches="tight"); plt.close()

# ---- Fig 3: efecto del preprocesamiento (una muestra NIR) ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
s = Xn[0]
ax[0].plot(wn, s, color="#333"); ax[0].set_title("Cruda (reflectancia)")
ax[1].plot(wn, snv(s[None])[0], color="#2c7fb8"); ax[1].set_title("SNV")
ax[2].plot(wn, savgol_filter(s, 25, 3, deriv=1), color="#d95f02"); ax[2].set_title("Savitzky-Golay 1a derivada")
for a in ax: a.set_xlabel("Longitud de onda (nm)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_preprocesamiento.png", bbox_inches="tight"); plt.close()

# ---- Fig 4: correlacion |r| banda-COS (NIR) ----
ylog = np.log(yn)
corr = np.array([abs(np.corrcoef(savgol_filter(Xn, 25, 3, deriv=1)[:, i], ylog)[0, 1])
                 for i in range(Xn.shape[1])])
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(wn, corr, color="#2c7fb8", lw=0.9)
ax.fill_between(wn, corr, alpha=0.3, color="#2c7fb8")
ax.set_xlabel("Longitud de onda (nm)"); ax.set_ylabel("|r| con log(COS)")
ax.set_title(f"Correlacion banda-COS (NIR, SG1).  Maximo |r| = {corr.max():.3f}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_correlacion_bandas.png", bbox_inches="tight"); plt.close()

# ---- Fig 5: comparacion de modelos (evolucion honesta, CV) ----
modelos = ["ResNet 1D / BiLSTM (base)", "LucasResNet-16", "Attention-CNN (top-80)",
           "PLS-R (SG1+SNV)", "SVR-RBF", "Meta-ensamble (Ridge)"]
r2 = [0.38, 0.477, 0.615, 0.626, 0.643, 0.658]
rpd = [1.29, 1.39, 1.63, 1.63, 1.67, 1.71]
colors = ["#7fb3d5"]*3 + ["#f5b041", "#58d68d", "#2e7d32"]
fig, ax = plt.subplots(figsize=(9, 4.2))
b = ax.barh(modelos, r2, color=colors, edgecolor="black", lw=0.4)
ax.axvline(0.80, color="red", ls=":", lw=1.5, label="Objetivo R$^2$=0.80")
ax.axvline(0.68, color="gray", ls="--", lw=1.2, label="Techo estimado (~0.68-0.75)")
ax.set_xlabel("R$^2$ (validacion cruzada 5-fold)"); ax.set_title("Evolucion del desempeno por iteracion experimental")
for bar, v in zip(b, r2):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
ax.legend(loc="lower right"); ax.set_xlim(0, 0.9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_comparacion_modelos.png", bbox_inches="tight"); plt.close()

# ---- Fig 6: efecto del rango (hoy, holdout por muestra) ----
rangos = ["VIS", "NIR", "VISNIR"]
vgg = [0.164, 0.450, 0.254]
resnet = [0.080, 0.371, 0.247]
x = np.arange(3); w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(x - w/2, vgg, w, label="LucasVGG16", color="#2c7fb8", edgecolor="black", lw=0.4)
ax.bar(x + w/2, resnet, w, label="LucasResNet16", color="#d95f02", edgecolor="black", lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(rangos); ax.set_ylabel("R$^2$ (prueba, por muestra)")
ax.set_title("Efecto del rango espectral y la fusion VIS-NIR"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig_efecto_rango.png", bbox_inches="tight"); plt.close()

# ---- Fig 7: dispersion medido vs predicho (mejor modelo disponible) ----
pred_path = "results/NIR__LucasVGG16__raw__seed0/predictions.csv"
if os.path.exists(pred_path):
    p = pd.read_csv(pred_path)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(p.y_true, p.y_pred, alpha=0.6, s=18, color="#2c7fb8", edgecolor="white", lw=0.3)
    lim = [min(p.y_true.min(), p.y_pred.min()), max(p.y_true.max(), p.y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.3, label="1:1")
    from sklearn.metrics import r2_score
    r2v = r2_score(p.y_true, p.y_pred)
    ax.set_xlabel("COS medido (\\%)"); ax.set_ylabel("COS predicho (\\%)")
    ax.set_title(f"Medido vs. predicho (LucasVGG16, NIR)\\nR$^2$={r2v:.3f}")
    ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_dispersion_mejor.png", bbox_inches="tight"); plt.close()

# ---- Fig 8: importancia de longitudes de onda ----
imp_path = "results/NIR__LucasVGG16__raw__seed0/wavelength_importance.npz"
if os.path.exists(imp_path):
    d = np.load(imp_path, allow_pickle=True)
    wl, imp = d["wavelengths"], d["importance"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(wl, imp, color="#d95f02", lw=1)
    ax.fill_between(wl, imp, alpha=0.25, color="#d95f02")
    for b in [1390, 2300]:
        ax.axvline(b, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Longitud de onda (nm)"); ax.set_ylabel("Importancia (norm.)")
    ax.set_title("Importancia de longitudes de onda para COS (NIR)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_importancia_bandas.png", bbox_inches="tight"); plt.close()

# ---- Fig 9: techo R2 vs ruido del metodo de referencia ----
mean, std = yn.mean(), yn.std()
cv = np.linspace(0.01, 0.30, 100)
ceil = np.clip(1 - (cv * mean / std) ** 2, 0, 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cv * 100, ceil, color="#2c7fb8", lw=2)
ax.axhspan(0.68, 0.75, color="green", alpha=0.12, label="Techo estimado (0.68-0.75)")
ax.axhline(0.658, color="red", ls="--", lw=1.3, label="Mejor modelo (0.658)")
ax.axvspan(5, 8, color="orange", alpha=0.12, label="CV tipico dicromato (5-8\\%)")
ax.set_xlabel("CV del metodo de referencia (\\%)"); ax.set_ylabel("R$^2$ maximo teorico")
ax.set_title("Techo estadistico vs. ruido del metodo de laboratorio")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_techo_r2.png", bbox_inches="tight"); plt.close()

print("Figuras generadas en:", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print("  ", f)
