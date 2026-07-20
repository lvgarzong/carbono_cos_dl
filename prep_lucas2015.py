"""
prep_lucas2015.py
=================
Prepara el conjunto LUCAS 2015 (estructura oficial ESDAC) para transferencia de
aprendizaje: une los espectros de absorbancia (archivos por pais) con el OC del
archivo de propiedades por Point_ID, filtra suelos minerales y guarda un .npz
(A=absorbancia, wl=longitudes de onda, oc=carbono organico en g/kg).
"""
import os, glob, csv, numpy as np, pandas as pd

LUCAS = r"C:\Users\lvgar\Music\COS\LUCAS"
SPECTRA_DIR = os.path.join(LUCAS, "LUCAS2015_spectra", "LUCAS2015_Soil_Spectra_EU28")
TOPSOIL = os.path.join(LUCAS, "LUCAS2015_topsoildata_20200323",
                       "LUCAS_Topsoil_2015_20200323.csv")
OUT = r"C:\Users\lvgar\Music\COS\carbono_cos_dl\lucas_prepared.npz"

N_PER_COUNTRY = 900      # muestras por pais (cap para acotar tiempo/memoria)
MAX_OC = 120.0           # g/kg: descartar suelos organicos (turbas) -> dominio mineral
MIN_OC = 2.0             # limite de deteccion

# --- 1. OC por Point_ID ---
prop = pd.read_csv(TOPSOIL, encoding="latin-1", usecols=["Point_ID", "OC"])
prop["OC"] = pd.to_numeric(prop["OC"], errors="coerce")
oc_map = dict(zip(prop["Point_ID"].astype(str), prop["OC"]))
print(f"Propiedades: {len(oc_map)} puntos con OC")

# --- 2. detectar columnas de longitud de onda (cabecera de un archivo) ---
files = sorted(glob.glob(os.path.join(SPECTRA_DIR, "spectra_*.csv")))
with open(files[0], encoding="latin-1") as f:
    header = next(csv.reader(f))
wl_cols, wls = [], []
for c in header:
    try:
        v = float(c)
    except ValueError:
        continue
    if 350 <= v <= 2550:
        wl_cols.append(c); wls.append(v)
wls = np.array(wls)
order = np.argsort(wls)
wls = wls[order]; wl_cols = [wl_cols[i] for i in order]
print(f"Bandas detectadas: {len(wls)} ({wls.min()}-{wls.max()} nm)")

# --- 3. leer cada pais, unir OC, filtrar ---
A_list, oc_list = [], []
pid_col = "PointID"
for p in files:
    try:
        df = pd.read_csv(p, encoding="latin-1", nrows=N_PER_COUNTRY,
                         usecols=[pid_col] + wl_cols)
    except Exception as e:
        print("  saltado", os.path.basename(p), e); continue
    pid = df[pid_col].astype(str).values
    oc = np.array([oc_map.get(i, np.nan) for i in pid], dtype=np.float64)
    A = df[wl_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(oc) & (oc >= MIN_OC) & (oc <= MAX_OC) & np.isfinite(A).all(1)
    if mask.sum():
        A_list.append(A[mask]); oc_list.append(oc[mask])
    print(f"  {os.path.basename(p):20} validas={int(mask.sum())}")

A = np.concatenate(A_list, 0); oc = np.concatenate(oc_list, 0)
print(f"Total tras filtrar: {len(oc)} muestras")

# --- 4. guardar ---
np.savez_compressed(OUT, A=A.astype(np.float32), wl=wls.astype(np.float32),
                    oc=oc.astype(np.float32))
print(f"Guardado {OUT}: A={A.shape}, OC {oc.min():.1f}-{oc.max():.1f} g/kg "
      f"(media {oc.mean():.1f})")
