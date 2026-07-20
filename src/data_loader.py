"""
data_loader.py
==============
Carga y prepara las firmas espectrales VIS / NIR y las etiquetas de COS.

Mejoras frente al pipeline original:
  * Una sola firma por muestra (mediana robusta de las ~114 replicas).
  * Recorte de bordes ruidosos (VIS_CUT / NIR_CUT) -> firmas limpias.
  * Fusion VIS-NIR por concatenacion (contribucion i).
  * Cache en .npz -> la segunda vez carga en segundos.
  * Logging de cada archivo faltante o corrupto.
  * Devuelve tambien el vector de longitudes de onda (clave para interpretar
    la importancia de bandas, contribucion ii).
"""
from __future__ import annotations
import glob
import os
import warnings
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

import config

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


@dataclass
class SpectralDataset:
    """Contenedor del dataset cargado."""
    X: np.ndarray            # (n_muestras, n_bandas) reflectancia agregada
    y: np.ndarray            # (n_muestras,) COS en %
    ids: np.ndarray          # (n_muestras,) codigo de muestra
    wavelengths: np.ndarray  # (n_bandas,) longitudes de onda en nm
    range_name: str          # "VIS" | "NIR" | "VISNIR"

    def __len__(self):
        return len(self.y)


# --------------------------------------------------------------------------
# Lectura de archivos .mat
# --------------------------------------------------------------------------
_VAR = {
    "VIS": ("Reflectancia", "wavelengths"),
    "NIR": ("ReflectanciaX", "wavelengthsX"),
}


def _read_mat(path: str, refl_var: str, wl_var: str, logger=None):
    """Lee reflectancia (n_scans, n_bandas) y longitudes de onda de un .mat.

    Soporta tanto formato HDF5 (v7.3) como .mat clasico.
    """
    try:
        with h5py.File(path, "r") as f:
            refl = np.array(f[refl_var])
            wl = np.array(f[wl_var]).ravel()
        return refl, wl
    except Exception as e1:
        try:
            d = loadmat(path)
            return d[refl_var].T, d[wl_var].ravel()
        except Exception as e2:
            if logger:
                logger.error("No se pudo leer %s | h5:%s | mat:%s", path, e1, e2)
            return None, None


def _scans(refl: np.ndarray) -> np.ndarray:
    """Devuelve los escaneos validos (descarta posibles referencias blanco/negro)."""
    return refl[2:, :] if refl.shape[0] > 4 else refl


def _aggregate(refl: np.ndarray, agg: str) -> np.ndarray:
    """Agrega las replicas (filas) en una sola firma."""
    sig = _scans(refl)
    if agg == "mean":
        out = np.nanmean(sig, axis=0)
    else:  # median (robusto, por defecto)
        out = np.nanmedian(sig, axis=0)
    return np.nan_to_num(out)


def _select_scans(refl: np.ndarray, max_scans, seed: int) -> np.ndarray:
    """Selecciona hasta max_scans escaneos (aleatorio reproducible) para augment."""
    sig = np.nan_to_num(_scans(refl))
    if max_scans is not None and sig.shape[0] > max_scans:
        rng = np.random.default_rng(seed)
        idx = rng.choice(sig.shape[0], size=max_scans, replace=False)
        sig = sig[np.sort(idx)]
    return sig


# --------------------------------------------------------------------------
# Etiquetas (bitacora de laboratorio)
# --------------------------------------------------------------------------
def load_labels(logger=None) -> pd.Series:
    """Lee la bitacora y devuelve una Serie {codigo: COS%} limpia."""
    df = pd.read_excel(
        config.BITACORA_FILE,
        sheet_name=config.BITACORA_SHEET,
        skiprows=config.BITACORA_SKIPROWS,
    )
    df = df[[config.ID_COLUMN, config.TARGET_COLUMN]].copy()
    df[config.ID_COLUMN] = pd.to_numeric(df[config.ID_COLUMN], errors="coerce")
    df[config.TARGET_COLUMN] = pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce")
    df = df.dropna()
    df[config.ID_COLUMN] = df[config.ID_COLUMN].astype(int)
    s = df.set_index(config.ID_COLUMN)[config.TARGET_COLUMN]
    s = s[~s.index.duplicated(keep="first")]
    if logger:
        logger.info("Etiquetas COS validas: %d (min=%.2f max=%.2f media=%.2f)",
                    len(s), s.min(), s.max(), s.mean())
    return s


# --------------------------------------------------------------------------
# Construccion del dataset
# --------------------------------------------------------------------------
def _list_ids(folder, suffix):
    ids = {}
    for p in glob.glob(os.path.join(str(folder), f"*{suffix}")):
        try:
            i = int(os.path.basename(p).split("_")[0])
            ids[i] = p
        except ValueError:
            continue
    return ids


def _build_single_range(range_name: str, labels: pd.Series, agg: str, logger=None):
    """Construye X, y, ids, wavelengths para un rango VIS o NIR."""
    refl_var, wl_var = _VAR[range_name]
    folder = config.VIS_DIR if range_name == "VIS" else config.NIR_DIR
    suffix = "_Vis.mat" if range_name == "VIS" else "_Nir.mat"
    cut = config.VIS_CUT if range_name == "VIS" else config.NIR_CUT

    files = _list_ids(folder, suffix)
    wl_ref = None
    rows_X, rows_y, rows_id = [], [], []

    common = sorted(set(files) & set(labels.index) - set(config.BLACKLIST))
    if logger:
        logger.info("[%s] archivos=%d, en comun con etiquetas (sin blacklist)=%d",
                    range_name, len(files), len(common))

    for i in common:
        refl, wl = _read_mat(files[i], refl_var, wl_var, logger)
        if refl is None:
            continue
        if wl_ref is None:
            wl_ref = wl[cut[0]:cut[1]]
        if agg == "replicas":
            # Cada escaneo es un ejemplo (mismo id de muestra -> mismo grupo).
            for sc in _select_scans(refl, config.REPLICA_MAX_SCANS, seed=i):
                rows_X.append(sc[cut[0]:cut[1]])
                rows_y.append(float(labels[i]))
                rows_id.append(i)
        else:
            rows_X.append(_aggregate(refl, agg)[cut[0]:cut[1]])
            rows_y.append(float(labels[i]))
            rows_id.append(i)

    X = np.asarray(rows_X, dtype=np.float32)
    y = np.asarray(rows_y, dtype=np.float32)
    ids = np.asarray(rows_id, dtype=np.int32)
    if logger:
        logger.info("[%s] dataset -> X%s y%s, bandas %.1f-%.1f nm",
                    range_name, X.shape, y.shape, wl_ref.min(), wl_ref.max())
    return X, y, ids, wl_ref


def load_dataset(range_name: str, agg: str | None = None,
                 use_cache: bool = True, logger=None) -> SpectralDataset:
    """Carga (o construye) el dataset para el rango pedido.

    range_name: "VIS", "NIR" o "VISNIR" (fusion).
    """
    agg = agg or config.SPECTRUM_AGG
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = config.CACHE_DIR / f"dataset_{range_name}_{agg}.npz"

    if use_cache and cache_file.exists():
        d = np.load(cache_file, allow_pickle=True)
        if logger:
            logger.info("Dataset %s cargado desde cache (%s muestras)",
                        range_name, d["X"].shape[0])
        return SpectralDataset(d["X"], d["y"], d["ids"], d["wavelengths"], range_name)

    labels = load_labels(logger)

    if range_name in ("VIS", "NIR"):
        X, y, ids, wl = _build_single_range(range_name, labels, agg, logger)
    elif range_name == "VISNIR":
        Xv, yv, idv, wlv = _build_single_range("VIS", labels, agg, logger)
        Xn, yn, idn, wln = _build_single_range("NIR", labels, agg, logger)
        # Agrupa filas por id de muestra (1 fila si median/mean, varias si replicas).
        from collections import defaultdict
        gv, gn = defaultdict(list), defaultdict(list)
        for k, i in enumerate(idv):
            gv[int(i)].append(k)
        for k, i in enumerate(idn):
            gn[int(i)].append(k)
        common = sorted(set(gv) & set(gn))
        rX, ry, rid = [], [], []
        for i in common:
            # Empareja escaneos VIS<->NIR por indice, hasta el minimo disponible.
            n = min(len(gv[i]), len(gn[i]))
            for j in range(n):
                rX.append(np.concatenate([Xv[gv[i][j]], Xn[gn[i][j]]]))
                ry.append(yv[gv[i][j]])
                rid.append(i)
        X = np.asarray(rX, dtype=np.float32)
        y = np.asarray(ry, dtype=np.float32)
        ids = np.asarray(rid, dtype=np.int32)
        wl = np.concatenate([wlv, wln])
        if logger:
            logger.info("[VISNIR] fusion -> X%s (VIS %d + NIR %d bandas), %d filas / %d muestras",
                        X.shape, len(wlv), len(wln), len(y), len(common))
    else:
        raise ValueError(f"range_name invalido: {range_name}")

    ds = SpectralDataset(X, y, ids, wl, range_name)
    if use_cache:
        np.savez_compressed(cache_file, X=X, y=y, ids=ids, wavelengths=wl)
        if logger:
            logger.info("Dataset %s cacheado en %s", range_name, cache_file)
    return ds


if __name__ == "__main__":
    # Smoke test manual
    from src.logging_utils import get_logger
    log = get_logger("data_test")
    for r in ["VIS", "NIR", "VISNIR"]:
        ds = load_dataset(r, use_cache=False, logger=log)
        log.info("%s -> %s muestras, %s bandas", r, len(ds), ds.X.shape[1])
