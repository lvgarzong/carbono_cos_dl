"""
prepare_lucas.py
================
Convierte los archivos crudos de la libreria espectral LUCAS al formato que
espera run_transfer_lucas.py:
    - lucas_spectra.parquet : columnas = longitudes de onda (nm), valores en ABSORBANCIA
    - lucas_oc.parquet      : columna 'OC' (carbono organico, g/kg)

------------------------------------------------------------------------------
 COMO CONSEGUIR LUCAS (gratis, requiere registro):
   1. Entra a:  https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data
      (o la version 2009: https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data)
   2. Rellena el formulario de acceso (correo institucional). Te llega un enlace.
   3. Descarga:
        a) el/los archivo(s) de ESPECTROS Vis-NIR (400-2500 nm, ~4200 columnas),
        b) el archivo de PROPIEDADES del suelo (incluye OC / carbono organico).
   4. Descomprime los .zip. Tendras CSV grandes.
------------------------------------------------------------------------------

Este script NO asume nombres de columna fijos: AUTO-DETECTA
   - las columnas de longitud de onda (cabeceras numericas en 350-2550 nm),
   - la columna de OC (busca 'oc', 'soc', 'organic'),
   - la columna de ID para unir espectros<->propiedades,
   - si los espectros vienen en reflectancia (0-1 o 0-100%) o ya en absorbancia,
     y los convierte a absorbancia A = log10(1/R).

Uso tipico:
    python prepare_lucas.py --spectra LUCAS2015_Spectra.csv \
        --properties LUCAS2015_topsoil.csv --out-dir .

    # si espectros y propiedades estan en el MISMO archivo:
    python prepare_lucas.py --spectra LUCAS_all.csv --out-dir .

    # filtrar suelos minerales (recomendado para transferir a citricos):
    python prepare_lucas.py --spectra ... --properties ... --max-oc 120
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_utils import get_logger

WL_MIN, WL_MAX = 350.0, 2550.0
OC_PAT = re.compile(r"\b(oc|soc|organic[_ ]?carbon|carbon)\b", re.I)
ID_CANDIDATES = ["pointid", "point_id", "id", "sample_id", "sampleid",
                 "lc1", "lucas_id", "ordateid", "barcode"]


def _read_any(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    # sniff separador comun en exports europeos (a veces ';')
    head = open(path, "r", encoding="latin-1").readline()
    if head.count(";") > head.count(sep):
        sep = ";"
    return pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)


def _wavelength_columns(df: pd.DataFrame):
    """Devuelve (lista_cols, lista_wl) de las columnas cuya cabecera es un nm valido."""
    cols, wls = [], []
    for c in df.columns:
        s = str(c).replace(",", ".").strip().lstrip("Xx_").replace("nm", "")
        try:
            v = float(s)
        except ValueError:
            continue
        if WL_MIN <= v <= WL_MAX:
            cols.append(c)
            wls.append(v)
    order = np.argsort(wls)
    return [cols[i] for i in order], np.array([wls[i] for i in order])


def _find_oc(df: pd.DataFrame):
    for c in df.columns:
        if OC_PAT.search(str(c)):
            return c
    return None


def _find_id(df: pd.DataFrame):
    low = {str(c).lower(): c for c in df.columns}
    for cand in ID_CANDIDATES:
        if cand in low:
            return low[cand]
    return None


def _to_absorbance(A: np.ndarray, logger) -> np.ndarray:
    """Heuristica: detecta si son reflectancia (0-1 / 0-100) o absorbancia."""
    finite = A[np.isfinite(A)]
    mx, mn = np.nanpercentile(finite, 99), np.nanpercentile(finite, 1)
    if mx > 10:                       # reflectancia en % (0-100)
        logger.info("Espectros detectados como reflectancia %% (0-100) -> absorbancia")
        R = np.clip(A / 100.0, 1e-4, 1.0)
        return np.log10(1.0 / R)
    if mx <= 1.5 and mn >= 0:          # reflectancia fraccion (0-1)
        logger.info("Espectros detectados como reflectancia (0-1) -> absorbancia")
        R = np.clip(A, 1e-4, 1.0)
        return np.log10(1.0 / R)
    logger.info("Espectros detectados como absorbancia (se usan tal cual)")
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", required=True, help="CSV/parquet con espectros")
    ap.add_argument("--properties", default=None,
                    help="CSV con propiedades (OC). Omitir si OC esta en --spectra")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-oc", type=float, default=None,
                    help="descartar OC > este valor (g/kg). Util: 120 (suelos minerales)")
    ap.add_argument("--assume", choices=["auto", "reflectance", "reflectance100",
                                         "absorbance"], default="auto")
    args = ap.parse_args()

    logger = get_logger("prep_lucas", log_dir="logs")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Leyendo espectros: %s", args.spectra)
    spec = _read_any(args.spectra)
    wl_cols, wls = _wavelength_columns(spec)
    if not wl_cols:
        raise SystemExit("No se detectaron columnas de longitud de onda (350-2550 nm). "
                         "Revisa el archivo o sus cabeceras.")
    logger.info("Detectadas %d bandas: %.1f-%.1f nm", len(wls), wls.min(), wls.max())

    A = spec[wl_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    # --- OC y join ---
    if args.properties:
        props = _read_any(args.properties)
        oc_col = _find_oc(props)
        sid_s = _find_id(spec)
        sid_p = _find_id(props)
        if oc_col is None:
            raise SystemExit("No se encontro columna de OC en --properties.")
        if sid_s and sid_p:
            logger.info("Uniendo por ID: espectros[%s] <-> props[%s]", sid_s, sid_p)
            key = spec[sid_s].astype(str).values
            mp = dict(zip(props[sid_p].astype(str), pd.to_numeric(props[oc_col], errors="coerce")))
            oc = np.array([mp.get(k, np.nan) for k in key], dtype=np.float64)
        else:
            logger.warning("Sin columna ID comun; se asume MISMO orden de filas.")
            n = min(len(spec), len(props))
            A = A[:n]
            oc = pd.to_numeric(props[oc_col], errors="coerce").to_numpy()[:n]
    else:
        oc_col = _find_oc(spec)
        if oc_col is None:
            raise SystemExit("No hay --properties y no se hallo OC en --spectra.")
        logger.info("OC tomado de la misma tabla de espectros: %s", oc_col)
        oc = pd.to_numeric(spec[oc_col], errors="coerce").to_numpy(dtype=np.float64)

    # --- Conversion a absorbancia ---
    if args.assume == "absorbance":
        pass
    elif args.assume == "reflectance":
        A = np.log10(1.0 / np.clip(A, 1e-4, 1.0))
    elif args.assume == "reflectance100":
        A = np.log10(1.0 / np.clip(A / 100.0, 1e-4, 1.0))
    else:
        A = _to_absorbance(A, logger)

    # --- Limpieza ---
    mask = np.isfinite(oc)
    if args.max_oc is not None:
        mask &= (oc <= args.max_oc)
    mask &= np.isfinite(A).all(axis=1)
    A, oc = A[mask], oc[mask]
    logger.info("Muestras validas tras limpieza: %d (OC %.1f-%.1f g/kg, media %.1f)",
                len(oc), np.nanmin(oc), np.nanmax(oc), np.nanmean(oc))

    # --- Guardado (.npz: sin dependencias extra, guarda espectros+wl+OC juntos) ---
    npz_path = out / "lucas_prepared.npz"
    np.savez_compressed(npz_path, A=A.astype(np.float32),
                        wl=wls.astype(np.float32), oc=oc.astype(np.float32))
    logger.info("Guardado: %s  (A=%s, %d bandas, %d muestras)",
                npz_path, A.shape, len(wls), len(oc))
    logger.info("Listo. Ahora corre:")
    logger.info("  python run_transfer_lucas.py --lucas %s --model LucasResNet16 --range NIR",
                npz_path)


if __name__ == "__main__":
    main()
