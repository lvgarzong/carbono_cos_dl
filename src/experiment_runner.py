"""
experiment_runner.py
====================
Ejecuta la rejilla de experimentos: MODELOS x RANGOS x PREPROCESADOS x SEMILLAS.

Guarda un CSV maestro (results/all_results.csv) que es la entrada del reporte
final. Es robusto: si un experimento falla, registra el traceback completo en el
log y continua con los demas.
"""
from __future__ import annotations
import itertools
import time
import traceback
from pathlib import Path

import pandas as pd

import config
from src.data_loader import load_dataset
from src.train import train_one


def run_grid(models=None, ranges=None, preprocess=None, seeds=None,
             epochs=None, logger=None, use_cache=True, aggregate=None) -> pd.DataFrame:
    models = models or config.MODELS
    ranges = ranges or config.RANGES
    preprocess = preprocess or config.PREPROCESS_METHODS
    seeds = seeds or config.SEEDS
    aggregate = aggregate or config.SPECTRUM_AGG

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    master_csv = config.RESULTS_DIR / "all_results.csv"

    # Acumula sobre ejecuciones previas (no sobrescribe el historico).
    prior = pd.read_csv(master_csv).to_dict("records") if master_csv.exists() else []

    # Carga (cacheada) de los datasets por rango una sola vez.
    datasets = {r: load_dataset(r, agg=aggregate, use_cache=use_cache, logger=logger)
                for r in ranges}

    combos = list(itertools.product(ranges, models, preprocess, seeds))
    total = len(combos)
    logger.info("=" * 60)
    logger.info("Iniciando rejilla: %d experimentos (agregacion=%s)", total, aggregate)
    logger.info("  modelos=%s", models)
    logger.info("  rangos=%s", ranges)
    logger.info("  preprocesados=%s", preprocess)
    logger.info("  semillas=%s  epocas=%s", seeds, epochs or config.EPOCHS)
    logger.info("=" * 60)

    records = []
    t0 = time.time()
    for k, (rng, model_name, prep, seed) in enumerate(combos, 1):
        exp_name = f"{rng}__{model_name}__{prep}__{aggregate}__seed{seed}"
        exp_dir = config.RESULTS_DIR / exp_name
        logger.info("[%d/%d] %s", k, total, exp_name)
        try:
            rec = train_one(datasets[rng], model_name, prep, seed, exp_dir,
                            logger, epochs=epochs)
            rec["exp_name"] = exp_name
            rec["aggregate"] = aggregate
            rec["status"] = "ok"
            records.append(rec)
        except Exception:
            logger.error("FALLO en %s\n%s", exp_name, traceback.format_exc())
            records.append({"exp_name": exp_name, "model": model_name,
                            "range": rng, "preprocess": prep, "seed": seed,
                            "aggregate": aggregate, "status": "error"})
        # Guardado incremental + dedup (ultimo gana) sobre el historico.
        merged = pd.DataFrame(prior + records)
        merged = merged.drop_duplicates(subset="exp_name", keep="last")
        merged.to_csv(master_csv, index=False)

    dt = time.time() - t0
    logger.info("Rejilla terminada en %.1f min. Resultados -> %s",
                dt / 60, master_csv)
    return pd.DataFrame(records)
