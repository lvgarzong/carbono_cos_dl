"""
run_experiments.py
==================
Punto de entrada principal. Ejecuta la rejilla de experimentos definida en
config.py y genera el reporte HTML final con el analisis de las contribuciones.

Uso basico (parametros desde config.py):
    python run_experiments.py

Uso con overrides rapidos (utiles para pruebas):
    python run_experiments.py --epochs 5 --seeds 0 --models SpectralNet --ranges VISNIR --preprocess raw
    python run_experiments.py --smoke      # prueba rapida de humo (1 combo, pocas epocas)

Despues de correr, abre results/reporte_final.html (o enviamelo) para interpretar
los resultados y proponer mejoras.
"""
from __future__ import annotations
import argparse

import config
from src.logging_utils import get_logger
from src.experiment_runner import run_grid
from src.report import generate_report
from src.wavelength_analysis import analyze_best


def parse_args():
    p = argparse.ArgumentParser(description="Experimentos COS deep learning")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--ranges", nargs="+", default=None)
    p.add_argument("--preprocess", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--aggregate", default=None, choices=["median", "mean", "replicas"],
                   help="median/mean = 1 firma por muestra; replicas = augmentation")
    p.add_argument("--no-cache", action="store_true",
                   help="recalcular datasets (ignorar cache .npz)")
    p.add_argument("--smoke", action="store_true",
                   help="prueba rapida: 1 modelo, 1 rango, 1 prep, 1 semilla, 5 epocas")
    return p.parse_args()


def main():
    args = parse_args()
    logger = get_logger("cos", log_dir=config.LOGS_DIR)

    if args.smoke:
        models, ranges, preprocess, seeds, epochs = (
            ["SpectralNet"], ["NIR"], ["raw"], [0], 5)
        logger.info(">>> MODO SMOKE TEST <<<")
    else:
        models, ranges, preprocess, seeds, epochs = (
            args.models, args.ranges, args.preprocess, args.seeds, args.epochs)

    df = run_grid(models=models, ranges=ranges, preprocess=preprocess,
                  seeds=seeds, epochs=epochs, logger=logger,
                  use_cache=not args.no_cache, aggregate=args.aggregate)

    if (df.get("status") == "ok").any():
        try:
            analyze_best(logger=logger)        # importancia de bandas del mejor modelo
        except Exception as e:
            logger.warning("Analisis de bandas omitido: %s", e)
        out = generate_report(logger=logger)
        logger.info("LISTO. Abre el reporte: %s", out)
    else:
        logger.error("Ningun experimento exitoso; revisa el log para el traceback.")


if __name__ == "__main__":
    main()
