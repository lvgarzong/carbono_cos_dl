"""
analyze_results.py
==================
Regenera el analisis de bandas y el reporte HTML a partir de resultados ya
calculados (results/all_results.csv), sin reentrenar.

Uso:
    python analyze_results.py
"""
from __future__ import annotations
import config
from src.logging_utils import get_logger
from src.report import generate_report
from src.wavelength_analysis import analyze_best


def main():
    logger = get_logger("analyze", log_dir=config.LOGS_DIR)
    try:
        analyze_best(logger=logger)
    except Exception as e:
        logger.warning("Analisis de bandas omitido: %s", e)
    out = generate_report(logger=logger)
    logger.info("Reporte: %s", out)


if __name__ == "__main__":
    main()
