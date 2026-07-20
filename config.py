"""
config.py
=========
Configuracion central del proyecto de estimacion de Carbono Organico del Suelo (COS).

Todo lo que quieras variar entre ejecuciones se controla desde aqui:
  - rutas a los datos
  - rango espectral (VIS / NIR / VISNIR  -> contribucion i: fusion de datos)
  - preprocesamiento espectral          (-> contribucion i: efecto del preprocesado)
  - modelos a comparar                  (-> contribucion iii: comparacion de modelos)
  - hiperparametros de entrenamiento
  - rejilla (grid) de experimentos

Editar este archivo (o pasar overrides por linea de comandos) es la forma
recomendada de lanzar "muchos casos" para despues analizar los resultados.
"""
from __future__ import annotations
import os
from pathlib import Path

# --------------------------------------------------------------------------
# 1. RUTAS
# --------------------------------------------------------------------------
# Carpeta con los datos originales (firmas VIS/NIR + bitacora de laboratorio).
# Ajusta DATA_ROOT si mueves los datos.
DATA_ROOT = Path(
    os.environ.get(
        "COS_DATA_ROOT",
        r"C:\Users\lvgar\Music\COS\carbono_machine_learning-main"
        r"\carbono_machine_learning-main-ORIGINAL\data",
    )
)

BITACORA_FILE = DATA_ROOT / "ACTUAL-Bitacora muestras -graficos.xlsx"
SPECTRA_DIR = DATA_ROOT / "datos  VIS NIR  SOC" / "112_Firmas"   # ojo: doble espacio
VIS_DIR = SPECTRA_DIR / "VIS"
NIR_DIR = SPECTRA_DIR / "NIR"

# Carpetas de salida (se crean solas)
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"

# --------------------------------------------------------------------------
# 2. DATOS Y ETIQUETAS
# --------------------------------------------------------------------------
TARGET_COLUMN = "T_Dicro"     # Carbono organico (% por metodo Walkley-Black / dicromato)
ID_COLUMN = "Codigo"
BITACORA_SHEET = "Report"
BITACORA_SKIPROWS = 1

# Recorte de bandas ruidosas en los bordes (indices sobre el vector original).
# VIS original: 2048 bandas 338-1022 nm  -> [163:1665) deja 401-911 nm (1502 bandas)
# NIR original:  512 bandas 901-2514 nm  -> [5:491)    deja 917-2449 nm (486 bandas)
VIS_CUT = (163, 1665)
NIR_CUT = (5, 491)

# Como tratar las ~114 replicas (escaneos) de cada muestra:
#   "median"   -> una firma por muestra, mediana robusta (limpio, pocas muestras)
#   "mean"     -> una firma por muestra, promedio
#   "replicas" -> CADA escaneo es un ejemplo de entrenamiento (data augmentation).
#                 Multiplica los datos ~x20-100. El split es POR MUESTRA (sin fuga)
#                 y la evaluacion promedia las predicciones por muestra.
SPECTRUM_AGG = "median"

# En modo "replicas": maximo de escaneos por muestra (None = todos ~112).
# Limitar acelera el entrenamiento sin perder mucha diversidad.
REPLICA_MAX_SCANS = 20

# Muestras descartadas (ruido / errores de laboratorio) heredadas del estudio original.
BLACKLIST = [
    8, 21, 23, 24, 35, 279, 285, 291, 309, 344, 373, 418, 486, 492, 503, 513,
    519, 522, 523, 530, 537, 565, 589, 592, 593, 594, 596, 597, 598, 602, 603,
    606, 612, 617, 620, 622, 627, 628, 634,
]

# --------------------------------------------------------------------------
# 3. PREPROCESAMIENTO ESPECTRAL  (contribucion i)
# --------------------------------------------------------------------------
# Metodos disponibles (ver src/preprocessing.py):
#   "raw"      -> reflectancia sin transformar
#   "snv"      -> Standard Normal Variate (corrige scattering)
#   "sg0"      -> Savitzky-Golay suavizado (orden 0)
#   "sg1"      -> 1a derivada Savitzky-Golay
#   "sg2"      -> 2a derivada Savitzky-Golay
#   "sg1_snv"  -> SG1 + SNV
#   "msc"      -> Multiplicative Scatter Correction
PREPROCESS_METHODS = ["raw", "snv", "sg1"]

SG_WINDOW = 25         # ventana Savitzky-Golay (impar)
SG_POLYORDER = 3

# Normalizacion de la entrada al modelo (sobre cada banda, ajustada en train):
#   "zscore" (recomendado) | "minmax" | "none"
FEATURE_SCALING = "zscore"

# Normalizacion del objetivo (COS). Se invierte para reportar en % reales.
#   "zscore" (recomendado) | "minmax" | "none"
TARGET_SCALING = "zscore"

# --------------------------------------------------------------------------
# 4. RANGO ESPECTRAL  (contribucion i: fusion VIS-NIR)
# --------------------------------------------------------------------------
# "VIS" | "NIR" | "VISNIR" (fusion por concatenacion)
RANGES = ["VIS", "NIR", "VISNIR"]

# --------------------------------------------------------------------------
# 5. MODELOS  (contribucion iii)
# --------------------------------------------------------------------------
# Nombres validos (ver src/models.py -> get_model):
#   "LucasVGG16"      -> CNN VGG 1D del paper Zhong et al. 2021
#   "LucasResNet16"   -> ResNet 1D del paper (mejor del paper)
#   "LSTMPaper"       -> LSTM 1D (Singh & Kasana 2019, referencia del paper)
#   "SpectralNet"     -> MODELO NUEVO propuesto (multi-escala + atencion SE)
#   "CNN_LSTM"        -> hibrido CNN + LSTM
#   "SpectralResNetSE"-> ResNet 1D con bloques Squeeze-Excitation (propuesta avanzada)
MODELS = ["LucasResNet16", "LucasVGG16", "LSTMPaper", "SpectralNet"]

# --------------------------------------------------------------------------
# 6. ENTRENAMIENTO
# --------------------------------------------------------------------------
TEST_RATIO = 0.15
VAL_RATIO = 0.15
STRATIFY_BINS = 5          # estratifica el split por quantiles de COS

BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 5e-4
OPTIMIZER = "nadam"        # "nadam" | "adam"
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 40
REDUCE_LR_PATIENCE = 15
L2_REG = 1e-4

# Semillas para repetir cada experimento (robustez estadistica).
SEEDS = [0, 42, 707]

# --------------------------------------------------------------------------
# 7. GRID DE EXPERIMENTOS
# --------------------------------------------------------------------------
# El runner hace producto cartesiano de: MODELS x RANGES x PREPROCESS_METHODS x SEEDS
# Numero de experimentos = len(MODELS)*len(RANGES)*len(PREPROCESS_METHODS)*len(SEEDS)
# Con los valores por defecto: 4 * 3 * 3 * 3 = 108 experimentos.

# Objetivo de desempeno (R2) que se quiere superar.
TARGET_R2 = 0.80
