# Estimación de Carbono Orgánico del Suelo (COS) con aprendizaje profundo

Código de la tesis de maestría *«Desarrollo de un modelo de estimación de carbono
orgánico del suelo mediante el análisis de firmas espectrales y la implementación de
técnicas de aprendizaje profundo»* (Universidad Nacional de Colombia).

Estima el **carbono orgánico del suelo (COS)** a partir de **firmas espectrales
VIS-NIR** de suelos de cultivos de cítricos, comparando arquitecturas de aprendizaje
profundo (CNN, LSTM, MLP, una arquitectura propia llamada *SpectralNet*) con métodos
quimiométricos de referencia (PLSR, SVR, Random Forest, meta-ensambles). Está escrito
sobre **TensorFlow 2.16+ / Keras 3** y **scikit-learn**, sin dependencias obsoletas.

## Resultado principal (en una línea)

Bajo un protocolo de validación riguroso (partición **por muestra** + validación
cruzada de 5 particiones + métricas en unidades reales de COS), el mejor modelo de
aprendizaje profundo es un **MLP optimizado** (NIR, SG1+SNV, PCA-50) con
**R² ≈ 0.64** y **RPIQ = 2.30**, estadísticamente equivalente al mejor método
quimiométrico (SVR-RBF / meta-ensamble, R² ≈ 0.65). Todos los modelos convergen a
un **techo de R² ≈ 0.68–0.75** impuesto por el tamaño del conjunto (~780 muestras) y
la relación señal-ruido, no por la arquitectura. El repositorio permite reproducir
esos experimentos de principio a fin.

> Nota histórica: la propuesta partió de una meta de R² ≥ 0.80. Uno de los aportes
> del trabajo fue mostrar, con validación estricta, que esa meta no es alcanzable con
> este conjunto de datos, y caracterizar por qué. Las cifras aquí reportadas son
> deliberadamente honestas y reproducibles.

---

## 1. Datos (no incluidos en el repositorio)

Los espectros crudos y la bitácora de laboratorio **no se versionan** (son pesados y
de origen experimental). El código los lee desde una ruta configurable:

- Define la variable de entorno `COS_DATA_ROOT` con la carpeta que contiene los datos, o
- edita `DATA_ROOT` en [`config.py`](config.py).

Estructura esperada de esa carpeta (ver `config.py`):

```
<COS_DATA_ROOT>/
├── ACTUAL-Bitacora muestras -graficos.xlsx     # etiquetas de laboratorio (T_Dicro = COS %)
└── datos  VIS NIR  SOC/112_Firmas/             # (ojo: doble espacio en el nombre)
    ├── VIS/   *.mat   (2048 bandas, ~112 réplicas por muestra)
    └── NIR/   *.mat   (512 bandas,  ~112 réplicas por muestra)
```

Los datos de la tesis pueden solicitarse a la autora. Para el experimento de
transferencia se usa además **LUCAS 2015 Topsoil** (público, con registro):
<https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data>.

```bash
# Ejemplo: preparar el env var en Windows PowerShell
$env:COS_DATA_ROOT = "D:\ruta\a\los\datos"
```

---

## 2. Instalación

```bash
git clone [<url-del-repositorio>](https://github.com/lvgarzong/carbono_cos_dl.git)
cd carbono_cos_dl
python -m venv .venv
.venv\Scripts\activate          # Windows   (source .venv/bin/activate en Linux/Mac)
pip install -r requirements.txt
```

Probado con **Python 3.11**. Todos los experimentos corren en **CPU** (no requiere GPU).

---

## 3. Estructura del repositorio

```
carbono_cos_dl/
├── config.py                  # Toda la configuración: rutas, rejilla de experimentos, hiperparámetros
├── requirements.txt
├── src/
│   ├── data_loader.py         # Carga VIS/NIR (.mat) + etiquetas + fusión + cache
│   ├── preprocessing.py       # SNV, MSC, Savitzky-Golay (SG0/1/2), escaladores
│   ├── models.py              # Arquitecturas: LucasVGG16, LucasResNet16, LSTM, SpectralNet, MLP, MLP_improved…
│   ├── train.py               # Entrena un experimento (early stopping, ReduceLR)
│   ├── experiment_runner.py   # Ejecuta la rejilla completa, robusta a fallos
│   ├── metrics.py             # R², RMSE, MAE, RPD, RPIQ (en % reales de COS)
│   ├── band_selection.py      # Selección de bandas por importancia VIP de PLS
│   ├── feature_importance.py  # Saliencia por gradiente + importancia por permutación
│   ├── wavelength_analysis.py # Bandas más influyentes del mejor modelo
│   ├── report.py              # Reporte HTML interactivo
│   └── logging_utils.py       # Logging a consola + archivo
│
├── run_experiments.py         # ENTRYPOINT principal: rejilla de experimentos + reporte
├── run_baselines.py           # Métodos quimiométricos de referencia (PLSR, SVR, RF)
├── run_cv_dl.py               # Modelos profundos bajo validación cruzada 5-fold (comparación justa)
├── run_best_dl.py             # Entrena el modelo entregado (MLP optimizado) + figuras OOF
├── run_explora.py … run_explora4.py  # Búsqueda sistemática de hiperparámetros del MLP
├── run_ensemble.py            # Ensambles + selección de bandas
├── run_transfer_lucas.py      # Transferencia de aprendizaje desde LUCAS
├── prep_lucas2015.py / prepare_lucas.py   # Preparación/homogeneización de LUCAS
├── gen_figuras_tesis*.py      # Genera las figuras del documento de tesis
├── gen_residuos_mlp.py        # Figura de análisis de residuos del modelo entregado
├── analyze_results.py         # Regenera el reporte sin reentrenar
│
├── cache/    results/    logs/   # Autogenerados (excluidos del repo por .gitignore)
└── docs/flujo_actual.html        # Análisis gráfico del código original
```

---

## 4. Cómo reproducir los resultados

El orden lógico para reproducir los hallazgos de la tesis:

```bash
# 0) Prueba de humo (1 combo, pocas épocas) — verifica que todo carga bien
python run_experiments.py --smoke

# 1) Rejilla principal de experimentos (arquitecturas × rangos × preprocesamientos)
#    -> results/all_results.csv  y  results/reporte_final.html
python run_experiments.py

# 2) Métodos quimiométricos de referencia (PLSR, SVR, RF, meta-ensamble)
python run_baselines.py

# 3) Comparación JUSTA: modelos profundos bajo el MISMO protocolo (5-fold CV)
python run_cv_dl.py

# 4) Modelo entregado: MLP optimizado (NIR, SG1) con predicciones OOF + figuras
python run_best_dl.py

# 5) Búsqueda sistemática de la arquitectura/hiperparámetros del MLP
python run_explora.py            # (y run_explora2/3/4.py)

# 6) Experimentos adicionales: ensamble y selección de bandas
python run_ensemble.py --range NIR --preprocess sg1 --models MLP --init-seeds 0 42 707
python run_ensemble.py --range NIR --preprocess sg1 --models MLP --select-bands --top-k 150

# 7) Transferencia de aprendizaje desde LUCAS (requiere LUCAS preparado, ver §1)
python prepare_lucas.py --spectra LUCAS_Spectra.csv --properties LUCAS_props.csv --max-oc 120
python run_transfer_lucas.py --lucas lucas_prepared.npz --model LucasResNet16 --range NIR
```

Overrides útiles de `run_experiments.py` (ver `--help`):

```bash
# Solo un modelo/rango/preprocesamiento, una semilla, menos épocas
python run_experiments.py --models MLP --ranges NIR --preprocess sg1 --seeds 0 --epochs 100
# Aumento de datos por réplicas (cada escaneo como ejemplo, split por muestra)
python run_experiments.py --models MLP --ranges NIR --preprocess sg1 --aggregate replicas
```

El reporte final interactivo queda en **`results/reporte_final.html`**.

---

## 5. Modelos disponibles (`src/models.py`)

| Nombre | Tipo | Origen |
|---|---|---|
| `LucasVGG16` | CNN 1D tipo VGG | Zhong et al. (2021) |
| `LucasResNet16` | ResNet 1D | Zhong et al. (2021) |
| `LSTMPaper` | CNN + LSTM 1D | Singh & Kasana (2019) |
| `MLP` / `MLP_improved` | Perceptrón multicapa | **Modelo entregado** (el `improved` es el óptimo) |
| `SpectralNet` | CNN multiescala + atención SE | **Diseño propio** (ensambla bloques de Szegedy 2015, Hu 2018, Lin 2014) |
| `SpectralResNetSE` | ResNet 1D + SE + global pooling | Diseño propio (variante) |
| `CompactCNN` | CNN 1D pequeña y regularizada | Diseño propio (para pocos datos) |

`SpectralNet` **no** proviene de un artículo previo: es una arquitectura propia que
combina componentes establecidos de la literatura (módulos multiescala tipo *Inception*,
atención *Squeeze-and-Excitation* y *global average pooling*) adaptados a firmas 1D.

---

## 6. Protocolo de validación (por qué las cifras son fiables)

- **Partición por muestra**: todas las réplicas de un suelo caen en la misma partición
  (evita la fuga por muestras correlacionadas).
- **Validación cruzada estratificada de 5 particiones** (no una sola partición).
- El escalado y el PCA se ajustan **solo con el conjunto de entrenamiento** de cada fold.
- Todas las métricas se reportan en **unidades reales de COS (%)**, no normalizadas.

---

## 7. Referencias principales

- Zhong, L., Guo, X., Xu, Z., Ding, M. (2021). *Soil properties: Their prediction and
  feature extraction from the LUCAS spectral library using deep convolutional neural
  networks.* **Geoderma 402, 115366.**
- Ward, K. J. et al. (2019); Padarian, J. et al. (2019, 2020); Hu, J. et al. (2018,
  Squeeze-and-Excitation); Szegedy, C. et al. (2015, Inception). Ver la bibliografía
  completa en el documento de tesis.
