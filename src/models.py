"""
models.py
=========
Modelos de deep learning para regresion de COS a partir de firmas 1D.

Todos usan la API funcional de Keras 3 (TF 2.16+), entrada (L, 1):
  - LucasVGG16      : CNN VGG 1D                (Zhong et al. 2021, paper)
  - LucasResNet16   : ResNet 1D con res-blocks  (Zhong et al. 2021, paper, mejor)
  - LSTMPaper       : LSTM 1D                    (Singh & Kasana 2019, referencia)
  - SpectralNet     : PROPUESTO  -> CNN multi-escala + atencion Squeeze-Excitation
  - CNN_LSTM        : hibrido CNN + BiLSTM
  - SpectralResNetSE: PROPUESTO avanzado -> ResNet 1D + SE + global pooling

Diseno fiel al paper: activacion tanh en capas internas, salida lineal para
regresion (mejor que sigmoid cuando el objetivo esta estandarizado), optimizador
Nadam, dropout en las capas densas.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input

import config


def _reg():
    return regularizers.l2(config.L2_REG) if config.L2_REG else None


# ==========================================================================
#  LucasVGGNet-16 (paper)  -- 13 conv + 5 pool + 3 densas
# ==========================================================================
def build_lucas_vgg16(input_len: int, dropout: float = 0.3) -> Model:
    inp = Input(shape=(input_len, 1), name="spectrum")

    def conv(x, f):
        return layers.Conv1D(f, 3, padding="same", activation="tanh",
                             kernel_initializer="glorot_uniform",
                             kernel_regularizer=_reg())(x)

    x = conv(inp, 6); x = conv(x, 6); x = layers.MaxPooling1D(2)(x)
    x = conv(x, 12); x = conv(x, 12); x = layers.MaxPooling1D(2)(x)
    x = conv(x, 24); x = conv(x, 24); x = conv(x, 24); x = layers.MaxPooling1D(2)(x)
    x = conv(x, 48); x = conv(x, 48); x = conv(x, 48); x = layers.MaxPooling1D(2)(x)
    x = conv(x, 48); x = conv(x, 48); x = conv(x, 48); x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(200, activation="tanh")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(100, activation="tanh")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear", name="cos")(x)
    return Model(inp, out, name="LucasVGG16")


# ==========================================================================
#  LucasResNet-16 (paper)  -- 4 res-blocks + 2 pool + 3 densas
# ==========================================================================
def _res_block(x, filters, with_proj=False):
    f1, f2, f3 = filters
    shortcut = x
    y = layers.Conv1D(f1, 1, padding="same", activation="tanh", kernel_regularizer=_reg())(x)
    y = layers.Conv1D(f2, 3, padding="same", activation="tanh", kernel_regularizer=_reg())(y)
    y = layers.Conv1D(f3, 1, padding="same", activation="tanh", kernel_regularizer=_reg())(y)
    if with_proj or shortcut.shape[-1] != f3:
        shortcut = layers.Conv1D(f3, 1, padding="same", kernel_regularizer=_reg())(shortcut)
    return layers.Add()([y, shortcut])


def build_lucas_resnet16(input_len: int, dropout: float = 0.3) -> Model:
    inp = Input(shape=(input_len, 1), name="spectrum")
    x = layers.Conv1D(6, 7, strides=2, padding="same", activation="tanh",
                      kernel_regularizer=_reg())(inp)
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = _res_block(x, [6, 6, 12], with_proj=True)
    x = _res_block(x, [6, 6, 12])
    x = _res_block(x, [12, 12, 24], with_proj=True)
    x = _res_block(x, [12, 12, 24])
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(200, activation="tanh")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(100, activation="tanh")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear", name="cos")(x)
    return Model(inp, out, name="LucasResNet16")


# ==========================================================================
#  LSTM 1D (Singh & Kasana 2019, referencia del paper)
# ==========================================================================
def build_lstm_paper(input_len: int, dropout: float = 0.1) -> Model:
    inp = Input(shape=(input_len, 1), name="spectrum")
    # Submuestreo conv para que la secuencia sea manejable por el LSTM.
    x = layers.Conv1D(16, 7, strides=4, padding="same", activation="tanh")(inp)
    x = layers.MaxPooling1D(4)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=dropout)(x)
    x = layers.LSTM(64, dropout=dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="cos")(x)
    return Model(inp, out, name="LSTMPaper")


# ==========================================================================
#  SpectralNet (PROPUESTO)
#  CNN multi-escala (kernels 3/7/15 capturan picos finos y bandas anchas)
#  + atencion Squeeze-Excitation + global average pooling.
# ==========================================================================
def _se_block(x, ratio=8):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(ch // ratio, 4), activation="relu")(s)
    s = layers.Dense(ch, activation="sigmoid")(s)
    s = layers.Reshape((1, ch))(s)
    return layers.Multiply()([x, s])


def _multiscale(x, f):
    a = layers.Conv1D(f, 3, padding="same", activation="relu", kernel_regularizer=_reg())(x)
    b = layers.Conv1D(f, 7, padding="same", activation="relu", kernel_regularizer=_reg())(x)
    c = layers.Conv1D(f, 15, padding="same", activation="relu", kernel_regularizer=_reg())(x)
    x = layers.Concatenate()([a, b, c])
    x = layers.BatchNormalization()(x)
    x = _se_block(x)
    return x


def build_spectralnet(input_len: int, dropout: float = 0.3) -> Model:
    inp = Input(shape=(input_len, 1), name="spectrum")
    x = _multiscale(inp, 16); x = layers.MaxPooling1D(2)(x)
    x = _multiscale(x, 32); x = layers.MaxPooling1D(2)(x)
    x = _multiscale(x, 64); x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=_reg())(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear", name="cos")(x)
    return Model(inp, out, name="SpectralNet")


# ==========================================================================
#  CNN + BiLSTM hibrido
# ==========================================================================
def build_cnn_lstm(input_len: int, dropout: float = 0.3) -> Model:
    inp = Input(shape=(input_len, 1), name="spectrum")
    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=dropout))(x)
    x = layers.Bidirectional(layers.LSTM(32, dropout=dropout))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear", name="cos")(x)
    return Model(inp, out, name="CNN_LSTM")


# ==========================================================================
#  SpectralResNetSE (PROPUESTO avanzado)
# ==========================================================================
def _res_se_block(x, f, k=5):
    shortcut = x
    y = layers.Conv1D(f, k, padding="same", kernel_regularizer=_reg())(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(f, k, padding="same", kernel_regularizer=_reg())(y)
    y = layers.BatchNormalization()(y)
    y = _se_block(y)
    if shortcut.shape[-1] != f:
        shortcut = layers.Conv1D(f, 1, padding="same")(shortcut)
    y = layers.Add()([y, shortcut])
    return layers.Activation("relu")(y)


def build_spectral_resnet_se(input_len: int, dropout: float = 0.3) -> Model:
    inp = Input(shape=(input_len, 1), name="spectrum")
    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = _res_se_block(x, 32); x = layers.MaxPooling1D(2)(x)
    x = _res_se_block(x, 64); x = layers.MaxPooling1D(2)(x)
    x = _res_se_block(x, 128); x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear", name="cos")(x)
    return Model(inp, out, name="SpectralResNetSE")


# ==========================================================================
#  CompactCNN (PROPUESTO para datasets pequenos)
#  CNN 1D pequena y fuertemente regularizada (BatchNorm + dropout + L2 + GAP).
#  Pocos parametros -> generaliza mejor con ~550 muestras que VGG/ResNet.
# ==========================================================================
def build_mlp(input_len: int, dropout: float = 0.3) -> Model:
    """Perceptron multicapa (MLP) sobre el espectro aplanado.

    Modelo de referencia/control (como en el borrador previo). Sin convoluciones:
    capas densas sobre el vector espectral completo.
    """
    inp = Input(shape=(input_len, 1), name="spectrum")
    x = layers.Flatten()(inp)
    x = layers.Dense(256, activation="relu", kernel_regularizer=_reg())(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=_reg())(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="cos")(x)
    return Model(inp, out, name="MLP")


def build_mlp_improved(input_len: int, dropout: float = 0.4) -> Model:
    """MLP mejorado: BatchNorm + L2 + dropout alto + capas decrecientes.

    Diseñado para reducir el sobreajuste del MLP simple sobre espectros de alta
    dimension y pocas muestras.
    """
    inp = Input(shape=(input_len, 1), name="spectrum")
    x = layers.Flatten()(inp)
    for units in (128, 64, 32):
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(1e-3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear", name="cos")(x)
    return Model(inp, out, name="MLP_improved")


def build_compact_cnn(input_len: int, dropout: float = 0.3) -> Model:
    inp = Input(shape=(input_len, 1), name="spectrum")

    def block(x, f, k):
        x = layers.Conv1D(f, k, padding="same", kernel_regularizer=_reg())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return layers.MaxPooling1D(2)(x)

    x = block(inp, 16, 7)
    x = block(x, 32, 5)
    x = block(x, 64, 3)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=_reg())(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear", name="cos")(x)
    return Model(inp, out, name="CompactCNN")


# ==========================================================================
#  Fabrica
# ==========================================================================
_BUILDERS = {
    "LucasVGG16": build_lucas_vgg16,
    "LucasResNet16": build_lucas_resnet16,
    "LSTMPaper": build_lstm_paper,
    "SpectralNet": build_spectralnet,
    "CNN_LSTM": build_cnn_lstm,
    "SpectralResNetSE": build_spectral_resnet_se,
    "CompactCNN": build_compact_cnn,
    "MLP": build_mlp,
    "MLP_improved": build_mlp_improved,
}


def get_model(name: str, input_len: int, dropout: float | None = None) -> Model:
    if name not in _BUILDERS:
        raise ValueError(f"Modelo desconocido: {name}. Validos: {list(_BUILDERS)}")
    dropout = config.DROPOUT if dropout is None else dropout
    return _BUILDERS[name](input_len, dropout)


def available_models():
    return list(_BUILDERS)
