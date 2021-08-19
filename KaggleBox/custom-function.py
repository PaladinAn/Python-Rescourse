import numpy as np
import pandas as pd


# -----------------------------------
# 使用相似的自定義目標函數，評価指標，取代 MAE的最佳化 (MAE不易受極端值，影養)
# 當 斜率不連續 且 二階微分値可能是 0 ， MAE不適用
# -----------------------------------

# Fair 函數
def fair(preds, dtrain):
    x = preds - dtrain.get_labels()  # 残差取得
    c = 1.0                          # Fair函數的參數
    den = abs(x) + c                 # 計算「斜率」公式的分母
    grad = c * x / den               # 斜率
    hess = c * c / den ** 2          # 二階微分値
    return grad, hess


# Pseudo-Huber 函數
def psuedo_huber(preds, dtrain):
    d = preds - dtrain.get_labels()  # 残差取得
    delta = 1.0                      # Pseudo-Huber函數的參數
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt            # 斜率
    hess = 1 / scale / scale_sqrt    # 二階微分値
    return grad, hess
