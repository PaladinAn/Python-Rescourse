# ---------------------------------
# 自定義評價指標、目標函數
# ----------------------------------
import numpy as np
import pandas as pd

# train_x 訓練的標籤、train_y 訓練的特徵、test_x 讀取測試資料

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 將訓練資料分成，訓練資料 跟 驗證資料
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# xgboost 自定義評價指標、目標函數
# （参考）https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss

# 將特徴資料、與標籤資料，轉換成適合 xgboost 的結構
# 訓練的特徵資料、目標資料 tr_x, tr_y、驗證的特徴資料、目標資料 va_x, va_y
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)


# 自定義目標函數（實作 logloss、 等同於 xgboost 的 'binary:logistic'）
def logregobj(preds, dtrain):
    labels = dtrain.get_label()  # 取得實際值
    preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid函數
    grad = preds - labels  # 斜率
    hess = preds * (1.0 - preds)  # 二階微分値
    return grad, hess


# 自定義評価指標（誤答率）
def evalerror(preds, dtrain):
    labels = dtrain.get_label()  # 取得實際值的標籤
    return 'custom-error', float(sum(labels != (preds > 0.0))) / len(labels)


# 超參數設定
params = {'silent': 1, 'random_state': 71}
num_round = 50
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# 開始對模型進行訓練
bst = xgb.train(params, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)

# 自定義目標函數與 binary:logistic 為目標函數不同
pred_val = bst.predict(dvalid)           # 模型輸出，預測值
pred = 1.0 / (1.0 + np.exp(-pred_val))   # 輸入 Sigmoid 函數，進行轉換
logloss = log_loss(va_y, pred)
print(logloss)

# （参考）一般訓練方法，使用指定 binary:logistic為目標函數
params = {'silent': 1, 'random_state': 71, 'objective': 'binary:logistic'}
bst = xgb.train(params, dtrain, num_round, watchlist)

pred = bst.predict(dvalid)
logloss = log_loss(va_y, pred)
print(logloss)
