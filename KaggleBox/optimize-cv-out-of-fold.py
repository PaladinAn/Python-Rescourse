import numpy as np
import pandas as pd

# -----------------------------------
# out-of-fold 閾値最適化
# out-of-fold 將訓練資料分割，一份作為預測，一份做為訓練使用
# -----------------------------------
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# 隨機資料生成準備
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# 實際値與預測値，train_y, train_pred_prob
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# 交叉驗證範圍內，閾値求得
thresholds = []
scores_tr = []
scores_va = []

kf = KFold(n_splits=4, random_state=71, shuffle=True)            # 訓練資料，打散後，切割成 4 組
for i, (tr_idx, va_idx) in enumerate(kf.split(train_pred_prob)):
    tr_pred_prob, va_pred_prob = train_pred_prob[tr_idx], train_pred_prob[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 最適化目標函數設定
    def f1_opt(x):
        return -f1_score(tr_y, tr_pred_prob >= x)

    # 訓練的資料閾値最適化、使用驗證資料來評價
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    threshold = result['x'].item()                                       # 訓練資料中，找到的最佳化閾値
    score_tr = f1_score(tr_y, tr_pred_prob >= threshold)                 # 閾値計算訓練資料的 f1
    score_va = f1_score(va_y, va_pred_prob >= threshold)                 # 閾値計算驗證資料的 f1
    print(threshold, score_tr, score_va)

    thresholds.append(threshold)
    scores_tr.append(score_tr)
    scores_va.append(score_va)

# 各 fold 最佳化閾値平均，在使用於測試資料
threshold_test = np.mean(thresholds)
print(threshold_test)
