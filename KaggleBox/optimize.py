import numpy as np
import pandas as pd

# -----------------------------------
# 閾値的最佳化
# -----------------------------------
from sklearn.metrics import f1_score
from scipy.optimize import minimize

# 樣本數據生成的準備(10000筆樣本、機率直)
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# 隨機產生10000資料，train_y 具有以下實際值和預測值 , 都與 train_pred_prob的機率值比較，小於就是負例，大於為正，以此作為實際值的標籤
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)

# 數列範圍 0 和 1之間
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# 起始閾値 0.5、F1 為 0.722
init_threshold = 0.5
init_score = f1_score(train_y, train_pred_prob >= init_threshold)
print(init_threshold, init_score)


# 建立最適化的目標函數，最佳化的值 = x
def f1_opt(x):
    return -f1_score(train_y, train_pred_prob >= x)


# scipy.optimize 套件提供的 minimize() 中，指定 'Nelder-Mead' 演算法 來求得佳閾値
# 最佳閾値下，計算  F1，求得、0.756
result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
best_threshold = result['x'].item()
best_score = f1_score(train_y, train_pred_prob >= best_threshold)
print(best_threshold, best_score)
