# ---------------------------------
# 標準化的準備
# ----------------------------------
import numpy as np
import pandas as pd

# train_x 訓練函數、train_y 目標函數、test_x 測試函數

train = pd.read_csv('/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../sample-data/test_preprocessed.csv')


train_x_saved = train_x.copy()
test_x_saved = test_x.copy()



def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x


# 調整数値變數名稱
num_cols = ['age', 'height', 'weight', 'amount',
            'medical_info_a1', 'medical_info_a2', 'medical_info_a3', 'medical_info_b1']

# -----------------------------------
# 標準化
# -----------------------------------
# StandardScaler 模組
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 對上面所指定的欄位標準化
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# 標準化的資料來置換個欄位的原資料
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 要轉換資料中所的數值時，該使用訓練資料還是測試資料?
# 法一、使用訓練資料，來計算平均、標準差。以此來對[訓練+測試]資料，進行標準化
# 法二、結合訓練資料及測試資料後，計算平均、標準差。以此來對[訓練+測試]資料，進行標準化
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 法一、使用訓練資料，來計算平均、標準差。以此來對[訓練+測試]資料，進行標準化
scaler = StandardScaler()
scaler.fit(pd.concat([train_x[num_cols], test_x[num_cols]]))

# 進行標準化、置換各欄位原先的數值
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 法二、結合訓練資料及測試資料後，計算平均、標準差。以此來對[訓練+測試]資料，進行標準化
scaler_train = StandardScaler()
scaler_train.fit(train_x[num_cols])
train_x[num_cols] = scaler_train.transform(train_x[num_cols])
scaler_test = StandardScaler()
scaler_test.fit(test_x[num_cols])
test_x[num_cols] = scaler_test.transform(test_x[num_cols])

# -----------------------------------
# Min-Max 縮放方法，統一變數的尺度 (資料轉換)
# 優點：平均不會剛好是0、不太受極端質影響；線性轉換，不會改變特徵分布的形狀
# -----------------------------------
# 
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import MinMaxScaler

# 以訓練資料，定義 Min-Max 縮放
scaler = MinMaxScaler()
scaler.fit(train_x[num_cols])

# 進行標準化、置換各欄位原先的數值
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 非線性轉換
# 法1 對數轉換
# 法2 Box-Cox轉換、Yeo-Johnson 轉換
# 優點：會改變特徵分布的形狀，將偏態轉常態
# -----------------------------------
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

# 取對數
x1 = np.log(x)

# 加 1 後取對數
x2 = np.log1p(x)

# 取絶対値的對數，加上原本的符號
x3 = np.sign(x) * np.log(np.abs(x))

# -----------------------------------
# Box-Cox轉換
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------

# 只將正値的變數納入清單，轉換
# 若有缺失值，採用 (~(train_x[c] <= 0.0)).all() 方法
pos_cols = [c for c in num_cols if (train_x[c] > 0.0).all() and (test_x[c] > 0.0).all()]

from sklearn.preprocessing import PowerTransformer

# 以訓練資料，來進行 Box-Cox 轉換
pt = PowerTransformer(method='box-cox')
pt.fit(train_x[pos_cols])

# 已轉換後的資料，置換各欄位原先的數值
train_x[pos_cols] = pt.transform(train_x[pos_cols])
test_x[pos_cols] = pt.transform(test_x[pos_cols])

# -----------------------------------
# Yeo-Johnson 轉換
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------

from sklearn.preprocessing import PowerTransformer

# 以訓練資料，來進行 Yeo-Johnson 轉換
pt = PowerTransformer(method='yeo-johnson')
pt.fit(train_x[num_cols])

# 已轉換後的資料，置換各欄位原先的數值
train_x[num_cols] = pt.transform(train_x[num_cols])
test_x[num_cols] = pt.transform(test_x[num_cols])

# -----------------------------------
# 變數中，可能含有極值
# 可以設定上下限，藉此排除極值
# -----------------------------------
# clipping (上下限切割)
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
# 先以 quantile()，設定下限
# 
p01 = train_x[num_cols].quantile(0.01)   # 下限
p99 = train_x[num_cols].quantile(0.99)   # 上限

# 低於1％点以下的値，為Clipping 為下限值、99％以上的値為 clipping 上限值
train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)

# -----------------------------------
# binning (分組)
# -----------------------------------
x = [1, 7, 5, 4, 6, 3]

# 先 用 pandas 的 cut 函數 來進行切割

# 法一、 指定區間數量為 3
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0] - 顯示轉換後，的數值屬於於哪個區間

# 法二、 指定區間範圍時，（3.0以下、3.0 ~ 5.0、5.0以上）
bin_edges = [-float('inf'), 3.0, 5.0, float('inf')]
binned = pd.cut(x, bin_edges, labels=False)
print(binned)
# [0 2 1 1 2 0] - 顯示轉換後，的數值屬於於哪個區間

# -----------------------------------
# 順位轉換
# 優點：將茲瞭尺度控制在 0~1 之間
# -----------------------------------
x = [10, 20, 30, 0, 40, 40]

# 法一、 pandas 中 rank 函數 順位轉換
rank = pd.Series(x).rank()
print(rank.values)
# 從 1 開始、相同數值，的將排序，以平均值 顯示
# [2. 3. 4. 1. 5.5 5.5]  ， 5 跟 6 的 平均值是 5.5

# 法二、numpy 中 argsort 函數 排序轉換
order = np.argsort(x)
rank = np.argsort(order)
print(rank)
# 從 0 開始、相同數值，索引小的排序在前面
# [1 2 3 0 4 5]

# -----------------------------------
# RankGauss
# 將數值必涮，轉換為，排序
# 強制，常態分布
# 缺點：若， n_quantiles 太小，會用較多的線性內插法，來簡化，造成效果不理想
# -----------------------------------
# 資料讀取
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import QuantileTransformer

# 將訓練資料，的特徵欄位，進行 RankGauss 轉換
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

# 使用轉換後的資料，取代，各欄位的資料
train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])
