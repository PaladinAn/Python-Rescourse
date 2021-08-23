# ---------------------------------
# 準備
# ----------------------------------
import numpy as np
import pandas as pd

# train_x 訓練函數、train_y 目標函數、test_x 驗證函數
# pandas DataFrame, Series 

train = pd.read_csv('/sample-data/train.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('/sample-data/test.csv')

# 訓練資料保存
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()


# 
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x


# 設定變數欄位名稱
cat_cols = ['sex', 'product', 'medical_info_b2', 'medical_info_b3']

# -----------------------------------
# one-hot encoding
# 進行 one-hot encoding 後 大部份資料都是 0，為稀疏矩陣，由 sparse() 參數設定為 True，只記錄值為 1的位置，可以節省記憶體空間 
# 缺點：特徵的數量，會隨類別變數的項目增加而增加
#       項目過多，會產生很多變數=0，資訊很少，卻有大量特徵的情況；所需記憶體。會大幅增加，影響效能
# -----------------------------------
# 讀取
train_x, test_x = load_data()
# -----------------------------------

# 整合訓練資料、測試資料；執行 one-hot encoding
all_x = pd.concat([train_x, test_x])
all_x = pd.get_dummies(all_x, columns=cat_cols)

# 重新分割訓練資料、測試資料
train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)

# -----------------------------------
# 
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoder 建立 encoding 物件
ohe = OneHotEncoder(sparse=False, categories='auto')
ohe.fit(train_x[cat_cols])

# 建立虛擬變數欄位名稱
columns = []
for i, c in enumerate(cat_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 建立好的虛擬變數轉換成 dataframe
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[cat_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[cat_cols]), columns=columns)

# 轉換後的 dataframe 與其他變數結合
train_x = pd.concat([train_x.drop(cat_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(cat_cols, axis=1), dummy_vals_test], axis=1)

# -----------------------------------
# label encoding
# 透過字母、數字大小，直接轉換成整數
# 適用，決策樹為基礎，建模
# -----------------------------------
# 
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 將類別變數，進行 loop 迴圈 ， 並進行 label encoding
for c in cat_cols:
    # 以訓練資料來對 label encoding 物件進行定義 (fit)
    le = LabelEncoder()
    le.fit(train_x[c])
    train_x[c] = le.transform(train_x[c])
    test_x[c] = le.transform(test_x[c])

# -----------------------------------
# feature hashing
# 由 hash 函數，轉後的特徵，會小於類別項目個數(減少，轉後產生過多的特徵)
# -----------------------------------
# 
train_x, test_x = load_data()
# -----------------------------------
from sklearn.feature_extraction import FeatureHasher

# 對類別變數進行 feature hashing
for c in cat_cols:
    # FeatureHasher 方法與其他 encoder 不同

    fh = FeatureHasher(n_features=5, input_type='string')
    # 將變數，轉換成，文字列，可以適用 FeatureHasher
    hash_train = fh.transform(train_x[[c]].astype(str).values)
    hash_test = fh.transform(test_x[[c]].astype(str).values)
    # 轉換成 dataframe
    hash_train = pd.DataFrame(hash_train.todense(), columns=[f'{c}_{i}' for i in range(5)])
    hash_test = pd.DataFrame(hash_test.todense(), columns=[f'{c}_{i}' for i in range(5)])
    # 與原資料 的 dataframe 結合
    train_x = pd.concat([train_x, hash_train], axis=1)
    test_x = pd.concat([test_x, hash_test], axis=1)

# 刪除 原資料的 類別變數
train_x.drop(cat_cols, axis=1, inplace=True)
test_x.drop(cat_cols, axis=1, inplace=True)

# -----------------------------------
# frequency encoding
# 以各項目，出現次數、出現頻率，來代換，類別變數
# 適用：若，類別變數 和 各項目出現頻率，相關時，此方法有助於，反映「預測值」
# -----------------------------------
# 
train_x, test_x = load_data()
# -----------------------------------
# 對類別變數進行 frequency encoding
for c in cat_cols:
    freq = train_x[c].value_counts()
    # 將變數，代換為，類別出現的次數 
    train_x[c] = train_x[c].map(freq)
    test_x[c] = test_x[c].map(freq)

# -----------------------------------
# target encoding
# 類別變數轉換為數值變數
# 將類別變數中，相同項目的資料取出，計算平均後，取代原本的值
# 缺點：直接使用資料來取得平均，會造成本身標籤，被作為，類別變數來使用，進而造成資料外洩
#     解決方法：對資料先平均後，再進行轉換
# -----------------------------------
# 
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# 對每個類變數進行 target encoding
for c in cat_cols:
    # 以訓練資料計算每個標籤平均
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    target_mean = data_tmp.groupby(c)['target'].mean()
    # 轉換測試資料的類別
    test_x[c] = test_x[c].map(target_mean)

    # 設定訓練資料轉換後格式
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 分割訓練資料
    kf = KFold(n_splits=4, shuffle=True, random_state=72)
    for idx_1, idx_2 in kf.split(train_x):
        # 以 out-of-fold 來計算各類別變數 的標籤 平均值
        target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        # 在 暫定格式中 置入 轉換後後的値
        tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

    # 以主換後的資料，取代原変数
    train_x[c] = tmp

# -----------------------------------
# target encoding - 搭配交叉驗證 ，替 fold做編碼
# 在訓練資料分割成 target encoding 的 fold ，必須排除，要進行交叉驗證的資料
# -----------------------------------
# 
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# 對交叉驗證的每個 fold 重新執行 target encoding
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):

    # 將驗證資料，從訓練資料中，分離
    tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 進行每個類別変数 的 target encoding
    for c in cat_cols:
        # 計算，所有訓練資料中，各個項目的 標籤平均值， target 平均
        data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # 代替驗證資料的類別
        va_x.loc[:, c] = va_x[c].map(target_mean)

        # 設定訓練資料主患後的排列方式
        tmp = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72)
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            # 以 out-of-fold 方法，計算個類別變數的標籤平均值
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 將轉換後的值，傳回，暫定排列中
            tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)

        tr_x.loc[:, c] = tmp

    # 若必要 可以保存 encode 的特徴量

# -----------------------------------
# target encoding - 將訓練資料 & 驗證資料合併起來做；交叉驗證 fold 與 target encoding 的 fold 合併，就可以一次完成轉換
# -----------------------------------
# 
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# 交叉驗證 fold 定義
kf = KFold(n_splits=4, shuffle=True, random_state=71)

# 進行每個類別変数 的 target encoding
for c in cat_cols:

    # 加上 target
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    # 設定轉換後，置入數值的格式
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 將驗證資料從，訓練資料中分離
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        # 計算訓練資料中，各類別變數平均
        target_mean = data_tmp.iloc[tr_idx].groupby(c)['target'].mean()
        # 將轉換後的驗證資料數值置入暫定格式中
        tmp[va_idx] = train_x[c].iloc[va_idx].map(target_mean)

    # 以轉換後的資料代換 原資料
    train_x[c] = tmp
