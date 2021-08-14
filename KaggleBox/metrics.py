import numpy as np
import pandas as pd

# -----------------------------------
# 回帰
# RMSE 較容易受到極端質影響，必須事先排除極端值
# -----------------------------------
# rmse 方均跟誤差 MSE

from sklearn.metrics import mean_squared_error

# y_true 實際值、y_pred 預測値
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.5532

# -----------------------------------
# 二値分類
# 預測值為，正 or 負 (陽性、陰性)
# -----------------------------------
# 混合矩陣

from sklearn.metrics import confusion_matrix

# 0, 1 表示二値分類 正 負 預測値
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp],
                              [fn, tn]])
print(confusion_matrix1)
# array([[3, 1],
#        [2, 2]])

# 亦可用 scikit-learn的 metrics 的 confusion_matrix()
confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)
# array([[2, 1],
#        [2, 3]])

# -----------------------------------
# accuracy 準確率  (資料不均衡，結果無意義。正或負，機率在50%上下，才建議使用)

from sklearn.metrics import accuracy_score

# 0, 1 表示二値分類 正 負 預測値
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625

# -----------------------------------
# logloss 預測值，為正，的機率，採用的指標

from sklearn.metrics import log_loss

# 0, 1 表示二値分類 正 負 預測値
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.7136

# -----------------------------------
# 多變數 分類
# -----------------------------------
# multi-class logloss

from sklearn.metrics import log_loss

# 3類別分類的 實際值 預測値
y_true = np.array([0, 2, 1, 2, 2])
y_pred = np.array([[0.68, 0.32, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.60, 0.40, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.28, 0.12, 0.60]])
logloss = log_loss(y_true, y_pred)
print(logloss)
# 0.3626

# -----------------------------------
# 多標籤 分類
# -----------------------------------
# mean_f1, macro_f1, micro_f1

from sklearn.metrics import f1_score

# 多標籤分類的評價，將實際值與預測值，以 k-hot 編碼 表示 再進行計算
# 實際値 - [[1,2], [1], [1,2,3], [2,3], [3]]
y_true = np.array([[1, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [0, 1, 1],
                   [0, 0, 1]])

# 預測値 - [[1,3], [2], [1,3], [3], [3]]
y_pred = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

# 計算 mean-f1 評價 會先以資料為單會計算 F1-score 再取平均
mean_f1 = np.mean([f1_score(y_true[i, :], y_pred[i, :]) for i in range(len(y_true))])

# 計算 macro-f1 評價 會先以資料為單會計算 F1-score 再取平均
n_class = 3
macro_f1 = np.mean([f1_score(y_true[:, c], y_pred[:, c]) for c in range(n_class)])

# 計算 micro-f1 評價，以資料 X 為1組，計算 TP/TN/FP/FN ，求得 F1-score
micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))

print(mean_f1, macro_f1, micro_f1)
# 0.5933, 0.5524, 0.6250

# scikit-learn 也可以，直接在 f1_score 函式，加上，average，超參數來計算
mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

# -----------------------------------
# 多分類的評價指標
# -----------------------------------
# quadratic weighted kappa（各類別之間，有次序關係時，使用）

from sklearn.metrics import confusion_matrix, cohen_kappa_score


# quadratic weighted kappa 計算
def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            wij = ((i - j) ** 2.0)
            oij = c_matrix[i, j]
            eij = c_matrix[i, :].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij

    return 1.0 - numer / denom


# y_true 實際值 類別的 list 、 y_pred 預測値 類別的  list
y_true = [1, 2, 3, 4, 3]
y_pred = [2, 2, 4, 4, 5]

# 計算混淆矩陣
c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

# quadratic weighted kappa 計算
kappa = quadratic_weighted_kappa(c_matrix)
print(kappa)
# 0.6153

# scikit-learn 也可以直接計算 quadratic weighted kappa，不用先算混淆矩陣
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

# -----------------------------------
# 評價，推薦任務
# -----------------------------------
# MAP@K 平均精確率均值

# K=3、資料數 5個、類別 4種類
K = 3

# 每筆資料的實際値
y_true = [[1, 2], [1, 2], [4], [1, 2, 3, 4], [3, 4]]

# 每筆資料的預測値 - 因為 K=3，所以、通常每筆資料，會預測出，最有可能的 3 筆數據
y_pred = [[1, 2, 4], [4, 1, 2], [1, 4, 3], [1, 2, 3], [1, 2, 4]]


# 建立函數，計算 每筆資料的 average precision 平均準確率
def apk(y_i_true, y_i_pred):
    # y_pred 的長度必須在 K 以下 ，且元素不能重覆
    assert (len(y_i_pred) <= K)
    assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision

    return sum_precision / min(len(y_i_true), K)


# MAP@K 建立
def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])


# MAP@K 求得
print(mapk(y_true, y_pred))
# 0.65

# 即使，預測值內的正解與正解的數量相同、但是，只要，預測值排名不同，分數就會不同
print(apk(y_true[0], y_pred[0]))
print(apk(y_true[1], y_pred[1]))
# 1.0, 0.5833
