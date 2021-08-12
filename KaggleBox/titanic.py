import numpy as np
import pandas as pd

# -----------------------------------
# 
# -----------------------------------
# 
train = pd.read_csv('..train.csv')
test = pd.read_csv('..test.csv')

# 取出，訓練資料的特徵和標籤
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

# 備份一個，原測試資料
test_x = test.copy()

# -----------------------------------
# 建立特徴，先使用label encoding，將文字轉換成數字
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 去除 Passenger 欄位名稱
train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)

# 去除 Name, Ticket, Cabin 欄位名稱
train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 對 Sex、Embarked 進行 label encoding
for c in ['Sex', 'Embarked']:
    # 決定資料轉換
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    # 訓練資料、學習資料，轉換，若有缺失值，填入NA
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# -----------------------------------
# 使用XGBC套件(XGBoost)，使用GBDT演算法，建立模型
# -----------------------------------
from xgboost import XGBClassifier

# 建立模型、訓練資料投入進行學習
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# 投入測試資料、輸出預測值 (介於0~1的機率值)
pred = model.predict_proba(test_x)[:, 1]

# 大於 0.5 = 1 、小於 0.5 = 0
pred_label = np.where(pred > 0.5, 1, 0)

# 建立提交的檔案
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)
# 預測分數：0.7799

# -----------------------------------
# 對模型，進行評價  (交叉驗證)
# -----------------------------------
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

# 用 list，保存 fold 的 accuracy、logloss分數
scores_accuracy = []
scores_logloss = []

# 進行交叉驗證
# 訓練資料分 4 等分 ，其中 1 作為 驗證資料，且不斷，論替驗證資料
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    # 將資料分成，訓練資料、驗證資料 (含標籤)
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 建立 XGBoost模型，投入訓練資料、標籤，進行學習
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    # 對，驗證資料進行預測，輸出，預測值的準確率
    va_pred = model.predict_proba(va_x)[:, 1]

    # 計算，驗證資料，預測值的評價指標，用 accuracy、logloss 來計算誤差
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # 保存評價指標
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

# 輸出，各評價指標的平均值
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
# logloss: 0.4270, accuracy: 0.8148  (用GBDT演算法，預測分數：0.7799 accuracy: 0.8148 ，理論上要相似，差異原因可能來自，樣本數)

# -----------------------------------
# 模型調整，(訓練前指定，1組，超參數；採 Grid Search，暴力搜尋法)
# -----------------------------------
import itertools        #處裡排列組合，很方便的套件

# 準備用於調整的超參數
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

# 產生，超參數，max_depth，min_child_weight，的所有組合
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

# 使用 List 保存 超參數組合、超參數組合的分數
params = []
scores = []

# 對超參數組合的模型，進行，交叉驗證
for max_depth, min_child_weight in param_combinations:

    score_folds = []
    # 進行，交叉驗證
    # 訓練資料分 4 等分 ，其中 1 作為 驗證資料，且不斷，論替驗證資料
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        # 將資料分成，訓練資料、驗證資料
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # 建立 XGBoost 模型，且進行訓練
        model = XGBClassifier(n_estimators=20, random_state=71,
                              max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # 驗證資料的預測值、logloss 評價指標、保存
        va_pred = model.predict_proba(va_x)[:, 1]  # 產生，驗證資料，的預測值
        logloss = log_loss(va_y, va_pred)          # 產生，logloss 評價指標
        score_folds.append(logloss)                # 保存，logloss 評價指標

    # 評價指標，取平均數
    score_mean = np.mean(score_folds)

    # 保存，超參數組合、評價指標
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

# 找出，評價指標分數，最佳超參數的組合
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# max_depth=7, min_child_weight=2.0 (logloss 最低分，所以是個好指標)


# -----------------------------------
# 集成學習 (Ensemble Learning)
# Logistic Regression的特徵訓練資料，建立
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# 訓練、測試資料
train_x2 = train.drop(['Survived'], axis=1)  #訓練資料，去除 Survived 欄位
test_x2 = test.copy()

# 去除 訓練、測試資料中 Passenger欄位
train_x2 = train_x2.drop(['PassengerId'], axis=1)
test_x2 = test_x2.drop(['PassengerId'], axis=1)

# 去除 訓練、測試資料中 Name, Ticket, Cabin欄位
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 建立 one-hot encoding 的物件
cat_cols = ['Sex', 'Embarked', 'Pclass']
ohe = OneHotEncoder(categories='auto', sparse=False)   #建立 one hot 物件
ohe.fit(train_x2[cat_cols].fillna('NA'))               #缺失，補 NA

# 對指定的欄位 進行 one-hot encoding 編碼
ohe_columns = []
for i, c in enumerate(cat_cols):
    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# one-hot encoding 編碼後，將結果存在一個新的 Dataframe
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# one-hot encoding 去除 原資料中，指定欄位的數據
train_x2 = train_x2.drop(cat_cols, axis=1)
test_x2 = test_x2.drop(cat_cols, axis=1)

# 將 one-hot encoding 編碼後的 Dataframe 與原資料進行合併
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)
test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)

# 將欄位中的缺失值，用整個欄位的，平均值取代
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# 對，Fare 欄位的數據，取 log  
# np.log1p 是資料 +1 後取對數，作用為，使資料平滑化
train_x2['Fare'] = np.log1p(train_x2['Fare'])
test_x2['Fare'] = np.log1p(test_x2['Fare'])

# -----------------------------------
# 進行，集成學習
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# xgboost模型、訓練與預測
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# 建立 Logistic Regression 模型，進行訓練、預測
# 因為，必須放入 與 xgboost 不同的特徵，所以，建立，train_x2, test_x2
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# 取得加權平均後的預測值
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)


# 建立提交的檔案
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_ensemble.csv', index=False)
