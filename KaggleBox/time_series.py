import numpy as np
import pandas as pd

# -----------------------------------
# 處裡時間序誒資料
# -----------------------------------

# 讀取寬表格
df_wide = pd.read_csv('../sample-data/time_series_wide.csv', index_col=0)
# 將索引，轉換成，日期型態
df_wide.index = pd.to_datetime(df_wide.index)

print(df_wide.iloc[:5, :3])
'''
              A     B     C
date
2016-07-01  532  3314  1136
2016-07-02  798  2461  1188
2016-07-03  823  3522  1711
2016-07-04  937  5451  1977
2016-07-05  881  4729  1975
'''

# 轉換成長表格
df_long = df_wide.stack().reset_index(1)
df_long.columns = ['id', 'value']

print(df_long.head(10))
'''
           id  value
date
2016-07-01  A    532
2016-07-01  B   3314
2016-07-01  C   1136
2016-07-02  A    798
2016-07-02  B   2461
2016-07-02  C   1188
2016-07-03  A    823
2016-07-03  B   3522
2016-07-03  C   1711
2016-07-04  A    937
...
'''

# 還原成。寬表格
df_wide = df_long.pivot(index=None, columns='id', values='value')

# -----------------------------------
# lag特徵函數
# 設置，寬格式數據
# x 為 寬表格的 dataframe
# -----------------------------------
# index 為日期等時間、列為使用者或店面等資料，值則為營業額等我們關注的變數
x = df_wide
# -----------------------------------
# 
# 

# 取得 1期前 lag 的特徵
x_lag1 = x.shift(1)

# 取得 7期前 lag 的特徵
x_lag7 = x.shift(7)

# -----------------------------------
# 移動平均 與 其他 lag 特徵
# 計算 前 1~3 單位  期間 移動平均
x_avg3 = x.shift(1).rolling(window=3).mean()

# -----------------------------------
# 計算 前 1期 ~ 前 7 單位期間 的最大値
x_max7 = x.shift(1).rolling(window=7).max()

# -----------------------------------
# 將前 7期 , 前 14期 , 前 21期, 前 28期 値 進行平均
x_e7_avg = (x.shift(7) + x.shift(14) + x.shift(21) + x.shift(28)) / 4.0

# 當資料，長時間趨勢沒有太大的變化，統計長期的資料比較好
# lead 特徵，是未來的值
# -----------------------------------
#  取得 後 1期 的値
x_lead1 = x.shift(-1)

# -----------------------------------
# 資料與時間做連結
# 讀取數據
# -----------------------------------
# 
train_x = pd.read_csv('../sample-data/time_series_train.csv')
event_history = pd.read_csv('../sample-data/time_series_events.csv')
train_x['date'] = pd.to_datetime(train_x['date'])
event_history['date'] = pd.to_datetime(event_history['date'])
# -----------------------------------

# train_x 訓練資料、活動資訊、舉辦特價 ID, DataFrame
# event_history 過去舉辦過的活動資訊，包含日期、活動欄位的 DataFrame

# occurrences 含有日期、是否舉辦特價的欄位的 DataFrame
dates = np.sort(train_x['date'].unique())
occurrences = pd.DataFrame(dates, columns=['date'])
sale_history = event_history[event_history['event'] == 'sale']
occurrences['sale'] = occurrences['date'].isin(sale_history['date'])

# 累積和 來表現每個日期，累積出現次數
# occurrences 含有日期、拍賣的累積出現次數的 DataFrame
occurrences['sale'] = occurrences['sale'].cumsum()

# 以日期為 key 來結合訓練資料
train_x = train_x.merge(occurrences, on='date', how='left')
