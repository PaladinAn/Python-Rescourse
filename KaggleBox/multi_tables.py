import numpy as np
import pandas as pd

# -----------------------------------
# 報表結合
# -----------------------------------
# 讀取
train = pd.read_csv('../sample-data/multi_table_train.csv')
product_master = pd.read_csv('../sample-data/multi_table_product.csv')
user_log = pd.read_csv('../sample-data/multi_table_log.csv')

# -----------------------------------
# 假設資料框架
# train         : 訓練資料（使用者 ID, 商品ID, 標籤欄位）
# product_master: 商品清單（商品 ID 商品資訊）
# user_log      : 使用者活動的紀錄檔資料（使用者 ID 和各種活動資訊）

# 合併商品清單和訓練資料
train = train.merge(product_master, on='product_id', how='left')

# 先整合每個使用者活動的紀錄檔欄位，再與訓練資料合併
user_log_agg = user_log.groupby('user_id').size().reset_index().rename(columns={0: 'user_count'})
train = train.merge(user_log_agg, on='user_id', how='left')
