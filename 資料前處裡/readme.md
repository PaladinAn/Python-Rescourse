# 資料前處理
   sklearn 中四種不同資料前處理方式
   
## StandardScaler (平均值和標準差)
    所有特徵標準化，也就是常態分佈。使得數據的平均值為0，變異數為1
    適合的使用時機於當有些特徵的變異數過大時，使用標準化能夠有效地讓模型快速收斂。

## MinMaxScaler(最小最大值標準化)
    給定了一個明確的最大值與最小值。每個特徵中的最小值變成了0，最大值變成了1。數據會縮放到到[0,1]之間。
## MaxAbsScaler（絕對值最大標準化）
    所有數據都會除以該列絕對值後的最大值，數據會縮放到到[-1,1]之間。

## RobustScaler (中位數和四分位數標準化)
    縮放帶有outlier的數據，透過Robust如果數據中含有異常值在縮放中會捨去。

# 缺失值處理
  ## 1）缺失值刪除（dropna）
  ### ①刪除例項
  ### ②刪除特徵
  ## 2）缺失值填充（fillna）
  ### ①用固定值填充
對於特徵值缺失的一種常見的方法就是可以用固定值來填充，例如0，9999， -9999, 例如下面對灰度分這個特徵缺失值全部填充為-99

        ex. data['某某'] = data['某某'].fillna('-99')
  ### ②用均值填充
對於數值型的特徵，其缺失值也可以用未缺失資料的均值填充，下面對灰度分這個特徵缺失值進行均值填充

        data['某某'] = data['某某'].fillna(data['某某'].mean()))
  ### ③用眾數填充
與均值類似，可以用未缺失資料的眾數來填充缺失值

        data['某某'] = data['某某'].fillna(data['某某'].mode()))
  ### ④用上下資料進行填充
  #### 用前一個數據進行填充
    
        data['某某'] = data['某某'].fillna(method='pad')
        
  #### 用後一個數據進行填充

        data['某某'] = data['某某'].fillna(method='bfill')
        
  ### ⑤用插值法填充
        data['某某'] = data['某某'].interpolate()
        
  ### ⑥用KNN進行填充
        from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
        dataset = KNN(k=3).complete(dataset)
