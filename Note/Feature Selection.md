
# Feature Selection 特徵選擇

 特徵愈多，愈容易得到好的預測效果。
 -----------------------------------
 But，當特徵過多時，運算成本增加，更容易過度擬和、維度災難
 --------------------------------------------------------
 How to do?
 
 
 
    特徵選擇模組
 
    from sklearn import feature_selection as fs
    
    # Removing features with low variance
    透過設定特徵變異數的門檻值，當特徵的變異數小於門檻則遭到剔除
    當變異數愈小，代表特徵間差異愈小，也就是，特徵間同質性很高，所以較不具鑑別度
    
    
    sel=fs.VarianceThreshold(threshold=0.8*(1-0.8))            設定不到，變異數80%，就剃除
    sel.fit_transform(X)
    
    
    # Univariate feature selection
    單獨計算每個特徵的統計值來決定最重要的K個特徵(SelectKBest)或排名前多少百分比的特徵(SelectPercentile)
    
    對於連續型的目標(regression)，
      可以採用 f_regression, mutual_info_regression
    對於離散型的目標(classification)，
      可以採用chi2, f_classif, mutual_info_classif
    
    
    X_new = fs.SelectKBest(fs.chi2, k=3).fit_transform(X, y)        現在有4個特徵，但我只要3個特徵
    
    X_new = fs.SelectPercentile(fs.mutual_info_classif, percentile = 50).fit_transform(X, y)
                                                                       取前 50%
    
    Recursive feature elimination, Feature selection using SelectFromModel
