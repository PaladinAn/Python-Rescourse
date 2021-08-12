# Ensemble Learning(集成學習)主要是透過結合多個機器學習器而成的大模型。
---------------------------------------------------------------------
  集成學習，就是多個機器學習器，綜合起來得到最終的結果。
  ----------------------------------------------------------------------------
  方法主要包含三種，分別為Bagging, Bosting及Stacking。
  ------------------------------------------------------------------------
  # Bagging:
  ---------------------------
  從大袋中(紫色圈圈)隨機抽取樣本及特徵變成一個小袋子(綠色圈圈)，並以此袋子的內容物建模
 ----------------------
  接下來，將樣本放回，再抽一次，建構第二個小袋子與模型，以此類推。取後放回的抽取方式，我們稱之為bootstrap。
  -----------------------
  最後，透過每個模型產出的結果，投票決定最終結果，等權重加權
  -----------------------
  如果目標為連續型資料，則改採對每個學習器的結果取平均。
------------------------------------------
 #  How to do?
    引入套件
  
  
      from sklearn.ensemble import BaggingClassifier
      from sklearn import datasets
      from sklearn.model_selection import train_test_split
      import matplotlib.pyplot as plt
      %matplotlib inline

    下載資料集
     iris=datasets.load_iris()
     X=iris.data
     y=iris.target

    區分訓練集與測試集
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
     
     Bagging
    引入子模型套件
    from sklearn import tree
    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB

    選擇子模型
     clf=GaussianNB()

     Bagging
     bagging=BaggingClassifier(base_estimator=clf, n_estimators=10,
     bootstrap=True, bootstrap_features=True, max_features=3, max_samples=0.7)
     bagging.fit(X_train, y_train)
     bagging.predict(X_test)
     bagging.score(X_test, y_test)
