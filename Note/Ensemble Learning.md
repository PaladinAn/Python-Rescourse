# Ensemble Learning(集成學習)主要是透過結合多個機器學習器而成的大模型。
---------------------------------------------------------------------
  集成學習，就是多個機器學習器，綜合起來得到最終的結果。
  ----------------------------------------------------------------------------
  方法主要包含三種，分別為Bagging, Bosting及Stacking。
  ------------------------------------------------------------------------
  # Bagging:
  ---------------------------
  從母體中，隨機抽取樣本及特徵，變成一個小樣本，並以此樣本的內容物建模
 ----------------------
  接下來，將樣本放回，再抽一次，建構第二個小樣本與模型，以此類推。取後放回的抽取方式，我們稱之為bootstrap。
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



 ------------------------------------------------------------------------
  # Boosting:
  ------------------------------
  會針對錯誤不斷訓練的
  ---------------------------
  Boosting裡的分類器則會由前一個分類器的結果而做更進一步的修正，因此每個分類器皆有所關連。 
  ---------------------
  因為，每次抽取結束都有調整權重，所以，每次抽取機率不同
  -----------------------
  最後，而在Boosting裡面則是菁英體制，準確度高的給予較高權重。
  -----------------------
  缺點：也很容易受到極值影響
  -----------------------
   #  Bagging 與 Boosting的主要差異
                Bagging	                       Boosting
      --------------------------------------------------------
      樣本抽樣    取後放回，每次抽取都是獨立的    依據前一次模型表現調整
      --------------------------------------------------------
      樣本權重    相同	                        前一次分類錯誤率愈高，則權重愈大
      --------------------------------------------------------
      子模型權重  相同	                         子模型表現錯誤率愈小，權重愈大
      --------------------------------------------------------
      平行運算    可以	                        不可
      --------------------------------------------------------
      用途	      減小Variance	                減小Bias
--------------------------------------------------------------
 #  How to do?
     引入套件
      在sklearn.ensemble裡面，有著各式各樣已寫好的boosting方法。

      from sklearn.ensemble import AdaBoostClassifier
      from sklearn.ensemble import GradientBoostingClassifier

    下載資料
      資料的部分，大家可以採用自己的真實資料，將格式分別整理成特徵及標籤兩個部分即可。在這邊，那我們採用sklearn裡面所提供的鳶尾花資料集。

      from sklearn import datasets
      iris=datasets.load_iris()
      X=iris.data
      y=iris.target

    區分訓練集與測試集
    有了完整的資料以後，我們會將這些樣本區分為訓練集以及測試集。用訓練集的資料來建模，並用測試集資料來驗證結果，以避免過度擬和的問題。

      X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0)

    Boosting
    在這邊，我們會用兩個方法進行Boosting的演示，分別為Ada Boost與Gradient Boos.

    Ada Boost
    分類問題可以採用AdaBoostClassifier，連續型問題可以採用AdaBoostRegressor。

      adb= AdaBoostClassifier()
      adb.fit(X_train,y_train)
      adb.predict(X_test)
      adb.score(X_test,y_test)


    在AdaBoostClassifier裡面，有幾個重要的參數
    base_estimator: 子模型 (弱學習器)
    n_estimators: 子模型數量
    learning_rate: 子模型權重縮減係數

    Gradient Boost
    分類問題可以採用GradientBoostingClassifier，連續型問題可以採用GradientBoostingRegressor。

      gb=GradientBoostingClassifier()
      gb.fit(X_train,y_train)
      gb.predict(X_test)
      gb.score(X_test,y_test)

    其中，在GradientBoostingClassifier裡面，有幾個重要參數如下:
    loss: 損失函數
    n_estimators: 子模型數量
    learning_rate: 子模型縮減係數

