# Python-Rescourse
  -------------------------------------------------
  
  
  Magic儲存指令
    %history -n 1-4
  
  NumPy
    
    
    查看資料型態 type()   參數放 []  或" "
    np.arrar([, , , , ])  建立陣列   若需要明確設定，dtype ，np.arrar([, , , , ], dtype="float32") 
      3X5 亂數      np.random.random((3, 5))
      3X5 亂數常態  np.random.nornmal((3, 5))
      3X5 亂數整數  np.random.randint((3, 5))
      3X5 亂數整數，範圍[0, 10]  np.random.randint(0, 10, (3, 5))
      3X5 亂數整數，1維  np.random.randint((3, 5), size=6)
      3X5 亂數整數，2維  np.random.randint((3, 5), size=(3, 4)
      3X5 亂數整數，3維  np.random.randint((3, 5), size=(3, 4, 5)
    切片符號 :
      前面5個   X[:5]
      5後面     X[5:]
      間隔2     X[::2]   結果array([0, 2, 4, 6, 8])
      反轉      X[::-1]  結果array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
      殂5後面開始反轉間隔2    X[5::-2]     結果array([5, 3, 1])
      
      多維切片  原式     array([12, 5, 6, 7], 
                              [2, 3, 6, 7], 
                              [0, 2, 3, 8])   
      X[:2, :3] 就是，把 2X3 框起來
                        array([12, 5, 6], 
                              [2, 3, 6], 
      X[:3, ::2] 匡3X2，但是後面的2是切第二號位置
                        array([12, 6], 
                              [2, 6], 
                              [0, 3])  
      切第一欄 (直)  print(:, 0)   array([12, 2, 0]
      切第一列 (横)  print(0, :)   array([12, 5, 6, 7]
      
     重塑 reshape()
     把1~9放入3X3     np.arrange(1,10).reshape((3, 3))
                             array[[1, 2, 3]
                                   [4, 5, 6]
                                   [7, 8, 9]]
            x = np.array[1, 2, 3] 透過  reshape(3, 1)
                                        變成3X1
                                            array([1], 
                                                  [2], 
                                                  [3])
     串接  
            np.concatente([])
      X = np.array([1, 2, 3])
      Y = np.array([4, 5, 6])
               np.concatente([X, Y])             =>array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        串2維，  也可以用 .vstack()
          X1 = np.array([1, 2],
                        [3, 4])
               np.concatente([X1, X1])           =>array([1, 2], 
                                                         [3, 4], 
                                                         [1, 2],
                                                         [3, 4])
        水平串， 也可以用 .hstack()
               np.concatente([X1, X1], axis=1)           =>array([1, 2, 1, 2],
                                                                 [3, 4, 3, 4])
                                                                 
      分割
             np.split([])
       X = [1, 2, 3, 99, 99, 4, 5, 6]
             np.split(X, [3, 5])    切第3、第5的節點     =>[1 2 3][99 99][4 5 6]
         切多維
                    array([1, 2],
                          [3, 4], 
                          [1, 2],
                          [3, 4])
               np.split(X, [2])      =>
                                                 [[1, 2],
                                                  [3, 4]] 
                                                 [[1, 2]
                                                  [3, 4]]
      常用函式
        np.sum          加總
        np.prod         乘積
        np.mean         平均值
        np.std          標準差
        np.var          變異量
        np.min          最小值     np.argmin  索引
        np.max          最大值     np.argman  索引
        np.median       中位數
        np.precentile   排名統計
        np.any          任一值，是Ture 或 != 傳Ture
        np.all          所有值，是Ture 或 != 傳Ture
      
      排序
        選擇排序法
          def selection_sort(x):
            for i in range(len(x)):
            swap = i + np.argmin(x[i:])
            (x[i, x[swap]) = (x[swap], x[i])
          x = np.array([2, 1, 3, 4, 6])
          selection_sort(x)                              =>array([1, 2, 3, 4, 6])
         
         或 Big-O
          def bogosort(x):
            while np.any(x[:-1] > x[1:]):
              np.random.shufflie(x)
              return x
            x = np.array([2, 1, 3, 4, 6])
          bogosort(x)                                    =>array([1, 2, 3, 4, 6])
         
         快速排序法   np.sort()
            x = np.array([2, 1, 3, 4, 6])
            np.sort(x)                                  =>array([1, 2, 3, 4, 6])
            
          欄(上下)小到大，排序  axis=0
            rand = np.random.RandomState(42)
               X = rand.randint(0, 10, (4, 6))
               print(X)                     => [[6 3 7 4 6 9]
                                                [2 6 7 4 3 7]
                                                [7 2 5 4 1 7]
                                                [5 1 4 0 9 5]]
                 np.sort(X, axis=0)
                          =>array([2, 1, 4, 0, 1, 5],
                                  [5, 2, 5, 4, 3, 7], 
                                  [6, 3, 7, 4, 6, 7], 
                                  [7, 6, 7, 4, 9, 9]) 
                                  
           列(左右)小到大，排序  axis=1
            rand = np.random.RandomState(42)
               X = rand.randint(0, 10, (4, 6))
               print(X)                     => [[6 3 7 4 6 9]
                                                [2 6 7 4 3 7]
                                                [7 2 5 4 1 7]
                                                [5 1 4 0 9 5]]
                 np.sort(X, axis=1)
                          =>array([3, 4, 6, 6, 7, 9],
                                  [2, 3, 4, 6, 7, 7], 
                                  [1, 2, 4, 5, 7, 7], 
                                  [0 ,1, 4, 5, 5, 9]) 
         ------------------------------------------------                         
  Pandas
    
                 
