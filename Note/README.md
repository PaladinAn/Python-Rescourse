# Python-Rescourse
  Magic儲存指令
    %history -n 1-4
  
  NumPy
    查看資料型態
    type()   參數放 []  或" "
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
