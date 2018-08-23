

```python
# k近邻的k值是事先设定的，K值不同对模型的表现性能有巨大影响
```

### 代码25：读取Iris数据集


```python
from sklearn.datasets import load_iris
iris = load_iris()
iris.data.shape
```




    (150, 4)




```python
# 查看数据说明，对于一名机器学习的实践者来说，这是一个好习惯
print(iris.DESCR)
```

    Iris Plants Database
    ====================
    
    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML iris datasets.
    http://archive.ics.uci.edu/ml/datasets/Iris
    
    The famous Iris database, first used by Sir R.A Fisher
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    References
    ----------
       - Fisher,R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...
    
    

### 代码26： 数据集分割


```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 33)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

### 代码27:使用K近邻分类器对鸢尾花数据进行类别预测


```python
# 导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
```


```python
# 对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
```


```python
# 模型初始化
knc = KNeighborsClassifier()
# 训练
knc.fit(X_train, y_train)
# 预测
y_predict = knc.predict(X_test)
```

### 代码28：预测性能评估


```python
# 使用模型自带的评估函数进行准确性测评
print('The Accuracy of K-Nearest Neighbor Classifier is:', knc.score(X_test, y_test))
```

    The Accuracy of K-Nearest Neighbor Classifier is: 0.8947368421052632
    


```python
# 依然使用sklearn.metrics 里面的classification_report模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names = iris.target_names))
```

                 precision    recall  f1-score   support
    
         setosa       1.00      1.00      1.00         8
     versicolor       0.73      1.00      0.85        11
      virginica       1.00      0.79      0.88        19
    
    avg / total       0.92      0.89      0.90        38
    
    

***K近邻属于无参数模型（Nonparametric model）中非常简单的一种，懒惰学习，正是这样的决策算法，导致了其非常高的计算复杂度和内存消耗，因为改模型每处理一个测试样本，都需要对所有预先加载在内存的训练样本进行遍历，逐一计算相似度，排序并且选取K个最近邻训练样本的标记，进而做出分类决策。这是平方级别的算法复杂度，一旦数据规模稍大，使用者便需要权衡更多计算时间的代价***
