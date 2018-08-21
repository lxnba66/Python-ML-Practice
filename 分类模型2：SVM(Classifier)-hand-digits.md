
### 代码17：手写体数据读取代码样例


```python
# 导入手写体数字加载器
from sklearn.datasets import load_digits
# 从通过数据加载器获得手写体数字的数码图像数据并储存在digits变量中
digits = load_digits()
# 检查数据规模和特征维度
digits.data.shape
```




    (1797, 64)



### 代码18：手写体数据分割代码样例


```python
from sklearn.cross_validation import train_test_split
X_train,Ｘ_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 33)
# 分别检查训练集与测试集数据规模
y_train.shape,y_test.shape
```




    ((1347,), (450,))



### 代码19：使用支持向量机（分类）对手写体数字图像进行识别


```python
from sklearn.preprocessing import StandardScaler
# 导入基于线性假设的支持向量机分类器
from sklearn.svm import LinearSVC
# 对特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
```


```python
# 初始化线性假设的支持向量机分类器
lsvc = LinearSVC()
# 进行模型训练
lsvc.fit(X_train, y_train)
# 预测
y_predict = lsvc.predict(X_test)
```

### 代码20：支持向量机（分类）模型对手写体数码图像识别能力的评估


```python
# 使用模型自带的评估函数进行准确性测评
print('The Accuracy of Linear SVC is:', lsvc.score(X_test, y_test))
# 精确率、召回率和F1值
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names = digits.target_names.astype(str)))
```

    The Accuracy of Linear SVC is: 0.9511111111111111
                 precision    recall  f1-score   support
    
              0       0.92      1.00      0.96        35
              1       0.95      0.98      0.96        54
              2       0.98      1.00      0.99        44
              3       0.93      0.93      0.93        46
              4       0.97      1.00      0.99        35
              5       0.94      0.94      0.94        48
              6       0.96      0.98      0.97        51
              7       0.92      1.00      0.96        35
              8       0.98      0.83      0.90        58
              9       0.95      0.91      0.93        44
    
    avg / total       0.95      0.95      0.95       450
    
    
