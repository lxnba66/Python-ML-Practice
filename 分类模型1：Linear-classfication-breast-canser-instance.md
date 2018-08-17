
### 代码13：良/恶性乳腺癌肿瘤数据预处理


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
import numpy as np
```


```python
# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
```


```python
# 从互联网读取指定数据
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names = column_names)
```


```python
#将？替换为标准缺失值表示
data = data.replace(to_replace = '?', value = np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）
data = data.dropna(how = 'any')
# 输出data的数据量和维度
data.shape
```




    (683, 11)




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample code number</th>
      <th>Clump Thickness</th>
      <th>Uniformity of Cell Size</th>
      <th>Uniformity of Cell Shape</th>
      <th>Marginal Adhesion</th>
      <th>Single Epithelial Cell Size</th>
      <th>Bare Nuclei</th>
      <th>Bland Chromatin</th>
      <th>Normal Nucleoli</th>
      <th>Mitoses</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1016277</td>
      <td>6</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1017023</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### 代码14：准备良/恶性乳腺癌肿瘤训练、测试数据


```python
# 使用 sklearn.cross_validation 里面的train_test_split模块用于分割数据
from sklearn.cross_validation import train_test_split
# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size = 0.25, random_state = 33)
```


```python
# 查验训练样本的数量和类别分布
y_train.value_counts()
```




    2    344
    4    168
    Name: Class, dtype: int64




```python
#查验测试样本的数量和类别分布
y_test.value_counts()
```




    2    100
    4     71
    Name: Class, dtype: int64



### 代码15：使用线性分类模型从事良/恶性肿瘤预测任务


```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
```


```python
# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
```


```python
# 模型初始化
lr = LogisticRegression()
sgdc = SGDClassifier()
```


```python
# 模型训练
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)
```

### 代码16：使用线性分类模型从事良/恶性肿瘤预测任务的性能分析


```python
from sklearn.metrics import classification_report

# 使用逻辑斯谛回归模型自带的评分函数score获得模型在测试集上的准确性结果
print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
# 获得其他三个指标的结果
print(classification_report(y_test, lr_y_predict, target_names = ['Benign', 'Malignant']))
```

    Accuracy of LR Classifier: 0.9883040935672515
                 precision    recall  f1-score   support
    
         Benign       0.99      0.99      0.99       100
      Malignant       0.99      0.99      0.99        71
    
    avg / total       0.99      0.99      0.99       171
    
    


```python
# 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果
print('Accuarcy of SGD Classifier:', sgdc.score(X_test, y_test))
# 获得其他三个指标的结果
print(classification_report(y_test, sgdc_y_predict, target_names = ['Benign', 'Malignant']))
```

    Accuarcy of SGD Classifier: 0.9649122807017544
                 precision    recall  f1-score   support
    
         Benign       1.00      0.94      0.97       100
      Malignant       0.92      1.00      0.96        71
    
    avg / total       0.97      0.96      0.97       171
    
    

  ***逻辑斯蒂精确解析，随机梯度是用梯度估计法估计参数，所以准确率较低一些。  
  对于10万量级以上的数据，考虑到时间的耗用，笔者更推荐使用随机梯度算法对模型参数进行估计。  
  尽管受限于数据特征与分类目标之间的线性假设，我们仍然可以在科学研究和工程实践中把线性分类器的表现性能作为基准。***
