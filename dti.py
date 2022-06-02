import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
import pydotplus
from IPython.display import Image

from functools import reduce
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch import is_anomaly_enabled
from sklearn import tree

dtr = tree.DecisionTreeClassifier( )  # 对树进行分类
df1 = pd.read_csv('train-normalized.csv', usecols=[0, 1, 2, 3])
xy1 = df1.iloc[:, [1, 2]]
label1 = df1['is_anomaly']

dtr.fit(xy1,label1)

df1 = df1.values


column_names = xy1.columns.tolist()   # 提取训练集特征名称

xy1 = xy1.values

#构造用于可视化决策树的dot数据
# dot_data = \
#     tree.export_graphviz(
#         dtr,                # 决策树模型名称
#         out_file = None,   # 输出文件的句柄或名称，默认值是None
#         feature_names = column_names,  # 训练所用特征名称
#         filled = True,     # 指定是否给节点填充不同的颜色
#         impurity = True,   # 指定在节点中显示不纯度，如MSE、MAE等
#         rounded = True     # 设置节点在拐点处的形状为圆形
#     )

# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())

# here do the predict work

df2 = pd.read_csv('test-normalized.csv', usecols=[0, 1, 2, 3])
xy2 = df2.iloc[:, [1, 2]]
predict = dtr.predict(xy2)

label2 = df2['is_anomaly']
df2 = df2.values
xy2 = xy2.values

plt.figure(figsize=(21, 21))
plt.subplot(1, 2, 1)
plt.scatter(xy2[:, 0], xy2[:, 1],c=predict)
plt.xlabel('CPC')
plt.ylabel('CPM')
plt.title('The predictions of the test set based on Decision Tree Induction')

plt.subplot(1, 2, 2)
plt.scatter(xy2[:, 0], xy2[:, 1], c=label2)
plt.xlabel('CPC')
plt.ylabel('CPM')
plt.title('The real labels of the test set')


plt.show()

print(dtr.score(xy2))
print(accuracy_score(label2,predict))
