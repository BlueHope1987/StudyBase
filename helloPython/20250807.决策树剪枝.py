'''
https://mp.weixin.qq.com/s/0pPVM-_XSRv5GSu6l2bbaQ
一个强大算法模型，决策树剪枝 ！

1. 决策树基础
决策树是一种用于分类和回归的机器学习模型，通过树状结构对数据进行分割和判断。在构建决策树时，为了提高模型的泛化能力，我们常常需要进行剪枝操作。
2. 决策树剪枝原理
决策树剪枝的目标是在保持模型泛化性能的同时，减小树的复杂度，防止过拟合。剪枝主要分为预剪枝（Pre-pruning）和后剪枝（Post-pruning）两种。
2.1 预剪枝
在决策树构建过程中，提前设定停止条件，例如限制树的深度、节点的样本数或信息增益的阈值，以防止过拟合。这种方法简单直接，但可能导致欠拟合。
2.2 后剪枝
后剪枝是在构建完整颗树后，通过剪掉一些节点来提高泛化性能。后剪枝的核心思想是通过交叉验证，评估剪枝前后模型性能，选择能够提高泛化性能的剪枝方案。
3. 决策树剪枝核心公式
假设剪枝前树的损失函数为L0(T)，剪枝后树的损失函数为Lα(T)，其中α是剪枝参数。则剪枝的目标是找到一个最小的α，使得Lα(T)≤L0(T)+α。
具体计算公式为：
Cα(T)=L(T)+α|T|
其中，是树的损失函数，是树的叶子节点数目。

4. 决策树剪枝步骤
    1.从决策树底部开始，自底向上计算每个内部节点的损失函数。
    2.对每个内部节点，计算剪枝前后的损失函数差异。
    3.对每个内部节点，计算剪枝参数。
    4.选择最小的，进行剪枝。
5. 优缺点
5.1 优点
·提高决策树的泛化性能。
·减小模型的复杂度，避免过拟合。
5.2 缺点
·需要进行交叉验证来选择最优的剪枝参数，计算较为复杂。
·可能导致一些节点的信息丢失，对于某些训练数据过于敏感。
6. 适用场景
决策树剪枝适用于决策树构建后出现过拟合的情况，通过剪枝来提高模型的泛化能力。
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# 数据加载
iris = datasets.load_iris()
X = iris.data
y = iris.target

'''
鸢尾花数据集
Iris 数据集是一个经典的多变量分析数据集，广泛用于分类任务和机器学习实验。
该数据集包含 150 个样本，分为 3 个类别：Setosa、Versicolour 和 Virginica，每类各有 50 个样本。
数据结构
Iris 数据集以 150x4 的二维数组形式存储，其中：
每一行代表一个样本。
每一列代表一个特征。
特征包括：
    萼片长度 (Sepal Length)：以厘米为单位。
    萼片宽度 (Sepal Width)：以厘米为单位。
    花瓣长度 (Petal Length)：以厘米为单位。
    花瓣宽度 (Petal Width)：以厘米为单位。
目标变量为类别标签，分别对应 3 种鸢尾花类型：
    Iris Setosa
    Iris Versicolour
    Iris Virginica
数据特点
    线性可分性：Setosa 类别与其他两类线性可分，而 Versicolour 和 Virginica 存在部分重叠。
    特征分布：Setosa 的萼片较短且宽，而 Versicolour 和 Virginica 的特征值范围更接近。
'''


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义决策树模型
clf = DecisionTreeClassifier(random_state=42)

# 定义参数网格
param_grid = {'max_depth': range(1, 11), 'min_samples_split': range(2, 11)}

# 使用网格搜索进行交叉验证选择最优参数
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最优参数
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
# 结果：Best Parameters: {'max_depth': 4, 'min_samples_split': 2}

# 构建带有最优参数的决策树模型
best_clf = DecisionTreeClassifier(random_state=42, **best_params)
best_clf.fit(X_train, y_train)

# 可视化最优决策树
plt.figure(figsize=(12, 8))
plot_tree(best_clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

'''
在上述代码中，我们首先定义了一个决策树模型，然后定义了一个参数网格，包含了我们希望搜索的最大深度和最小样本分割数的范围。接着，我们使用GridSearchCV进行交叉验证，并在网格搜索的结果中选择具有最优参数的模型。最后，我们用最优参数构建了决策树模型，并将其可视化输出。
这样，我们就可以通过交叉验证选择最优的剪枝参数，并使用最优参数构建决策树模型，以提高模型的泛化性能。
'''