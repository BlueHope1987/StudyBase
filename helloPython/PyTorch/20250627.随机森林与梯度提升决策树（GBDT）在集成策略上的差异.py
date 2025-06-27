#https://mp.weixin.qq.com/s/7iaXCgSNE3TIOu_7qNKxgw
#通透！随机森林 vs GBDT！！

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

'''
Bagging和Boosting，是随机森林和GBDT为代表
Bagging 通过对训练集进行Bootstrap采样（有放回随机采样）产生多个子数据集，然后分别训练多个弱学习器，并通过平均（回归）或投票（分类）集成预测结果。
Boosting 将多个弱学习器串行构建，每一步都拟合前一步残差，逐步修正模型的预测误差，从而降低偏差。
GBDT 是 Boosting 的一种形式，使用 CART 决策树作为弱学习器。
Boosting 强调的是逐步降低偏差，尤其在模型能力不足时具有优势。
但由于模型不断依赖之前的学习器，易产生过拟合风险，因此需要控制树的深度、学习率、早停等策略。

比对表格：

维度    随机森林（Bagging） GBDT（Boosting）
训练方式    并行训练多个树  顺序训练多个树
抽样机制    样本 Bootstrap，有放回；特征随机采样    不抽样/全样本，每轮针对残差
集成方法    投票 / 平均 加法模型（加权残差）
偏差    高（未减小）    低（逐步减小）
方差    低（多模型平均）    相对较高（依赖顺序）
抗过拟合    更强（依赖随机性）  稍弱（需正则手段）
超参数敏感性    不高    较高（需调参）
常用场景    大数据、高维、鲁棒性要求    小样本、精度要求高、复杂非线性关系

虚拟数据集：使用 make_moons 生成简单的二分类数据
Python 代码：包含训练、预测及可视化分析
数据分析图形及其意义
每种算法优势与适用场景

GBDT 在复杂边界拟合和预测精度上更具优势，而随机森林在训练速度、模型稳定性和泛化能力方面表现更好。
尽管 GBDT 能持续优化残差降低误差，但更易过拟合且需精细调参；相比之下，随机森林对参数不敏感，适合快速构建稳健模型。
在处理非线性数据时，两者均有良好表现，但适用场景侧重点不同。
'''

# 数据集
X, y = make_moons(n_samples=2000, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义模型
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, oob_score=True) #随机森林
gbdt = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)

# 训练模型
rf.fit(X_train, y_train)
gbdt.fit(X_train, y_train)

# 准备绘图网格
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                     np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# 计算决策边界
Z_rf = rf.predict(grid).reshape(xx.shape)
Z_gbdt = gbdt.predict(grid).reshape(xx.shape)

# 1. Decision Boundaries
plt.figure()
plt.contourf(xx, yy, Z_rf, alpha=0.3, cmap='Wistia')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, edgecolor='k', cmap='rainbow')
plt.title("RF Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.figure()
plt.contourf(xx, yy, Z_gbdt, alpha=0.3, cmap='viridis')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, edgecolor='k', cmap='rainbow')
plt.title("GBDT Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 2. Accuracy vs Number of Trees
n_trees = list(range(1, 51))
acc_rf = []
acc_gbdt = []
for n in n_trees:
    rf_n = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42)
    gbdt_n = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1, max_depth=3, random_state=42)
    rf_n.fit(X_train, y_train)
    gbdt_n.fit(X_train, y_train)
    acc_rf.append(accuracy_score(y_test, rf_n.predict(X_test)))
    acc_gbdt.append(accuracy_score(y_test, gbdt_n.predict(X_test)))

plt.figure()
plt.plot(n_trees, acc_rf, label="RF Accuracy", linewidth=2, color='magenta')
plt.plot(n_trees, acc_gbdt, label="GBDT Accuracy", linewidth=2, color='cyan')
plt.title("Accuracy vs Number of Trees")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.legend()

# 3. Feature Importance
fi_rf = rf.feature_importances_
fi_gbdt = gbdt.feature_importances_
features = ['Feature 1', 'Feature 2']

x = np.arange(len(features))
width = 0.35

plt.figure()
plt.bar(x - width/2, fi_rf, width, label='RF', color='orange')
plt.bar(x + width/2, fi_gbdt, width, label='GBDT', color='lime')
plt.xticks(x, features)
plt.title("Feature Importance Comparison")
plt.ylabel("Importance")
plt.legend()

# 4. OOB Error vs Training Devience
# RF OOB error
oob_error = 1 - rf.oob_score_
# GBDT training deviance
test_deviance = np.zeros((50,), dtype=np.float64)
for i, pred in enumerate(gbdt.staged_predict_proba(X_test)):
    test_deviance[i] = gbdt.loss_(y_test, pred[:, 1]) #AttributeError: 'GradientBoostingClassifier' object has no attribute 'loss_'

plt.figure()
plt.plot([50], [oob_error], 'o', label='RF OOB Error', markersize=8, color='red')
plt.plot(n_trees, test_deviance, label='GBDT Test Deviance', linewidth=2, color='blue')
plt.title("RF OOB Error vs GBDT Test Deviance")
plt.xlabel("Number of Trees")
plt.ylabel("Error / Deviance")
plt.legend()

plt.show()