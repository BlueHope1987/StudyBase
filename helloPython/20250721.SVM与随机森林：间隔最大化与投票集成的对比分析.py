'''
https://mp.weixin.qq.com/s/q0kQq2DY_VmZ8HAnLWJ3dw
通透！SVM vs 随机森林 ！！

这是一个经典的机器学习对比问题，SVM 与随机森林分别代表间隔最大化的判别式模型与基于集成学习的投票式模型。
SVM：间隔最大化
SVM通过找到一个最优超平面，使得不同类别的样本被划分开，同时最大化到超平面的最小间隔（Margin）。

随机森林：投票集成（Ensemble of Decision Trees）
随机森林是由多个决策树组成的集成方法，通过：
Bagging（自助采样）：训练集随机采样
随机特征选择：每个节点分裂只在随机选取的特征子集上决策
最后通过多数投票（分类）或平均（回归）

核心差异：最大间隔 vs 投票集成

维度        SVM（最大间隔）                 随机森林（投票集成）
理论基础    结构风险最小化，最大间隔原则     Bagging，统计学习理论中的方差降低
决策边界    直接求全局最优边界              组合多棵树的局部划分
假设空间    假设存在一个理想的超平面        无需全局假设，靠集成优化
过拟合风险  对小样本，噪声敏感              通过Bagging降低方差，抗噪声好
特征尺度要求    特征需标准化                不需要特征标准化
可解释性    较差（尤其核SVM）               较强（单棵树易解释）
推理复杂度  高，尤其核方法                  低，易并行
支持非线性  通过核方法实现                  内在支持非线性划分

完整案例
这里，咱们试着通过对比实验，直观演示两种算法在不同数据条件下的决策边界、泛化性能、对噪声与过拟合的容忍度，帮助理解其适用场景。
算法配置：SVM：使用线性核与RBF核，随机森林：配置多棵树（如100棵）。

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import ListedColormap
import seaborn as sns


# 1. 生成线性和非线性数据
X_linear, y_linear = make_classification(n_samples=5000, n_features=2, 
                                         n_informative=2, n_redundant=0,
                                         n_clusters_per_class=1, class_sep=2, random_state=42)

X_nonlinear, y_nonlinear = make_circles(n_samples=5000, factor=0.5, noise=0.1, random_state=42)

# 2. 切分训练测试集
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_linear, y_linear, test_size=0.3, random_state=42)
X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(X_nonlinear, y_nonlinear, test_size=0.3, random_state=42)

# 3. 训练模型
svm_linear = SVC(kernel='linear', probability=True).fit(X_train_lin, y_train_lin)
svm_rbf = SVC(kernel='rbf', probability=True).fit(X_train_nl, y_train_nl)
rf_lin = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_lin, y_train_lin)
rf_nl = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_nl, y_train_nl)

# 辅助：绘制决策边界
def plot_decision_boundary(clf, X, y, ax, title):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#0000FF']
    
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor='k')
    ax.set_title(title)

# 4. 绘图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

plot_decision_boundary(svm_linear, X_test_lin, y_test_lin, axes[0,0], 'SVM (Linear Kernel) - Linear Data')
plot_decision_boundary(rf_lin, X_test_lin, y_test_lin, axes[0,1], 'Random Forest - Linear Data')
plot_decision_boundary(svm_rbf, X_test_nl, y_test_nl, axes[1,0], 'SVM (RBF Kernel) - Nonlinear Data')
plot_decision_boundary(rf_nl, X_test_nl, y_test_nl, axes[1,1], 'Random Forest - Nonlinear Data')

plt.tight_layout()
plt.show()

'''
决策边界图
展示不同模型对线性与非线性数据的决策边界。
SVM（线性核）在线性数据集上边界清晰直线；
SVM（RBF核）在非线性数据划分出圆形的非线性边界。
随机森林在两种情况下均能柔性拟合数据，但在高噪声下可能边界不够光滑。
'''



# 5. 混淆矩阵
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for clf, X_test, y_test, ax, title in zip(
    [svm_linear, rf_lin, svm_rbf, rf_nl],
    [X_test_lin, X_test_lin, X_test_nl, X_test_nl],
    [y_test_lin, y_test_lin, y_test_nl, y_test_nl],
    axes.flatten(),
    ['SVM Linear', 'RF Linear', 'SVM RBF', 'RF Nonlinear']
):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='plasma', colorbar=False)
    ax.set_title(f'Confusion Matrix - {title}')

plt.tight_layout()
plt.show()

'''
混淆矩阵
定量对比模型在测试集的分类效果。
直观了解各类的分类准确性、错误率。
'''

# 6. ROC曲线
def plot_roc_curve(ax, clf, X_test, y_test, title):
    y_score = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0,1], [0,1], color='navy', lw=1, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

plot_roc_curve(axes[0,0], svm_linear, X_test_lin, y_test_lin, 'SVM Linear Kernel ROC')
plot_roc_curve(axes[0,1], rf_lin, X_test_lin, y_test_lin, 'Random Forest Linear Data ROC')
plot_roc_curve(axes[1,0], svm_rbf, X_test_nl, y_test_nl, 'SVM RBF Kernel ROC')
plot_roc_curve(axes[1,1], rf_nl, X_test_nl, y_test_nl, 'Random Forest Nonlinear Data ROC')

plt.tight_layout()
plt.show()

'''
ROC曲线与AUC
评估模型的概率预测与分类门槛的性能。
AUC越接近1，模型越优秀。观察SVM在边界清晰时表现卓越，随机森林对非线性与噪声适应性好。
'''

# 7. 特征重要性（随机森林专属）
feature_names = ['Feature 1', 'Feature 2']
importances = rf_lin.feature_importances_

plt.figure(figsize=(8,6))
sns.barplot(x=feature_names, y=importances, palette='viridis')
plt.title('Feature Importance in Random Forest (Linear Data)')
plt.show()

'''
特征重要性（随机森林）
展示特征对分类的重要性评分。
SVM不提供此功能，随机森林有助于解释模型。

上面，咱们实验利用线性、非线性、带噪声的不同数据集，结合决策边界、混淆矩阵、ROC、特征重要性等多角度对比了SVM与随机森林的差异。
SVM胜在理论优雅、边界最优，但对超参数与核函数敏感。
随机森林胜在应用便捷、泛化好、解释性强，适合工业与快速迭代场景。
'''