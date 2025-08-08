'''
通透！随机森林vs.回归树 ！！
https://mp.weixin.qq.com/s/V2txH_CLK-Glb-Oi-b9WGg

再传统机器学习中，回归树（Regression Tree）和随机森林（Random Forest）是两种用于回归问题的模型。
回归树是一个单一模型，而随机森林是基于回归树的集成策略。
回归树用于预测连续数值型变量。它通过对输入空间进行递归分割，在每个叶节点上预测一个值（通常是训练样本的均值）。
随机森林是多个回归树的集成模型，采用Bootstrap聚合（Bagging）策略，并在每次分裂时进行特征子采样。

模型对比分析
1. 过拟合控制
回归树：
容易过拟合：每次分裂都会追求最优MSE降低，容易将训练数据拟合得过度。控制手段：剪枝（pre-pruning/post-pruning）、限制最大深度、最小样本数等
随机森林：天生具有抗过拟合能力，主要得益于以下机制：
Bagging：通过不同的数据子集训练，减少方差
特征随机性：降低每棵树的相关性，增强集成多样性
2. 稳定性
回归树：
不稳定模型，对数据的微小扰动敏感
一个小的样本更改可能导致树结构剧烈变化
随机森林：
通过集成多个模型，极大提高鲁棒性
在噪声样本存在时依然表现良好
3. 泛化性能
回归树：
若控制得当（如剪枝），在低噪声、小数据集下可能泛化较好
但在复杂场景下表现有限
随机森林：
泛化能力强，尤其在高维、大样本数据下
4. 可解释性与计算复杂度

属性        回归树      随机森林
可解释性    高（树结构清晰）低（为多个树的集合）
训练复杂度  低              高（需训练多个树）
预测时间    快              慢（需遍历多棵树）
调参复杂度  简单            中等（需调树数、深度、特征数等）

总的来说，若追求模型可解释性、数据量较小且结构简单 → 可考虑使用回归树
若追求高性能、强泛化能力、面对复杂或高维数据 → 首选随机森林


完整案例
在回归建模中，我们经常面临模型选择的挑战：使用一个结构清晰、快速的回归树，还是使用精度更高但复杂的随机森林？本项目将围绕一个虚拟回归任务，通过生成非线性噪声数据，分别使用回归树和随机森林进行建模，并从以下角度对比：
拟合效果
过拟合风险
模型稳定性
泛化能力
最后结合实际图像、定量指标和代码，给出清晰的建模决策建议。
数据集
我们构建一个包含噪声的非线性目标函数y=sin(1.5Πx)+e,e~N(0,0.2^2)
我们将生成训练集和测试集，确保能观察到模型的泛化差异。
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


np.random.seed(42)

# 构造数据
X = np.sort(np.random.rand(200, 1))
y = np.sin(1.5 * np.pi * X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# 拆分训练与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建模型
tree = DecisionTreeRegressor(max_depth=4)
forest = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)

# 拟合模型
tree.fit(X_train, y_train)
forest.fit(X_train, y_train)

# 预测
x_plot = np.linspace(0, 1, 500).reshape(-1, 1)
y_tree_pred = tree.predict(x_plot)
y_forest_pred = forest.predict(x_plot)

# 数据分析可视化
plt.figure(figsize=(18, 12))
sns.set_style("whitegrid")
colors = sns.color_palette("bright")

# 1. 真实函数与训练数据分布图
plt.subplot(2, 2, 1)
plt.title("Ground Truth vs. Noisy Training Data", fontsize=14)
plt.plot(x_plot, np.sin(1.5 * np.pi * x_plot), color=colors[0], label="True Function", linewidth=2)
plt.scatter(X_train, y_train, color=colors[1], alpha=0.6, label="Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 2. 回归树拟合图
plt.subplot(2, 2, 2)
plt.title("Decision Tree Prediction", fontsize=14)
plt.plot(x_plot, np.sin(1.5 * np.pi * x_plot), color=colors[0], linestyle='--', label="True Function")
plt.plot(x_plot, y_tree_pred, color=colors[2], label="Tree Prediction", linewidth=2)
plt.scatter(X_test, y_test, color=colors[3], label="Test Data", alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 3. 随机森林拟合图
plt.subplot(2, 2, 3)
plt.title("Random Forest Prediction", fontsize=14)
plt.plot(x_plot, np.sin(1.5 * np.pi * x_plot), color=colors[0], linestyle='--', label="True Function")
plt.plot(x_plot, y_forest_pred, color=colors[4], label="Forest Prediction", linewidth=2)
plt.scatter(X_test, y_test, color=colors[3], label="Test Data", alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 4. 残差分布图（Residual Plot）
tree_residuals = y_test - tree.predict(X_test)
forest_residuals = y_test - forest.predict(X_test)

plt.subplot(2, 2, 4)
plt.title("Residual Comparison", fontsize=14)
sns.histplot(tree_residuals, color=colors[2], label="Tree Residuals", kde=True, stat="density", bins=20, alpha=0.6)
sns.histplot(forest_residuals, color=colors[4], label="Forest Residuals", kde=True, stat="density", bins=20, alpha=0.6)
plt.xlabel("Residual")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

'''
真实函数 vs 训练数据：说明数据呈明显非线性趋势，并带有噪声。
回归树预测图：呈“阶梯状”预测，说明其是基于区间均值预测，容易在边界处产生剧烈跳跃。
随机森林预测图：预测曲线更光滑，趋近真实函数，说明其集成策略更稳定，抗噪声能力更强。
残差图（误差分布）：随机森林残差更集中于0附近，说明其拟合更稳定，方差小，泛化性能更好。
'''


# 评估性能指标

tree_mse = mean_squared_error(y_test, tree.predict(X_test))
forest_mse = mean_squared_error(y_test, forest.predict(X_test))

tree_r2 = r2_score(y_test, tree.predict(X_test))
forest_r2 = r2_score(y_test, forest.predict(X_test))

print("决策树 MSE:", tree_mse)
print("随机森林 MSE:", forest_mse)
print("决策树 R²:", tree_r2)
print("随机森林 R²:", forest_r2)

'''
在回归问题中，是否采用集成模型往往是性能表现的关键：
如果你的数据稳定、结构简单、对模型可解释性有较高要求，则回归树可能是更优选择；
如果你的任务要求鲁棒性强、准确率高，面对的是非线性、复杂数据结构，那么随机森林几乎是稳妥首选。
从今天的案例可以清晰看到，虽然回归树的表现已不俗，但在面对复杂函数形式与噪声影响时，随机森林凭借其集成策略，实现了更高的精度与泛化能力。
'''