# 最强特征选择，互信息法 ！！
# https://mp.weixin.qq.com/s/RrxntQY8NiqlGGvp3mfbeQ

'''
想象一下，你在猜一个秘密（目标变量），然后你有一些线索（特征）。
互信息法就像是在告诉你，这些线索（特征）能帮助你多大程度上猜到这个秘密（目标变量）。
如果一个特征能让你更好地猜到目标变量，那它就是有用的。如果它基本上对目标没有影响，那么它就不是很有用。

信息论的核心概念包括：
信息熵（Entropy）：用来衡量一个随机变量的不确定性或信息量
条件熵（Conditional Entropy）：表示在已知某个变量  的情况下，变量  的不确定性
联合熵（Joint Entropy）：表示两个变量  和  一起的熵，衡量了  和  的联合不确定性
互信息（Mutual Information）衡量的是两个变量之间的信息共享量。简单来说，互信息越大，两个变量之间的关系越强，它们所包含的信息越相似。
两个变量X和Y的互信息I(X;Y)等于它们的单独信息熵之和，减去它们的联合信息熵。
在特征选择中，我们的目标是选择与目标变量（标签）具有最大互信息的特征。
互信息法的特征选择流程大致如下：
1.计算特征与目标变量之间的互信息
2.选择互信息最大的特征：互信息值越高，意味着该特征和目标变量之间的关系越强，对预测目标变量更有帮助。
3.筛选特征： 根据设定的阈值，或者通过选择排名前 K 的特征（根据互信息的大小），来选择最终的特征子集。
互信息的估计方法：
在实际应用中，由于我们通常面对的是有限的数据集，因此需要对互信息进行估计。一些常用的互信息估计方法：
1.离散化法：将连续特征离散化为若干类别，之后计算互信息。
2.基于直方图的估计：通过构建数据的概率密度直方图来估计熵和联合熵。
3.K近邻估计法：利用 K 近邻算法估计概率密度，进而计算熵和互信息。
互信息法在特征选择中的关键在于如何计算互信息，并且在面对高维数据时需要有效地估计概率分布。
通过提到的这些方法，能够有效地提高模型的性能，减少冗余特征，提高计算效率。

案例

我们使用一个模拟数据集，该数据集包含一些与健康状况相关的特征，如年龄、体重、身高、饮食习惯、运动频率等。
变量是一个二分类标签，表示该人是否患有某种疾病。
目标：使用互信息法进行特征选择。通过可视化展示特征选择的过程。使用交叉验证优化模型。
数据集构建：
我们将构建一个虚拟的数据集，使用pandas和numpy生成数据。该数据集将包含以下特征：
age（年龄）height（身高）weight（体重）diet（饮食习惯，分类变量）exercise（运动频率，分类变量）health_status（健康状况，目标变量）
代码实现：包括数据生成、特征选择、模型训练与优化、可视化分析等步骤。

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

np.random.seed(42)

# 1. 构建虚拟数据集
'''
1. 数据集生成与预处理
在这个部分，我们首先创建了一个模拟数据集，其中包含五个特征（age, height, weight, diet, exercise）和一个目标变量（health_status）。
使用了LabelEncoder对分类变量（diet, exercise）进行编码，将其转化为数值型变量。
接着，我们对特征进行了标准化（使用StandardScaler），确保每个特征的值在同一尺度下，从而防止某些特征对模型训练的影响过大。
'''
def create_dataset():
    # 随机生成数据
    n_samples = 1000
    age = np.random.randint(18, 80, size=n_samples)
    height = np.random.randint(150, 200, size=n_samples)
    weight = np.random.randint(40, 120, size=n_samples)
    diet = np.random.choice(['Good', 'Average', 'Poor'], size=n_samples)
    exercise = np.random.choice(['None', 'Low', 'High'], size=n_samples)
    
    # 生成目标变量 health_status (0 = 健康, 1 = 患病)
    health_status = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # 将数据组织为一个DataFrame
    df = pd.DataFrame({
        'age': age,
        'height': height,
        'weight': weight,
        'diet': diet,
        'exercise': exercise,
        'health_status': health_status
    })
    
    return df

# 创建数据集
df = create_dataset()

# 2. 数据预处理

# 显示数据集的前几行
print("Dataset preview:")
print(df.head())

# 将分类变量转化为数值型变量
label_encoder = LabelEncoder()
df['diet'] = label_encoder.fit_transform(df['diet'])
df['exercise'] = label_encoder.fit_transform(df['exercise'])

# 将特征与目标分开
X = df.drop('health_status', axis=1)
y = df['health_status']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 计算互信息（Mutual Information）
'''
2. 互信息计算与特征选择
我们通过mutual_info_classif函数计算每个特征与目标变量之间的互信息值。
互信息值高的特征，意味着它们与目标变量之间的关系更强，因此是更有用的特征。
我们通过可视化展示了每个特征的互信息值，并选择互信息值最高的前三个特征来进行模型训练。
'''
# 计算互信息
mi = mutual_info_classif(X_scaled, y)

# 将互信息与特征对应
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})

# 按互信息值排序
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# 显示互信息排序结果
print("\nMutual Information of features:")
print(mi_df)

# 4. 特征选择：选择互信息最大的特征

# 可视化互信息
plt.figure(figsize=(10, 6))
sns.barplot(x='Mutual Information', y='Feature', data=mi_df, palette='Set2')
plt.title('Mutual Information of Features')
plt.xlabel('Mutual Information')
plt.ylabel('Feature')
plt.show()

# 根据互信息选择最重要的特征
top_features = mi_df['Feature'].head(3).values  # 选择前3个最相关的特征
print("\nTop 3 most relevant features:", top_features)

# 根据互信息选择特征
X_selected = X[top_features]

# 5. 数据集拆分

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 6. 模型训练与评估
'''
3. 模型训练与评估
我们使用RandomForestClassifier来训练一个分类模型，并通过交叉验证和混淆矩阵评估模型的性能。
交叉验证可以帮助我们更好地理解模型在不同数据集上的表现，混淆矩阵则能够直观展示模型在分类任务中的预测效果。
'''
# 训练一个随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 输出模型评估报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 交叉验证评分
cross_val_score_result = cross_val_score(model, X_selected, y, cv=5)
print("\nCross-validation scores:", cross_val_score_result)
print("Mean cross-validation score:", cross_val_score_result.mean())

# 7. 模型优化（可选）

# 使用GridSearchCV来优化随机森林的超参数
'''
4. 模型优化
我们使用GridSearchCV对模型进行超参数调优。通过调整随机森林分类器的参数（如n_estimators, max_depth, min_samples_split），
我们可以获得性能更好的模型。调优后的模型再次进行评估，并与未调优的模型进行对比，展示了优化带来的性能提升。
'''
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 输出最好的超参数组合
print("\nBest parameters from GridSearchCV:", grid_search.best_params_)

# 使用最好的超参数训练模型
best_model = grid_search.best_estimator_

# 评估优化后的模型
y_pred_optimized = best_model.predict(X_test)

'''
5. 可视化
在整个过程中，我们使用了matplotlib和seaborn来可视化数据和结果。
特别是互信息的条形图和混淆矩阵的热力图，帮助我们更好地理解数据，还能直观地展示模型的优劣。
'''

# 输出优化后的分类报告
print("\nOptimized Classification Report:")
print(classification_report(y_test, y_pred_optimized))

# 混淆矩阵
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.title('Optimized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 交叉验证评分
cross_val_score_result_optimized = cross_val_score(best_model, X_selected, y, cv=5)
print("\nOptimized cross-validation scores:", cross_val_score_result_optimized)
print("Mean optimized cross-validation score:", cross_val_score_result_optimized.mean())

'''
算法优化与建议
特征选择优化：可以尝试使用其他特征选择方法，如基于模型的特征选择（如L1正则化）或者递归特征消除（RFE）等。
数据增强：如果数据不够平衡，可以尝试使用过采样（如SMOTE）或欠采样来平衡类别分布。
模型优化：可以尝试其他模型（如支持向量机、XGBoost等）进行对比，选择最适合的数据模型。
交叉验证优化：使用更多的交叉验证折数（如10折）来提高模型的可靠性。
通过互信息法，我们能够筛选出与目标变量最相关的特征，这有助于提高模型的性能，减少计算复杂度。
'''