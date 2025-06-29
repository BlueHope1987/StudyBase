# https://mp.weixin.qq.com/s/Gy3EeNyI88LJ_Q01kes6Wg
# 超全面讲透一个算法模型，层次聚类 ！！
'''
什么是层次聚类？
1. 分组的过程：层次聚类是把数据分成一组一组的过程。它从“每个数据点自己是一个组”开始，慢慢地把距离近的组合并起来，最后变成一个大组，包含所有数据点。
或者反过来，从“所有数据点是一大组”开始，慢慢拆分，直到每个数据点都是一个单独的组。
2. 像盖房子或砍树：
如果是从小组开始逐步合并，像“盖房子”一样，把一块砖（数据点）接到一起，最后盖成一栋房子。
如果是从一个大组开始逐步拆分，像“砍树”一样，从树干逐渐分成树枝，再分到小树枝，最后到叶子。
3. 输出的结果：最终会得到一个树形结构（叫做树状图/树形图），它展示了所有分组的关系，比如哪两组是最早合并的，哪组是最后才合并的。

怎么做层次聚类？
1. 测量距离：先计算数据点之间的距离，比如两点之间的直线距离（欧几里得距离）或者其他方法。
2. 合并最近的点或组：找到距离最近的两个点或组，把它们合并成一个新组。
3. 重复步骤：继续计算新的组和其他点或组的距离，重复合并过程。
4. 直到所有数据在一个组里。

举个简单例子
假设你有5个朋友，他们的身高分别是：
A: 160 cm   B: 165 cm   C: 170 cm   D: 175 cm   E: 180 cm
目标：用层次聚类把他们分成组。
1. 计算身高差（距离）：
A 和 B 的差距是 5 cm，B 和 C 的差距是 5 cm，以此类推。
A 和 C 的差距是 10 cm，A 和 D 的差距是 15 cm，依此类推。
2. 找到最近的两个点合并：
A 和 B 的距离最小（5 cm），所以把 A 和 B 合并成一组（AB组）。
3. 更新组与其他人的距离：
AB组与 C 的距离可以用两种方法计算：
最小距离：AB 和 C 的距离是 10 cm（A 和 C 的差距）。
平均距离：AB 和 C 的距离是 (5+10)/2 = 7.5 cm。
继续选择方法，根据情况决定。
4. 继续合并最近的组：
AB 和 C 的距离最小，所以合并成 ABC组。
继续合并 D 和 E。
5. 直到所有人都合并成一个大组。
最终，你会得到一个树形图，展示了 A 和 B 最早合并，接着与 C 合并，最后与 D 和 E 合并的过程。

层次聚类详细推导
距离度量（欧几里得距离）
欧几里得距离是最常用的两点之间的距离度量
聚类间的距离计算方法
在层次聚类中，我们有多种方法来计算两个簇之间的距离。常见的三种方法是：
单链接（最小距离）：选择两个簇中距离最近的点之间的距离。
全链接（最大距离）：选择两个簇中距离最远的点之间的距离。
平均链接（均值距离）：计算两个簇之间所有点对的平均距离。

层次聚类的过程
1. 初始化：每个点开始时都是一个单独的簇。
2. 计算距离：计算所有簇之间的距离。
3. 合并簇：选择距离最小的两个簇进行合并。
4. 更新距离矩阵：合并簇后，更新簇间的距离。
5. 重复：重复步骤3和4，直到所有点合并为一个簇。

'''



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # seaborn 是一个基于 matplotlib 的 Python 数据可视化库，提供了一个高级接口，用于绘制吸引人且信息丰富的统计图形。
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# 1. 生成虚拟数据集
# 使用 make_blobs 生成一个包含15个样本的虚拟数据集，其中数据点分布在3个中心点周围。每个数据点都有一个标签，便于后续分析。
np.random.seed(42)
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)
labels = [f"Point {i}" for i in range(len(X))]

# 2. 执行层次聚类 
# 使用 sklearn 的 AgglomerativeClustering 类进行层次聚类。该方法使用距离阈值和簇数量控制聚类过程。
# 我们设置了 n_clusters=None 和 distance_threshold=0，让聚类继续合并直到形成一个完整的树。
agg_clust = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
agg_clust.fit(X)

# 3. 可视化
def plot_data(X, labels):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], color='blue', s=100)
    for i, label in enumerate(labels):
        plt.text(X[i, 0] + 0.2, X[i, 1], label, fontsize=9, color='darkred')
    plt.title("Scatter Plot of Points")
    plt.show()
#数据点分布：散点图展示了数据点的分布。通过观察图形，我们可以看到数据点相对聚集在几个区域，这为层次聚类提供了线索。

def plot_dendrogram(X):
    # 创建一个 linkage 矩阵
    Z = linkage(X, 'ward')
    plt.figure(figsize=(10, 8))
    dendrogram(Z, truncate_mode='level', p=3)
    plt.title("Dendrogram (Hierarchical Clustering Tree)")
    plt.show()
#层次聚类树状图：树状图揭示了聚类过程的细节。每次合并两个簇都会形成一个树节点。树状图显示了最先合并的点以及每次合并的距离，可以帮助我们选择聚类的合适数量。

def plot_cluster_result(X, labels, title="Clustered Data"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="Set1", s=100, legend='full')
    plt.title(title)
    plt.show()
#聚类结果：聚类后的数据点散点图，展示了层次聚类算法如何将数据点分成多个簇。图中使用不同的颜色来标识不同的簇。

def plot_distance_matrix(X):
    from sklearn.metrics import pairwise_distances
    dist_matrix = pairwise_distances(X)
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Distance Matrix")
    plt.show()
#距离矩阵热力图：热力图展示了数据点之间的距离。通过观察热力图，我们可以清晰地看到哪些数据点相互接近，哪些点之间的距离较远，这对于理解聚类效果至关重要。

# 图1：数据点分布
plot_data(X, labels)

# 图2：层次聚类树状图
plot_dendrogram(X)

# 图3：聚类结果
plot_cluster_result(X, agg_clust.labels_, title="Clustered Data (Agglomerative)")

# 图4：距离矩阵热力图
plot_distance_matrix(X)