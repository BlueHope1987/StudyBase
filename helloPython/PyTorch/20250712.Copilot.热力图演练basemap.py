import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.stats import gaussian_kde

# 1. 生成模拟数据：北京范围内的随机点
np.random.seed(42)
num_points = 300
# 北京经纬度范围（大致）
lon_min, lon_max = 116.2, 116.6
lat_min, lat_max = 39.7, 40.1
lons = np.random.uniform(lon_min, lon_max, num_points)
lats = np.random.uniform(lat_min, lat_max, num_points)
values = np.random.normal(loc=100, scale=20, size=num_points)  # 模拟数值

# 2. 创建北京地图底图
fig, ax = plt.subplots(figsize=(10, 8))
m = Basemap(
    llcrnrlon=lon_min, llcrnrlat=lat_min,
    urcrnrlon=lon_max, urcrnrlat=lat_max,
    resolution='i', projection='merc', ax=ax
)
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='beige', lake_color='lightblue')
m.drawcoastlines()
m.drawcountries()
m.drawrivers(color='blue')
m.drawparallels(np.arange(lat_min, lat_max, 0.05), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(lon_min, lon_max, 0.05), labels=[0,0,0,1], fontsize=10)

# 3. 热力图数据准备：核密度估计
x, y = m(lons, lats)
xy = np.vstack([x, y])
kde = gaussian_kde(xy, weights=values)
xi, yi = np.meshgrid(
    np.linspace(x.min(), x.max(), 300),
    np.linspace(y.min(), y.max(), 300)
)
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

# 4. 绘制热力图
heatmap = m.imshow(
    zi.reshape(xi.shape),
    cmap='hot', alpha=0.6,
    extent=[x.min(), x.max(), y.min(), y.max()],
    interpolation='bilinear'
)

# 5. 叠加原始点
m.scatter(x, y, c=values, cmap='cool', edgecolor='k', s=40, alpha=0.7, label='数据点')

# 6. 添加色带、标题等
cbar = plt.colorbar(heatmap, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('热力强度', fontsize=12)
plt.title('北京市随机数据热力图示例', fontsize=16)
plt.legend(loc='lower left')

# 7. 保存和展示
plt.tight_layout()
plt.savefig('beijing_heatmap_demo.png', dpi=200)
plt.show()

# =========================
# 教程说明：
# - 本示例演示了如何在北京地图上叠加热力图。
# - 使用Basemap绘制底图，核密度估计生成热力分布。
# - 可调整点数、色带、透明度等参数实现不同效果。
# - 适合教学和数据可视化入门。
