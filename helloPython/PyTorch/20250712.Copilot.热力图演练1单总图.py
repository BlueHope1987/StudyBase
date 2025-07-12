import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde

# 1. 生成模拟数据：北京范围内的随机点
np.random.seed(42)
num_points = 300
lon_min, lon_max = 116.2, 116.6
lat_min, lat_max = 39.7, 40.1
lons = np.random.uniform(lon_min, lon_max, num_points)
lats = np.random.uniform(lat_min, lat_max, num_points)
values = np.random.normal(loc=100, scale=20, size=num_points)  # 模拟数值

# 2. Cartopy地图设置
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='beige')
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.RIVERS, edgecolor='blue')
ax.add_feature(cfeature.COASTLINE)
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# 3. 热力图数据准备：核密度估计
xy = np.vstack([lons, lats])
kde = gaussian_kde(xy, weights=values)
xi, yi = np.meshgrid(
    np.linspace(lon_min, lon_max, 300),
    np.linspace(lat_min, lat_max, 300)
)
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

# 4. 绘制热力图（叠加在地图上）
heatmap = ax.imshow(
    zi.reshape(xi.shape),
    origin='lower',
    extent=[lon_min, lon_max, lat_min, lat_max],
    transform=ccrs.PlateCarree(),
    cmap='hot', alpha=0.6, interpolation='bilinear'
)

# 5. 叠加原始点
sc = ax.scatter(
    lons, lats, c=values, cmap='cool', edgecolor='k', s=40, alpha=0.7,
    transform=ccrs.PlateCarree(), label='数据点'
)

# 6. 添加色带、标题等
cbar = plt.colorbar(heatmap, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('热力强度', fontsize=12)
plt.title('北京市随机数据热力图示例（Cartopy版）', fontsize=16)
plt.legend(loc='lower left')

# 7. 保存和展示
plt.tight_layout()
plt.savefig('beijing_heatmap_cartopy_demo.png', dpi=200)
plt.show()

# =========================
# 教程说明：
# - 本示例演示了如何用Cartopy在北京地图上叠加热力图。
# - 使用核密度估计生成空间热点分布。
# - 可调整点数、色带、透明度等参数实现不同效果。
# - Cartopy是主流地理可视化库，推荐用于新项目。
# - 适合教学和数据可视化入门。
