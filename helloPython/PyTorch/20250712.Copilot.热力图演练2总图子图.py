import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde
import random

# 设置中文字体（如微软雅黑），确保中文正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成马卡龙色系
def macaron_colors(n):
    base_colors = [
        "#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA",
        "#B5B2FF", "#B2FFFF", "#FFD6E0", "#FFECB2", "#B2FFEC"
    ]
    return [random.choice(base_colors) for _ in range(n)]

# 1. 生成模拟数据：北京范围内的随机点
np.random.seed(42)
num_points = 300
lon_min, lon_max = 116.2, 116.6
lat_min, lat_max = 39.7, 40.1
lons = np.random.uniform(lon_min, lon_max, num_points)
lats = np.random.uniform(lat_min, lat_max, num_points)
values = np.random.normal(loc=100, scale=20, size=num_points)

# 2. 创建总图和子图布局
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.15)

# 3. 总图：中国地图，标记北京区域，省辖区随机马卡龙色填充
ax_main = fig.add_subplot(gs[0], projection=ccrs.Mercator())
ax_main.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())
ax_main.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
ax_main.add_feature(cfeature.BORDERS, linestyle=':')
ax_main.add_feature(cfeature.COASTLINE)

# 尝试加载省界 shapefile（如无则跳过，仅用LAND）
try:
    from cartopy.io.shapereader import Reader
    import cartopy.feature as cfeature
    # 省界 shapefile 路径（需自行下载中国省界数据）
    shp_path = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )
    # 读取省界多边形
    reader = Reader(shp_path.path)
    provinces = list(reader.geometries())
    colors = macaron_colors(len(provinces))
    for geom, color in zip(provinces, colors):
        ax_main.add_geometries([geom], ccrs.PlateCarree(), facecolor=color, edgecolor='gray', linewidth=0.5, alpha=0.8)
except Exception:
    # 若无省界数据则用LAND填充
    ax_main.add_feature(cfeature.LAND.with_scale('50m'), facecolor=random.choice(macaron_colors(1)), alpha=0.8)

ax_main.set_title('中国地图与北京区域', fontsize=14)
# 用红色矩形框标记北京区域
rect = mpatches.Rectangle(
    (lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
    linewidth=2, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree()
)
ax_main.add_patch(rect)
ax_main.text(lon_min, lat_max+0.5, '北京', color='red', fontsize=12, transform=ccrs.PlateCarree())

# 4. 子图：北京热力图
ax = fig.add_subplot(gs[1], projection=ccrs.Mercator())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='beige')
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.RIVERS, edgecolor='blue')
ax.add_feature(cfeature.COASTLINE)
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# 热力图数据准备
xy = np.vstack([lons, lats])
kde = gaussian_kde(xy, weights=values)
xi, yi = np.meshgrid(
    np.linspace(lon_min, lon_max, 300),
    np.linspace(lat_min, lat_max, 300)
)
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

# 绘制热力图
heatmap = ax.imshow(
    zi.reshape(xi.shape),
    origin='lower',
    extent=[lon_min, lon_max, lat_min, lat_max],
    transform=ccrs.PlateCarree(),
    cmap='hot', alpha=0.6, interpolation='bilinear'
)

# 叠加原始点
sc = ax.scatter(
    lons, lats, c=values, cmap='cool', edgecolor='k', s=40, alpha=0.7,
    transform=ccrs.PlateCarree(), label='数据点'
)

# 添加色带、标题等
cbar = plt.colorbar(heatmap, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('热力强度', fontsize=12)
ax.set_title('北京市随机数据热力图（局部放大）', fontsize=16)
ax.legend(loc='lower left')

# 5. 连线：总图北京框体与子图区域连线
# 获取总图和子图的坐标系转换
def get_display_coords(ax, lon, lat):
    # 经纬度转为显示坐标
    return ax.transData.transform(ax.projection.transform_point(lon, lat, ccrs.PlateCarree()))

# 总图北京框体右上角
main_xy = get_display_coords(ax_main, lon_max, lat_max)
# 子图左上角
sub_xy = get_display_coords(ax, lon_min, lat_max)

# 在figure坐标系下绘制连线
fig.canvas.draw()  # 确保坐标转换有效
line = plt.Line2D(
    [main_xy[0], sub_xy[0]], [main_xy[1], sub_xy[1]],
    transform=fig.transFigure, color='red', linewidth=2, linestyle='--', alpha=0.7
)
fig.lines.append(line)

# 6. 布局与保存
plt.tight_layout()
plt.savefig('beijing_heatmap_cartopy_overview_macaron.png', dpi=200)
plt.show()

# =========================
# 教程说明：
# - 总图各省辖区随机马卡龙色填充，增强辨识性和美观性。
# - 总图北京区域与子图热力图区域通过连线关联，空间关系清晰。
# - 设置中文字体，确保中文标题和标注正常显示。
# - 适合空间数据可视化、地理教学演示。
