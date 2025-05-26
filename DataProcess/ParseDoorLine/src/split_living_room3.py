from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deal_labelme1 import find_living_room_label

def find_convex_regions(polygon):
    """
    找出多边形中的所有凸起区域
    """
    convex_regions = []
    n = len(polygon.exterior.coords) - 1  # 减去重复的起始点

    for i in range(n):
        p1 = polygon.exterior.coords[i]
        p2 = polygon.exterior.coords[(i + 1) % n]
        p3 = polygon.exterior.coords[(i + 2) % n]

        # 计算向量
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # 计算向量的叉积
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        if cross_product > 0:  # 凸起
            # 找到凸起区域的边界
            region_points = [p2]

            # 向左扩展
            left_index = i
            while left_index >= 0 and polygon.exterior.coords[(left_index + 1) % n][1] == p2[1]:
                region_points.append(polygon.exterior.coords[left_index])
                left_index -= 1

            # 向右扩展
            right_index = (i + 2) % n
            while right_index != left_index and polygon.exterior.coords[right_index][1] == p2[1]:
                region_points.append(polygon.exterior.coords[right_index])
                right_index = (right_index + 1) % n

            # 添加底部点
            region_points.append(polygon.exterior.coords[(right_index + 1) % n])
            region_points.append(polygon.exterior.coords[(left_index + 1) % n])

            # 创建凸起区域的多边形
            convex_region = Polygon(region_points)
            convex_regions.append(convex_region)

    return convex_regions

def plot_polygon(polygon, color='blue', label=None):
    """
    绘制多边形
    """
    x, y = polygon.exterior.xy
    plt.plot(x, y, color=color, label=label)
    plt.fill(x, y, color=color, alpha=0.3)

# 示例多边形（凸字形状）
# polygon = Polygon([
#     (0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (0, 3)
# ])
room = find_living_room_label()
polygon = room['poly']

convex_regions = find_convex_regions(polygon)

# 绘制原始多边形
plt.figure()
plot_polygon(polygon, color='blue', label='Original Polygon')

# 绘制凸起区域
colors = ['red', 'green', 'purple']  # 不同颜色用于区分凸起区域
for i, region in enumerate(convex_regions):
    plot_polygon(region, color=colors[i % len(colors)], label=f'Convex Region {i + 1}')

# 添加图例和显示图形
plt.legend()
plt.title("Polygon and Convex Regions")
plt.grid(True)
plt.axis('equal')  # 保持比例
plt.show()