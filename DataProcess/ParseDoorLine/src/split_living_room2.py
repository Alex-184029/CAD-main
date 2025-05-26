import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from deal_labelme1 import find_living_room_label
import numpy as np

def calc_convex(polygon):
    # 1. 定义一个带有凸起的多边形
    # coords = [(1, 1), (5, 1), (5, 3), (6, 3), (6, 6), (2, 6), (2, 4), (1, 4), (1, 1)]
    # polygon = Polygon(coords)

    # 2. 计算凸包
    convex_hull = polygon.convex_hull

    # 3. 计算“向外凸起”部分
    convex_diff = polygon.difference(convex_hull)

    # 4. 输出凸起区域的 Polygon 表示
    if not convex_diff.is_empty:
        if convex_diff.geom_type == "Polygon":
            protrusions = [convex_diff]  # 只有一个凸起区域
        else:
            protrusions = list(convex_diff.geoms)  # 多个凸起区域
        print("检测到的凸起区域:")
        for i, protrusion in enumerate(protrusions, start=1):
            print(f"凸起区域 {i}: {protrusion}")
    else:
        print("未检测到明显的凸起区域。")

    # 5. 可视化多边形及其凸起部分
    fig, ax = plt.subplots()
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'b-', label="Original Polygon")

    x, y = convex_hull.exterior.xy
    ax.plot(x, y, 'g--', label="Convex Hull")

    # 可视化凸起部分
    if not convex_diff.is_empty:
        for geom in protrusions:
            x, y = geom.exterior.xy
            ax.fill(x, y, 'r', alpha=0.5, label="Protrusions")

    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Convex Detection")
    plt.show()

def angle_between(p1, p2, p3):
    """计算 p2 处的内角（单位：度）"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_prod = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_prod / norm) if norm != 0 else 0
    return np.degrees(angle)

def detect_protrusions(polygon, angle_threshold=90, visualize=True):
    """
    识别并可视化多边形的凸起部分
    :param polygon: 输入的 shapely.geometry.Polygon 对象
    :param angle_threshold: 角度阈值（小于该值的角认为是凸起）
    :param visualize: 是否可视化结果
    :return: 凸起区域的 Polygon 列表
    """
    coords = list(polygon.exterior.coords)
    protrusion_points = []
    protrusions = []

    # 1. 遍历所有顶点，计算夹角
    for i in range(1, len(coords) - 1):
        ang = angle_between(coords[i - 1], coords[i], coords[i + 1])
        if ang < angle_threshold:  # 设定阈值，找出凸起的角
            protrusion_points.append(coords[i])

    # 2. 生成凸起区域 Polygon
    for point in protrusion_points:
        prev_point = coords[coords.index(point) - 1]  # 之前的点
        next_point = coords[coords.index(point) + 1]  # 之后的点
        protrusion_poly = Polygon([prev_point, point, next_point])  # 近似三角形凸起区域
        protrusions.append(protrusion_poly)

    # 3. 输出凸起区域
    if protrusions:
        print("检测到的凸起区域:")
        for i, protrusion in enumerate(protrusions, start=1):
            print(f"凸起区域 {i}: {protrusion}")
    else:
        print("未检测到明显的凸起区域。")

    # 4. 可视化
    if visualize:
        fig, ax = plt.subplots()
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-', label="Polygon")

        # 标记凸起点
        for px, py in protrusion_points:
            ax.scatter(px, py, color='r', s=100, label="Protrusion Points")

        # 可视化凸起区域
        for protrusion in protrusions:
            x, y = protrusion.exterior.xy
            ax.fill(x, y, 'r', alpha=0.5, label="Protrusions")

        ax.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Convex Detection")
        plt.show()

    return protrusions

def find_and_plot_convexities(polygon):
    # 计算凸包
    convex_hull = polygon.convex_hull
    
    # 计算差异区域
    difference = convex_hull.difference(polygon)
    
    # 可视化
    fig, ax = plt.subplots()
    
    # 绘制原始多边形
    x, y = polygon.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='blue', label='Original Polygon')
    
    # 绘制凸包
    x, y = convex_hull.exterior.xy
    ax.plot(x, y, 'r--', label='Convex Hull')
    
    # 绘制凸起部分
    if isinstance(difference, MultiPolygon):
        for geom in difference.geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='green', label='Convexity')
    elif isinstance(difference, Polygon):
        x, y = difference.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='green', label='Convexity')
    
    ax.legend()
    plt.show()
    return difference

def find_convex_corners(polygon):
    coords = list(polygon.exterior.coords)
    convex_corners = []
    
    for i in range(1, len(coords)-1):
        prev = coords[i-1]
        curr = coords[i]
        next_p = coords[i+1]
        
        # 计算叉积确定拐角方向
        cross = (curr[0] - prev[0]) * (next_p[1] - curr[1]) - (curr[1] - prev[1]) * (next_p[0] - curr[0])
        
        if cross < 0:  # 凸拐角
            convex_corners.append(curr)

    plt.figure()
    x, y = polygon.exterior.xy
    plt.fill(x, y, alpha=0.5, fc='blue')
    plt.plot(*zip(*convex_corners), 'ro', markersize=10, label='Convex Corners')
    plt.legend()
    plt.show()
    
    return convex_corners

def find_convexities_by_rectangles(polygon, min_area=1):
    # 获取多边形的边界
    min_x, min_y, max_x, max_y = polygon.bounds
    
    # 创建网格
    x = min_x
    rectangles = []
    while x < max_x:
        y = min_y
        while y < max_y:
            # 创建小矩形
            rect = Polygon([(x, y), (x+min_area, y), (x+min_area, y+min_area), (x, y+min_area)])
            if polygon.contains(rect):
                rectangles.append(rect)
            y += min_area
        x += min_area
    
    # 合并相邻矩形
    merged = unary_union(rectangles)
    
    # 差异部分即为凸起区域
    convexities = polygon.difference(merged)
    
    # 可视化
    fig, ax = plt.subplots()
    x, y = polygon.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='blue', label='Original Polygon')
    
    if isinstance(merged, MultiPolygon):
        for geom in merged.geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='red', label='Main Body')
    else:
        x, y = merged.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='red', label='Main Body')
    
    if isinstance(convexities, MultiPolygon):
        for geom in convexities.geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='green', label='Convexities')
    elif isinstance(convexities, Polygon):
        x, y = convexities.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='green', label='Convexities')
    
    ax.legend()
    plt.show()
    
    return convexities

def find_protrusions(ortho_polygon):
    """
    找出正交多边形中的所有凸起部分
    返回一个多边形列表，每个多边形代表一个独立的凸起区域
    """
    # 计算整个多边形的凸包
    convex_hull = ortho_polygon.convex_hull
    
    # 计算凸包与原始多边形的差异
    difference = convex_hull.difference(ortho_polygon)
    
    # 将差异部分分解为独立的多边形
    protrusions = []
    if isinstance(difference, MultiPolygon):
        protrusions = list(difference.geoms)
    elif isinstance(difference, Polygon):
        if not difference.is_empty:
            protrusions = [difference]
    
    return protrusions

def visualize(original, protrusions):
    """可视化原始多边形和凸起部分"""
    fig, ax = plt.subplots()
    
    # 绘制原始多边形
    x, y = original.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='blue', label='Original Polygon')
    
    # 绘制每个凸起部分
    colors = ['red', 'green', 'yellow', 'purple', 'orange']
    for i, protrusion in enumerate(protrusions):
        x, y = protrusion.exterior.xy
        ax.fill(x, y, alpha=0.5, fc=colors[i % len(colors)], 
                label=f'Protrusion {i+1}')
    
    ax.legend()
    plt.axis('equal')
    plt.show()

def find_protrusions2(polygon, visualize=True):
    """
    识别并可视化多边形中的凸起部分
    :param polygon: 输入的 shapely.geometry.Polygon（仅包含水平和竖直边）
    :param visualize: 是否可视化凸起区域
    :return: 凸起区域的 Polygon 列表
    """
    # 计算凸包（Convex Hull）
    convex_hull = polygon.convex_hull

    # 计算“凸起”部分（原始多边形 - 凸包）
    convex_diff = polygon.difference(convex_hull)

    # 提取凸起区域
    protrusions = []
    if not convex_diff.is_empty:
        if convex_diff.geom_type == "Polygon":
            protrusions = [convex_diff]  # 只有一个凸起区域
        elif convex_diff.geom_type == "MultiPolygon":
            protrusions = list(convex_diff.geoms)  # 多个凸起区域

    # 输出凸起区域
    if protrusions:
        print("检测到的凸起区域:")
        for i, protrusion in enumerate(protrusions, start=1):
            print(f"凸起区域 {i}: {protrusion}")
    else:
        print("未检测到凸起区域。")

    # 可视化
    if visualize:
        fig, ax = plt.subplots()
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-', label="Origin Part")

        x, y = convex_hull.exterior.xy
        ax.plot(x, y, 'g--', label="Hull")

        # 可视化凸起部分
        for protrusion in protrusions:
            x, y = protrusion.exterior.xy
            ax.fill(x, y, 'r', alpha=0.5, label="Convex Part")

        ax.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Convex Detection")
        plt.show()

    return protrusions


if __name__ == '__main__':
    room = find_living_room_label()
    # find_convexities_by_rectangles(room['poly'])

    protrusions = find_protrusions2(room['poly'])
    # print(f"Found {len(protrusions)} protrusions in complex shape")
    # visualize(room['poly'], protrusions)

