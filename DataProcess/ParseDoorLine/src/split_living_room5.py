from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt

def find_protrusions(polygon):
    """
    找出由水平边和垂直边组成的多边形中的所有凸起区域
    
    参数:
        polygon: shapely.geometry.Polygon 输入多边形
        
    返回:
        list: 包含所有凸起区域的Polygon对象列表
    """
    # 获取多边形的外环坐标
    coords = list(polygon.exterior.coords)
    
    # 移除重复的最后一个点(如果与第一个点相同)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    
    protrusions = []
    n = len(coords)
    
    # 遍历所有顶点寻找凹点
    for i in range(n):
        prev = coords[(i-1)%n]
        curr = coords[i]
        next_ = coords[(i+1)%n]
        
        # 计算向量
        v1 = (curr[0] - prev[0], curr[1] - prev[1])  # 前一边的向量
        v2 = (next_[0] - curr[0], next_[1] - curr[1])  # 当前边的向量
        
        # 计算叉积确定转向(因为是水平垂直边，所以叉积只有0, ±1)
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        
        # 如果叉积为正，是左转(凸点)；为负是右转(凹点)
        if cross < 0:  # 这是一个凹点
            # 根据边的方向确定凸起类型
            if v1[0] == 0:  # 前一边是垂直边
                if v1[1] > 0:  # 向上
                    if v2[0] > 0:  # 向右
                        # 这是一个朝向右上的凹点，可能对应左侧凸起
                        protrusion = find_rectangle_protrusion(polygon, curr, 'left')
                    else:  # 向左
                        # 这是一个朝向左上的凹点，可能对应右侧凸起
                        protrusion = find_rectangle_protrusion(polygon, curr, 'right')
                else:  # 向下
                    if v2[0] > 0:  # 向右
                        # 这是一个朝向右下的凹点，可能对应左侧凸起
                        protrusion = find_rectangle_protrusion(polygon, curr, 'left')
                    else:  # 向左
                        # 这是一个朝向左下的凹点，可能对应右侧凸起
                        protrusion = find_rectangle_protrusion(polygon, curr, 'right')
            else:  # 前一边是水平边
                if v1[0] > 0:  # 向右
                    if v2[1] > 0:  # 向上
                        # 这是一个朝向右上的凹点，可能对应下方凸起
                        protrusion = find_rectangle_protrusion(polygon, curr, 'bottom')
                    else:  # 向下
                        # 这是一个朝向右下的凹点，可能对应上方凸起
                        protrusion = find_rectangle_protrusion(polygon, curr, 'top')
                else:  # 向左
                    if v2[1] > 0:  # 向上
                        # 这是一个朝向左上的凹点，可能对应下方凸起
                        protrusion = find_rectangle_protrusion(polygon, curr, 'bottom')
                    else:  # 向下
                        # 这是一个朝向左下的凹点，可能对应上方凸起
                        protrusion = find_rectangle_protrusion(polygon, curr, 'top')
            
            if protrusion and protrusion not in protrusions:
                protrusions.append(protrusion)
    
    return protrusions

def find_rectangle_protrusion(polygon, corner, direction):
    """
    从给定的角落点向指定方向搜索矩形凸起
    
    参数:
        polygon: 原始多边形
        corner: 角落点坐标 (x, y)
        direction: 搜索方向 ('top', 'bottom', 'left', 'right')
        
    返回:
        Polygon: 表示凸起的多边形，如果没有找到则返回None
    """
    x, y = corner
    min_size = 0.1  # 最小凸起尺寸，避免找到太小的凸起
    
    if direction == 'top':
        # 向上搜索，寻找水平边
        test_y = y + min_size
        test_point = Point(x, test_y)
        if polygon.contains(test_point):
            # 找到上边界
            max_y = find_edge(polygon, x, y, 'top')
            # 找到左右边界
            left_x = find_edge(polygon, x, (y + max_y)/2, 'left')
            right_x = find_edge(polygon, x, (y + max_y)/2, 'right')
            if abs(max_y - y) > min_size and abs(right_x - left_x) > min_size:
                return Polygon([(left_x, y), (right_x, y), 
                                (right_x, max_y), (left_x, max_y)])
    
    elif direction == 'bottom':
        # 向下搜索
        test_y = y - min_size
        test_point = Point(x, test_y)
        if polygon.contains(test_point):
            min_y = find_edge(polygon, x, y, 'bottom')
            left_x = find_edge(polygon, x, (y + min_y)/2, 'left')
            right_x = find_edge(polygon, x, (y + min_y)/2, 'right')
            if abs(min_y - y) > min_size and abs(right_x - left_x) > min_size:
                return Polygon([(left_x, min_y), (right_x, min_y), 
                                (right_x, y), (left_x, y)])
    
    elif direction == 'left':
        # 向左搜索
        test_x = x - min_size
        test_point = Point(test_x, y)
        if polygon.contains(test_point):
            min_x = find_edge(polygon, x, y, 'left')
            top_y = find_edge(polygon, (x + min_x)/2, y, 'top')
            bottom_y = find_edge(polygon, (x + min_x)/2, y, 'bottom')
            if abs(min_x - x) > min_size and abs(top_y - bottom_y) > min_size:
                return Polygon([(min_x, bottom_y), (x, bottom_y), 
                               (x, top_y), (min_x, top_y)])
    
    elif direction == 'right':
        # 向右搜索
        test_x = x + min_size
        test_point = Point(test_x, y)
        if polygon.contains(test_point):
            max_x = find_edge(polygon, x, y, 'right')
            top_y = find_edge(polygon, (x + max_x)/2, y, 'top')
            bottom_y = find_edge(polygon, (x + max_x)/2, y, 'bottom')
            if abs(max_x - x) > min_size and abs(top_y - bottom_y) > min_size:
                return Polygon([(x, bottom_y), (max_x, bottom_y), 
                               (max_x, top_y), (x, top_y)])
    
    return None

def find_edge(polygon, x, y, direction):
    """
    从点(x,y)向指定方向搜索多边形的边界
    
    参数:
        polygon: 原始多边形
        x, y: 起始点坐标
        direction: 搜索方向 ('top', 'bottom', 'left', 'right')
        
    返回:
        float: 边界坐标
    """
    step = 0.1
    max_iter = 1000
    threshold = 0.01
    
    if direction == 'top':
        current = y + step
        for _ in range(max_iter):
            if not polygon.contains(Point(x, current)):
                return binary_search_edge(polygon, x, current - step, current, 'y')
            current += step
        return current
    
    elif direction == 'bottom':
        current = y - step
        for _ in range(max_iter):
            if not polygon.contains(Point(x, current)):
                return binary_search_edge(polygon, x, current, current + step, 'y')
            current -= step
        return current
    
    elif direction == 'left':
        current = x - step
        for _ in range(max_iter):
            if not polygon.contains(Point(current, y)):
                return binary_search_edge(polygon, current, current + step, y, 'x')
            current -= step
        return current
    
    elif direction == 'right':
        current = x + step
        for _ in range(max_iter):
            if not polygon.contains(Point(current, y)):
                return binary_search_edge(polygon, current - step, current, y, 'x')
            current += step
        return current

def binary_search_edge(polygon, low, high, const, axis):
    """
    使用二分法精确查找边界位置
    
    参数:
        polygon: 原始多边形
        low: 区间下限(在多边形内)
        high: 区间上限(在多边形外)
        const: 另一个坐标的固定值
        axis: 搜索的坐标轴 ('x' 或 'y')
        
    返回:
        float: 精确的边界位置
    """
    threshold = 0.001
    for _ in range(20):  # 最多迭代20次
        mid = (low + high) / 2
        if axis == 'x':
            point = Point(mid, const)
        else:
            point = Point(const, mid)
        
        if polygon.contains(point):
            low = mid
        else:
            high = mid
        
        if high - low < threshold:
            break
    
    return (low + high) / 2

def main():
    # 创建一个"凸"字形状的多边形
    tu_coords = [
        (0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (3, 2), (3, 3), (0, 3),
        (0, 2), (1, 2), (1, 1), (0, 1), (0, 0)
    ]

    tu_polygon = Polygon(tu_coords)

    # 查找凸起区域
    protrusions = find_protrusions(tu_polygon)

    print(f"找到 {len(protrusions)} 个凸起区域:")
    for i, protrusion in enumerate(protrusions, 1):
        print(f"凸起 {i}: {list(protrusion.exterior.coords)}")

    # 可视化
    fig, ax = plt.subplots()
    x, y = tu_polygon.exterior.xy
    ax.plot(x, y, 'k-', label="Origin Polygon")

    # 绘制凸起区域
    for p in protrusions:
        x, y = p.exterior.xy
        ax.fill(x, y, color='red', alpha=0.5, label="Convex Part")

    # for p in angle_ao:
    #     ax.scatter(p[0], p[1], color='blue', label="Angle AO")

    ax.set_aspect('equal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()