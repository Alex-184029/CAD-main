from shapely.geometry import Polygon, Point, LineString, MultiPoint, MultiPolygon
from shapely.ops import split, unary_union
import matplotlib.pyplot as plt
import copy

def split_polygon(polygon, start_point, direction='up', tolerance=1e-6):
    """
    从多边形边上的点出发向指定方向切割多边形
    
    参数:
        polygon: shapely.geometry.Polygon - 要切割的多边形
        start_point: tuple (x, y) - 多边形边上的起始点
        direction: str - 延伸方向 ('up', 'down', 'left', 'right')
        tolerance: float - 用于点比较的容差
        
    返回:
        对于上下切割: tuple (上方多边形, 下方多边形)
        对于左右切割: tuple (左侧多边形, 右侧多边形)
        如果切割失败返回 (None, None)
    """
    # 验证输入点是否在多边形边上
    if not is_point_on_polygon_boundary(polygon, start_point, tolerance):
        print("错误：起始点不在多边形边上")
        return None, None
    
    x, y = start_point
    bounds = polygon.bounds
    
    # 根据方向确定延伸线
    if direction == 'up':
        line = LineString([(x, y), (x, bounds[3] + 10)])
    elif direction == 'down':
        line = LineString([(x, y), (x, bounds[1] - 10)])
    elif direction == 'left':
        line = LineString([(x, y), (bounds[0] - 10, y)])
    elif direction == 'right':
        line = LineString([(x, y), (bounds[2] + 10, y)])
    else:
        print("错误：方向参数必须是 'up', 'down', 'left' 或 'right'")
        return None, None
    
    # 找到与多边形轮廓的第一个交点（不包括起点）
    intersection = find_first_intersection(polygon, line, start_point, direction, tolerance)
    
    if not intersection:
        print("错误：找不到交点")
        return None, None
    
    # 创建切割线
    split_line = LineString([start_point, intersection])
    
    # 分割多边形
    result = split(polygon, split_line)
    
    # 根据方向确定返回的上下或左右多边形
    if direction in ('up', 'down'):
        return determine_left_right_polygons(result, split_line)
    else:
        return determine_upper_lower_polygons(result, split_line)

def split_polygon2(polygon, start_point, direction='up', tolerance=1e-6):
    """
    从多边形边上的点出发向指定方向切割多边形
    
    参数:
        polygon: shapely.geometry.Polygon - 要切割的多边形
        start_point: tuple (x, y) - 多边形边上的起始点
        direction: str - 延伸方向 ('up', 'down', 'left', 'right')
        tolerance: float - 用于点比较的容差
        
    返回:
        对于上下切割: tuple (上方多边形, 下方多边形)
        对于左右切割: tuple (左侧多边形, 右侧多边形)
        如果切割失败返回 (None, None)
    """
    # 验证输入点是否在多边形边上
    if not is_point_on_polygon_boundary(polygon, start_point, tolerance):
        print("错误：起始点不在多边形边上")
        return None, None
    
    x, y = start_point
    # bounds = polygon.bounds
    max_extend = 300      # 最长延申限制：4m
    
    # 根据方向确定延伸线
    if direction == 'up':
        line = LineString([(x, y), (x, y + max_extend)])
    elif direction == 'down':
        line = LineString([(x, y), (x, max(1, y - max_extend))])
    elif direction == 'left':
        line = LineString([(x, y), (max(1, x - max_extend), y)])
    elif direction == 'right':
        line = LineString([(x, y), (x + max_extend, y)])
    else:
        print("错误：方向参数必须是 'up', 'down', 'left' 或 'right'")
        return None, None
    
    # 找到与多边形轮廓的第一个交点（不包括起点）
    intersection = find_first_intersection(polygon, line, start_point, direction, tolerance)
    
    if not intersection:
        print("错误：找不到交点")
        return None, None
    
    # 创建切割线
    split_line = LineString([start_point, intersection])
    
    # 分割多边形
    result = split(polygon, split_line)
    
    # 根据方向确定返回的上下或左右多边形
    if direction in ('up', 'down'):
        return determine_left_right_polygons(result, split_line)
    else:
        return determine_upper_lower_polygons(result, split_line)

def is_point_on_polygon_boundary(polygon, point, tolerance=1e-6):
    """检查点是否在多边形边上"""
    x, y = point
    point_obj = Point(x, y)
    boundary = polygon.boundary
    return boundary.distance(point_obj) < tolerance

def find_first_intersection0(polygon, line, start_point, direction, tolerance=1e-6):
    """找到线与多边形轮廓的第一个交点（不包括起点）"""
    boundary = polygon.boundary
    intersections = boundary.intersection(line)
    
    if intersections.is_empty:
        return None
    elif isinstance(intersections, Point):
        points = [intersections]
    elif isinstance(intersections, LineString):
        coords = list(intersections.coords)
        points = [Point(p) for p in coords]
    else:  # MultiPoint
        points = list(intersections.geoms)
    
    # 过滤掉与起点相同的点
    start_pt = Point(start_point)
    filtered_points = [p for p in points if p.distance(start_pt) > tolerance]
    
    if not filtered_points:
        return None
    
    # 根据方向确定正确的交点
    if direction == 'up':
        # 向上取y最小的交点（最近的上方）
        return min(filtered_points, key=lambda p: p.y).coords[0]
    elif direction == 'down':
        # 向下取y最大的交点（最近的下方）
        return max(filtered_points, key=lambda p: p.y).coords[0]
    elif direction == 'left':
        # 向左取x最小的交点（最近的左侧）
        return min(filtered_points, key=lambda p: p.x).coords[0]
    else:  # 'right'
        # 向右取x最大的交点（最近的右侧）
        return max(filtered_points, key=lambda p: p.x).coords[0]

def find_first_intersection(polygon, line, start_point, direction, tolerance=1e-6):
    """
    找到线与多边形轮廓的第一个交点（不包括起点）
    
    参数:
        polygon: 目标多边形
        line: 延伸线
        start_point: 起始点 (x, y)
        direction: 延伸方向 ('up', 'down', 'left', 'right')
        tolerance: 坐标比较容差
        
    返回:
        tuple: (x, y) 交点坐标 或 None
    """
    boundary = polygon.boundary
    intersections = boundary.intersection(line)
    
    if intersections.is_empty:
        return None
    
    # 收集所有候选点坐标
    candidate_points = []
    
    if isinstance(intersections, Point):
        candidate_points.append((intersections.x, intersections.y))
    elif isinstance(intersections, LineString):
        # 添加线段的所有顶点
        candidate_points.extend(intersections.coords)
    elif isinstance(intersections, MultiPoint):
        # 添加多点集合中的所有点
        candidate_points.extend([(p.x, p.y) for p in intersections.geoms])
    else:
        for geo in intersections.geoms:
            if isinstance(geo, Point):
                candidate_points.append((geo.x, geo.y))
            elif isinstance(geo, LineString):
                candidate_points.extend(geo.coords)
            elif isinstance(geo, MultiPoint):
                candidate_points.extend([(p.x, p.y) for p in geo.geoms])
            else:
                print(f"其它类型: {type(geo)}")

    # 过滤掉与起点太近的点
    start_x, start_y = start_point
    filtered_points = [
        (x, y) for x, y in candidate_points
        if abs(x - start_x) > tolerance or abs(y - start_y) > tolerance
    ]
    
    if not filtered_points:
        return None
    
    # 根据方向选择正确的交点
    if direction == 'up':
        return min(filtered_points, key=lambda p: p[1])  # 最小y（最上方）
    elif direction == 'down':
        return max(filtered_points, key=lambda p: p[1])  # 最大y（最下方）
    elif direction == 'left':
        return min(filtered_points, key=lambda p: p[0])  # 最小x（最左侧）
    else:  # 'right'
        return max(filtered_points, key=lambda p: p[0])  # 最大x（最右侧）

def determine_left_right_polygons(geoms, split_line):
    """确定分割后的左右多边形"""
    if geoms.is_empty:
        return None, None
    
    geometries = list(geoms.geoms)
    
    if len(geometries) < 2:
        return None, None
    
    line_coords = list(split_line.coords)
    vec = (line_coords[1][0] - line_coords[0][0], line_coords[1][1] - line_coords[0][1])
    vec = 0, abs(line_coords[1][1] - line_coords[0][1])
    # print('lr vec:', vec)
    
    left_polys = []
    right_polys = []
    
    for geom in geometries:
        if not isinstance(geom, Polygon):
            continue
        
        centroid = geom.centroid
        point_vec = (centroid.x - line_coords[0][0], centroid.y - line_coords[0][1])
        cross = vec[0] * point_vec[1] - vec[1] * point_vec[0]
        
        if cross > 0:
            left_polys.append(geom)
        elif cross < 0:
            right_polys.append(geom)
    
    left_poly = unary_union(left_polys) if left_polys else None
    right_poly = unary_union(right_polys) if right_polys else None
    
    return left_poly, right_poly

def determine_upper_lower_polygons(geoms, split_line):
    """确定分割后的上下多边形"""
    if geoms.is_empty:
        return None, None
    
    geometries = list(geoms.geoms)
    
    if len(geometries) < 2:
        return None, None
    
    line_coords = list(split_line.coords)
    # vec = (line_coords[1][0] - line_coords[0][0], line_coords[1][1] - line_coords[0][1])
    vec = -abs(line_coords[1][0] - line_coords[0][0]), 0
    # print('ud vec:', vec)
    
    upper_polys = []
    lower_polys = []
    
    for geom in geometries:
        if not isinstance(geom, Polygon):
            continue
        
        centroid = geom.centroid
        point_vec = (centroid.x - line_coords[0][0], centroid.y - line_coords[0][1])
        cross = vec[0] * point_vec[1] - vec[1] * point_vec[0]
        
        if cross < 0:  # 点在线上方
            upper_polys.append(geom)
        elif cross > 0:  # 点在线下方
            lower_polys.append(geom)
    
    upper_poly = unary_union(upper_polys) if upper_polys else None
    lower_poly = unary_union(lower_polys) if lower_polys else None
    
    return upper_poly, lower_poly

def filter_contained_polygons(polygons):
    # 标记多边形是否被包含
    num = len(polygons)
    is_contained = [False] * num
    
    for i, poly_i in enumerate(polygons):
        for j, poly_j in enumerate(polygons):
            if is_contained[j]:
                continue
            if i != j and poly_j.contains(poly_i):  # 如果 poly_i 被 poly_j 包含
                is_contained[i] = True
                break
    
    polygons_ans = [polygons[i] for i in range(num) if not is_contained[i]]
    # 返回未被包含多边形
    return polygons_ans

def filter_multi_polygons(polygons):
    ans = []
    for p in polygons:
        if isinstance(p, Polygon):
            ans.append(p)
        elif isinstance(p, MultiPolygon):
            for poly in p.geoms:
                if isinstance(poly, Polygon):
                    ans.append(poly)
                else:
                    print('例外类型：', type(poly))
        else:
            print('例外类型：', type(p))
    return ans

def get_polygon(polygon, p2, direction, side):
    poly = None
    if side == 'left' or side == 'up':
        poly, _ = split_polygon(polygon, p2, direction, tolerance=1e-6)
    elif side == 'right' or side == 'down':
        _, poly = split_polygon(polygon, p2, direction, tolerance=1e-6)
    if poly is None:
        print('获取polygon失败，p2: %s, direction: %s, side: %s' % (p2, direction, side))
    return poly

def get_polygon2(polygon, p2, direction, side):
    poly1, poly2 = None, None
    if side == 'left' or side == 'up':
        poly1, poly2 = split_polygon2(polygon, p2, direction, tolerance=1e-6)
    elif side == 'right' or side == 'down':
        poly1, poly2 = split_polygon2(polygon, p2, direction, tolerance=1e-6)
    if poly1 is None or poly2 is None:
        print('获取polygon失败，p2: %s, direction: %s, side: %s' % (p2, direction, side))
        return None
    return poly1 if poly1.area < poly2.area else poly2

def find_protrusions(polygon):
    coords = list(polygon.exterior.coords[:-1])  # 取出所有顶点（去掉重复的起点终点）
    angle_ao = []
    n = len(coords)
    polygons_convex = []

    # 遍历所有角点，寻找270度的“凹角”
    for i in range(len(coords)):
        p1 = coords[i - 1]  # 前一个点
        p2 = coords[i]      # 当前点
        p3 = coords[(i + 1) % n]  # 下一个点

        # 计算是否是凹角（顺时针方向270°）
        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
        dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
        cross_product = dx1 * dy2 - dy1 * dx2  # 叉积
        # if cross_product < 0:  # 说明是270度凹角
        if cross_product > 0:    # 说明是270度凹角，注意：这里与轮廓顺时针还是逆时针有关
            # 处理p1
            angle_ao.append(p2)
            if dx1 == 0 and dy1 != 0:
                direction = 'up' if dy1 > 0 else 'down'
                side = 'right' if dx2 > 0 else 'left'
                poly = get_polygon2(polygon, p2, direction, side)
                if not poly is None:
                    polygons_convex.append(poly)
            elif dy1 == 0 and dx1 != 0:
                direction = 'right' if dx1 > 0 else 'left'
                side = 'up' if dy2 > 0 else 'down'
                poly = get_polygon2(polygon, p2, direction, side)
                if not poly is None:
                    polygons_convex.append(poly)
            # 处理p2
            if dx2 == 0 and dy2 != 0:
                direction = 'up' if dy2 < 0 else 'down'
                side = 'right' if dx1 < 0 else 'left'
                poly = get_polygon2(polygon, p2, direction, side)
                if not poly is None:
                    polygons_convex.append(poly)
            elif dy2 == 0 and dx2 != 0:
                direction = 'right' if dx2 < 0 else 'left'
                side = 'up' if dy1 < 0 else 'down'
                poly = get_polygon2(polygon, p2, direction, side)
                if not poly is None:
                    polygons_convex.append(poly)

    return polygons_convex, angle_ao

def find_convex_regions(polygon):
    """
    找出多边形中的所有凸起区域
    """
    convex_regions = []
    n = len(polygon.exterior.coords) - 1  # 减去重复的起始点
    angle_ao = []

    for i in range(n):
        p1 = polygon.exterior.coords[i]
        p2 = polygon.exterior.coords[(i + 1) % n]
        p3 = polygon.exterior.coords[(i + 2) % n]

        # 计算向量
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # 计算向量的叉积
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        if cross_product < 0:  # 凸起
            angle_ao.append([p2[0], p2[1]])
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

    return convex_regions, angle_ao


def main():
    # 示例多边形（“凸”字形的近似表示）
    polygon = Polygon([(0,0), (6,0), (6,2), (4,2), (4,4), (6,4), (6,6), (0,6), (0,4), (2,4), (2,2), (0,2)])
    # polygon = Polygon([(0,0), (6,0), (6,2), (4,2), (4,4), (2,4), (2,2), (0,2)])

    # 找出凸起区域
    protrusions, angle_ao = find_protrusions(polygon)
    # protrusions, angle_ao = find_convex_regions(polygon)
    print("len protrussions 1:", len(protrusions))
    print("凹点：", len(angle_ao), angle_ao)
    protrusions = filter_contained_polygons(protrusions)
    print("len protrussions 2:", len(protrusions))

    # 可视化
    fig, ax = plt.subplots()
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'k-', label="Origin Polygon")

    # 绘制凸起区域
    for p in protrusions:
        x, y = p.exterior.xy
        ax.fill(x, y, color='red', alpha=0.5, label="Convex Part")

    for p in angle_ao:
        ax.scatter(p[0], p[1], color='blue', label="Angle AO")

    ax.set_aspect('equal')
    plt.legend()
    plt.show()

def find_living_room_index(rooms):
    if rooms is None or len(rooms) == 0:
        return None
    for i, room in enumerate(rooms):
        if 'living' in room['function']:
            return i
    return None

def handle_living_room_partition(rooms):
    # 定位、筛选living_room
    living_index = find_living_room_index(rooms)
    if living_index is None:
        print('No living room found.')
        return rooms
    room_living = rooms[living_index]
    print('room_living:', room_living)
    polygon = Polygon(room_living['poly'])

    # 找出凸起区域
    protrusions, _ = find_protrusions(polygon)
    # protrusions, angle_ao = find_convex_regions(polygon)
    # print("凹点：", len(angle_ao), angle_ao)
    # print("len protrussions 1:", len(protrusions))
    protrusions = filter_multi_polygons(protrusions)
    # print("len protrussions 2:", len(protrusions))
    protrusions = filter_contained_polygons(protrusions)
    # print("len protrussions 3:", len(protrusions))

    # 客厅标签
    labels = room_living['labels']
    funcs = room_living['function']

    # poly_living = copy.copy(polygon)    # 这里会破坏polygon数据，如果后续需要用到最好做一个备份
    poly_living = polygon

    # 这里还有很多问题：如果区域没有被这种方法分隔开，需要其它处理
    rooms2, cnt = [], 1
    for poly in protrusions:
        for i, label in enumerate(labels):
            if 'x' in label and 'y' in label and poly.contains(Point(label['x'], label['y'])):
                room2 = {}
                room2['poly'] = list(map(list, poly.exterior.coords))
                room2['id'] = cnt
                room2['area'] = poly.area / 1e4     # 单位平方米
                room2['perimeter'] = poly.length / 1e2
                func = [funcs[i]]
                room2['function'] = func
                room2['labels'] = [label]
                rooms2.append(room2)
                poly_living = poly_living.difference(poly)
                cnt += 1

    label_living_id = funcs.index('living')
    label_living = [labels[label_living_id]]
    room_living = {'poly': list(map(list, poly_living.exterior.coords)), 'id': cnt, 'area': poly_living.area / 1e4, 'perimeter': poly_living.length / 1e2, 'function': ['living'], 'labels': label_living}
    rooms2.append(room_living)

    # 添加到rooms，id重排
    del rooms[living_index]
    rooms += rooms2
    for i, room in enumerate(rooms):
        room['id'] = i + 1
    return rooms
