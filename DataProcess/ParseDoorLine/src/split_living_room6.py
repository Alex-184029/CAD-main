from shapely.geometry import Polygon, Point, LineString, MultiPoint
from shapely.ops import split, unary_union

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


def main():
    # 创建一个矩形多边形
    rect = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

    # 1. 向上切割
    print("向上切割:")
    upper, lower = split_polygon(rect, (1, 0), 'up')
    print(f"左侧多边形: {upper}\n右侧多边形: {lower}")

    # 2. 向下切割
    print("\n向下切割:")
    upper, lower = split_polygon(rect, (1, 2), 'down')
    print(f"左侧多边形: {upper}\n右侧多边形: {lower}")

    # 3. 向左切割
    print("\n向左切割:")
    left, right = split_polygon(rect, (2, 1), 'left')
    print(f"上方多边形: {left}\n下方多边形: {right}")

    # 4. 向右切割
    print("\n向右切割:")
    left, right = split_polygon(rect, (0, 1), 'right')
    print(f"上方多边形: {left}\n下方多边形: {right}")


def test():
    polygon = Polygon([(0,0), (6,0), (6,2), (4,2), (4,4), (2,4), (2,2), (0,2)])
    p = (2, 2)
    upper, lower = split_polygon(polygon, p, 'right')
    print('upper:', upper)
    print('lower:', lower)


if __name__ == '__main__':
    # main()
    test()
