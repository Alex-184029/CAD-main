import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union, polygonize, split
from scipy.spatial import Voronoi
from deal_labelme1 import find_living_room_label, save_to_labelme2
from scipy.spatial import KDTree
from collections import defaultdict

def create_voronoi_subregions(room_polygon: Polygon, labels: dict):
    """
    使用Voronoi细分方式划分客厅区域，尽量使用水平或竖直的辅助线。
    
    参数:
    - room_polygon: Shapely的Polygon对象，表示客厅连通区域
    - labels: 列表，包含 (标签, (x, y)) 的元组
    
    返回:
    - subregions: 字典，键是区域标签，值是子区域的Polygon
    """
    # 提取所有标签的坐标
    # points = np.array([coord for _, coord in labels])
    points = np.array([(label['x'], label['y']) for label in labels])

    # 计算Voronoi图
    vor = Voronoi(points)

    # 存储所有的划分线（只保留在 room_polygon 内的）
    all_lines = []

    # 遍历Voronoi边界线
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue  # 忽略无穷远边界

        p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
        line = LineString([p1, p2])

        # 裁剪到房间边界内
        clipped_line = line.intersection(room_polygon)
        if not clipped_line.is_empty:
            all_lines.append(clipped_line)

    # 组合所有划分线，生成子区域
    merged_lines = unary_union(all_lines)
    subregions = list(polygonize(merged_lines.union(room_polygon.boundary)))

    # 生成区域标签字典
    region_dict = {}
    for label in labels:
        for region in subregions:
            # if region.contains(Polygon([point])):  # 确保点在子区域内
            if region.contains(Point(label['x'], label['y'])):
                region_dict[label['txt']] = region
                break

    return region_dict

def split_polygon_by_labels(polygon: Polygon, labels: dict):
    """
    用水平和竖直辅助线划分多边形，并根据标签位置分配子区域。
    
    参数:
        polygon: shapely.Polygon, 待划分的多边形。
        labels: list of tuples, 格式如 [("客厅", x1, y1), ("餐厅", x2, y2), ...]。
    
    返回:
        dict: 键为标签名，值为对应的子多边形。
    """
    if not polygon.is_valid:
        raise ValueError("输入的多边形无效！")
    
    # 提取所有标签的x和y坐标
    x_coords = [label['x'] for label in labels]
    y_coords = [label['y'] for label in labels]
    
    # 生成水平和竖直分割线（去重并排序）
    vertical_lines = sorted(list(set(x_coords)))   # 竖直分割线（x坐标）
    horizontal_lines = sorted(list(set(y_coords))) # 水平分割线（y坐标）
    
    # 初始化待分割的多边形集合
    polygons_to_split = [polygon]
    
    # 第一步：用竖直分割线划分
    for x in vertical_lines:
        line = LineString([(x, polygon.bounds[1]), (x, polygon.bounds[3])])
        new_polygons = []
        for poly in polygons_to_split:
            split_result = split(poly, line)
            if split_result.is_empty:
                new_polygons.append(poly)
            else:
                new_polygons.extend(split_result.geoms)
        polygons_to_split = new_polygons
    
    # 第二步：用水平分割线划分
    for y in horizontal_lines:
        line = LineString([(polygon.bounds[0], y), (polygon.bounds[2], y)])
        new_polygons = []
        for poly in polygons_to_split:
            split_result = split(poly, line)
            if split_result.is_empty:
                new_polygons.append(poly)
            else:
                new_polygons.extend(split_result.geoms)
        polygons_to_split = new_polygons
    
    # 将子多边形分配到最近的标签
    label_points = {label['txt']: Point(label['x'], label['y']) for label in labels}
    result = {name: [] for name in label_points.keys()}
    
    for sub_poly in polygons_to_split:
        if sub_poly.is_empty:
            continue
        # 找到子多边形的中心点
        center = sub_poly.centroid
        # 计算中心点到所有标签点的距离，选择最近的标签
        closest_label = min(
            label_points.keys(),
            key=lambda name: center.distance(label_points[name])
        )
        result[closest_label].append(sub_poly)
    
    # 合并同一标签的子多边形（可能被多次分割）
    for name in result:
        if result[name]:
            result[name] = unary_union(result[name])
        else:
            result[name] = None
    
    return result

def create_nearest_neighbor_subregions(room_polygon: Polygon, labels: dict, grid_size=2.0):
    """
    使用最近邻扩展方式划分客厅区域，尽量使用水平或竖直的辅助线。
    
    参数:
    - room_polygon: Shapely的Polygon对象，表示客厅连通区域
    - labels: 列表，包含 (标签, (x, y)) 的元组
    - grid_size: 生成网格点的步长

    返回:
    - subregions: 字典，键是区域标签，值是子区域的Polygon
    """
    # 提取标签点和名称
    # label_points = np.array([coord for _, coord in labels])
    # label_names = [name for name, _ in labels]
    label_points = np.array([(label['x'], label['y']) for label in labels])
    label_names = [label['txt'] for label in labels]

    # 构建 KDTree 以进行快速最近邻搜索
    tree = KDTree(label_points)

    # 生成规则化网格点
    minx, miny, maxx, maxy = room_polygon.bounds
    x_vals = np.arange(minx, maxx + grid_size, grid_size)
    y_vals = np.arange(miny, maxy + grid_size, grid_size)

    # 存储所有网格点的最近邻标签
    grid_points = []
    point_to_label = {}

    for x in x_vals:
        for y in y_vals:
            p = Point(x, y)
            if room_polygon.contains(p):  # 只保留在多边形内部的点
                dist, idx = tree.query((x, y))  # 查询最近邻标签点
                label = label_names[idx]
                grid_points.append(p)
                point_to_label[p] = label

    # 生成分割线
    all_lines = []
    for p1 in grid_points:
        for p2 in grid_points:
            if p1 == p2:
                continue
            # 水平或垂直相邻
            if (p1.x == p2.x and abs(p1.y - p2.y) == grid_size) or (p1.y == p2.y and abs(p1.x - p2.x) == grid_size):
                if point_to_label[p1] != point_to_label[p2]:  # 交界处生成分割线
                    line = LineString([p1, p2])
                    all_lines.append(line)

    # 合并所有分割线，得到子区域
    merged_lines = unary_union(all_lines)
    subregions = list(polygonize(merged_lines.union(room_polygon.boundary)))

    # 生成区域标签字典
    region_dict = defaultdict(list)
    for region in subregions:
        # 计算区域中心点
        centroid = region.centroid
        dist, idx = tree.query((centroid.x, centroid.y))
        label = label_names[idx]
        region_dict[label].append(region)

    return region_dict

def main_old():
    # 示例：创建客厅区域（假设它是一个矩形）
    room_polygon = Polygon([(2, 2), (10, 2), (10, 8), (2, 8)])

    # 定义标签及其坐标
    labels = [
        ("客厅", (4, 3)),
        ("餐厅", (8, 3)),
        ("玄关", (2, 6)),
        ("走廊", (6, 7))
    ]

    # 执行划分
    subregions = create_voronoi_subregions(room_polygon, labels)

    # 输出子区域
    for name, poly in subregions.items():
        print(f"{name}: {poly}")
    
def main():
    json_origin = r'../data/tmp_res/tmp_region2.json'
    json_output = r'../data/tmp_res/tmp_living3.json'
    room = find_living_room_label()
    # region_dict = create_voronoi_subregions(room['poly'], room['labels'])
    region_dict = create_nearest_neighbor_subregions(room['poly'], room['labels'])
    # for name, poly in region_dict.items():
    #     print(f"{name}: {poly}")
    rooms_living = []
    for name, poly in region_dict.items():
        room_living = dict()
        room_living['function'] = name
        room_living['poly'] = poly
        # print(f"{name}: {poly}")
        rooms_living.append(room_living)
    save_to_labelme2(json_origin, rooms_living, json_output)
    print('finish')
    


if __name__ == "__main__":
    main()
