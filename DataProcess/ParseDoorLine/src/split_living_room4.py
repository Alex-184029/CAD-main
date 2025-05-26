from shapely.geometry import Polygon, MultiPolygon, Point
import matplotlib.pyplot as plt
from split_living_room6 import split_polygon, split_polygon2
from deal_labelme1 import find_living_room_label, classify_room, save_to_labelme2, room_type
import copy

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

def test1():
    input_json = r'../data/labels_Room/(T3) 12#楼105户型平面图（镜像）-3_Structure2.json'
    output_json = r'../data/tmp_res/tmp_living3.json'
    room = find_living_room_label()
    polygon = room['poly']
    # 示例多边形（“凸”字形的近似表示）
    # polygon = Polygon([(0,0), (6,0), (6,2), (4,2), (4,4), (6,4), (6,6), (0,6), (0,4), (2,4), (2,2), (0,2)])
    # polygon = Polygon([(0,0), (6,0), (6,2), (4,2), (4,4), (2,4), (2,2), (0,2)])

    # 找出凸起区域
    protrusions, angle_ao = find_protrusions(polygon)
    # protrusions, angle_ao = find_convex_regions(polygon)
    print("凹点：", len(angle_ao), angle_ao)
    print("len protrussions 1:", len(protrusions))
    protrusions = filter_multi_polygons(protrusions)
    print("len protrussions 2:", len(protrusions))
    protrusions = filter_contained_polygons(protrusions)
    print("len protrussions 3:", len(protrusions))

    # 与labels映射
    labels = room['labels']
    print('labels:', labels)
    room_types = list(room_type.keys()) + ['default']
    poly_living = copy.copy(polygon)

    rooms2, cnt = [], 1
    for poly in protrusions:
        room2 = {}
        room2['poly'] = poly
        room2['id'] = cnt
        room2['area'] = poly.area / 1e4     # 单位平方米
        room2['function'] = []
        for label in labels:
            if poly.contains(Point(label['x'], label['y'])):
                funcs = classify_room(label['txt'])
                if len(funcs) > 0 and not 0 in funcs:
                    funcs = [room_types[func] for func in funcs]
                    room2['function'] += funcs
                    room2['label'] = label['txt']
                    rooms2.append(room2)
                    poly_living = poly_living.difference(poly)
                    cnt += 1
    room_living = {'id': cnt, 'poly': poly_living, 'area': poly_living.area / 1e4, 'function': ['living'], 'label': '客厅'}
    rooms2.append(room_living)

    for room2 in rooms2:
        print('room: %d, area: %.3f, function: %s, label: %s' % (room2['id'], room2['area'], room2['function'], room2['label']))
    save_to_labelme2(input_json, rooms2, output_json)
    print('Write json to:', output_json)



    # 可视化
    fig, ax = plt.subplots()
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'k-', label="Origin Polygon")

    # 绘制凸起区域
    for p in protrusions:
        x, y = p.exterior.xy
        ax.fill(x, y, color='red', alpha=0.5, label="Convex Part")

    for p in angle_ao:
        ax.scatter(p[0], p[1], color='blue')

    ax.set_aspect('equal')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # main()
    test1()
