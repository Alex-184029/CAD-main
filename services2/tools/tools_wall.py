import networkx as nx
from shapely.geometry import Polygon
import numpy as np
import json

def split_segments_until_done(segments):
    """
    持续切分线段，直到没有需要切分的线段为止。
    """
    # -- 线段相交切分
    def calculate_intersection(seg1, seg2):
        """
        计算两条线段是否相交，并返回交点。
        如果相交，返回 (x, y) 坐标；否则返回 None。
        """
        x1, y1, x2, y2 = seg1
        x3, y3, x4, y4 = seg2

        # 计算线段的方向向量
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3

        # 计算行列式
        det = dx1 * dy2 - dy1 * dx2
        if det == 0:
            return None  # 两线段平行或共线

        # 参数 t 和 u，用于确定交点是否在线段内
        t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
        u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det

        # 判断交点是否在两条线段内
        if 0 <= t <= 1 and 0 <= u <= 1:
            # 计算交点坐标
            ix = x1 + t * dx1
            iy = y1 + t * dy1
            return (ix, iy)

        return None

    def split_segments_once(segments):
        """
        对线段集合进行一次切分操作。
        返回新的线段集合和一个布尔值，表示是否有线段被切分。
        """
        new_segments = []
        split_occurred = False
        n = len(segments)
        processed = [False] * n

        for i in range(n):
            if processed[i]:
                continue

            seg1 = segments[i]

            for j in range(i + 1, n):
                if processed[j]:
                    continue

                seg2 = segments[j]
                intersection = calculate_intersection(seg1, seg2)

                if intersection:
                    ix, iy = intersection
                    ix, iy = round(ix), round(iy)

                    if (ix, iy) != (seg2[0], seg2[1]) and (ix, iy) != (seg2[2], seg2[3]):   # seg2被切割
                        # 切分两条线段
                        new_segments.append([seg2[0], seg2[1], ix, iy])
                        new_segments.append([ix, iy, seg2[2], seg2[3]])
                        # 标记为已切分
                        processed[j] = True
                        split_occurred = True

                    if (ix, iy) != (seg1[0], seg1[1]) and (ix, iy) != (seg1[2], seg1[3]):   # seg1被切割
                        # 切分两条线段
                        new_segments.append([seg1[0], seg1[1], ix, iy])
                        new_segments.append([ix, iy, seg1[2], seg1[3]])
                        # 标记为已切分
                        processed[i] = True
                        split_occurred = True
                        break              # seg1被切后就不能再切seg1了

            if not processed[i]:
                new_segments.append(seg1)

        return new_segments, split_occurred

    while True:
        segments, split_occurred = split_segments_once(segments)
        if not split_occurred:
            break
    return segments


def simplify_segments(segments):     # 线段简化
    """
    Simplify a list of line segments based on the given rules:
    1. Remove segments that are contained within other segments.
    2. Merge collinear overlapping segments into a single segment.

    Args:
        segments (list of tuples): List of segments represented as (x1, y1, x2, y2).

    Returns:
        list of tuples: Simplified list of segments.
    """
    def is_contained(seg1, seg2):    # 只考虑了水平垂直情况
        """Check if seg1 is contained within seg2."""
        x1, y1, x2, y2 = seg1
        a1, b1, a2, b2 = seg2

        if y1 == y2 == b1 == b2:  # Horizontal segments
            return min(a1, a2) <= min(x1, x2) <= max(x1, x2) <= max(a1, a2)
        elif x1 == x2 == a1 == a2:  # Vertical segments
            return min(b1, b2) <= min(y1, y2) <= max(y1, y2) <= max(b1, b2)
        return False

    def can_merge(seg1, seg2):
        """Check if two segments can be merged."""
        x1, y1, x2, y2 = seg1
        a1, b1, a2, b2 = seg2

        if y1 == y2 == b1 == b2:  # Horizontal segments
            c1 = min(x1, x2)
            c2 = max(x1, x2)
            c3 = min(a1, a2)
            c4 = max(a1, a2)
            if (max(c1, c3) <= min(c2, c4)):
                return True
        elif x1 == x2 == a1 == a2:  # Vertical segments
            c1 = min(y1, y2)
            c2 = max(y1, y2)
            c3 = min(b1, b2)
            c4 = max(b1, b2)
            if (max(c1, c3) <= min(c2, c4)):
                return True
        return False

    def merge_segments(seg1, seg2):
        """Merge two overlapping collinear segments."""
        x1, y1, x2, y2 = seg1
        a1, b1, a2, b2 = seg2

        if y1 == y2 == b1 == b2:  # Horizontal segments
            return [min(x1, x2, a1, a2), y1, max(x1, x2, a1, a2), y2]
        elif x1 == x2 == a1 == a2:  # Vertical segments
            return [x1, min(y1, y2, b1, b2), x2, max(y1, y2, b1, b2)]

    # Step 1: Remove contained segments    和步骤二重叠
    # remaining_segments = []
    # for seg1 in segments:
    #     contained = False
    #     for seg2 in segments:
    #         if seg1 != seg2 and is_contained(seg1, seg2):
    #             contained = True
    #             break
    #     if not contained:
    #         remaining_segments.append(seg1)
    remaining_segments = segments

    # Step 2: Merge overlapping collinear segments
    merged_segments = []
    used = [False] * len(remaining_segments)

    num = len(remaining_segments)
    for i in range(num):
        if used[i]:
            continue
        seg1 = remaining_segments[i]
        # 检查是否有收尾相同点
        x1, y1, x2, y2 = seg1
        if x1 == x2 and y1 == y2:
            used[i] = True
            continue
        # 重叠合并
        for j in range(i + 1, num):
            if not used[j] and can_merge(seg1, remaining_segments[j]):
                seg1 = merge_segments(seg1, remaining_segments[j])
                used[j] = True
        merged_segments.append(seg1)

    return merged_segments

def organize_segments(segments):
    """
    Organize line segments into horizontal, vertical, and other segments with specific sorting rules.

    Args:
        segments (list of tuples): List of segments represented as (x1, y1, x2, y2).

    Returns:
        list of tuples: Organized list of segments.
    """
    # Helper functions
    def is_horizontal(seg):
        return seg[1] == seg[3]

    def is_vertical(seg):
        return seg[0] == seg[2]

    # 筛选
    horizontal_segments = [seg for seg in segments if is_horizontal(seg)]
    vertical_segments = [seg for seg in segments if is_vertical(seg)]
    other_segments = [seg for seg in segments if not is_horizontal(seg) and not is_vertical(seg)]

    # 排序
    horizontal_segments.sort(key=lambda seg: (seg[1], min(seg[0], seg[2])))
    vertical_segments.sort(key=lambda seg: (seg[0], min(seg[1], seg[3])))
    print('nums1:', len(horizontal_segments), len(vertical_segments), len(other_segments))

    # 简化
    horizontal_segments = simplify_segments(horizontal_segments)
    vertical_segments = simplify_segments(vertical_segments)
    print('nums2:', len(horizontal_segments), len(vertical_segments), len(other_segments))

    # Combine all segments
    organized_segments = horizontal_segments + vertical_segments + other_segments

    return organized_segments

def construct_graph(segments):
    """
    Construct an undirected graph based on the given segments.
    Segments are represented as (x1, y1, x2, y2).
    
    Args:
        segments (list of tuples): List of segments in the format (x1, y1, x2, y2).
    
    Returns:
        networkx.Graph: The constructed undirected graph.
    """
    G = nx.Graph()
    
    # Add endpoints of segments as graph nodes
    for x1, y1, x2, y2 in segments:
        G.add_edge((x1, y1), (x2, y2))
    
    return G

def add_leaf_edge(graph):      # 查找挂点
    def euclidean_distance(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    # 找出度为1的节点（挂点）
    leaf_nodes = [node for node, degree in graph.degree() if degree == 1]

    # 设置距离阈值
    distance_threshold_min = 10    # 10px对应50mm
    distance_threshold_max = 50    # origin is 100, 100px对应500mm，最大墙厚

    # 遍历所有挂点对，检查最短路径长度
    leaf_num = len(leaf_nodes)
    cnt = 0
    is_used_leaf = [False] * leaf_num
    print('leaf_nodes:', leaf_num)
    # print('distance thred:', distance_threshold_min, distance_threshold_max)
    for i in range(leaf_num):
        if is_used_leaf[i]:
            continue
        for j in range(i + 1, leaf_num):
            if is_used_leaf[j]:
                continue
            # 获取两个挂点
            node1, node2 = leaf_nodes[i], leaf_nodes[j]
            # 计算最短路径长度
            # distance = nx.shortest_path_length(graph, source=node1, target=node2)   # 图中最短径距离
            distance = euclidean_distance(node1, node2)
            # if distance <= distance_threshold_max:
            #     print('add edge %d: %s, %s, %.3f'% (cnt, node1, node2, distance))
            # 如果距离在阈值内，添加边
            if distance_threshold_min <= distance <= distance_threshold_max:
                is_used_leaf[i] = True
                is_used_leaf[j] = True
                cnt += 1
                print('add edge %d: %s, %s, %.3f'% (cnt, node1, node2, distance))
                graph.add_edge(node1, node2)
    print('add leaf edge finish, ', cnt)
    return graph

def find_cycles(graph):
    """
    Find all simple cycles in the graph using NetworkX's cycle_basis method.

    Args:
        graph (networkx.Graph): The input undirected graph.

    Returns:
        list of lists: Each cycle is represented as a list of nodes in traversal order.
    """
    # cycles = []
    # for cycle in nx.cycle_basis(graph):
    #     ordered_cycle = order_cycle(graph, cycle)
    #     cycles.append(ordered_cycle)
    # return cycles

    # directed_graph = graph.to_directed()    # 转换有向图

    # 使用 simple_cycles 查找所有简单环路
    # cycles = list(nx.simple_cycles(directed_graph))
    cycles = list(nx.simple_cycles(graph))    # 可以直接从无向图
    print('len cycles:', len(cycles))

    # 将环路中的节点排序并去重（因无向图中环路可以从任意节点开始）
    unique_cycles, unique_cycles_order = [], []
    for cycle in cycles:
        sorted_cycle = sorted(cycle)
        if sorted_cycle not in unique_cycles:
            unique_cycles.append(sorted_cycle)
            unique_cycles_order.append(cycle)
    print('len unique_cycles_order:', len(unique_cycles_order))
    return unique_cycles_order

def simplify_cycles(cycles):
    # 去除点集数量小于3的cycle
    return [cycle for cycle in cycles if len(cycle) > 3]

def filter_nested_cycles(cycles):
    """
    过滤掉被其他环路包含的环路。

    参数:
    cycles: list[list[int]] -- 简单环路的列表，每个环路是一个顶点序列。

    返回:
    list[list[int]] -- 过滤后的环路列表。
    """
    def polygon_area(points):
        # 计算多边形面积
        x, y = zip(*points)
        return 0.5 * abs(sum(x[i] * y[i-1] - x[i-1] * y[i] for i in range(len(points))))

    # 将环路转换为集合以便于比较
    cycle_sets = [set(cycle) for cycle in cycles]
    # 标记需保留的环路
    keep = [True] * len(cycles)
    # 两两比较环路
    for i in range(len(cycle_sets)):
        for j in range(len(cycle_sets)):
            if i != j and cycle_sets[i] < cycle_sets[j]:     # 如果环路 i 是环路 j 的真子集
                keep[i] = False
                break
            elif i != j and cycle_sets[i] == cycle_sets[j]:  # 点数相等比较面积
                area_i = polygon_area(cycles[i])
                area_j = polygon_area(cycles[j])
                if area_i < area_j:
                    keep[i] = False
                    break
    
    # 返回过滤后的环路
    return [cycles[i] for i in range(len(cycles)) if keep[i]]

def filter_contained_cycles(cycles):
    """
    从多边形环路集合中剔除被包含的多边形。
    
    参数：
        cycles: list[list]，环路的点集列表，每个环路是点的有序列表。
        graph: networkx.Graph，无向图，节点包含坐标属性 'pos'。
    
    返回：
        list[list]，过滤后的环路点集列表。
    """
    # 转化为shapely多边形对象
    polygons = [Polygon(cycle) for cycle in cycles]
    
    # 标记多边形是否被包含
    is_contained = [False] * len(polygons)
    
    for i, poly_i in enumerate(polygons):
        for j, poly_j in enumerate(polygons):
            if i != j and poly_j.contains(poly_i):  # 如果 poly_i 被 poly_j 包含
                is_contained[i] = True
                break
    
    # 返回未被包含的环路点集
    filtered_cycles = [cycles[i] for i in range(len(cycles)) if not is_contained[i]]
    return filtered_cycles

def load_line_json(line_path):
    with open(line_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lines = []
    for shape in data['shapes']:
        line = shape['points']
        lines.append([line[0][0], line[0][1], line[1][0], line[1][1]])
    return lines

def cycles_to_json(contours, json_origin, save_path):
    if contours is None or len(contours) == 0:
        print('data error for:', save_path)
        return
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = []
    for contour in contours:
        shape = {
            'label': 'WallArea1', 
            'points': contour,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)

    data['shapes'] = shapes
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print('Write json to,', save_path)

def parse_wall_tool(lines: list):
    # 线段预处理
    lines_simplify = organize_segments(lines)
    lines_split = split_segments_until_done(lines_simplify) # 相交分割

    # 构建无向图
    graph = construct_graph(lines_split)
    graph = add_leaf_edge(graph)    # 挂点处理 

    cycles = find_cycles(graph)
    cycles = simplify_cycles(cycles)
    cycles = filter_nested_cycles(cycles)
    cycles = filter_contained_cycles(cycles)

    return cycles


def main_old():
    # line_path = '../data/labels_line_ori/01 1-6号住宅楼标准层A户型平面图-5.json'
    # line_path = '../data/labels_line_ori/(T3) 12#楼105户型平面图（镜像）-2.json'
    # line_path = '../data/labels_line_ori/(T3) 12#楼105户型平面图（镜像）-3.json'
    line_path = '../data/labels_line_ori/01 1-6号住宅楼标准层A户型平面图-2.json'
    lines = load_line_json(line_path)
    print('line num1:', len(lines))
    lines_simplify = organize_segments(lines)
    print('line num2:', len(lines_simplify))
    lines_split = split_segments_until_done(lines_simplify)
    print('line num3:', len(lines_split))

    # select_line(lines_split)
    # lines_to_json(lines_split, line_path, '../data/labels_line_ori/tmp21.json')

    graph = construct_graph(lines_split)
    graph = add_leaf_edge(graph)

    cycles = find_cycles(graph)
    print('Cycles:', len(cycles), cycles[0])
    cycles = simplify_cycles(cycles)
    print('Cycles1:', len(cycles), cycles[0])
    cycles = filter_nested_cycles(cycles)
    print('Cycles2:', len(cycles), cycles[0])
    cycles = filter_contained_cycles(cycles)
    print('Cycles3:', len(cycles), cycles[0])

    # cycles = [order_cycle(graph, cycle) for cycle in cycles]

    # visualize_graph_and_cycles(graph, cycles)
    cycles_to_json(cycles, line_path, '../data/tmp_res2/tmp2.json')