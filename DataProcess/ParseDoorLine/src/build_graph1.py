# -- 构建无向图并简化
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import numpy as np

# 定义一个颜色映射列表，这些颜色映射通常包含视觉上区分良好的颜色
colormaps = [
    plt.cm.tab10,  # 10种颜色
    plt.cm.tab20,  # 20种颜色
    plt.cm.Set1,   # 9种颜色
    plt.cm.Set2,   # 8种颜色
    plt.cm.Set3,   # 12种颜色
    plt.cm.Paired, # 12种颜色
    plt.cm.Accent, # 8种颜色
    plt.cm.Dark2,  # 8种颜色
    plt.cm.Pastel1,# 9种颜色
    plt.cm.Pastel2 # 8种颜色
]

# 选择一个颜色映射
cmap = np.random.choice(colormaps)

# 生成一个随机的颜色索引
color_index = np.random.rand()

def get_random_color():
    global cmap, color_index
    # 生成一个随机的颜色
    color = cmap(color_index)
    # 更新颜色索引，以便下次生成不同的颜色
    color_index = np.random.rand()
    return color

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
        # G.add_node((x1, y1))
        # G.add_node((x2, y2))
        G.add_edge((x1, y1), (x2, y2))
    
    # Check for intersections between all segment pairs
    # for i, (x1, y1, x2, y2) in enumerate(segments):
    #     line1 = LineString([(x1, y1), (x2, y2)])
    #     for j, (x3, y3, x4, y4) in enumerate(segments):
    #         if i >= j:  # Avoid duplicate checks
    #             continue
    #         line2 = LineString([(x3, y3), (x4, y4)])
    #         if line1.intersects(line2):
    #             intersection = line1.intersection(line2)
    #             if isinstance(intersection, Point):  # Ensure it's a point
    #                 ix, iy = intersection.x, intersection.y
    #                 G.add_node((ix, iy))
    #                 G.add_edge((x1, y1), (ix, iy))
    #                 G.add_edge((x2, y2), (ix, iy))
    #                 G.add_edge((x3, y3), (ix, iy))
    #                 G.add_edge((x4, y4), (ix, iy))
    
    return G

def visualize_graph(G):
    """
    Visualize the graph using matplotlib.
    
    Args:
        G (networkx.Graph): The graph to visualize.
    """
    pos = {node: node for node in G.nodes()}  # Use coordinates as positions
    nx.draw(
        G, pos, with_labels=True, node_size=10, font_size=4, node_color="skyblue", edge_color="gray"
    )
    plt.title("Undirected Graph with Intersection Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

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
    # print('len cycles:', len(cycles))

    # 将环路中的节点排序并去重（因无向图中环路可以从任意节点开始）
    unique_cycles, unique_cycles_order = [], []
    for cycle in cycles:
        sorted_cycle = sorted(cycle)
        if sorted_cycle not in unique_cycles:
            unique_cycles.append(sorted_cycle)
            unique_cycles_order.append(cycle)
    # print('len unique_cycles_order:', len(unique_cycles_order))
    return unique_cycles_order

def find_cycles2(graph):
    """
    基于 nx.cycle_basis() 生成的环路基，计算无向图中的所有可能简单环路。
    """
    # 获取环路基
    cycle_basis = nx.cycle_basis(graph)
    # print('len cycle_basis:', len(cycle_basis))
    all_cycles = set()  # 用集合存储去重后的环路

    # 将基础环添加到集合中
    for cycle in cycle_basis:
        all_cycles.add(tuple(sorted(cycle)))

    # 尝试扩展环路基
    def extend_cycles(current_cycle, visited_edges):
        """
        递归地扩展环路，找到所有可能的环。
        """
        for edge in graph.edges(current_cycle[-1]):  # 从当前环的末尾出发
            neighbor = edge[1] if edge[0] == current_cycle[-1] else edge[0]

            if neighbor == current_cycle[0] and len(current_cycle) > 2:
                # 如果回到起点形成环
                new_cycle = tuple(sorted(current_cycle))
                all_cycles.add(new_cycle)
            elif neighbor not in current_cycle:
                # 如果邻居未被访问，继续扩展
                extend_cycles(current_cycle + [neighbor], visited_edges | {tuple(sorted(edge))})

    # 从每条环路基出发，尝试扩展
    for cycle in cycle_basis:
        extend_cycles(cycle, set())

    # print('len all_cycles:', len(all_cycles))
    # 返回所有环路的列表
    return [list(cycle) for cycle in all_cycles]

def order_cycle(graph, cycle):    # 有一定复杂度
    """
    Order the nodes in the cycle based on traversal order.

    Args:
        graph (networkx.Graph): The graph containing the cycle.
        cycle (list): The list of nodes forming a cycle.

    Returns:
        list: Ordered nodes of the cycle.
    """
    ordered = [cycle[0]]
    visited = set(ordered)

    while len(ordered) < len(cycle):
        current = ordered[-1]
        for neighbor in graph.neighbors(current):
            if neighbor in cycle and neighbor not in visited:
                ordered.append(neighbor)
                visited.add(neighbor)
                break

    return ordered

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

def simplify_cycles(cycles):
    # 去除点集数量小于3的cycle
    # return [cycle for cycle in cycles if len(cycle) > 3]
    # 筛选点集数为4的cycle（门扇）
    return [cycle for cycle in cycles if len(cycle) == 4]

def find_rect_cycles(cycles):
    rect_cycles = []
    cycles = [cycle for cycle in cycles if len(cycle) > 3]
    for cycle in cycles:
        x1, y1, x2, y2 = cycle[0][0], cycle[0][1], cycle[0][0], cycle[0][1]
        for point in cycle[1:]:
            x1 = min(x1, point[0])
            y1 = min(y1, point[1])
            x2 = max(x2, point[0])
            y2 = max(y2, point[1])
        sign = True
        for x, y in cycle:
            if (x == x1 or x == x2) and y1 <= y <= y2:
                continue
            elif (y == y1 or y == y2) and x1 <= x <= x2:
                continue
            sign = False
        if sign:
            rect_cycles.append(cycle)

    return rect_cycles

def visualize_graph_and_cycles(graph, cycles):
    """
    Visualize the graph and its cycles.

    Args:
        graph (networkx.Graph): The input graph.
        cycles (list of lists): The list of cycles.
    """
    pos = {node: node for node in graph.nodes()}  # Use coordinates as positions

    # Draw the graph
    plt.figure(figsize=(10, 10))
    nx.draw(
        graph, pos, with_labels=False, node_size=100, font_size=8, node_color="skyblue", edge_color="gray"
    )

    # Highlight the cycles
    for cycle in cycles:
        cycle_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        # nx.draw_networkx_edges(graph, pos, edgelist=cycle_edges, edge_color="red", width=2)
        nx.draw_networkx_edges(graph, pos, edgelist=cycle_edges, edge_color=get_random_color(), width=1)

    plt.title("Graph with Highlighted Cycles")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(False)
    plt.show()

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
    # print('leaf_nodes:', leaf_num, leaf_nodes[0])
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
                # print('add edge %d: %s, %s, %.3f'% (cnt, node1, node2, distance))
                graph.add_edge(node1, node2)
    # print('add leaf edge finish, ', cnt)
    return graph


if __name__ == '__main__':
    lines = [
        [1187.0, 1331.0, 1187.0, 1351.0],
        [1187.0, 1351.0, 1187, 1391],
        [1177, 1331, 1187, 1331],
        [1167.0, 1351.0, 1187, 1351],
        [1202, 1391, 1187, 1391],
        [1202, 1411, 1167, 1411],
        [1167, 1411, 1167, 1351],
        [1202, 1391, 1202, 1411],
    ]
    graph = construct_graph(lines)
    cycles = find_cycles2(graph)
    print('Cycles:', len(cycles), cycles[0])
    # cycles = simplify_cycles(cycles)
    # print('Cycles1:', len(cycles), cycles[0])
    # cycles = filter_nested_cycles(cycles)
    # print('Cycles2:', len(cycles), cycles[0])

    # cycles = [order_cycle(graph, cycle) for cycle in cycles]

    visualize_graph_and_cycles(graph, cycles)

