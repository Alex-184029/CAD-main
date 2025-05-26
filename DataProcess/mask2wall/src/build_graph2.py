# 另一种方法构建无向图，没用到
from collections import defaultdict
import matplotlib.pyplot as plt

def build_graph(segments):
    """
    根据线段构建图结构。
    :param segments: List of segments, each defined as [x1, y1, x2, y2].
    :return: Graph as an adjacency list.
    """
    graph = defaultdict(list)
    for x1, y1, x2, y2 in segments:
        p1, p2 = (x1, y1), (x2, y2)
        graph[p1].append(p2)
        graph[p2].append(p1)
    return graph

def find_cycles(graph):
    """
    使用深度优先搜索 (DFS) 查找图中的所有闭合环路。
    :param graph: Graph as an adjacency list.
    :return: List of all closed loops.
    """
    def dfs(node, start, path, visited_edges):
        path.append(node)
        for neighbor in graph[node]:
            edge = tuple(sorted([node, neighbor]))
            if edge in visited_edges:
                continue

            visited_edges.add(edge)

            if neighbor == start and len(path) > 2:
                cycles.append(path[:])
            elif neighbor not in path:
                dfs(neighbor, start, path, visited_edges)

            visited_edges.remove(edge)
        path.pop()

    cycles = []
    visited_edges = set()
    for start_node in graph:
        dfs(start_node, start_node, [], visited_edges)

    # 消除重复环路（因为环路可能从不同的起点生成）
    unique_cycles = []
    seen = set()
    for cycle in cycles:
        sorted_cycle = tuple(sorted(cycle))
        if sorted_cycle not in seen:
            seen.add(sorted_cycle)
            unique_cycles.append(cycle)

    return unique_cycles

def visualize_cycles(segments, cycles):
    """
    可视化所有闭合环路。
    :param segments: Original list of segments.
    :param cycles: List of closed loops.
    """
    plt.figure(figsize=(8, 8))

    # 绘制线段
    for x1, y1, x2, y2 in segments:
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)

    # 为每个环路选择不同颜色并绘制
    colors = plt.cm.tab10(range(len(cycles)))
    for i, cycle in enumerate(cycles):
        x_coords, y_coords = zip(*(cycle + [cycle[0]]))  # 闭合环路
        plt.plot(x_coords, y_coords, '-', label=f'Cycle {i + 1}', color=colors[i])

    plt.axis('equal')
    plt.legend()
    plt.title('Closed Loops Visualization')
    plt.show()

def do_test(segments):
    # 构建图
    graph = build_graph(segments)

    # 查找所有闭合环路
    cycles = find_cycles(graph)

    print("闭合环路:")
    for cycle in cycles:
        print(cycle)

    # 可视化环路
    visualize_cycles(segments, cycles)


if __name__ == "__main__":
    # 示例线段列表
    segments = [
        [0, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 2, 0],
        [2, 0, 2, 1],
        [2, 1, 1, 1]
    ]

    # 构建图
    graph = build_graph(segments)

    # 查找所有闭合环路
    cycles = find_cycles(graph)

    print("闭合环路:")
    for cycle in cycles:
        print(cycle)

    # 可视化环路
    visualize_cycles(segments, cycles)
