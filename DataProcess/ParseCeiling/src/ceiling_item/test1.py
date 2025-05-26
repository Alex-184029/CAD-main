import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher, categorical_node_match, numerical_edge_match
from typing import List

def are_graphs_equal(g1, g2):
    # 匹配节点 Type 属性
    node_match = categorical_node_match('Type', None)
    # 匹配边 weight 属性
    edge_match = numerical_edge_match('weight', None)
    matcher = DiGraphMatcher(g1, g2, node_match=node_match, edge_match=edge_match)
    return matcher.is_isomorphic()

# 子图同构方法
def deduplicate_digraphs1(graph_list):
    unique_graphs = []
    for g in graph_list:
        if not any(are_graphs_equal(g, existing) for existing in unique_graphs):
            unique_graphs.append(g)
    return unique_graphs

# 哈希摘要方法
def deduplicate_digraphs(graphs: List[nx.DiGraph]) -> List[nx.DiGraph]:
    """
    对有向图序列进行去重
    :param graphs: 输入的有向图列表
    :return: 去重后的列表（保持原始顺序）
    """
    unique_graphs = []
    seen_hashes = set()
    
    for graph in graphs:
        # 生成图的唯一标识（基于排序后的节点和边特征）
        graph_hash = tuple(sorted((graph[u].get('Type'), graph[v].get('Type'), data['weight']) for u, v, data in graph.edges(data=True)))
        # print('graph_hash:', graph_hash)
        
        if graph_hash not in seen_hashes:
            seen_hashes.add(graph_hash)
            unique_graphs.append(graph)
    
    return unique_graphs

def test():
    # 构建三个图，其中 g1 和 g2 是结构相同但节点名称不同
    g1 = nx.DiGraph()
    g1.add_node("A", Type="X")
    g1.add_node("B", Type="Y")
    g1.add_edge("A", "B", weight=1)

    g2 = nx.DiGraph()
    g2.add_node("X", Type="X")
    g2.add_node("Y", Type="Y")
    g2.add_edge("X", "Y", weight=1)

    g3 = nx.DiGraph()
    g3.add_node("P", Type="X")
    g3.add_node("Q", Type="Y")
    g3.add_edge("P", "Q", weight=2)  # 不同的边权

    graph_seq = [g1, g2, g3]

    unique_graphs = deduplicate_digraphs(graph_seq)
    print(f"Original count: {len(graph_seq)}")
    print(f"Unique count: {len(unique_graphs)}")

    G = g1
    for u, v, data in G.edges(data=True):
        source_type = G.nodes[u].get("Type", "N/A")
        target_type = G.nodes[v].get("Type", "N/A")
        weight = data.get("weight", "N/A")
        print(f"{u}({source_type}) -> {v}({target_type}), weight = {weight}")


if __name__ == '__main__':
    test()
