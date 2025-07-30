import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher, categorical_node_match, numerical_edge_match
import json
import os
import cv2
import numpy as np
import re
from typing import List
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import requests

def classify_legend(text):
    # url = 'http://127.0.0.1:5006/classify_room2'
    url = 'http://10.112.227.114:5006/classify_legend'
    data = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return None

def get_ceiling_item(json_path, item_name):
    if not os.path.exists(json_path):
        print('json_path not exist:', json_path)
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if item_name in data and 'range' in data:
        return data[item_name], data['range']
    return None

def get_json_attribute(json_path, att_name):
    if not os.path.exists(json_path):
        print('json_path not exist:', json_path)
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if att_name in data:
        return data[att_name]
    print(f'Attribute {att_name} not exist.')
    return None

def visualize_digraph(G):
    """
    可视化有向图（显示节点Type和边weight）
    :param G: NetworkX DiGraph（需包含节点Type和边weight属性）
    """
    # 1. 设置图形布局
    plt.figure(figsize=(10, 8))
    
    # 2. 为不同Type分配颜色
    unique_types = {data['Type'] for _, data in G.nodes(data=True)}
    type_colors = ListedColormap(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])  # 自定义颜色
    type_to_color = {t: type_colors(i) for i, t in enumerate(unique_types)}
    
    # 3. 绘制节点（按Type着色）
    node_colors = [type_to_color[G.nodes[n]['Type']] for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42)  # 布局算法（可改为circular/spiral等）
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        node_size=800,
        alpha=0.9
    )
    
    # 4. 绘制边（线宽反映weight）
    edge_weights = [G.edges[u, v]['weight'] + 1 for u, v in G.edges()]
    edges = nx.draw_networkx_edges(
        G, pos, 
        width=[w * 0.4 for w in edge_weights],  # 缩放权重到线宽
        edge_color='#555555', 
        arrowstyle='->',
        arrowsize=15,
        alpha=0.7
    )
    
    # 5. 添加标签
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    # 6. 显示边权重
    edge_labels = {(u, v): f"{G.edges[u, v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos, 
        edge_labels=edge_labels,
        font_size=9,
        label_pos=0.4  # 标签位置（0-1）
    )
    
    # 7. 添加图例和标题
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=type_to_color[t], markersize=10, label=t)
        for t in unique_types
    ]
    plt.legend(handles=legend_handles, title='Node Types')
    plt.title("Directed Graph with Node Types and Edge Weights", pad=20)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def reindex_subgraph(G):
    """
    将有向图的节点重新编号为从0开始的连续整数，保持原始编号的相对顺序
    :param G: NetworkX DiGraph（需保证节点编号可比较）
    :return: 新编号的图（原图不变）
    """
    # 1. 获取原始节点编号并排序
    original_nodes = sorted(G.nodes())
    
    # 2. 创建新旧编号映射字典
    mapping = {old: new for new, old in enumerate(original_nodes)}
    
    # 3. 创建新图并复制所有属性
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    
    return new_G

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgReadGray(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)

def build_graph(nodes, get_weight):
    """
    构建无向图，并记录每个节点的 Type 类型
    :param nodes: 节点列表，每个节点是字典类型（必须包含 'Type' 键）
    :param get_weight: 函数，接收两个节点字典，返回权重（-1表示无边）
    :return: NetworkX 无向图对象（节点含 Type 属性）
    """
    G = nx.Graph()
    
    # 添加所有节点，显式存储 Type 和其他属性
    for i, node in enumerate(nodes):
        if 'Type' not in node or 'Text' not in node:
            raise ValueError(f"每个节点必须包含'Type'和'Text'属性，error node: {i, node}")
        G.add_node(i, {'Type': node['Type'] + node['Text']})  # 将整个 node 字典存入节点属性
    
    num = len(nodes)
    # 添加边（基于 get_weight 函数）
    for i in range(num):
        for j in range(i + 1, num):
            weight = get_weight(nodes[i], nodes[j])
            if weight != -1:
                G.add_edge(i, j, weight=weight)
    
    return G

def build_digraph(nodes, get_weight, thred=800):
    """
    构建有向图，并记录每个节点的 Type 类型
    :param nodes: 节点列表，每个节点是字典类型（必须包含 'Type'）
    :param get_weight: 函数，接收两个节点字典，返回 A→B 的权重（-1表示无此方向边）
    :return: NetworkX 有向图对象（含 Type 属性）
    """
    # print('nodes num 1:', len(nodes))
    # nodes = sort_nodes(nodes)      # 结点排序
    # print('nodes num 2:', len(nodes))
    G = nx.DiGraph()
    
    # 添加所有节点
    for i, node in enumerate(nodes):   # 需要对结点预先排序吗？
        if 'Type' not in node or 'Text' not in node:
            raise ValueError(f"每个节点必须包含'Type'和'Text'属性，error node: {i, node}")
        dict_type = {'Type': node['Type'] + node['Text']}
        G.add_node(i, **dict_type)  # 将整个 node 字典存入节点属性
    
    # 添加有向边
    num = len(nodes)
    for i in range(num):
        for j in range(i + 1, num):
            res = get_weight(nodes[i], nodes[j], thred=thred)
            if res is not None:
                weight1, weight2 = res
                G.add_edge(i, j, weight=weight1)
                G.add_edge(j, i, weight=weight2)
    
    return G

def find_connected_components(G, connection_type='weak'):
    """
    查找有向图的连通分量，并返回节点索引和对应的 Type 信息
    :param G: 有向图（NetworkX DiGraph）
    :param connection_type: 'weak'（弱连通，默认）或 'strong'（强连通）
    :return: 列表，格式为 [(节点索引列表, 对应的Type列表), ...]
    """
    if connection_type == 'weak':
        components = list(nx.weakly_connected_components(G))
    elif connection_type == 'strong':
        components = list(nx.strongly_connected_components(G))
    else:
        raise ValueError("connection_type 必须是 'weak' 或 'strong'")
    
    result = []
    for component in components:
        node_indices = list(component)
        types = [G.nodes[n]['Type'] for n in component]
        result.append((node_indices, types))
    return result

# 子图查找，需要验证是否有效
def find_directed_subgraphs(G, subgraph_template):
    """
    在有向图 G 中查找所有与模板拓扑相同且 Type 匹配的子图
    :param G: 原始有向图（含 Type 属性）
    :param subgraph_template: 有向子图模板（NetworkX DiGraph，含 Type 属性）
    :return: 匹配的子图节点索引列表
    """
    node_match = categorical_node_match('Type', default=None)
    edge_match = numerical_edge_match('weight', default=None)
    matcher = DiGraphMatcher(G, subgraph_template, node_match=node_match, edge_match=edge_match)
    return [list(mapping.keys()) for mapping in matcher.subgraph_isomorphisms_iter()]

# 根据索引序列分离子图，用于图例子图构建
def extract_subgraph_from_component(G, component_indices):
    """
    从有向图中提取指定连通分量子图
    :param G: 原始有向图（NetworkX DiGraph）
    :param component_indices: 连通分量的节点索引列表（如 [0, 1, 2]）
    :return: 子图（NetworkX DiGraph）
    """
    # 提取子图（自动保留节点和边的属性）
    subgraph = G.subgraph(component_indices).copy()
    return subgraph

def extract_subgraph(G, node_list):
    # 创建一个新的空有向图
    G_sub = nx.DiGraph()

    # 添加节点及其 Type 属性
    for node in node_list:
        if node in G:
            # print('node:', G.nodes[node])
            G_sub.add_node(node, **G.nodes[node])

    # 添加边及其 weight 属性（仅在 node_list 内部的边）
    for u in node_list:
        for v in node_list:
            if G.has_edge(u, v):
                # print('weight: (%s, %s) -> %s' % (u, v, G.nodes[node]))
                G_sub.add_edge(u, v, **G.edges[u, v])

    # print('G_sub hash 0:', get_digraph_hash(G_sub))
    # G = G_sub
    # for u, v, data in G.edges(data=True):
    #     source_type = G.nodes[u].get("Type", "N/A")
    #     target_type = G.nodes[v].get("Type", "N/A")
    #     weight = data.get("weight", "N/A")
    #     print(f"{u}({source_type}) -> {v}({target_type}), weight = {weight}")

    # graph = G
    # graph_hash = tuple(sorted((graph.nodes[u].get('Type'), graph.nodes[v].get('Type'), data['weight']) for u, v, data in graph.edges(data=True)))
    # print('G_sub hash 1:', graph_hash)

    return G_sub

# 构造单一节点子图，用于匹配
def build_subgraph_from_single_node(node):
    G = nx.DiGraph()
    dict_type = {'Type': node['Type'] + node['Text']}
    G.add_node(0, **dict_type)
    return G

# 常见角度旋转，90、180、270
def create_rotate_digraph(G):
    G_90 = G.copy()
    G_180 = G.copy()
    G_270 = G.copy()
    # 90°处理
    for u, v, data in G_90.edges(data=True):
        if 'weight' in data and data['weight'] != 8:
            w = data['weight']
            w_10, w_1 = w // 10, w % 10
            data['weight'] = w_10 * 10 + (w_1 + 2) % 8

    # 180°处理
    for u, v, data in G_180.edges(data=True):
        if 'weight' in data and data['weight'] != 8:
            w = data['weight']
            w_10, w_1 = w // 10, w % 10
            data['weight'] = w_10 * 10 + (w_1 + 4) % 8

    # 270°处理
    for u, v, data in G_270.edges(data=True):
        if 'weight' in data and data['weight'] != 8:
            w = data['weight']
            w_10, w_1 = w // 10, w % 10
            data['weight'] = w_10 * 10 + (w_1 + 6) % 8
    
    return [G, G_90, G_180, G_270]

# 有向图序列去重
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
        # graph_hash = (
        #     tuple(sorted((n, data['Type']) for n, data in graph.nodes(data=True))),
        #     tuple(sorted((u, v, data['weight']) for u, v, data in graph.edges(data=True)))
        # )
        graph_hash = tuple(sorted((graph.nodes[u].get('Type'), graph.nodes[v].get('Type'), data['weight']) for u, v, data in graph.edges(data=True)))
        # print('graph_hash:', graph_hash)
        
        if graph_hash not in seen_hashes:
            seen_hashes.add(graph_hash)
            unique_graphs.append(graph)
    
    return unique_graphs


def get_weight(node1: dict, node2: dict, thred=800):        # 距离阈值暂定1米，应该是有点大了，后续再调整
    try:
        # if node1['BlockName'] != node2['BlockName'] or node1['LayerName'] != node2['LayerName']:      # 暂时要求同块同层
        if node1['BlockName'] != node2['BlockName']:  # 考虑去除层次筛选，plan_1中图例和模型空间都出现了同图元但不同层的现象
            return None
        # 图例元素时的共行处理
        if 'line_id' in node1 and 'line_id' in node2 and node1['line_id'] != node2['line_id']:
            return None
        rect1 = node1['Rect']
        rect2 = node2['Rect']
        # 判定两个rect间距是否小于阈值
        if min_distance_between_rectangles(rect1, rect2) > thred:
            return None
        # 计算方向权值
        x1, y1 = (rect1[0] + rect1[2]) / 2, (rect1[1] + rect1[3]) / 2
        x2, y2 = (rect2[0] + rect2[2]) / 2, (rect2[1] + rect2[3]) / 2
        dx, dy = x2 - x1, y2 - y1
        # 计算中心点是否在区域内
        flag_in1, flag_in2 = False, False
        if rect1[0] <= x2 <= rect1[2] and rect1[1] <= y2 <= rect1[3]:
            flag_in1 = True
        if rect2[0] <= x1 <= rect2[2] and rect2[1] <= y1 <= rect2[3]:
            flag_in2 = True
        weight, float_thred = -1, 1e-5
        if dx < -float_thred and dy > float_thred:
            weight = 1
        elif abs(dx) < float_thred and dy > float_thred:
            weight = 2
        elif dx > float_thred and dy > float_thred:
            weight = 3
        elif dx > float_thred and abs(dy) < float_thred:
            weight = 4
        elif dx > float_thred and dy < -float_thred:
            weight = 5
        elif abs(dx) < float_thred and dy < -float_thred:
            weight = 6
        elif dx < -float_thred and dy < -float_thred:
            weight = 7
        elif dx < -float_thred and abs(dy) < float_thred:
            weight = 0
        elif abs(dx) < float_thred and abs(dy) < float_thred:
            weight = 8
        else:
            raise ValueError(f"Error data for dx, dy: ({dx, dy})")
        # print('rect:', rect1, rect2)
        # print('x1: %.5f, y1: %.5f, x2: %.5f, y2: %.5f' % (x1, y1, x2, y2))
        # print('dx: %.5f, dy: %.5f, weight: %d, flag_in1: %s, flag_in2: %s' % (dx, dy, weight, flag_in1, flag_in2))
        if weight == -1:
            return None
        if weight == 8:
            return weight, weight
        elif 0 <= weight < 4:
            weight2 = weight + 4
            weight += 0 if flag_in1 else 10
            weight2 += 0 if flag_in2 else 10
            return weight, weight2
        elif 4 <= weight < 8:
            weight2 = weight - 4
            weight += 0 if flag_in1 else 10
            weight2 += 0 if flag_in2 else 10
            return weight, weight2
        else:
            raise ValueError(f"Error data for weight: {weight}")    # 抛出异常

    except Exception as e:
        print(f"Error: {e}")
        return None

def min_distance_between_rectangles(rect1, rect2):
    # 解包矩形的坐标
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # 计算 x 轴上的最小距离
    x_min_dist = max(0, max(x1_1, x1_2) - min(x2_1, x2_2))
    # 计算 y 轴上的最小距离
    y_min_dist = max(0, max(y1_1, y1_2) - min(y2_1, y2_2))
    d = (x_min_dist ** 2 + y_min_dist ** 2) ** 0.5

    # 如果两个矩形相交，距离为 0
    if x_min_dist == 0 and y_min_dist == 0:
        return 0
    # 如果只有一个方向有距离，返回该方向的距离
    elif x_min_dist == 0:
        return y_min_dist
    elif y_min_dist == 0:
        return x_min_dist
    # 如果两个方向都有距离，根据勾股定理计算对角线距离
    return (x_min_dist ** 2 + y_min_dist ** 2) ** 0.5

def get_indices_rect(nodes, indices):
    if nodes is None or indices is None or len(indices) < 1:
        return None
    x1, y1, x2, y2 = nodes[indices[0]]['Rect']
    num = len(indices)
    for i in range(1, num):
        xx1, yy1, xx2, yy2 = nodes[indices[i]]['Rect']
        x1 = min(x1, xx1)
        y1 = min(y1, yy1)
        x2 = max(x2, xx2)
        y2 = max(y2, yy2)
    return [x1, y1, x2, y2]

def filter_components(nodes, components):
    rects, labels = [], []
    for indices, types in components:
        if len(indices) > 1 or len(indices) == 1 and (types[0].startswith('Block') or types[0].startswith('Hatch')):    # and优先级更高
            rect = get_indices_rect(nodes, indices)
            if rect is None:
                print(f'Error indices: {indices}, types: {types}')
                continue
            rects.append(rect)
            labels.append(nodes[indices[0]]['Type'] + nodes[indices[0]]['Text'])

    return rects, labels

def filter_components2(nodes, components):
    '''
    图例元素组合
        参数nodes：原始节点
        参数components：查找分量得到的索引元组、类型元组
    '''
    nodes_legend = []
    for indices, types in components:
        # 独立的Block和Hatch类型
        if len(indices) == 1 and (types[0].startswith('Block') or types[0].startswith('Hatch')):   # and优先级更高
            nodes_legend.append(nodes[indices[0]])
        # 组合类型
        elif len(indices) > 1:
            rect = get_indices_rect(nodes, indices)
            if rect is None:
                print(f'Error indices: {indices}, types: {types}')
                continue
            node_refer = nodes[indices[0]]
            node = {
                'Type': 'Merge',
                'Text': '',
                'Indices': indices,
                'Rect': rect,
                'BlockName': node_refer['BlockName'],
                'LayerName': node_refer['LayerName'],
                'line_id': node_refer['line_id']
            }
            nodes_legend.append(node)

    return nodes_legend

def filter_legend_text(legend_items):
    '''
    图例中文本类型筛选：
        1.带中文，候选标签
        2.无中文且长度小于3，可能图例结构
        3.无中文长度大于等于3，说明文本需要剔除
    '''
    # 辅助函数：判断字串中是否包含中文字符
    def has_chinese_char(s):
        pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(pattern.search(s))

    items_remove, items_label = [], []
    for i, item in enumerate(legend_items):
        if item['Type'] == 'Text':
            txt = item['Text']
            if has_chinese_char(txt):
                items_label.append(i)
            elif len(txt) > 2:      # 长度大于2的全英文
                items_remove.append(i)

    label_items, legend_ans = [], []
    num = len(legend_items)
    for i in range(num):
        if i in items_label:
            label_items.append(legend_items[i])
        elif not i in items_remove:
            legend_ans.append(legend_items[i])
    return legend_ans, label_items

def do_map_rects(rects, box, imgWidth=1600, imgHeight=1280):
    if rects is None or len(rects) == 0 or box is None or len(box) != 4:
        print('Here is do_map_rects, input error.')
        return []
    imgCenterX = imgWidth / 2
    imgCenterY = imgHeight / 2
    rangeWidth = box[2] - box[0]
    rangeHeight = box[3] - box[1]
    rangeCenterX = (box[2] + box[0]) / 2
    rangeCenterY = (box[3] + box[1]) / 2

    k1 = imgHeight * 1. / imgWidth
    k2 = rangeHeight * 1. / rangeWidth 
    scale = (imgWidth * 1. / rangeWidth) if k1 > k2 else (imgHeight * 1. / rangeHeight)

    rects_ans = []
    for rect in rects:
        x1, y1, x2, y2 = rect
        xx1 = round(imgCenterX + (x1 - rangeCenterX) * scale)
        yy1 = imgHeight - round(imgCenterY + (y1 - rangeCenterY) * scale)
        xx2 = round(imgCenterX + (x2 - rangeCenterX) * scale)
        yy2 = imgHeight - round(imgCenterY + (y2 - rangeCenterY) * scale)
        if xx1 < 0 or xx1 > imgWidth or xx2 < 0 or xx2 > imgWidth or yy1 < 0 or yy1 > imgHeight or yy2 < 0 or yy2 > imgHeight:
            print('Out of range: (%.2f, %.2f, %.2f, %.2f) -> (%d, %d, %d, %d)' % (x1, y1, x2, y2, xx1, xx2, yy1, yy2))
            continue
        rects_ans.append([min(xx1, xx2), min(yy1, yy2), max(xx1, xx2), max(yy1, yy2)])
    return rects_ans

def rects_to_json(rects, save_path, img_path, label='CeilingItem'):
    if rects is None or len(rects) == 0:
        print('data none for:', save_path)
        return

    im = imgRead(img_path)
    h, w, _ = im.shape
    data = {
        "version": "5.5.0",
        "flags": {},
        "imagePath": img_path,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    shapes = []
    for rect in rects:
        x1, y1, x2, y2 = rect
        shape = {
            'label': label, 
            'points': [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)

    data['shapes'] = shapes
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def dict_to_json(match_dict, save_path, img_path):
    if match_dict is None or len(match_dict) == 0:
        print('data none for:', save_path)
        return

    im = imgRead(img_path)
    h, w, _ = im.shape
    data = {
        "version": "5.5.0",
        "flags": {},
        "imagePath": img_path,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    shapes = []
    for label in match_dict:
        rects = match_dict[label]['rects']
        for rect in rects:
            x1, y1, x2, y2 = rect
            shape = {
                'label': label, 
                'points': [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
            shapes.append(shape)

    data['shapes'] = shapes
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_legend_lines(nodes):
    '''
    给定连通查找后的图例元素，输出图例行排列
    '''
    lines = dict()
    # 输入判断
    if nodes is None or len(nodes) == 0:
        print('Get legend lines, input is None.')
        return lines
    # 按行放入
    for node in nodes:
        line_id = node['line_id']
        if not line_id in lines:
            lines[line_id] = []
        lines[line_id].append(node)
    # 剔除全文本/全图例
    for line_id in list(lines.keys()):    # 创建键列表副本
        line = lines[line_id]
        cnt_legend, cnt_text = 0, 0
        for item in line:
            if item['Type'] == 'Text':
                cnt_text += 1
            else:
                cnt_legend += 1
        if cnt_legend == 0 or cnt_text == 0:
            del lines[line_id]
    # 坐标排序
    for line_id in lines:
        line = lines[line_id]
        lines[line_id] = sorted(line, key=lambda x: (-x['Rect'][1], x['Rect'][0]))
    return lines

def get_legend_dict(lines):
    '''
    输入图例行，解析行结构，输出键值对
    '''
    # 输入判断
    if not lines:
        print('Get legend dict, input is None.')
        return dict()
    # 逐行构造
    legend_dicts = {}
    for line_id in lines:
        line = lines[line_id]
        num, i = len(line), 0
        while i < num:
            sign = False
            type_id1 = 0 if line[i]['Type'] == 'Text' else 1
            for j in range(i + 1, num):
                type_id2 = 0 if line[j]['Type'] == 'Text' else 1
                if type_id1 != type_id2:
                    # 构造键值对
                    if type_id1 == 0:
                        key_id, value_id = i, j
                    else:
                        key_id, value_id = j, i
                    # print('key_id: %d, key_type: %s, key_text: %s' % (key_id, line[key_id]['Type'], line[key_id]['Text']))
                    key_text = line[key_id]['Text']
                    if not key_text in legend_dicts:
                        legend_dicts[key_text] = []
                    legend_dicts[key_text].append(line[value_id])
                    # 结束条件
                    i = j + 1
                    sign = True
                    break
            if not sign:
                i += 1

    return legend_dicts

def sort_nodes(nodes):
    return sorted(nodes, key=lambda x: (-x['Rect'][1], x['Rect'][0]))

def build_ceiling_item_graph(json_path):
    res = get_ceiling_item(json_path, 'ceiling_item')
    if res is None:
        print('Get None in get_ceiling_item.')
        return
    nodes, _ = res
    if nodes is None or len(nodes) == 0:
        print('Error nodes:', nodes)
        return
    nodes = sort_nodes(nodes)
    # 构建图并计算连通分量
    G = build_digraph(nodes, get_weight)
    return nodes, G

def is_rect_overlap(rect1, rect2, thred=0.001):      # 距离判定阈值
    if rect1 is None or rect2 is None or len(rect1) != 4 or len(rect2) != 4:
        print('Error: input errror in is_rect_close.')
    if all(abs(rect1[i] - rect2[i]) < thred for i in range(4)):
        return True
    return False

def remove_hatch_line(components, nodes):
    num = len(components)
    for i in range(num):
        indices, types = components[i]
        n = len(types)
        remove_index = []
        hatch_index, line_index, hi_lines = [], [], []
        for j in range(n):
            if types[j].startswith('Hatch'):
                hatch_index.append(j)
            if types[j] == 'Line':
                line_index.append(j)
        if len(hatch_index) == 0 or len(line_index) == 0:
            continue
        for hi in hatch_index:
            x1, y1, x2, y2 = nodes[indices[hi]]['Rect']
            hi_lines.append([x1, y1, x1, y2])
            hi_lines.append([x2, y1, x2, y2])
            hi_lines.append([x1, y2, x2, y2])
            hi_lines.append([x1, y1, x2, y1])
        for li in line_index:
            line = nodes[indices[li]]['Rect']
            flag = False
            for line2 in hi_lines:
                if is_rect_overlap(line, line2):
                    flag = True
                    break
            if flag:
                remove_index.append(li)
        if len(remove_index) != 0:
            # print('Remove index:', *remove_index)
            indices = [indices[i] for i in range(n) if not i in remove_index]
            types = [types[i] for i in range(n) if not i in remove_index]
            components[i] = indices, types

    return components

def remove_included_rects(match_dict):
    '''
    功能：去除match_dict中被包含的检测框结果
    思路：先根据结点数量逆序排序，再依次记录，去除被包含的
    '''
    def is_contained(rect0, rects):
        x1, y1, x2, y2 = rect0
        for rect in rects:
            rx1, ry1, rx2, ry2 = rect
            if (x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2):
                return True
        return False
    match_dict = dict(sorted(match_dict.items(), key=lambda item: item[1]['node_num'], reverse=True))
    rects_record = []
    for label in list(match_dict.keys()):
        rects = match_dict[label]['rects']
        rects_remove, num_total = [], len(rects)
        for i, rect in enumerate(rects):
            if is_contained(rect, rects_record):
                rects_remove.append(i)
            else:
                rects_record.append(rect)
        if len(rects_remove) == num_total:
            del match_dict[label]     # 全部删除
        elif len(rects_remove) > 0:
            match_dict[label]['rects'] = [rects[i] for i in range(num_total) if not i in rects_remove]

    return match_dict

def test():
    json_path = r'C:\Users\DELL\Desktop\test2\tmp0.json'
    img_path = r'C:\Users\DELL\Desktop\test2\tmp1.jpg'
    json_out = r'C:\Users\DELL\Desktop\test2\tmp11.json'
    res = get_ceiling_item(json_path, 'ceiling_item')
    if res is None:
        print('Get None in get_ceiling_item.')
        return
    nodes, box = res
    if nodes is None or len(nodes) == 0:
        print('Error nodes:', nodes)
        return
    print('nodes:', len(nodes), nodes[0])

    rects0 = []
    for node in nodes:
        rects0.append(node['Rect'])
    print('rects 0:', len(rects0), rects0[0])
    
    # 构建图并计算连通分量
    G = build_digraph(nodes, get_weight)
    components = find_connected_components(G)
    
    # 输出结果
    # print("图的节点属性:", G.nodes(data=True))
    # print("连通分量（索引和Type）:")
    # for indices, types in components:
    #     print(f"节点索引: {indices}, Types: {types}")

    rects, _ = filter_components(nodes, components)
    print('rects:', len(rects), rects[0])
    # for i, rect in enumerate(rects):
    #     print(i + 1, rect)

    # rects = rects0
    
    im = imgRead(img_path)
    h, w, _ = im.shape
    rects = do_map_rects(rects, box, w, h)
    print('rects 2:', len(rects), rects[0])
    rects_to_json(rects, json_out, img_path)
    print('----- finish -----')

    '''
    TODO: 
        1.对图例元素进行聚类，构建键值对，构建子图
        2.多角度旋转的子图匹配算法
        3.图例图元的包含关系处理
    '''

def match_legends(json_legend, json_ceiling):
    # json_out = r'C:\Users\DELL\Desktop\test3\outs\plan_2-out1.json'
    res = get_ceiling_item(json_legend, 'legend_item')
    if res is None:
        print('Get legend none in get_ceiling_item.')
        return
    nodes, _ = res
    nodes, label_items = filter_legend_text(nodes)
    if nodes is None or len(nodes) == 0 or label_items is None or len(label_items) == 0:
        print('Error data:', nodes, label_items)
        return
    nodes = sort_nodes(nodes)
    # print('nodes:', len(nodes), nodes[0])
    # print('labels:', len(label_items), label_items[0])

    yd = get_json_attribute(json_legend, 'yd')
    box = get_json_attribute(json_legend, 'range')
    if yd is None or box is None:
        print('Get att None.')
        return
    # print('yd:', yd)

    # -- 构建图并计算连通分量
    G = build_digraph(nodes, get_weight, thred=yd)    # 构建图
    components = find_connected_components(G)         # 查找连通分量

    # 图例空间填充周围线去除
    components = remove_hatch_line(components, nodes)

    # 输出结果
    # print("图的节点属性:", G.nodes(data=True))
    # print("连通分量（索引和Type）:")
    # for indices, types in components:
    #     print(f"节点索引: {indices}, Types: {types}")

    nodes = filter_components2(nodes, components)     # 连通分量组合为图例元素
    nodes += label_items

    # -- 构建图例表格行
    lines = get_legend_lines(nodes)
    # print('lines num:', len(lines))
    # print('lines name:', list(lines.keys()))

    # -- 构建图例对
    legend_dict = get_legend_dict(lines)
    # print('legend dict num:', len(legend_dict))
    # for legend in legend_dict:
    #     print('key: %s, value: %s' % (legend, legend_dict[legend]))

    # -- 子图匹配
    # 获取天花图元有向图
    nodes_ceiling, G_ceiling = build_ceiling_item_graph(json_ceiling)
    match_dict = dict()    # 匹配结果存储
    for legend_label in list(legend_dict.keys()):
        rects, G_subs = [], []    # 输出矩形框、待匹配子图列表
        node_num = 0              # 子图结点数量
        for item in legend_dict[legend_label]:
            if not item['Type'] == 'Merge':
                G_sub = build_subgraph_from_single_node(item)
                G_subs.append(G_sub)
                node_num = max(node_num, 1)
            else:
                G_sub = extract_subgraph_from_component(G, item['Indices'])
                # G_sub = extract_subgraph(G, item['Indices'])    # 另一个子图匹配方法
                # 生成常见旋转子图
                G_subs += create_rotate_digraph(G_sub)
                node_num = max(node_num, len(item['Indices']))
        # print('Label: %s, G_subs num origin: %d' % (legend_label, len(G_subs)))
        G_subs = deduplicate_digraphs(G_subs)      # 有向图列表去重
        # print('Label: %s, G_subs num deduplicate: %d' % (legend_label, len(G_subs)))
        # num = len(G_subs)
        for i, G_sub in enumerate(G_subs):
            # print('%d / %d' % (i + 1, num))
            match_res = find_directed_subgraphs(G_ceiling, G_sub)
            for indices in match_res:
                rects.append(get_indices_rect(nodes_ceiling, indices))
        # print('Label: %s, get rects num: %d' % (legend_label, len(rects)))
        if len(rects) > 0:
            match_dict[legend_label] = {'node_num': node_num, 'rects': rects}
    
    num_rects = 0
    for label in match_dict:
        num_rects += len(match_dict[label]['rects'])
    print('legend num: %d, rects num: %d' % (len(match_dict), num_rects))

    # 去除被包含的匹配结果
    match_dict = remove_included_rects(match_dict)

    num_rects = 0
    for label in match_dict:
        num_rects += len(match_dict[label]['rects'])
    print('legend num: %d, rects num: %d' % (len(match_dict), num_rects))
    return match_dict
        
    # -- 结果可视化为labelme标注文件
    # im = imgRead(img_path)
    # h, w, _ = im.shape
    # for label in list(match_dict.keys()):
    #     match_dict[label]['rects'] = do_map_rects(match_dict[label]['rects'], box, w, h)
    # dict_to_json(match_dict, json_out, img_path)
    # print('----- finish -----')

def get_vps_name(input_dir):
    jsons = os.listdir(input_dir)
    vps = []
    # print('num:', len(jsons))
    for j in jsons:
        if j.endswith('-model.json') and '--&&&--' in j:
            vp = j[:-11].split('--&&&--')[1]
            # print('vp:', vp)
            vps.append(vp)
    return vps

def match_legends_batch(input_dir, dwg_name):
    legend_dir = os.path.join(input_dir, 'legends')
    vps = get_vps_name(legend_dir)
    if len(vps) == 0: 
        print('Vps is empty.')
        return
    match_data = dict()
    num = len(vps)
    if '.' in dwg_name:
        dwg_name = os.path.splitext(dwg_name)[0]
    for i, vp_name in enumerate(vps):
        print('%d / %d, %s' % (i + 1, num, vp_name))
        json_legend = os.path.join(legend_dir, dwg_name + '--&&&--' + vp_name + '-legend.json')
        json_model = os.path.join(legend_dir, dwg_name + '--&&&--' + vp_name + '-model.json')
        match_dict = match_legends(json_legend, json_model)
        match_data[vp_name] = match_dict
    legend_data = dict()
    legend_data['dwg_name'] = dwg_name
    legend_data['match_data'] = match_data
    with open(os.path.join(input_dir, dwg_name + '-legend-match.json'), 'w', encoding='utf-8') as f:
        json.dump(legend_data, f, indent=2, ensure_ascii=False)


def parse_matched_legends(legend_match_json, out_path):
    '''
    @Function: 从匹配好图例全集中提取图例、去重、归类，生成清单表及备份
    '''
    if not os.path.exists(legend_match_json) or not os.path.exists(out_path): return
    with open(legend_match_json, 'r', encoding='utf-8') as f:
        legend_data = json.load(f)
    dwg_name = legend_data['dwg_name']
    match_data = legend_data['match_data']

    legend_list = []   # 记视口、记两类、记名称
    for vp_name in list(match_data.keys()):
        vp_data = match_data[vp_name]
        for item_name in list(vp_data.keys()):
            classify_result = classify_legend(item_name)
            if classify_result is None or not 'subject' in classify_result or not 'cate' in classify_result:
                print('Error: 图例分类接口分类失败。text:', item_name)
                continue
            item_dict = dict(
                name = item_name,
                subject = classify_result['subject'],
                cate = classify_result['cate'],
                vp_name = vp_name,
                data = vp_data[item_name]
            )
            legend_list.append(item_dict)

    # 去重复，多视口能包含相同图例，取多的
    legend_list2 = []
    num = len(legend_list)
    signs = [False for i in range(num)]
    for i in range(num):
        if signs[i]: continue
        for j in range(i + 1, num):
            if signs[j]: continue
            item_i, item_j = legend_list[i], legend_list[j]
            if item_i['name'] == item_j['name']:
                if item_i['vp_name'] == item_j['vp_name']:
                    print('Notice: repeat item and repeat vp_name:%s %s' % (item_i['name'], item_i['vp_name']))
                else:
                    print('Notices: repeat item but not repeat vp_name: %s %s %s' % (item_i['name'], item_i['vp_name'], item_j['vp_name']))
                    signs[i], signs[j] = True, True
                    if len(item_i['data']['rects']) < len(item_j['data']['rects']):
                        legend_list2.append(item_j)
                    else:
                        legend_list2.append(item_i)
        if not signs[i]:
            legend_list2.append(signs[i])
    print('legend list num:', len(legend_list), len(legend_list2))

    legend_bill = dict()
    for item in legend_list2:
        subject, cate = item['subject'], item['cate']    
        name, num = item['name'], len(item['data']['rects'])
        item_dict = {'name': name, 'num': num}
        if not subject in legend_bill:
            cate_dict = {cate: item_dict}
            legend_bill[subject] = [cate_dict]
        elif not cate in legend_bill[subject]:
            cate_dict = {cate: item_dict}
            legend_bill[subject].append(cate_dict)
        else:
            legend_bill[subject][cate_dict].append(item_dict)

    out_json = os.path.join(out_path, dwg_name + '-legend-bill.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(legend_bill, f, ensure_ascii=False)
    print('----- finish -----')
    

def test2():
    # json_path = r'C:\Users\DELL\Desktop\test2\tmp21.json'
    # json_ceiling = r'C:\Users\DELL\Desktop\test2\tmp0.json'
    # img_path = r'C:\Users\DELL\Desktop\test2\tmp1.jpg'
    # json_out = r'C:\Users\DELL\Desktop\test2\res\tmp2-out1.json'
    json_path = r'C:\Users\DELL\Desktop\test3\jsons\plan_2-legend.json'
    json_ceiling = r'C:\Users\DELL\Desktop\test3\jsons\plan_2-ceiling.json'
    img_path = r'C:\Users\DELL\Desktop\test3\imgs\plan_2-4.jpg'
    json_out = r'C:\Users\DELL\Desktop\test3\outs\plan_2-out1.json'
    res = get_ceiling_item(json_path, 'legend_item')
    if res is None:
        print('Get None in get_ceiling_item.')
        return
    nodes, _ = res
    nodes, label_items = filter_legend_text(nodes)
    if nodes is None or len(nodes) == 0 or label_items is None or len(label_items) == 0:
        print('Error data:', nodes, label_items)
        return
    nodes = sort_nodes(nodes)
    print('nodes:', len(nodes), nodes[0])
    print('labels:', len(label_items), label_items[0])

    yd = get_json_attribute(json_path, 'yd')
    box = get_json_attribute(json_path, 'range')
    if yd is None or box is None:
        print('Get att None.')
        return
    print('yd:', yd)

    # -- 构建图并计算连通分量
    G = build_digraph(nodes, get_weight, thred=yd)    # 构建图
    components = find_connected_components(G)         # 查找连通分量

    # 图例空间填充周围线去除
    components = remove_hatch_line(components, nodes)

    # 输出结果
    print("图的节点属性:", G.nodes(data=True))
    print("连通分量（索引和Type）:")
    for indices, types in components:
        print(f"节点索引: {indices}, Types: {types}")

    nodes = filter_components2(nodes, components)     # 连通分量组合为图例元素
    nodes += label_items

    # -- 构建图例表格行
    lines = get_legend_lines(nodes)
    print('lines num:', len(lines))
    print('lines name:', list(lines.keys()))

    # -- 构建图例对
    legend_dict = get_legend_dict(lines)
    print('legend dict num:', len(legend_dict))
    for legend in legend_dict:
        print('key: %s, value: %s' % (legend, legend_dict[legend]))

    # -- 子图匹配
    # 获取天花图元有向图
    nodes_ceiling, G_ceiling = build_ceiling_item_graph(json_ceiling)
    match_dict = dict()    # 匹配结果存储
    for legend_label in list(legend_dict.keys()):
        rects, G_subs = [], []    # 输出矩形框、待匹配子图列表
        node_num = 0              # 子图结点数量
        for item in legend_dict[legend_label]:
            if not item['Type'] == 'Merge':
                G_sub = build_subgraph_from_single_node(item)
                G_subs.append(G_sub)
                node_num = max(node_num, 1)
            else:
                G_sub = extract_subgraph_from_component(G, item['Indices'])
                # G_sub = extract_subgraph(G, item['Indices'])    # 另一个子图匹配方法
                # 生成常见旋转子图
                G_subs += create_rotate_digraph(G_sub)
                node_num = max(node_num, len(item['Indices']))
        # print('Label: %s, G_subs num origin: %d' % (legend_label, len(G_subs)))
        G_subs = deduplicate_digraphs(G_subs)      # 有向图列表去重
        print('Label: %s, G_subs num deduplicate: %d' % (legend_label, len(G_subs)))
        num = len(G_subs)
        for i, G_sub in enumerate(G_subs):
            print('%d / %d' % (i + 1, num))
            match_res = find_directed_subgraphs(G_ceiling, G_sub)
            for indices in match_res:
                rects.append(get_indices_rect(nodes_ceiling, indices))
        print('Label: %s, get rects num: %d' % (legend_label, len(rects)))
        if len(rects) > 0:
            match_dict[legend_label] = {'node_num': node_num, 'rects': rects}
    
    num_rects = 0
    for label in match_dict:
        num_rects += len(match_dict[label]['rects'])
    print('legend num: %d, rects num: %d' % (len(match_dict), num_rects))

    # 去除被包含的匹配结果
    match_dict = remove_included_rects(match_dict)

    num_rects = 0
    for label in match_dict:
        num_rects += len(match_dict[label]['rects'])
    print('legend num 1: %d, rects num 1: %d' % (len(match_dict), num_rects))
        
    # -- 结果可视化为labelme标注文件
    im = imgRead(img_path)
    h, w, _ = im.shape
    for label in list(match_dict.keys()):
        match_dict[label]['rects'] = do_map_rects(match_dict[label]['rects'], box, w, h)
    dict_to_json(match_dict, json_out, img_path)
    print('----- finish -----')

def test3():
    input_dir = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public3\dwgs2\2c34a2b5-88c3-4d78-a42b-5623cf225044\legend_data'
    dwg_name = 'plan_2.dwg'
    match_legends_batch(input_dir, dwg_name)

def test_parse_matched_legends():
    input_dir = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public3\dwgs2\2c34a2b5-88c3-4d78-a42b-5623cf225044\legend_data\plan_2-legend-match.json'
    out_path = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public3\dwgs2\2c34a2b5-88c3-4d78-a42b-5623cf225044\legend_data'

    parse_matched_legends(input_dir, out_path)



if __name__ == '__main__':
    test_parse_matched_legends()
