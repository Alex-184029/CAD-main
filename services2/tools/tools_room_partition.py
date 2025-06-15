import json
from shapely.geometry import MultiPolygon, Point
from shapely.ops import polygonize, unary_union
import re
import requests

room_type = {
    'living': ['客厅', '起居室', '家庭厅'],
    'bolcany': ['阳台', '露台'],
    'bed': ['卧室', '主卧', '次卧', '客卧', '主人房', '老人房', '孩房', '儿童房', '客房', '长辈房'],
    'wash': ['清洗间', '家务间', '家政间', '家政房', '家政区', '洗衣房', '洗衣区', '盥洗房', '盥洗室', '盥洗区'],
    'kitchen': ['厨房'],
    'canteen': ['餐厅'],
    'rest': ['卫生间', '主卫', '客卫', '次卫', '公卫', '洗手间', '厕所', '浴池', '浴室', '淋浴间'],
    'study': ['书房', '工作室'],
    'hall': ['玄关', '门厅', '走廊', '过道', '门廊', '走道'],
    'play': ['娱乐室', '休闲区', '茶室', '健身房', '游戏厅'],
    'court': ['庭院', '花园', '花池'],
    'others': ['垃圾房', '设备间', '壁橱', '衣帽间', '保姆房', '电梯', '楼梯', '避难间', '避难区', '前室', '化妆间', '储藏室', '储物间', '多功能房', '多功能间', '多功能室', '多功能厅']
}

def keep_chinese_characters(input_string):     # 只保留汉字字符
    # 使用正则表达式匹配汉字字符
    chinese_characters = re.findall(r'[\u4e00-\u9fff]+', input_string)
    # 将匹配到的汉字字符列表合并成一个字符串
    result = ''.join(chinese_characters)
    return result

def filterTxts(txts):
    txts_new = []
    for txt in txts:
        txt_ch = keep_chinese_characters(txt)
        num = len(txt_ch)
        if num > 1 and num < 8:
            txts_new.append(txt_ch)
    return txts_new

def filterTxts2(labels):
    labels_new = []
    for label in labels:
        txt = label['txt']
        txt_ch = keep_chinese_characters(txt)
        num = len(txt_ch)
        if num > 1 and num < 8:
            labels_new.append(label)
    return labels_new

def construct_constraint_graph(polygons):
    # height, width = img_size
    # boundary = box(0, 0, width, height)  # 图像边界，考虑不需要加入图像边界的考量
    # edges = [boundary.exterior] + [poly.exterior for poly in polygons]
    edges = [poly.exterior for poly in polygons]
    
    # 计算所有约束边界的组合
    merged_edges = unary_union(edges)
    constrained_polygons = list(polygonize(merged_edges))  # 计算所有闭合区域
    
    return constrained_polygons

def filter_unlabeled_regions(constrained_polygons, labeled_polygons):
    labeled_multi = MultiPolygon(labeled_polygons)
    unlabeled_regions = [poly for poly in constrained_polygons if not labeled_multi.contains(poly)]

    # 过滤掉触及图像边界的区域
    # filtered_regions = [poly for poly in unlabeled_regions if not poly.intersects(boundary.exterior)]

    return unlabeled_regions

def save_to_labelme(json_origin, unlabeled_polygons, output_json):
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)

    shapes = []
    for poly in unlabeled_polygons:
        shape = {
            "label": "UnlabeledRegion",
            "points": list(map(list, poly.exterior.coords)),
            "shape_type": "polygon",
            "group_id": None,
            "description": '',
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
    data['shapes'] = shapes
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_to_labelme2(json_origin, rooms, output_json):
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)

    shapes = []
    for room in rooms:
        shape = {
            "label": '-'.join(room['function']),
            "points": list(map(list, room['poly'].exterior.coords)),
            "shape_type": "polygon",
            "group_id": None,
            "description": '',
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
    data['shapes'] = shapes
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def classify_room(text):
    url = 'http://127.0.0.1:5006/classify_room2'
    data = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result['res']
    else:
        return []

def handle_room_partition(labeled_polygons, labels):
    constrained_polygons = construct_constraint_graph(labeled_polygons)
    unlabeled_polygons = filter_unlabeled_regions(constrained_polygons, labeled_polygons)
    room_types = list(room_type.keys()) + ['default']
    rooms = []
    # print('labels 0:', labels[0])
    for i, poly in enumerate(unlabeled_polygons):
        room = {}
        room['poly'] = list(map(list, poly.exterior.coords))
        room['id'] = i + 1
        room['area'] = poly.area / 1e4        # 单位平方米
        room['perimeter'] = poly.length / 1e2 # 单位米
        room['function'] = []
        room['labels'] = []   # 可能为空
        for label in labels:
            # print('label:', label)
            if 'x' in label and 'y' in label and poly.contains(Point(label['x'], label['y'])):
                funcs = classify_room(label['text'])
                if len(funcs) > 0:
                    funcs = [room_types[func] for func in funcs]
                    room['function'] += funcs
                    room['labels'].append(label)
        if len(room['function']) == 0:
            room['function'].append('default')
        rooms.append(room)

    # for room in rooms:
    #     print('room: %d, area: %.3f, function: %s' % (room['id'], room['area'], room['function']))
    return rooms