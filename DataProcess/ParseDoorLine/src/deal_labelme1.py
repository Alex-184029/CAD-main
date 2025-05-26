import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box, Point
from shapely.ops import polygonize, unary_union
import os
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

def readTxt(txtpath):
    if not os.path.exists(txtpath):
        print('Txt path not exist, ', txtpath)
        return
    with open(txtpath, 'r', encoding='utf-8') as f:
        datas = f.readlines()
    if len(datas) < 4:
        print('Blank data')
        return
    datas = [data.strip() for data in datas[3:]]
    txts = []
    # num = len(datas)
    # print('datas num:', num)
    for data in datas:
        index1 = 6
        index2 = data.find(', X: ')
        if index2 < index1:
            print('Error index:', index2)
            continue
        txts.append(data[index1:index2])
    txts = filterTxts(txts)
    return txts

def readTxt2(txtpath):
    if not os.path.exists(txtpath):
        print('Txt path not exist, ', txtpath)
        return
    with open(txtpath, 'r', encoding='utf-8') as f:
        datas = f.readlines()
    if len(datas) < 4:
        print('Blank data')
        return
    data_range = datas[1].strip()[7:].split(',')
    data_range = [float(num) for num in data_range]
    datas = [data.strip() for data in datas[3:]]
    labels = []
    for data in datas:
        label = {}
        index1 = 6
        index2 = data.find(', X: ')
        index3 = data.find(', Y: ')
        if index2 < index1 or index3 < index2:
            print('Error index:', index1, index2, index3)
            continue
        try:
            label['txt'] = data[index1:index2]
            label['x'] = float(data[index2 + 5:index3])
            label['y'] = float(data[index3 + 5:])
            labels.append(label)
        except:
            print('Error data:', data)
            continue
    labels = filterTxts2(labels)
    return {'labels': labels, 'box': data_range}

def doMapRangeLabels(data, imgWidth=1600, imgHeight=1280):
    if (data['box'] is None or len(data['box']) != 4 or len(data['labels']) == 0):
        return
    imgCenterX = imgWidth / 2
    imgCenterY = imgHeight / 2
    rangeWidth = data['box'][2] - data['box'][0]
    rangeHeight = data['box'][3] - data['box'][1]
    rangeCenterX = (data['box'][2] + data['box'][0]) / 2
    rangeCenterY = (data['box'][3] + data['box'][1]) / 2

    k1 = imgHeight * 1. / imgWidth
    k2 = rangeHeight * 1. / rangeWidth 
    scale = (imgWidth * 1. / rangeWidth) if k1 > k2 else (imgHeight * 1. / rangeHeight)
    
    labels_new = []
    for label in data['labels']:
        x0, y0 = label['x'], label['y']
        x1 = round(imgCenterX + (x0 - rangeCenterX) * scale)
        y1 = imgHeight - round(imgCenterY + (y0 - rangeCenterY) * scale)
        if x1 < 0 or y1 < 0 or x1 > imgWidth or y1 > imgHeight:
            continue
        labels_new.append({'txt': label['txt'], 'x': x1, 'y': y1})
    data['labels'] = labels_new

def load_labelme_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    polygons = [Polygon(shape['points']) for shape in data['shapes'] if shape['shape_type'] == 'polygon']
    img_size = (data['imageWidth'], data['imageHeight'])
    return polygons, img_size

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

def main():
    input_json = r'../data/labels_Room/(T3) 12#楼105户型平面图（镜像）-3_Structure2.json'
    output_json = r'../data/tmp_res/tmp_region2.json'
    labeled_polygons, img_size = load_labelme_json(input_json)
    constrained_polygons = construct_constraint_graph(labeled_polygons)
    unlabeled_polygons = filter_unlabeled_regions(constrained_polygons, labeled_polygons)
    save_to_labelme(input_json, unlabeled_polygons, output_json)
    print('Write json to:', output_json)

def classify_room(text):
    url = 'http://127.0.0.1:5050/classify_room2'
    data = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result['res']
    else:
        return []

def use_room_label():
    # input_json = r'../data/labels_Room/(T3) 12#楼105户型平面图（镜像）-3_Structure2.json'
    # room_label = r'../data/labels_Room/(T3) 12#楼105户型平面图（镜像）-3.txt'
    input_json = r'../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_Structure3.json'
    room_label = r'../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_RoomText.txt'
    output_json = r'../data/tmp_res/tmp_region4.json'

    data = readTxt2(room_label)
    print('box:', data['box'])
    # print('labels:', data['labels'])
    labeled_polygons, img_size = load_labelme_json(input_json)
    w, h = img_size
    doMapRangeLabels(data, w, h)
    # print('labels2:', data['labels'])

    constrained_polygons = construct_constraint_graph(labeled_polygons)
    unlabeled_polygons = filter_unlabeled_regions(constrained_polygons, labeled_polygons)
    room_types = list(room_type.keys()) + ['default']
    rooms = []
    labels = data['labels']
    for i, poly in enumerate(unlabeled_polygons):
        room = {}
        room['poly'] = poly
        room['id'] = i + 1
        room['area'] = poly.area / 1e4     # 单位平方米
        room['function'] = []
        for label in labels:
            if poly.contains(Point(label['x'], label['y'])):
                funcs = classify_room(label['txt'])
                if len(funcs) > 0:
                    funcs = [room_types[func] for func in funcs]
                    room['function'] += funcs
        if len(room['function']) == 0:
            room['function'].append('default')
        rooms.append(room)

    for room in rooms:
        print('room: %d, area: %.3f, function: %s' % (room['id'], room['area'], room['function']))
    save_to_labelme2(input_json, rooms, output_json)
    print('Write json to:', output_json)

def find_living_room_label():
    input_json = r'../data/labels_Room/(T3) 12#楼105户型平面图（镜像）-3_Structure2.json'
    room_label = r'../data/labels_Room/(T3) 12#楼105户型平面图（镜像）-3.txt'
    # output_json = r'../data/tmp_res/tmp_region3.json'

    data = readTxt2(room_label)
    print('box:', data['box'])
    # print('labels:', data['labels'])
    labeled_polygons, img_size = load_labelme_json(input_json)
    w, h = img_size
    doMapRangeLabels(data, w, h)
    # print('labels2:', data['labels'])

    constrained_polygons = construct_constraint_graph(labeled_polygons)
    unlabeled_polygons = filter_unlabeled_regions(constrained_polygons, labeled_polygons)
    room_types = list(room_type.keys()) + ['default']
    rooms = []
    labels = data['labels']
    for i, poly in enumerate(unlabeled_polygons):
        room = {}
        room['poly'] = poly
        room['id'] = i + 1
        room['area'] = poly.area / 1e4     # 单位平方米
        room['function'] = []
        room['labels'] = []
        for label in labels:
            if poly.contains(Point(label['x'], label['y'])):
                funcs = classify_room(label['txt'])
                if len(funcs) > 0:
                    funcs = [room_types[func] for func in funcs]
                    room['function'] += funcs
                    room['labels'].append(label)
        if len(room['function']) == 0:
            room['function'].append('default')
        if ('living' in room['function']):
            rooms.append(room)
    
    # print('rooms:', rooms)
    if len(rooms) == 0:
        print('Living rooms num is 0.')
        return
    room_living = rooms[0]
    print('living room:', room_living)
    return room_living


    # for room in rooms:
    #     print('room: %d, area: %.3f, function: %s' % (room['id'], room['area'], room['function']))
    # save_to_labelme2(input_json, rooms, output_json)
    # print('Write json to:', output_json)



if __name__ == "__main__":
    # t0 = time.time()
    # main()
    # print('Finish, time: %.4f s' % (time.time() - t0))
    use_room_label()
    # find_living_room_label()
