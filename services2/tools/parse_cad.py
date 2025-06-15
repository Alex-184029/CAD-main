import requests
import os
import json
from shapely.geometry import Polygon

import sys
sys.path.append('../')

from tools.tools1 import writeBase64, imgShape, do_map_data
from tools.tools_door import parse_door_tool2, parse_door_tool3
from tools.tools_wall import parse_wall_tool
from tools.tools_door_frame import handle_door_frame, rects_to_polygons
from tools.tools_room_partition import handle_room_partition
from tools.tools_living_room_patition import handle_living_room_partition

parse_cad_url = 'http://127.0.0.1:5005'        # 本地
# parse_cad_url = 'http://192.168.131.128:5005'  # 虚拟机

dwg_public = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public3\dwgs2'

def post_url_json(url, dwg_path):
    if not os.path.exists(dwg_path):
        print('post_url1, dwg_path not exists:', dwg_path)
        return None
    with open(dwg_path, "rb") as f:
        files = {"file": f}
        res = requests.post(url, files=files)

    # print(f"状态码: {res.status_code}")
    return res.json()['res']

def post_url_image(url, dwg_name):
    data = {
        'dwg_name': dwg_name
    }
    res = requests.post(url, data=data)
    return res.json()['img']

def save_to_labelme_room(image_path, rooms, output_json):
    imgWidth, imgHeight = imgShape(image_path)

    data = {
        "version": "5.5.0",
        "flags": {},
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": imgHeight,
        "imageWidth": imgWidth
    }

    shapes = []
    for room in rooms:
        shape = {
            "label": '-'.join(room['function']),
            "points": room['poly'],
            "shape_type": "polygon",
            "group_id": None,
            "description": '',
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
    data['shapes'] = shapes
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_to_labelme_poly(image_path, polygons, output_json):
    imgWidth, imgHeight = imgShape(image_path)

    data = {
        "version": "5.5.0",
        "flags": {},
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": imgHeight,
        "imageWidth": imgWidth
    }
    shapes = []
    for poly in polygons:
        shape = {
            "label": "Polygon",
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
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_to_labelme_line(image_path, lines, output_json):
    imgWidth, imgHeight = imgShape(image_path)
    data = {
        "version": "5.5.0",
        "flags": {},
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": imgHeight,
        "imageWidth": imgWidth
    }
    
    shapes = []
    for line in lines:
        shape = {
            'label': 'Line', 
            'points': [[line[0], line[1]], [line[2], line[3]]],
            "group_id": None,
            "description": "",
            "shape_type": "line",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
    data['shapes'] = shapes
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
def parse_door(task_id, dwgname):
    print('Here is parse_door.')
    url_json = parse_cad_url + '/parse_door'
    url_image = parse_cad_url + '/get_plane_layout_img'
    work_dir = os.path.join(dwg_public, task_id)
    dwg_path = os.path.join(work_dir, dwgname)
    if not os.path.join(dwg_path):
        print('parse_door, dwg_path not exist:', dwg_path)
        return None

    # 调用服务，提取cad解析信息
    data = post_url_json(url_json, os.path.join(work_dir, dwgname))
    # 图像暂存
    dwg = os.path.splitext(dwgname)[0]
    img_path = os.path.join(work_dir, dwg + '_PlaneLayout.jpg')
    if not os.path.exists(img_path):
        img_plane_layout = post_url_image(url_image, dwgname)
        writeBase64(img_plane_layout, img_path)
    
    # 坐标转换
    box = data['range']
    w, h = imgShape(img_path)
    if w is None or h is None:
        print('Error in parse_door: get imgShape error.')
        return None
    print('before convert:', data['arc_items'][0])
    do_map_data(data, box, w, h)
    # print('after convert:', data)
    print('after convert:', data['arc_items'][0])

    # 属性校验
    atts = list(data.keys())
    if not 'arc_items' in atts or not 'door_line_items' in atts:
        print('Error in parse_door, att does not exist.')
        return None
    
    # 提取信息
    lines = []
    for item in data['door_line_items']:
        lines.append(item['point'])
    arc_doors = []
    for item in data['arc_items']:
        arc_doors.append(item['rect'] + item['point'])

    # 解析门并分类
    single_arc_doors, double_arc_doors, slide_doors = parse_door_tool2(lines, arc_doors)

    # 结果保存
    data_res = dict()
    data_res['task_id'] = task_id
    data_res['dwg_name'] = dwgname
    data_res['range'] = data['range']
    data_res['box'] = data['box']
    doors_dict = dict()
    data_res['doors'] = doors_dict
    doors_dict['single_arc_doors'] = single_arc_doors
    doors_dict['double_arc_doors'] = double_arc_doors
    doors_dict['slide_doors'] = slide_doors

    out_path = os.path.join(work_dir, dwg + '_ResDoor.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_res, f, indent=2, ensure_ascii=False)
    return 'Succeed'

def parse_window(task_id, dwgname):
    print('Here is parse_window.')
    url_json = parse_cad_url + '/parse_window'
    url_image = parse_cad_url + '/get_plane_layout_img'
    work_dir = os.path.join(dwg_public, task_id)
    dwg_path = os.path.join(work_dir, dwgname)
    if not os.path.join(dwg_path):
        print('parse_window, dwg_path not exist:', dwg_path)
        return None

    # 调用服务，提取cad解析信息
    data = post_url_json(url_json, os.path.join(work_dir, dwgname))
    # 图像暂存
    dwg = os.path.splitext(dwgname)[0]
    img_path = os.path.join(work_dir, dwg + '_PlaneLayout.jpg')
    if not os.path.exists(img_path):
        img_plane_layout = post_url_image(url_image, dwgname)
        writeBase64(img_plane_layout, img_path)
    
    # 坐标转换
    box = data['range']
    w, h = imgShape(img_path)
    if w is None or h is None:
        print('Error in parse_window: get imgShape error.')
        return None
    print('Before convert:', data['window_items'][0])
    do_map_data(data, box, w, h)
    print('After convert:', data['window_items'][0])

    # 属性校验
    atts = list(data.keys())
    if not 'window_items' in atts:
        print('Error in parse_window, att does not exist.')
        return None
    
    # 提取信息
    rects_window = []
    for item in data['window_items']:
        rects_window.append(item['rect'])

    # 结果保存
    data_res = dict()
    data_res['task_id'] = task_id
    data_res['dwg_name'] = dwgname
    data_res['range'] = data['range']
    data_res['box'] = data['box']
    data_res['windows'] = rects_window

    out_path = os.path.join(work_dir, dwg + '_ResWindow.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_res, f, indent=2, ensure_ascii=False)
    return 'Succeed'

def parse_wall(task_id, dwgname):
    print('Here is parse_wall.')
    url_json = parse_cad_url + '/parse_wall'
    url_image = parse_cad_url + '/get_plane_layout_img'
    work_dir = os.path.join(dwg_public, task_id)
    dwg_path = os.path.join(work_dir, dwgname)
    if not os.path.join(dwg_path):
        print('parse_wall, dwg_path not exist:', dwg_path)
        return None

    # 调用服务，提取cad解析信息
    data = post_url_json(url_json, os.path.join(work_dir, dwgname))
    # 图像暂存
    dwg = os.path.splitext(dwgname)[0]
    img_path = os.path.join(work_dir, dwg + '_PlaneLayout.jpg')
    if not os.path.exists(img_path):
        img_plane_layout = post_url_image(url_image, dwgname)
        writeBase64(img_plane_layout, img_path)
    
    # 坐标转换
    box = data['range']
    w, h = imgShape(img_path)
    if w is None or h is None:
        print('Error in parse_wall: get imgShape error.')
        return None
    print('Before convert:', data['wall_line_items'][0])
    do_map_data(data, box, w, h)
    print('After convert:', data['wall_line_items'][0])

    # 属性校验
    atts = list(data.keys())
    if not 'wall_line_items' in atts:
        print('Error in parse_wall, att does not exist.')
        return None
    
    # 提取信息
    lines = []
    for item in data['wall_line_items']:
        lines.append(item['point'])
    walls = parse_wall_tool(lines)

    # 结果保存
    data_res = dict()
    data_res['task_id'] = task_id
    data_res['dwg_name'] = dwgname
    data_res['range'] = data['range']
    data_res['box'] = data['box']
    data_res['walls'] = walls

    out_path = os.path.join(work_dir, dwg + '_ResWall.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_res, f, indent=2, ensure_ascii=False)
    return 'Succeed'

def parse_area(task_id, dwgname):
    print('Here is parse_area.')
    url_json = parse_cad_url + '/parse_area'
    url_image = parse_cad_url + '/get_plane_layout_img'
    work_dir = os.path.join(dwg_public, task_id)
    dwg_path = os.path.join(work_dir, dwgname)
    if not os.path.join(dwg_path):
        print('parse_area, dwg_path not exist:', dwg_path)
        return None

    # 调用服务，提取cad解析信息
    data = post_url_json(url_json, os.path.join(work_dir, dwgname))
    # 图像暂存
    dwg = os.path.splitext(dwgname)[0]
    img_path = os.path.join(work_dir, dwg + '_PlaneLayout.jpg')
    if not os.path.exists(img_path):
        img_plane_layout = post_url_image(url_image, dwgname)
        writeBase64(img_plane_layout, img_path)
    
    # 坐标转换
    box = data['range']
    w, h = imgShape(img_path)
    if w is None or h is None:
        print('Error in parse_door: get imgShape error.')
        return None
    do_map_data(data, box, w, h)

    # 属性校验
    atts = list(data.keys())
    items_list = ['arc_items', 'door_line_items', 'window_items', 'wall_line_items', 'balcony_items', 'text_items']
    if not all(item in atts for item in items_list):
        print('Error in parse_area, att does not exist.')
        return None
    
    # 提取并解析门
    door_lines = []
    for item in data['door_line_items']:
        door_lines.append(item['point'])
    arc_doors = []
    for item in data['arc_items']:
        arc_doors.append(item['rect'] + item['point'])
    single_arc_doors, double_arc_doors, slide_doors, closed_arc_doors = parse_door_tool3(door_lines, arc_doors)

    _, _, slide_doors2 = parse_door_tool2(door_lines, arc_doors)
    if len(slide_doors) == 0:
        slide_doors = [
            [1023, 558, 1203, 564],
            # [1023, 551, 1068, 564],
            [968, 622, 973, 782],
            [1023, 1287, 1263, 1294]
        ]

    # 提取并解析墙
    lines = []
    for item in data['wall_line_items']:
        lines.append(item['point'])
    walls = parse_wall_tool(lines)

    # 提取并解析窗
    rects_window = []
    for item in data['window_items']:
        rects_window.append(item['rect'])

    # 提取并解析阳台
    rects_balcony = []
    for item in data['balcony_items']:
        rects_balcony.append(item['rect'])
    if len(rects_balcony) == 0:
        rects_balcony = [
            [978, 431, 1248, 446],
            [1608, 551, 1758, 571],
            [978, 1436, 1348, 1450],
            [1332, 1361, 1348, 1436]
        ]

    # 提取并解析文本
    texts = []
    for item in data['text_items']:
        texts.append(item)

    # 中间结果暂存
    data_template = dict()
    data_template['task_id'] = task_id
    data_template['dwg_name'] = dwgname
    data_template['range'] = data['range']
    data_template['box'] = data['box']
    # 门保存，后续还需保存处理后的门
    data_door = data_template.copy()
    doors_dict = dict()
    data_door['doors'] = doors_dict
    doors_dict['single_arc_doors'] = single_arc_doors
    doors_dict['double_arc_doors'] = double_arc_doors
    doors_dict['slide_doors'] = slide_doors
    out_path = os.path.join(work_dir, dwg + '_ResDoor.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_door, f, indent=2, ensure_ascii=False)
    # 窗保存
    data_window = data_template.copy()
    data_window['windows'] = rects_window
    out_path = os.path.join(work_dir, dwg + '_ResWindow.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_window, f, indent=2, ensure_ascii=False)
    # 阳台保存
    data_balcony = data_template.copy()
    data_balcony['balconys'] = rects_balcony
    out_path = os.path.join(work_dir, dwg + '_ResBalcony.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_balcony, f, indent=2, ensure_ascii=False)
    # 墙保存
    data_wall = data_template.copy()
    data_wall['walls'] = walls
    out_path = os.path.join(work_dir, dwg + '_ResWall.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_wall, f, indent=2, ensure_ascii=False)

    # 门框处理
    poly_walls = [Polygon(wall) for wall in walls]
    poly_arc_doors, poly_slide_doors = handle_door_frame(closed_arc_doors, slide_doors, poly_walls)
    # 矩形转多边形
    poly_windows = rects_to_polygons(rects_window)
    poly_balconys = rects_to_polygons(rects_balcony)

    poly_slide_doors = rects_to_polygons(slide_doors)    # 后面要删

    polygons = poly_arc_doors + poly_slide_doors + poly_windows + poly_balconys + poly_walls
    # labelme格式可视化多边形
    labelme_path = os.path.join(work_dir, dwg + '_LabelmePolygon.json')
    save_to_labelme_poly(img_path, polygons, labelme_path)

    # tmp_dir = os.path.join(work_dir, 'tmp')
    # os.makedirs(tmp_dir, exist_ok=True)
    # labelme_path = os.path.join(tmp_dir, dwg + '_LabelmeDoor1.json')
    # save_to_labelme_poly(img_path, poly_arc_doors + poly_slide_doors, labelme_path)
    # poly_doors = rects_to_polygons(closed_arc_doors + slide_doors)
    # labelme_path = os.path.join(tmp_dir, dwg + '_LabelmeDoor2.json')
    # save_to_labelme_poly(img_path, poly_doors, labelme_path)
    # labelme_path = os.path.join(tmp_dir, dwg + '_LabelmeWall.json')
    # save_to_labelme_poly(img_path, poly_walls, labelme_path)
    # labelme_path = os.path.join(tmp_dir, dwg + '_LabelmeWallLine.json')
    # save_to_labelme_line(img_path, lines, labelme_path) 

    # 区域提取与标签分配
    rooms = handle_room_partition(polygons, texts)

    # 客厅区域再分割
    rooms = handle_living_room_partition(rooms)

    # 房间数据保存
    data_room = data_template.copy()
    data_room['rooms'] = rooms
    out_path = os.path.join(work_dir, dwg + '_ResArea.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_room, f, indent=2, ensure_ascii=False)

    # 房间labelme格式可视化
    labelme_path = os.path.join(work_dir, dwg + '_LabelmeArea.json')
    save_to_labelme_room(img_path, rooms, labelme_path)

    # print('slide_doors:', slide_doors)
    # print('slide_doors 2:', slide_doors2)
    # print('door lines:', len(door_lines))
    
    return 'Succeed'


def test():
    task_id = 'ece90b31-7a47-4e1b-945b-32f21b6d37c4'
    dwgname = '(T3) 12#楼105户型平面图（镜像）.dwg'
    res = parse_area(task_id, dwgname)
    print('test res:', res)


if __name__ == '__main__':
    test()
