import requests
import os
import json

import sys
sys.path.append('../')

from main_server4 import dwg_public
from tools1 import readBase64, writeBase64, imgShape, do_map_data
from tools_door import parse_door_tool
from tools_wall import parse_wall_tool

parse_cad_url = 'http://127.0.0.1:5005'        # 本地
# parse_cad_url = 'http://192.168.131.128:5005'  # 虚拟机

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
    single_arc_doors, double_arc_doors, slide_doors = parse_door_tool(lines, arc_doors)

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
    # to do ...
