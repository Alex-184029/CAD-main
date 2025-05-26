import json
from shapely.geometry import Polygon
import os
from deal_balcony2 import adjust_door_position

def round_polygon_coordinates(polygon):
    # 获取多边形的坐标点
    rounded_coords = [(round(x), round(y)) for x, y in polygon.exterior.coords]
    # 创建一个新的Polygon对象
    rounded_polygon = Polygon(rounded_coords)
    return rounded_polygon

def load_labelme_json(json_path):
    if not os.path.exists(json_path):
        print('Error: json_path not exists, json_path:', json_path)
        return None, None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    polygons = [Polygon(shape['points']) for shape in data['shapes'] if shape['shape_type'] == 'polygon']
    img_size = (data['imageWidth'], data['imageHeight'])
    return polygons, img_size

def save_to_labelme(json_origin, unlabeled_polygons, output_json, label='Region'):
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)

    shapes = []
    for poly in unlabeled_polygons:
        shape = {
            "label": label,
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

def save_polys_to_labelme(json_origin, polys: dict, output_json):
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)

    shapes = []
    poly_type = polys.keys()
    for item in poly_type:
        for poly in polys[item]:
            shape = {
                "label": item,
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

def remove_repeat():
    json_in = r'../../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_Balcony.json'
    json_out = r'../../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_Balcony2.json'
    polygons, _ = load_labelme_json(json_in)
    if polygons is None:
        return
    num = len(polygons)
    print('polygons num:', num)
    for i in range(num):
        for j in range(i + 1, num):
            if polygons[i].intersects(polygons[j]):
                tmp = polygons[j].difference(polygons[i])
                if tmp.is_empty or not tmp.is_valid:
                    print('Error: tmp is empty or not valid.')
                    continue
                polygons[j] = tmp
    save_to_labelme(json_in, polygons, json_out, label='Balcony2')

def read_structure(json_path):
    if not os.path.exists(json_path):
        print('data path not exists, json_path:', json_path)
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    poly_SlideDoor = [Polygon(shape['points']) for shape in data['shapes'] if shape['shape_type'] == 'polygon' and shape['label'] == 'SlideDoor']
    poly_ArcDoor = [Polygon(shape['points']) for shape in data['shapes'] if shape['shape_type'] == 'polygon' and shape['label'] == 'ArcDoor']
    poly_WallArea = [Polygon(shape['points']) for shape in data['shapes'] if shape['shape_type'] == 'polygon' and shape['label'] == 'WallArea1']
    poly_ParallelWindow = [Polygon(shape['points']) for shape in data['shapes'] if shape['shape_type'] == 'polygon' and shape['label'] == 'ParallelWindow']
    poly_Balcony = [Polygon(shape['points']) for shape in data['shapes'] if shape['shape_type'] == 'polygon' and shape['label'] == 'Balcony2']
    print('poly_SlideDoor:', len(poly_SlideDoor), 'poly_ArcDoor:', len(poly_ArcDoor), 'poly_WallArea:', len(poly_WallArea), 'poly_ParallelWindow:', len(poly_ParallelWindow), 'poly_Balcony:', len(poly_Balcony))

    poly_ArcDoor = [round_polygon_coordinates(poly) for poly in poly_ArcDoor]     # 四舍五入为整数

    poly_ArcDoorNew = []
    for poly in poly_ArcDoor:
        ans = adjust_door_position(poly, poly_WallArea)
        if ans is None:
            poly_ArcDoorNew.append(poly)
        else:
            poly_ArcDoorNew.append(ans)

    poly_SlideDoorNew = []
    for poly in poly_SlideDoor:
        ans = adjust_door_position(poly, poly_WallArea, method='slide')
        if ans is None:
            poly_SlideDoorNew.append(poly)
        else:
            poly_SlideDoorNew.append(ans)

    tmp_out = '../../data/labels_Balcony/tmp1.json'
    save_to_labelme(json_path, poly_ArcDoorNew + poly_WallArea + poly_SlideDoorNew, tmp_out)

    polys = {'SlideDoor': poly_SlideDoorNew, 'ArcDoor': poly_ArcDoorNew, 'Balcony': poly_Balcony, 'WallArea': poly_WallArea, 'ParallelWindow': poly_ParallelWindow}
    tmp_out = '../../data/labels_Balcony/tmp2.json'
    save_polys_to_labelme(json_path, polys, tmp_out)
    print('----- finish -----')


def test1():
    json_in = r'../../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_Structure2.json'
    read_structure(json_in)


if __name__ == '__main__':
    test1()
