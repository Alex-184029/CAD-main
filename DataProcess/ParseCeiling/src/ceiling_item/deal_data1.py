import json
import os

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

def diff_area_log():
    area_log = 'area1.log'
    with open(area_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    rooms, areas1, areas2 = [], [], []
    for line in lines:
        room, area1, area2 = line.strip().split(', ')
        rooms.append(room)
        areas1.append(float(area1))
        areas2.append(float(area2))
    num = len(rooms)
    diff_sum, diff_per_sum = 0, 0
    area_sum = 0
    for i in range(num):
        area_sum += areas1[i]
        diff = areas1[i] - areas2[i]
        diff_per = diff / areas1[i] * 100
        diff_sum += abs(diff)
        diff_per_sum += abs(diff_per)
        print('room %d %s: %f, %f, %.5f, %.4f' % (i + 1, rooms[i], areas1[i], areas2[i], diff, diff_per))
    print('ave diff: %.5f, ave diff_per: %f' % ((diff_sum / num), (diff_per_sum / num)))
    print('ave diff: %.5f, total diff: %.5f' % ((diff_sum / num), diff_sum))
    print('area1 sum: %.5f, total dif_per: %.5f' % (area_sum, diff_sum / area_sum))

def test():
    json_path = r'C:\Users\DELL\Desktop\tmp1.json'
    items = get_ceiling_item(json_path)
    if items is not None:
        print(len(items), items[0])


if __name__ == '__main__':
    # test()
    diff_area_log()