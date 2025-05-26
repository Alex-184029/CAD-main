# -- 构建墙体区域多边形整理流程
import json
import os
import numpy as np
from build_graph1 import construct_graph, find_cycles, find_cycles2, visualize_graph_and_cycles, simplify_cycles, filter_nested_cycles, filter_contained_cycles, add_leaf_edge, find_rect_cycles
from filter_line2 import split_segments_until_done
from filter_line3 import calculate_line_in_white_ratio, filter_line_in_white
import shutil
import multiprocessing

def load_line_json(line_path):
    with open(line_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lines = []
    for shape in data['shapes']:
        line = shape['points']
        if (len(line) == 2):
            lines.append([line[0][0], line[0][1], line[1][0], line[1][1]])
        else:
            print('error line:', line)
    return lines

def load_rect_json(rect_path):
    with open(rect_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rects = []
    for shape in data['shapes']:
        points = shape['points']
        num = len(points)
        if (num < 4):
            print('Error rect:', points)
            continue
        x1, y1, x2, y2 = points[0][0], points[0][1], points[0][0], points[0][1]
        for i in range(1, num):
            x1, y1, x2, y2 = min(x1, points[i][0]), min(y1, points[i][1]), max(x2, points[i][0]), max(y2, points[i][1])
        rects.append([x1, y1, x2, y2])
    return rects

def load_arcdoor_json(arcdoor_path):
    with open(arcdoor_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    arcs = []
    for shape in data['shapes']:
        points = shape['points']
        num = len(points)
        if (num < 4):
            print('Error rect:', points)
            continue
        x1, y1, x2, y2 = points[0][0], points[0][1], points[0][0], points[0][1]
        for i in range(1, num):
            x1, y1, x2, y2 = min(x1, points[i][0]), min(y1, points[i][1]), max(x2, points[i][0]), max(y2, points[i][1])
        xx1, yy1, xx2, yy2 = map(int, shape['description'].split())
        arcs.append([x1, y1, x2, y2, xx1, yy1, xx2, yy2])
    return arcs

def simplify_segments(segments):     # 线段简化
    """
    Simplify a list of line segments based on the given rules:
    1. Remove segments that are contained within other segments.
    2. Merge collinear overlapping segments into a single segment.

    Args:
        segments (list of tuples): List of segments represented as (x1, y1, x2, y2).

    Returns:
        list of tuples: Simplified list of segments.
    """
    def is_contained(seg1, seg2):    # 只考虑了水平垂直情况
        """Check if seg1 is contained within seg2."""
        x1, y1, x2, y2 = seg1
        a1, b1, a2, b2 = seg2

        if y1 == y2 == b1 == b2:  # Horizontal segments
            return min(a1, a2) <= min(x1, x2) <= max(x1, x2) <= max(a1, a2)
        elif x1 == x2 == a1 == a2:  # Vertical segments
            return min(b1, b2) <= min(y1, y2) <= max(y1, y2) <= max(b1, b2)
        return False

    def can_merge(seg1, seg2):
        """Check if two segments can be merged."""
        x1, y1, x2, y2 = seg1
        a1, b1, a2, b2 = seg2

        if y1 == y2 == b1 == b2:  # Horizontal segments
            c1 = min(x1, x2)
            c2 = max(x1, x2)
            c3 = min(a1, a2)
            c4 = max(a1, a2)
            if (max(c1, c3) <= min(c2, c4)):
                return True
        elif x1 == x2 == a1 == a2:  # Vertical segments
            c1 = min(y1, y2)
            c2 = max(y1, y2)
            c3 = min(b1, b2)
            c4 = max(b1, b2)
            if (max(c1, c3) <= min(c2, c4)):
                return True
        return False

    def merge_segments(seg1, seg2):
        """Merge two overlapping collinear segments."""
        x1, y1, x2, y2 = seg1
        a1, b1, a2, b2 = seg2

        if y1 == y2 == b1 == b2:  # Horizontal segments
            return [min(x1, x2, a1, a2), y1, max(x1, x2, a1, a2), y2]
        elif x1 == x2 == a1 == a2:  # Vertical segments
            return [x1, min(y1, y2, b1, b2), x2, max(y1, y2, b1, b2)]

    # Step 1: Remove contained segments    和步骤二重叠
    # remaining_segments = []
    # for seg1 in segments:
    #     contained = False
    #     for seg2 in segments:
    #         if seg1 != seg2 and is_contained(seg1, seg2):
    #             contained = True
    #             break
    #     if not contained:
    #         remaining_segments.append(seg1)
    remaining_segments = segments

    # Step 2: Merge overlapping collinear segments
    merged_segments = []
    used = [False] * len(remaining_segments)

    num = len(remaining_segments)
    for i in range(num):
        if used[i]:
            continue
        seg1 = remaining_segments[i]
        # 检查是否有收尾相同点
        x1, y1, x2, y2 = seg1
        if x1 == x2 and y1 == y2:
            used[i] = True
            continue
        # 重叠合并
        for j in range(i + 1, num):
            if not used[j] and can_merge(seg1, remaining_segments[j]):
                seg1 = merge_segments(seg1, remaining_segments[j])
                used[j] = True
        merged_segments.append(seg1)

    return merged_segments

def organize_segments(segments):
    """
    Organize line segments into horizontal, vertical, and other segments with specific sorting rules.

    Args:
        segments (list of tuples): List of segments represented as (x1, y1, x2, y2).

    Returns:
        list of tuples: Organized list of segments.
    """
    # Helper functions
    def is_horizontal(seg):
        return abs(seg[1] - seg[3]) < 0.5

    def is_vertical(seg):
        return abs(seg[0] - seg[2]) < 0.5

    # 筛选
    horizontal_segments = [seg for seg in segments if is_horizontal(seg) and not is_vertical(seg)]
    vertical_segments = [seg for seg in segments if is_vertical(seg) and not is_horizontal(seg)]
    # other_segments = [seg for seg in segments if not is_horizontal(seg) and not is_vertical(seg)]
    # print('segments num:', len(horizontal_segments), len(vertical_segments))

    # 排序
    horizontal_segments.sort(key=lambda seg: (seg[1], min(seg[0], seg[2])))
    vertical_segments.sort(key=lambda seg: (seg[0], min(seg[1], seg[3])))
    # print('nums1:', len(horizontal_segments), len(vertical_segments), len(other_segments))

    # 简化，必须做简化，否则可能线数量庞杂无法找环
    horizontal_segments = simplify_segments(horizontal_segments)
    vertical_segments = simplify_segments(vertical_segments)
    # print('segments nums2:', len(horizontal_segments), len(vertical_segments))

    # Combine all segments
    # organized_segments = horizontal_segments + vertical_segments + other_segments
    organized_segments = horizontal_segments + vertical_segments      # 只考虑水平和竖直的情况

    return organized_segments


def merge_all_rectangles(rectangles):
    def is_overlap(rect1, rect2):
        # 检查两个矩形是否有重叠
        x1, y1, x2, y2, type1 = rect1
        x3, y3, x4, y4, type2 = rect2
        if type1 != type2:
            return False
        return not (x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1)

    def merge_rectangles(rect1, rect2):
        # 合并两个矩形为一个更大的矩形
        x1, y1, x2, y2, type1 = rect1
        x3, y3, x4, y4, _ = rect2
        new_x1 = min(x1, x3)
        new_y1 = min(y1, y3)
        new_x2 = max(x2, x4)
        new_y2 = max(y2, y4)
        return [new_x1, new_y1, new_x2, new_y2, type1]

    # 合并所有重叠的矩形
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(rectangles):
            j = i + 1
            while j < len(rectangles):
                if is_overlap(rectangles[i], rectangles[j]):
                    # 合并矩形
                    merged_rect = merge_rectangles(rectangles[i], rectangles[j])
                    rectangles[i] = merged_rect
                    del rectangles[j]
                    changed = True
                else:
                    j += 1
            i += 1
    return rectangles

def find_rectangles(segments, l_min=50, l_max=150, w_max=15):    # 长度阈值与距离阈值（后续根据实际尺寸计算调整）
    # Helper functions
    def is_horizontal2(seg):
        return seg[1] == seg[3]

    def is_vertical2(seg):
        return seg[0] == seg[2]

    def is_horizontal(seg):
        return abs(seg[3] - seg[1]) <= 1

    def is_vertical(seg):
        return abs(seg[2] - seg[0]) <= 1

    def distance(seg):
        return np.sqrt((seg[2] - seg[0]) ** 2 + (seg[3] - seg[1]) ** 2)

    def seg_rect(seg1, seg2):
        if (is_horizontal(seg1) and is_horizontal(seg2)):
            if abs(seg1[1] - seg2[1]) > w_max:
                return None
            x11, x12, x21, x22 = min(seg1[0], seg1[2]), max(seg1[0], seg1[2]), min(seg2[0], seg2[2]), max(seg2[0], seg2[2])
            if x11 != x21 or x12 != x22:
                return None
            return [x11, min(seg1[1], seg2[1]), x12, max(seg1[1], seg2[1]), 0]
        elif (is_vertical(seg1) and is_vertical(seg2)):
            if abs(seg1[0] - seg2[0]) > w_max:
                return None
            y11, y12, y21, y22 = min(seg1[1], seg1[3]), max(seg1[1], seg1[3]), min(seg2[1], seg2[3]), max(seg2[1], seg2[3])
            if y11 != y21 or y12 != y22:
                return None
            return [min(seg1[0], seg2[0]), y11, max(seg1[0], seg2[0]), y12, 1]
        return None
    
    def rect_contain(rect1, rect2):
        if rect1[0] <= rect2[0] and rect1[1] <= rect2[1] and rect1[2] >= rect2[2] and rect1[3] >= rect2[3]:
            return 1    # rect1更大
        elif rect1[0] >= rect2[0] and rect1[1] >= rect2[1] and rect1[2] <= rect2[2] and rect1[3] <= rect2[3]:
            return 2    # rect2更大
        else:
            return 0
    
    def rect_simplify(rects):
        num = len(rects)
        signs = [False] * num
        for i in range(num):
            if signs[i]:
                continue
            x = i
            for j in range(i + 1, num):
                if not signs[j]:
                    if rect_contain(rects[x], rects[j]) == 1:
                        signs[j] = True
                    elif rect_contain(rects[x], rects[j]) == 2:
                        signs[x] = True
                        x = j
        return [rects[i] for i in range(num) if not signs[i]]

    # 筛选
    horizontal_segments = [seg for seg in segments if is_horizontal(seg) and l_min <= distance(seg) <= l_max]
    vertical_segments = [seg for seg in segments if is_vertical(seg) and l_min <= distance(seg) <= l_max]
    # other_segments = [seg for seg in segments if not is_horizontal(seg) and not is_vertical(seg)]

    # 排序
    horizontal_segments.sort(key=lambda seg: (seg[1], min(seg[0], seg[2])))
    vertical_segments.sort(key=lambda seg: (seg[0], min(seg[1], seg[3])))
    # print('line num:', len(horizontal_segments), len(vertical_segments))
    # 水平段
    segs = horizontal_segments
    num = len(segs)
    rects_horizontal = []
    for i in range(num):
        for j in range(i + 1, num):
            res = seg_rect(segs[i], segs[j])
            if not res is None:
                rects_horizontal.append(res)
    # 竖直段
    segs = vertical_segments
    num = len(segs)
    rects_vertical = []
    for i in range(num):
        for j in range(i + 1, num):
            res = seg_rect(segs[i], segs[j])
            if not res is None:
                rects_vertical.append(res)
    # print('rect num 00:', len(rects_horizontal), len(rects_vertical))

    # # 临时保存
    # line_path = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_DoorLine.json'
    # tmp_out = '../data/output/tmp21.json'
    # slide_doors_to_json(rects_horizontal + rects_vertical, line_path, tmp_out, 'DoorTmp', False)

    # tmp_out2 = '../data/output/tmp22.json'
    # lines_to_json(vertical_segments, line_path, tmp_out2, 'VerticalDoorLine')

    # 被包含滤除
    rects_horizontal = rect_simplify(rects_horizontal)
    rects_vertical = rect_simplify(rects_vertical)
    # print('rect num simplify:', len(rects_horizontal), len(rects_vertical))
    # 重叠合并（包括上面的包含滤除方法）
    # print('rects_horizontal0:', rects_horizontal)
    rects_horizontal = merge_all_rectangles(rects_horizontal)
    rects_vertical = merge_all_rectangles(rects_vertical)
    # print('rects_horizontal1:', rects_horizontal)
    # print('rect num merge:', len(rects_horizontal), len(rects_vertical))

    return rects_horizontal + rects_vertical


def lines_to_json(lines, json_origin, save_path, label='WallLine1'):
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = []
    for line in lines:
        shape = {
            'label': label, 
            'points': [[line[0], line[1]], [line[2], line[3]]],
            "group_id": None,
            "description": "",
            "shape_type": "line",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
    data['shapes'] = shapes
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print('Write json to,', save_path)

def cycles_to_json(contours, json_origin, save_path, label='WallArea1', is_append=False):
    if contours is None or len(contours) == 0:
        print('data error for:', save_path)
        return
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = data['shapes'] if is_append else []
    for contour in contours:
        shape = {
            'label': label, 
            'points': contour,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)

    data['shapes'] = shapes
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print('Write json to,', save_path)

def slide_doors_to_json(slide_doors, json_origin, save_path, label='SlideDoorArea1', is_append=False):
    if slide_doors is None or len(slide_doors) == 0:
        print('data none for:', save_path)
        return
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = data['shapes'] if is_append else []
    for slide_door in slide_doors:
        x1, y1, x2, y2, _ = slide_door
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
        json.dump(data, f, indent=2)
    # print('Write json to,', save_path)

def select_line(lines):
    range = [1162, 1348, 1206, 1412]
    line_select = []
    for line in lines:
        x1, y1, x2, y2 = line
        if min(x1, x2) >= range[0] and max(x1, x2) <= range[1] and min(y1, y2) >= range[2] and max(y1, y2) <= range[3]:
            line_select.append(line)
    print('----- begin -----')
    for line in line_select:
        print(line)
    print('----- end -----')

def cycles_to_rects(cycles, l_min=50, l_max=150, w_max=15):    # 门扇长边：500到1500，短边最宽150mm
    rects = []
    for cycle in cycles:
        x1, y1 = cycle[0]
        x2, y2 = cycle[0]
        for point in cycle[1:]:
            x1 = min(x1, point[0])
            x2 = max(x2, point[0])
            y1 = min(y1, point[1])
            y2 = max(y2, point[1])
        dx, dy = x2 - x1, y2 - y1
        if dx > l_min and dx < l_max and dy < w_max:
            rects.append([x1, y1, x2, y2, 0])
        elif dy > l_min and dy < l_max and dx < w_max:
            rects.append([x1, y1, x2, y2, 1])
        # else:
        #     print('error rect:', x1, y1, x2, y2)

    return rects

def rects_to_slide_doors(rects):      # 合并推拉门
    def to_slide_door(rect1, rect2, thred=10):
        x11, y11, x12, y12, type1 = rect1
        x21, y21, x22, y22, type2 = rect2
        if type1 != type2:
            return None
        if max(x11, x21) - min(x12, x22) > thred:
            return None
        if max(y11, y21) - min(y12, y22) > thred:
            return None
        if (y11 == y21 and y12 == y22) or (x11 == x21 and x12 == x22):    # 平行段不是推拉门
            return None
        x1, y1, x2, y2 = min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22)
        return [x1, y1, x2, y2, type1]

    num = len(rects)
    signs = [False] * num
    slide_doors = []
    for i in range(num):
        if signs[i]:
            continue
        j = i + 1
        while (j < num):
            if signs[j]:
                j += 1
                continue
            res = to_slide_door(rects[i], rects[j])
            if not res is None:
                signs[i] = True
                signs[j] = True
                rects[i] = res
                j = i + 1
                continue
            j += 1
        if signs[i]:
            slide_doors.append(rects[i])
    return slide_doors

def find_slide_doors(line_path, out_path):
    if not os.path.exists(line_path) or not os.path.exists(out_path):
        print('Path not exist:', line_path)
        return
    json_name = os.path.basename(line_path)
    print('Ready to deal:', json_name)
    try:
        lines = load_line_json(line_path)
        lines_simplify = organize_segments(lines)
        lines_split = split_segments_until_done(lines_simplify)    # 是否需要相交切分

        graph = construct_graph(lines_split)
        cycles = find_cycles(graph)
        cycles = find_rect_cycles(cycles)
        cycles = filter_contained_cycles(cycles)
        rects = cycles_to_rects(cycles)

        slide_doors = rects_to_slide_doors(rects)
        num = len(slide_doors)
        print('slide doors num:', num)
        if num == 0:
            return
        slide_doors_to_json(slide_doors, line_path, os.path.join(out_path, json_name))
    except Exception as e:
        print('Error:', e)

def find_slide_doors_batch():
    line_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\labels_line_total'
    out_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\labels_slide_door5'
    if not os.path.exists(line_path):
        print('Path not exist:', line_path)
        return
    os.makedirs(out_path, exist_ok=True)

    lines = os.listdir(line_path)
    for i, line in enumerate(lines):
        if i % 100 == 0:
            print('%d / %d' % (i, len(lines)))
        find_slide_doors3(os.path.join(line_path, line), out_path)
    print('finish')

# 使用 multiprocessing 实现超时控制
def run_with_timeout(func, args, timeout):
    process = multiprocessing.Process(target=func, args=args)
    process.start()
    process.join(timeout=timeout)  # 等待 timeout 秒

    if process.is_alive():
        process.terminate()  # 终止子进程
        process.join()
        raise TimeoutError("Function timed out")

def find_slide_doors_batch2():
    line_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset2-pdf\labels_line'
    out_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset2-pdf\labels_slide_door'
    err_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset2-pdf\labels_err1'
    if not os.path.exists(line_path):
        print('Path not exist:', line_path)
        return
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(err_path, exist_ok=True)

    lines = os.listdir(line_path)
    for i, line in enumerate(lines):
        if i % 100 == 0:
            print('%d / %d' % (i, len(lines)))
        try:
            run_with_timeout(find_slide_doors, (os.path.join(line_path, line), out_path), 20)  # 设置 10 秒超时
        except TimeoutError as e:
            print('Timeout', e)
            shutil.move(os.path.join(line_path, line), err_path)
    print('finish')

def find_slide_doors3(line_path, out_path):
    if not os.path.exists(line_path) or not os.path.exists(out_path):
        print('Path not exist:', line_path)
        return
    json_name = os.path.basename(line_path)
    print('Ready to deal:', json_name)
    lines = load_line_json(line_path)
    rects = find_rectangles(lines)
    # print('rects num:', len(rects))

    slide_doors = rects_to_slide_doors(rects)
    num = len(slide_doors)
    print('slide doors num:', num)
    if num == 0:
        return
    slide_doors_to_json(slide_doors, line_path, os.path.join(out_path, json_name))

def combine_with_arc_doors(rects, arc_doors):
    def do_combine(rect1, rect_arc2, thred=5):     # origin is 10
        x11, y11, x12, y12, _ = rect1
        x21, y21, x22, y22 = rect_arc2
        if max(x11, x21) - min(x12, x22) > thred:
            return None
        if max(y11, y21) - min(y12, y22) > thred:
            return None
        x1, y1, x2, y2 = min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22)
        return [x1, y1, x2, y2, 0]

    num = len(rects)
    signs = [False] * num
    
    rects_arc = []
    for i in range(num):
        for arc in arc_doors:
            res = do_combine(rects[i], arc)
            if not res is None:
                signs[i] = True
                rects_arc.append(res)
                break
    rects_arc = merge_all_rectangles(rects_arc)
    rects_slide = [rects[i] for i in range(num) if not signs[i]]

    return rects_slide, rects_arc

def combine_with_arc_doors2(rects, arc_doors):
    def point_to_rectangle_distance(x1, y1, x2, y2, xx1, yy1):
        # 如果点在矩形内部或边界上，距离为0
        if x1 <= xx1 <= x2 and y1 <= yy1 <= y2:
            return 0
        # 计算点到矩形边界的最近距离
        closest_x = max(x1, min(xx1, x2))
        closest_y = max(y1, min(yy1, y2))
        # 计算点到最近边界的距离
        distance_x = abs(xx1 - closest_x)
        distance_y = abs(yy1 - closest_y)
        # 返回最近距离
        return np.sqrt(distance_x ** 2 + distance_y ** 2)

    def point_to_line_distance(x1, y1, x2, y2, x0, y0):
        """计算点(x0, y0)到由(x1, y1)和(x2, y2)确定的直线的垂直距离"""
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return numerator / denominator
    
    def get_axis(x1, y1, x2, y2, xx1, yy1, xx2, yy2):     # 获取最远点（旋转轴心点）
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        ans = points[0]
        d0 = point_to_line_distance(xx1, yy1, xx2, yy2, ans[0], ans[1])
        for point in points[1:]:
            d = point_to_line_distance(xx1, yy1, xx2, yy2, point[0], point[1])
            if d > d0:
                ans = point
                d0 = d
        return ans

    def rotate_point(cx, cy, x, y, angle):
        """绕(cx, cy)点逆时针旋转angle弧度，返回旋转后的点坐标"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        nx = cos_a * (x - cx) - sin_a * (y - cy) + cx
        ny = sin_a * (x - cx) + cos_a * (y - cy) + cy
        return nx, ny

    def do_combine(rect1, rect_arc2, thred=5):     # 不太行，旋转角度不够
        x11, y11, x12, y12, _ = rect1
        x21, y21, x22, y22 = rect_arc2[:4]
        xx1, yy1, xx2, yy2 = rect_arc2[4:]
        if max(x11, x21) - min(x12, x22) > thred:
            return None
        if max(y11, y21) - min(y12, y22) > thred:
            return None
        d1 = point_to_rectangle_distance(x11, y11, x12, y12, xx1, yy1)
        d2 = point_to_rectangle_distance(x11, y11, x12, y12, xx2, yy2)
        source, target = (xx1, yy1), (xx2, yy2)
        if d1 > d2:
            source, target = (xx2, yy2), (xx1, yy1)
        pivot_x, pivot_y = get_axis(x11, y11, x12, y12, xx1, yy1, xx2, yy2)

        # 计算旋转角度
        angle1 = np.arctan2(source[1] - pivot_y, source[0] - pivot_x)
        angle2 = np.arctan2(target[1] - pivot_y, target[0] - pivot_x)
        rotation_angle = angle2 - angle1  # 计算旋转的角度
        
        # 计算矩形四个顶点
        corners = [(x11, y11), (x12, y11), (x12, y12), (x11, y12)]
        rotated_corners = [rotate_point(pivot_x, pivot_y, x, y, rotation_angle) for x, y in corners]
        
        return rotated_corners

    def do_combine2(rect1, rect_arc2, thred=5):     # origin is 10
        x11, y11, x12, y12, type1 = rect1           # 门框坐标，横纵类型（0横1纵）
        x21, y21, x22, y22 = rect_arc2[:4]          # 圆弧轮廓
        xx1, yy1, xx2, yy2 = rect_arc2[4:]          # 圆弧两端点坐标
        if max(x11, x21) - min(x12, x22) > thred:
            return None
        if max(y11, y21) - min(y12, y22) > thred:
            return None
        d1 = point_to_rectangle_distance(x11, y11, x12, y12, xx1, yy1)
        d2 = point_to_rectangle_distance(x11, y11, x12, y12, xx2, yy2)
        source, target = (xx1, yy1), (xx2, yy2)
        if d1 > d2:
            source, target = (xx2, yy2), (xx1, yy1)

        x1, y1, x2, y2 = min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22)
        if type1 == 0:
            width = abs(y12 - y11)
            if target[0] < source[0]:
                x2 = x1
                x1 -= width
            else:
                x1 = x2
                x2 += width
        elif type1 == 1:
            height = abs(x12 - x11)
            if target[1] < source[1]:
                y2 = y1
                y1 -= height
            else:
                y1 = y2
                y2 += height
        return [x1, y1, x2, y2, 0]

    num = len(rects)
    signs = [False] * num
    
    rects_arc = []
    for i in range(num):
        for arc in arc_doors:
            res = do_combine2(rects[i], arc)
            if not res is None:
                signs[i] = True
                rects_arc.append(res)
                break
    # rects_arc = merge_all_rectangles(rects_arc)
    rects_slide = [rects[i] for i in range(num) if not signs[i]]

    return rects_slide, rects_arc

def merge_jsons(json_paths, json_out):
    if json_paths is None or len(json_paths) == 0:
        print('Json paths none:', json_paths)
        return
    with open(json_paths[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = data['shapes']
    for json_path in json_paths[1:]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_tmp = json.load(f)
        shapes += data_tmp['shapes']

    data['shapes'] = shapes
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def test_merge_jsons():
    json_path1 = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_Door.json'
    json_path2 = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_Balcony2.json'
    json_path3 = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_WallArea.json'
    json_path4 = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_ParallelWindow.json'
    json_paths = [json_path1, json_path2, json_path3, json_path4]
    json_out = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_Structure1.json'
    merge_jsons(json_paths, json_out)

def test1():
    line_path = '../data/labels_line/01 1-6号住宅楼标准层A户型平面图-2.json'
    # line_path = '../data/labels_line/(T3) 12#楼105户型平面图（镜像）-3.json'
    # line_path = '../data/labels_line/01-100-b1户型平面图 2020.09.14-2.json'
    # line_path = '../data/labels_line/01-地上公区平面施工图-12.json'
    # line_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\labels_err\01.标准层3B户型精装修-平面图-1.json'
    lines = load_line_json(line_path)
    print('line num1:', len(lines))
    lines_simplify = organize_segments(lines)
    print('line num2:', len(lines_simplify))
    lines_split = split_segments_until_done(lines_simplify)    # 是否需要相交切分
    print('line num3:', len(lines_split))

    # select_line(lines_split)
    # lines_to_json(lines_split, line_path, '../data/labels_line_ori/tmp21.json')

    graph = construct_graph(lines_split)
    cycles = find_cycles(graph)
    print('Cycles:', len(cycles), cycles[0])
    # cycles = simplify_cycles(cycles)
    cycles = find_rect_cycles(cycles)
    print('Cycles1:', len(cycles))
    # cycles = filter_nested_cycles(cycles)
    # print('Cycles2:', len(cycles), cycles[0])
    cycles = filter_contained_cycles(cycles)
    print('Cycles3:', len(cycles))

    rects = cycles_to_rects(cycles)
    print('rects num:', len(rects))

    slide_doors = rects_to_slide_doors(rects)
    print('slide doors num:', len(slide_doors))
    if len(slide_doors) > 0:
        print('slide door0:', slide_doors[0])

    slide_doors_to_json(slide_doors, line_path, '../data/output/tmp13.json')
    # cycles_to_json(cycles, line_path, '../data/output/tmp11.json')

def test12():
    line_path = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_DoorLine.json'
    lines = load_line_json(line_path)
    rects = find_rectangles(lines)
    print('rects num:', len(rects))

    tmp_out = '../data/output/tmp20.json'
    slide_doors_to_json(rects, line_path, tmp_out, 'SlideDoorTmp', False)

    print('----- finish -----')

def test13():
    line_path = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_DoorLine.json'
    lines = load_line_json(line_path)
    rects = find_rectangles(lines)
    print('rects num:', len(rects))

    # 获取圆弧框
    arc_path = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_ArcDoor2.json'
    arc_doors = load_arcdoor_json(arc_path)
    print('arc_doors num:', len(arc_doors))
    if len(arc_doors) > 0:
        print(arc_doors[0])

    tmp_out = '../data/output/tmp20.json'
    slide_doors_to_json(rects, line_path, tmp_out, 'SlideDoorTmp', False)

    # 将rects与圆弧框进行对照
    rects_slide, rects_arc = combine_with_arc_doors2(rects, arc_doors)
    print('split rects num:', len(rects_slide), len(rects_arc))

    slide_doors = rects_to_slide_doors(rects_slide)
    print('slide doors num:', len(slide_doors))
    # if len(slide_doors) > 0:
    #     print('slide door0:', slide_doors[0])

    rect_out = '../data/labels_Balcony/01 1-6号住宅楼标准层A户型平面图-2_Door2.json'
    slide_doors_to_json(slide_doors, line_path, rect_out, 'SlideDoor', False)
    slide_doors_to_json(rects_arc, rect_out, rect_out, 'ArcDoor', True)
    print('----- finish -----')


if __name__ == '__main__':
    test13()
    # find_slide_doors_batch()
    # test_merge_jsons()
