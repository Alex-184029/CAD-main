# -- 构建墙体区域多边形整理流程
import json
import os
from build_graph1 import construct_graph, find_cycles, find_cycles2, visualize_graph_and_cycles, simplify_cycles, filter_nested_cycles, filter_contained_cycles, add_leaf_edge
from filter_line2 import split_segments_until_done
from filter_line3 import calculate_line_in_white_ratio, filter_line_in_white

def load_line_json(line_path):
    with open(line_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lines = []
    for shape in data['shapes']:
        line = shape['points']
        lines.append([line[0][0], line[0][1], line[1][0], line[1][1]])
    return lines

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
        return seg[1] == seg[3]

    def is_vertical(seg):
        return seg[0] == seg[2]

    # 筛选
    horizontal_segments = [seg for seg in segments if is_horizontal(seg)]
    vertical_segments = [seg for seg in segments if is_vertical(seg)]
    other_segments = [seg for seg in segments if not is_horizontal(seg) and not is_vertical(seg)]

    # 排序
    horizontal_segments.sort(key=lambda seg: (seg[1], min(seg[0], seg[2])))
    vertical_segments.sort(key=lambda seg: (seg[0], min(seg[1], seg[3])))
    print('nums1:', len(horizontal_segments), len(vertical_segments), len(other_segments))

    # 简化
    horizontal_segments = simplify_segments(horizontal_segments)
    vertical_segments = simplify_segments(vertical_segments)
    print('nums2:', len(horizontal_segments), len(vertical_segments), len(other_segments))

    # Combine all segments
    organized_segments = horizontal_segments + vertical_segments + other_segments

    return organized_segments

def lines_to_json(lines, json_origin, save_path):
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = []
    for line in lines:
        shape = {
            'label': 'WallLine1', 
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

def cycles_to_json(contours, json_origin, save_path):
    if contours is None or len(contours) == 0:
        print('data error for:', save_path)
        return
    with open(json_origin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = []
    for contour in contours:
        shape = {
            'label': 'WallArea1', 
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

    
def test1():
    # line_path = '../data/labels_line_ori/01 1-6号住宅楼标准层A户型平面图-5.json'
    # line_path = '../data/labels_line_ori/(T3) 12#楼105户型平面图（镜像）-2.json'
    # line_path = '../data/labels_line_ori/(T3) 12#楼105户型平面图（镜像）-3.json'
    line_path = '../data/labels_line_ori/01 1-6号住宅楼标准层A户型平面图-2.json'
    lines = load_line_json(line_path)
    print('line num1:', len(lines))
    lines_simplify = organize_segments(lines)
    print('line num2:', len(lines_simplify))
    lines_split = split_segments_until_done(lines_simplify)
    print('line num3:', len(lines_split))

    # select_line(lines_split)
    # lines_to_json(lines_split, line_path, '../data/labels_line_ori/tmp21.json')

    graph = construct_graph(lines_split)
    graph = add_leaf_edge(graph)

    cycles = find_cycles(graph)
    print('Cycles:', len(cycles), cycles[0])
    cycles = simplify_cycles(cycles)
    print('Cycles1:', len(cycles), cycles[0])
    cycles = filter_nested_cycles(cycles)
    print('Cycles2:', len(cycles), cycles[0])
    cycles = filter_contained_cycles(cycles)
    print('Cycles3:', len(cycles), cycles[0])

    # cycles = [order_cycle(graph, cycle) for cycle in cycles]

    # visualize_graph_and_cycles(graph, cycles)
    cycles_to_json(cycles, line_path, '../data/tmp_res2/tmp2.json')

def test2():
    # line_path = '../data/labels_line_ori/01 1-6号住宅楼标准层A户型平面图-5.json'
    line_path = '../data/labels_line_ori/(T3) 12#楼105户型平面图（镜像）-2.json'
    mask_dir = '../data/masks'
    img_name = os.path.splitext(os.path.basename(line_path))[0] + '.png'
    mask_path = os.path.join(mask_dir, img_name)
    lines = load_line_json(line_path)
    print('line num1:', len(lines))
    lines_simplify = organize_segments(lines)
    print('line num2:', len(lines_simplify))
    lines_split = split_segments_until_done(lines_simplify)
    print('line num3:', len(lines_split))

    lines_mask = filter_line_in_white(mask_path, lines_split)     # 通过掩膜图筛选线条
    print('line num4:', len(lines_mask))
    # lines_to_json(lines_mask, line_path, '../data/labels_line_ori/tmp22.json')
    lines_to_json(lines_simplify, line_path, '../data/tmp_res/tmp21-simplify.json')
    lines_to_json(lines_split, line_path, '../data/tmp_res/tmp22-split.json')
    lines_to_json(lines_mask, line_path, '../data/tmp_res/tmp23-mask.json')

    graph = construct_graph(lines_mask)
    graph = add_leaf_edge(graph)

    cycles = find_cycles(graph)
    print('Cycles:', len(cycles), cycles[0])
    cycles = simplify_cycles(cycles)
    print('Cycles1:', len(cycles), cycles[0])
    cycles_to_json(cycles, line_path, '../data/tmp_res/tmp24-simplify-cycles.json')
    cycles = filter_nested_cycles(cycles)
    print('Cycles2:', len(cycles), cycles[0])
    cycles = filter_contained_cycles(cycles)
    print('Cycles3:', len(cycles), cycles[0])

    # cycles = [order_cycle(graph, cycle) for cycle in cycles]

    # visualize_graph_and_cycles(graph, cycles)
    cycles_to_json(cycles, line_path, '../data/tmp_res/tmp25-cycles.json')

    print('----- finish -----')
    

if __name__ == '__main__':
    test1()
