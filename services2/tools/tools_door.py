import numpy as np

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
        return [x1, y1, x2, y2]

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

def merge_arc_rectangles(rectangles):    # 合并圆弧门，单开/双开
    def can_merge(rect1, rect2):
        """判断两个矩形是否可以合并"""
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2

        # 检查是否有完全相同的边（共享整条边）
        shared_edge = False

        # 检查rect1的左边是否与rect2的右边重合
        if x1_1 == x2_2 and (y1_1 <= y2_2 and y2_1 >= y1_2):
            shared_edge = True
        # 检查rect1的右边是否与rect2的左边重合
        elif x2_1 == x1_2 and (y1_1 <= y2_2 and y2_1 >= y1_2):
            shared_edge = True
        # 检查rect1的下边是否与rect2的上边重合
        elif y1_1 == y2_2 and (x1_1 <= x2_2 and x2_1 >= x1_2):
            shared_edge = True
        # 检查rect1的上边是否与rect2的下边重合
        elif y2_1 == y1_2 and (x1_1 <= x2_2 and x2_1 >= x1_2):
            shared_edge = True

        if not shared_edge:
            return False

        # 检查是否有重合区域（除共享边外）
        overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

        # 如果重叠区域面积为0或者仅为一条线，则认为没有重叠区域
        return overlap_x * overlap_y == 0

    def merge_rectangles(rect1, rect2):
        """合并两个矩形为它们的最小外接矩形"""
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2

        # 计算合并后的矩形边界
        new_x1 = min(x1_1, x1_2)
        new_y1 = min(y1_1, y1_2)
        new_x2 = max(x2_1, x2_2)
        new_y2 = max(y2_1, y2_2)

        return [new_x1, new_y1, new_x2, new_y2]

    n = len(rectangles)
    merged = [False] * n  # 标记每个矩形是否已被合并
    merged_rects = []     # 存储合并后的矩形

    # 遍历所有矩形对
    for i in range(n):
        if merged[i]:
            continue  # 已合并的矩形跳过
        for j in range(i + 1, n):
            if merged[j]:
                continue  # 已合并的矩形跳过
            if can_merge(rectangles[i], rectangles[j]):
                # 合并矩形
                merged_rect = merge_rectangles(rectangles[i], rectangles[j])
                merged_rects.append(merged_rect)
                merged[i] = True
                merged[j] = True
                break  # 每个矩形只考虑一次合并机会

    # 收集未合并的矩形
    unmerged_rects = [rectangles[i] for i in range(n) if not merged[i]]

    return merged_rects, unmerged_rects
   

def parse_door_tool(lines: list, arc_doors: list):
    rects = find_rectangles(lines)
    # 将rects与圆弧框进行对照
    rects_slide, rects_arc = combine_with_arc_doors2(rects, arc_doors)
    # print('split rects num:', len(rects_slide), len(rects_arc))
    slide_doors = rects_to_slide_doors(rects_slide) # 推拉门
    # print('slide doors num:', len(slide_doors))
    # 单开、双开门区分
    double_arc_doors, single_arc_doors = merge_arc_rectangles(rects_arc)

    return single_arc_doors, double_arc_doors, slide_doors


# 示例用法
if __name__ == "__main__":
    rectangles = [
        [0, 0, 2, 2],  # 可以和第二个矩形合并
        [2, 0, 4, 2],  # 可以和第一个矩形合并
        [5, 5, 7, 7],  # 无法合并
        [7, 5, 9, 7],  # 可以和第五个矩形合并
        [7, 7, 9, 9]   # 可以和第四个矩形合并
    ]

    merged, unmerged = merge_arc_rectangles(rectangles)
    print("可合并矩形:", merged)
    print("不可合并矩形:", unmerged) 
