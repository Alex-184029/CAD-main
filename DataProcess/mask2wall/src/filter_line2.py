# -- 线段相交切分
def calculate_intersection(seg1, seg2):
    """
    计算两条线段是否相交，并返回交点。
    如果相交，返回 (x, y) 坐标；否则返回 None。
    """
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2

    # 计算线段的方向向量
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3

    # 计算行列式
    det = dx1 * dy2 - dy1 * dx2
    if det == 0:
        return None  # 两线段平行或共线

    # 参数 t 和 u，用于确定交点是否在线段内
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
    u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det

    # 判断交点是否在两条线段内
    if 0 <= t <= 1 and 0 <= u <= 1:
        # 计算交点坐标
        ix = x1 + t * dx1
        iy = y1 + t * dy1
        return (ix, iy)

    return None

def split_segments_once(segments):
    """
    对线段集合进行一次切分操作。
    返回新的线段集合和一个布尔值，表示是否有线段被切分。
    """
    new_segments = []
    split_occurred = False
    n = len(segments)
    processed = [False] * n

    for i in range(n):
        if processed[i]:
            continue

        seg1 = segments[i]

        for j in range(i + 1, n):
            if processed[j]:
                continue

            seg2 = segments[j]
            intersection = calculate_intersection(seg1, seg2)

            if intersection:
                ix, iy = intersection
                ix, iy = round(ix), round(iy)

                if (ix, iy) != (seg2[0], seg2[1]) and (ix, iy) != (seg2[2], seg2[3]):   # seg2被切割
                    # 切分两条线段
                    new_segments.append([seg2[0], seg2[1], ix, iy])
                    new_segments.append([ix, iy, seg2[2], seg2[3]])
                    # 标记为已切分
                    processed[j] = True
                    split_occurred = True

                if (ix, iy) != (seg1[0], seg1[1]) and (ix, iy) != (seg1[2], seg1[3]):   # seg1被切割
                    # 切分两条线段
                    new_segments.append([seg1[0], seg1[1], ix, iy])
                    new_segments.append([ix, iy, seg1[2], seg1[3]])
                    # 标记为已切分
                    processed[i] = True
                    split_occurred = True
                    break              # seg1被切后就不能再切seg1了

        if not processed[i]:
            new_segments.append(seg1)

    return new_segments, split_occurred

def split_segments_until_done(segments):
    """
    持续切分线段，直到没有需要切分的线段为止。
    """
    while True:
        segments, split_occurred = split_segments_once(segments)
        if not split_occurred:
            break
    return segments

