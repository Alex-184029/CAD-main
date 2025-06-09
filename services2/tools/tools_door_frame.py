from shapely.geometry import Polygon
from shapely.affinity import translate

def get_min_extension(door, walls, direction, max_extend):
    """
    计算门扇在指定方向上的最小延申距离，使其与墙体贴合
    :param door: Polygon, 代表门扇
    :param walls: list[Polygon], 代表所有墙体
    :param direction: str, 延申方向 ("left", "right", "up", "down")
    :param max_extend: float, 最大延申距离
    :return: float, 需要的最小延申距离（如果无法贴合返回 max_extend）
    """
    minx, miny, maxx, maxy = door.bounds
    step = 1  # 细化搜索精度

    for delta in range(0, int(max_extend / step) + 1):
        delta_value = delta * step
        if direction == "left":
            extended_door = Polygon([(minx - delta_value, miny), (maxx, miny),
                                     (maxx, maxy), (minx - delta_value, maxy)])
        elif direction == "right":
            extended_door = Polygon([(minx, miny), (maxx + delta_value, miny),
                                     (maxx + delta_value, maxy), (minx, maxy)])
        elif direction == "up":
            extended_door = Polygon([(minx, miny - delta_value), (maxx, miny - delta_value),
                                     (maxx, maxy), (minx, maxy)])
        elif direction == "down":
            extended_door = Polygon([(minx, miny), (maxx, miny),
                                     (maxx, maxy + delta_value), (minx, maxy + delta_value)])

        # 检查是否有墙体相邻
        if any(extended_door.touches(wall) or extended_door.intersects(wall) for wall in walls):
            return delta_value  # 找到最小可行的延申距离

    return max_extend  # 如果无法贴合，返回最大延申

def is_door_adjacent_to_wall(door, walls):
    """检查门扇是否至少有一端贴合墙体"""
    return any(door.touches(wall) or door.intersects(wall) for wall in walls)

def adjust_door_position(door, walls, extend_threshold=5, move_threshold=5, method='arc'):
    """尝试延申和小范围平移门扇，使其贴合墙体"""
    minx, miny, maxx, maxy = door.bounds
    width = maxx - minx
    height = maxy - miny

    # 推拉门特殊处理
    if method == 'slide':
        move_threshold = 0    # 推拉门不做上下平移
        extend_threshold = max(width, height) + 5

    # 判断门是水平还是竖直
    is_horizontal = width > height
    
    if is_horizontal:
        left_extend = get_min_extension(door, walls, "left", extend_threshold)
        right_extend = get_min_extension(door, walls, "right", extend_threshold)
        adjusted_door = Polygon([
            (minx - left_extend, miny), (maxx + right_extend, miny),
            (maxx + right_extend, maxy), (minx - left_extend, maxy)
        ])
    else:
        up_extend = get_min_extension(door, walls, "up", extend_threshold)
        down_extend = get_min_extension(door, walls, "down", extend_threshold)
        adjusted_door = Polygon([
            (minx, miny - up_extend), (maxx, miny - up_extend),
            (maxx, maxy + down_extend), (minx, maxy + down_extend)
        ])

    # 如果延申后门扇已经贴合墙体，直接返回
    if is_door_adjacent_to_wall(adjusted_door, walls):
        return adjusted_door

    if method == 'slide':
        print('无法调整使其贴合墙体。')
        return None

    # 如果门扇仍然无法贴合墙体，尝试小范围平移再执行延申
    for move_delta in range(1, int(move_threshold) + 1):
        for move_direction in [-1, 1]:  # 先负方向（下/左），再正方向（上/右）
            moved_door = translate(door, xoff=move_delta * move_direction if not is_horizontal else 0,
                                   yoff=move_delta * move_direction if is_horizontal else 0)

            if is_horizontal:
                left_extend = get_min_extension(moved_door, walls, "left", extend_threshold)
                right_extend = get_min_extension(moved_door, walls, "right", extend_threshold)
                new_door = Polygon([
                    (moved_door.bounds[0] - left_extend, moved_door.bounds[1]),
                    (moved_door.bounds[2] + right_extend, moved_door.bounds[1]),
                    (moved_door.bounds[2] + right_extend, moved_door.bounds[3]),
                    (moved_door.bounds[0] - left_extend, moved_door.bounds[3])
                ])
            else:
                up_extend = get_min_extension(moved_door, walls, "up", extend_threshold)
                down_extend = get_min_extension(moved_door, walls, "down", extend_threshold)
                new_door = Polygon([
                    (moved_door.bounds[0], moved_door.bounds[1] - up_extend),
                    (moved_door.bounds[2], moved_door.bounds[1] - up_extend),
                    (moved_door.bounds[2], moved_door.bounds[3] + down_extend),
                    (moved_door.bounds[0], moved_door.bounds[3] + down_extend)
                ])

            # 检测是否贴合墙体
            if any(new_door.touches(wall) or new_door.intersects(wall) for wall in walls):
                return new_door

    print("Error: 无法调整门扇，使其贴合墙体！请检查门的位置或调整最大延申/平移阈值。")
    return None

def round_polygon_coordinates(polygon):
    # 获取多边形的坐标点
    rounded_coords = [(round(x), round(y)) for x, y in polygon.exterior.coords]
    # 创建一个新的Polygon对象
    rounded_polygon = Polygon(rounded_coords)
    return rounded_polygon

def handle_door_frame(poly_ArcDoor, poly_SlideDoor, poly_WallArea):
    poly_ArcDoor = [round_polygon_coordinates(poly) for poly in poly_ArcDoor]     # 四舍五入为整数，以防万一，正常传入的都是整数

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

    return poly_ArcDoorNew, poly_SlideDoorNew
    

def main_old():
    # 示例数据
    walls = [
        Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])  # 假设一个墙体
    ]
    door = Polygon([
        (3, 10.5), (3, 11), (7, 11), (7, 10.5)  # 初始门扇（可能有间隙）
    ])

    adjusted_door = adjust_door_position(door, walls, extend_threshold=1.0, move_threshold=2.0)
    print(adjusted_door)


if __name__ == "__main__":
    main_old()

