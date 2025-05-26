# -- 根据二值图筛选墙线
import cv2
import numpy as np
import os

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgReadGray(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)

def line_points(x1, y1, x2, y2):
    """生成线段上的所有像素点"""
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

def filter_line_in_white(mask_path, line_segments):
    ratio_thred = 0.5
    # binary_image = imgReadGray(mask_path)
    binary_image = dilate_mask(mask_path)
    """计算每条线段位于二值图中白色区域的部分占比"""
    filtered_lines = []
    for line in line_segments:
        x1, y1, x2, y2 = line
        points = line_points(x1, y1, x2, y2)
        total_points = len(points)
        white_points = 0

        for point in points:
            x, y = point
            if 0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]:
                if binary_image[y, x] == 255:
                    white_points += 1

        ratio = white_points / total_points if total_points > 0 else 0
        if ratio > ratio_thred:
            filtered_lines.append(line)
    return filtered_lines

def dilate_mask(mask_path, kernel_size=5):
    if not os.path.exists(mask_path):
        print('mask path not exist, ', mask_path)
        return None
    binary_image = imgReadGray(mask_path)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    opening_image = cv2.morphologyEx(dilated_image, cv2.MORPH_OPEN, kernel)
    imgWrite('../data/tmp_res/res-dilated.png', dilated_image)
    imgWrite('../data/tmp_res/res-opening2.png', opening_image)

    return opening_image

def calculate_line_in_white_ratio(mask_path, line_segments):
    binary_image = imgReadGray(mask_path)
    """计算每条线段位于二值图中白色区域的部分占比"""
    results = []
    for line in line_segments:
        x1, y1, x2, y2 = line
        points = line_points(x1, y1, x2, y2)
        total_points = len(points)
        white_points = 0

        for point in points:
            x, y = point
            if 0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]:
                if binary_image[y, x] == 255:
                    white_points += 1

        ratio = white_points / total_points if total_points > 0 else 0
        results.append(ratio)

    return results

def test():
    # 读取二值图像
    binary_image = cv2.imread('path_to_your_binary_image.png', cv2.IMREAD_GRAYSCALE)

    # 检查图片是否成功读取
    if binary_image is None:
        print("Error: Could not read the image.")
    else:
        # 定义线段列表，每个线段以 [x1, y1, x2, y2] 形式存储
        line_segments = [
            [10, 10, 100, 100],
            [50, 50, 150, 150],
            [200, 200, 300, 300]
        ]

        # 计算每条线段位于二值图中白色区域的部分占比
        ratios = calculate_line_in_white_ratio(binary_image, line_segments)

        # 打印结果
        for i, ratio in enumerate(ratios):
            print(f"Line segment {i+1} ratio in white area: {ratio:.2f}")

        # 显示原始二值图像
        cv2.imshow('Binary Image', binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
