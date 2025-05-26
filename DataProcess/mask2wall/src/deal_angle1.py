# 掩模图矢量化尝试：掩模图骨架化与查找角点
import cv2
import numpy as np
from deal_mask1 import imgReadGray, imgWrite, imgRead
from skimage.morphology import skeletonize

def line_intersection(line1, line2):
    """计算两条直线的交点"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    A1, B1, C1 = y2 - y1, x1 - x2, x2 * y1 - x1 * y2
    A2, B2, C2 = y4 - y3, x3 - x4, x4 * y3 - x3 * y4
    det = A1 * B2 - A2 * B1
    if det == 0:  # 平行或重叠
        return None
    x = (B1 * C2 - B2 * C1) / det
    y = (C1 * A2 - C2 * A1) / det
    return int(x), int(y)

def find_angle1():
    # 读取二值图像
    maskpath = './data/masks/(T3) 12#楼105户型平面图（镜像）-2.png'
    image = imgReadGray(maskpath)
    edges = cv2.Canny(image, 50, 150)
    print('step1')

    # 霍夫直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    lines = [line[0] for line in lines]  # 提取直线坐标
    print('step2')

    # 找到直线的交点
    points = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            intersection = line_intersection(line1, line2)
            if intersection:
                points.append(intersection)

    # 去除重复点
    points = np.array(points)
    # unique_points = cv2.groupRectangles([list(p) for p in points], groupThreshold=1, eps=5)[0]
    unique_points = points
    print("直线数量:", len(lines))
    print("角点数量:", len(unique_points))
    print('step3')

    # 可视化
    # 绘制直线和角点
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for x, y in unique_points:
        cv2.circle(image_rgb, (x, y), 5, (0, 0, 255), -1)

    print('step4')
    # 保存或显示结果
    cv2.imshow('Hough Transform Corners', image_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 检测骨架上的角点
def find_corners(skeleton):
    corners = []
    h, w = skeleton.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 1:
                neighbors = skeleton[y - 1:y + 2, x - 1:x + 2].sum() - 1
                if neighbors == 1:  # 端点
                    corners.append((x, y))
                elif neighbors > 2:  # 分叉点
                    corners.append((x, y))
    return corners

def find_angle2():
    # 骨架化
    maskpath = './data/masks/(T3) 12#楼105户型平面图（镜像）-2.png'
    binary = imgReadGray(maskpath)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binary = (gray > 0).astype(np.uint8)
    skeleton = skeletonize(binary)

    corners = find_corners(skeleton)

    # 绘制骨架和角点
    skeleton_image = (skeleton * 255).astype(np.uint8)
    skeleton_image = cv2.cvtColor(skeleton_image, cv2.COLOR_GRAY2BGR)
    for x, y in corners:
        cv2.circle(skeleton_image, (x, y), 5, (0, 0, 255), -1)

    # 保存或显示结果
    imgWrite('./data/tmp_res/res1.png', skeleton_image)
    # cv2.imshow('Skeleton Corners', skeleton_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

def find_angle3():
    maskpath = './data/masks/(T3) 12#楼105户型平面图（镜像）-2.png'
    gray = imgReadGray(maskpath)
    edges = cv2.Canny(gray, 50, 150)

    # Harris 角点检测
    dst = cv2.cornerHarris(edges, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)  # 增强角点
    threshold = 0.01 * dst.max()
    corners = np.argwhere(dst > threshold)  # 阈值化

    # 绘制角点
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for y, x in corners:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    # 保存结果
    imgWrite('./data/tmp_res/res2.png', image)

def find_contours():
    maskpath = './data/masks/(T3) 12#楼105户型平面图（镜像）-2.png'
    binary_image = imgReadGray(maskpath)

    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 创建一个新的与原始图像尺寸相同的二值图像，初始值为0（黑色）
    contour_image = np.zeros_like(binary_image)
    # 在新图像上绘制轮廓，轮廓颜色设置为255（白色）
    # cv2.drawContours(contour_image, contours, -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(contour_image, contours, -1, 255, thickness=2)

    imgWrite('./data/tmp_res/res3.png', contour_image)

    # 尝试骨架化
    # skeleton = skeletonize(contour_image)

    # corners = find_corners(skeleton)

    # # 绘制骨架和角点
    # skeleton_image = (skeleton * 255).astype(np.uint8)
    # skeleton_image = cv2.cvtColor(skeleton_image, cv2.COLOR_GRAY2BGR)
    # for x, y in corners:
    #     cv2.circle(skeleton_image, (x, y), 5, (0, 0, 255), -1)

    # imgWrite('./data/tmp_res/res3.png', skeleton_image)

def find_angle4():
    # 读取二值图像
    # image = cv2.imread('binary_image.png')
    maskpath = './data/masks/(T3) 12#楼105户型平面图（镜像）-2.png'
    gray = imgReadGray(maskpath)

    # Step 1: 使用 findContours 检测轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Harris 角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)  # 增强角点响应
    harris_threshold = 0.01 * dst.max()
    harris_corners = np.argwhere(dst > harris_threshold)  # 角点坐标集合 [(y, x)]

    # 转换为 (x, y) 格式
    harris_corners = [(x, y) for y, x in harris_corners]

    # Step 3: 将 Harris 角点与轮廓点集匹配
    polygons = []  # 存储每个多边形的角点
    for contour in contours:
        # 将轮廓点集转换为 Python 列表
        contour_points = contour.reshape(-1, 2)

        # 筛选 Harris 角点中属于当前轮廓的点
        matched_corners = []
        for corner in harris_corners:
            distances = np.linalg.norm(contour_points - corner, axis=1)
            if np.min(distances) < 5:  # 距离阈值，控制角点与轮廓的匹配精度
                matched_corners.append(corner)

        # 去重并排序
        matched_corners = np.unique(matched_corners, axis=0)
        polygons.append(matched_corners)

    # Step 4: 在图像上绘制多边形角点
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    output_image = image.copy()
    for polygon in polygons:
        cv2.polylines(output_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)  # 绘制轮廓
        for x, y in polygon:
            cv2.circle(output_image, (x, y), 2, (0, 0, 255), -1)  # 绘制角点

    # 显示结果
    # cv2.imshow('Combined Method - Harris + Contours', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    imgWrite('./data/tmp_res/res4.png', output_image)

    # 输出每个多边形的角点
    # for idx, polygon in enumerate(polygons):
    #     print(f"Polygon {idx + 1} corners: {polygon}")

def find_angle5():
    # 读取二值图像
    maskpath = './data/masks/(T3) 12#楼105户型平面图（镜像）-2.png'
    imgpath = './data/images/(T3) 12#楼105户型平面图（镜像）-2.png'
    image_ori = imgRead(imgpath)
    gray = imgReadGray(maskpath)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Step 1: 使用 findContours 检测轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Harris 角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)  # 增强角点响应
    harris_threshold = 0.01 * dst.max()
    harris_corners = np.argwhere(dst > harris_threshold)  # 角点坐标集合 [(y, x)]

    # 转换为 (x, y) 格式
    harris_corners = [(x, y) for y, x in harris_corners]

    # Step 3: 匹配 Harris 角点与轮廓
    polygons = []  # 存储每个多边形的角点
    for contour in contours:
        # 将轮廓点集转换为 Python 列表
        contour_points = contour.reshape(-1, 2)

        # 匹配 Harris 角点与当前轮廓
        matched_corners = []
        for corner in harris_corners:
            # 找到轮廓上与角点最近的点及其索引
            distances = np.linalg.norm(contour_points - corner, axis=1)
            min_index = np.argmin(distances)
            if distances[min_index] < 5:  # 距离阈值，控制角点与轮廓的匹配精度
                matched_corners.append((corner, min_index))

        # 按轮廓点索引排序（逆时针顺序）
        matched_corners = sorted(matched_corners, key=lambda x: x[1])

        # 提取排序后的角点
        sorted_corners = [corner for corner, _ in matched_corners]
        polygons.append(np.array(sorted_corners, dtype=np.int32))

    # Step 4: 在图像上绘制多边形角点
    output_image = image_ori.copy()
    for polygon in polygons:
        cv2.polylines(output_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)  # 绘制轮廓
        for x, y in polygon:
            cv2.circle(output_image, (x, y), 3, (0, 0, 255), -1)  # 绘制角点

    # 显示结果
    # cv2.imshow('Sorted Corners by Contour Order', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    imgWrite('./data/tmp_res/res5.png', output_image)

    # 输出每个多边形的角点
    # for idx, polygon in enumerate(polygons):
    #     print(f"Polygon {idx + 1} corners (sorted by contour order): {polygon}")


if __name__ == '__main__':
    find_angle5()
    # find_contours()
