import cv2
import numpy as np
import os
import random

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

def is_binary_image(image):
    """
    判断一个灰度图像是否是二值图。

    参数:
    image (numpy.ndarray): 输入的灰度图像。

    返回:
    bool: 如果图像是二值图，返回True；否则返回False。
    """
    # 检查图像是否为单通道
    if len(image.shape) != 2:
        return False
    
    # 获取图像中的唯一像素值
    unique_values = np.unique(image)
    
    # 检查唯一像素值是否只包含0和255
    if np.array_equal(unique_values, np.array([0, 255])):
        return True
    elif np.array_equal(unique_values, np.array([0])):
        return True
    elif np.array_equal(unique_values, np.array([255])):
        return True
    else:
        # 找出不是0和255的其他取值
        other_values = [value for value in unique_values if value not in [0, 255]]
        return False, other_values

def test_binary():
    imgpath = '../data/tmp_res/img_reverse.jpg'
    img = imgReadGray(imgpath)
    res = is_binary_image(img)
    print('res:', res)

def filter_contours(contours, min_area=10):
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours

def get_binary_contours():
    imgpath = '../res/img_opening2.jpg'
    img = imgReadGray(imgpath)
    if img is None:
        print('Read img failed.')
        return

    # 加入高斯模糊步骤
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow('Gaussian', img)
    # 查找轮廓
    # contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('contours num1:', len(contours), type(contours))
    contours = filter_contours(contours)
    print('contours num2:', len(contours), type(contours))
    # 创建一个空白图像用于绘制轮廓
    contour_image = np.zeros_like(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 绘制轮廓
    cv2.drawContours(img_rgb, contours, -1, (0, 0, 255), 2)
    cv2.drawContours(contour_image, contours, -1, (255), 1)

    # 显示结果图像
    # cv2.imshow('Contours', img_rgb)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    imgWrite('../res/img_contours.jpg', contour_image)

def find_rooms():
    maskpath = '../data/masks/(T3) 12#楼105户型平面图（镜像）-3_Structure2.jpg'
    img = imgReadGray(maskpath)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)     # 读取图片后必须二值化
    # img = cv2.bitwise_not(img)   # 黑白反转

    # contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('contours num1:', len(contours), type(contours))
    contours = filter_contours(contours)
    print('contours num2:', len(contours), type(contours))
    # 创建一个空白图像用于绘制轮廓
    # contour_image = np.zeros_like(img)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 绘制轮廓
    # cv2.drawContours(img_rgb, contours, -1, (0, 0, 255), 2)
    # cv2.drawContours(contour_image, contours, -1, (255), 1)

    img_revise = cv2.bitwise_not(img)
    con1 = contours[0]
    # 创建一个与二值图像尺寸相同的掩码
    mask = np.zeros_like(img_revise)
    # 绘制轮廓到掩码中，轮廓以内的区域标记为1，轮廓上置0
    cv2.drawContours(mask, [con1], -1, 255, thickness=cv2.FILLED)
    # cv2.drawContours(mask, [con1], -1, 0, thickness=1)     # findContours函数查找到的轮廓有厚度，必须设置特别大才行，需要小于最小墙厚

    result_image = cv2.bitwise_and(img_revise, mask)
    imgWrite('../data/tmp_res/img_contours2.jpg', result_image)
    # result_image = imgReadGray('../data/tmp_res/img_contours2.jpg')

    contours2, _ = cv2.findContours(result_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('contours2 num1:', len(contours2))
    contours2 = filter_contours(contours2)
    print('contours2 num2:', len(contours2))

    # 创建与二值图像尺寸相同的RGB画板
    rgb_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # 为每个轮廓分配一个随机颜色并绘制
    for contour in contours2:
        # 生成随机颜色 (B, G, R)
        color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
        print('color:', color)
        # 绘制轮廓
        # cv2.drawContours(rgb_image, [contour], -1, color, 2)
        cv2.drawContours(rgb_image, [contour], -1, color, thickness=cv2.FILLED)

    # imgWrite('../data/tmp_res/img_reverse.jpg', img)
    imgWrite('../data/tmp_res/img_contours3.jpg', rgb_image)

def get_connected_region():
    # imgpath = '../data/tmp_res/img_contours2.jpg'
    imgpath = '../data/tmp_res/img_reverse.jpg'
    binary_image = imgReadGray(imgpath)
    res = is_binary_image(binary_image)
    print('is binary res:', res)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    res = is_binary_image(binary_image)
    print('is binary res2:', res)
    # 1. 使用连通组件分析标记每个白色区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    # 2. 创建一个空白图像用于绘制轮廓
    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)  # 将二值图转为彩色图以便绘制轮廓

    # 3. 遍历每个连通区域（跳过背景，标签为0）
    for label in range(1, num_labels):
        # 创建当前连通区域的掩码
        mask = (labels == label).astype(np.uint8) * 255

        # 在当前掩码上查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在原图上绘制当前连通区域的轮廓
        color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
        # cv2.drawContours(output_image, contours, -1, color, 5)  # 绘制轮廓，绿色线条
        cv2.drawContours(output_image, contours, -1, color, cv2.FILLED)  # 绘制轮廓，绿色线条
    
    imgWrite('../data/tmp_res/img_contours30.jpg', binary_image)
    imgWrite('../data/tmp_res/img_contours31.jpg', output_image)

    # 打印找到的连通区域数量
    print(f"找到的独立连通区域数量: {num_labels - 1}")


if __name__ == '__main__':
    find_rooms()
    # get_connected_region()
    # test_binary()
