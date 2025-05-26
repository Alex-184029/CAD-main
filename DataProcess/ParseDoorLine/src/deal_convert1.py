import cv2
import numpy as np
import json
from deal_mask1 import imgRead, imgReadGray, imgWrite, is_binary_image

def load_labelme_polygons(json_file):
    """
    从 labelme JSON 文件中加载多边形标注。
    
    参数:
        json_file (str): labelme JSON 文件路径。
    
    返回:
        list: 多个多边形的列表，每个多边形是一个 NumPy 数组，表示顶点坐标。
    """
    with open(json_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    polygons = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "polygon":
            points = np.array(shape["points"], dtype=np.int32)
            polygons.append(points)
    
    return polygons

def apply_mask_to_image(image, polygons):
    """
    将多边形区域外的部分涂成白色，保留多边形区域内的颜色。
    
    参数:
        image (numpy.ndarray): 输入图像（BGR 格式）。
        polygons (list): 多个多边形的列表，每个多边形是一个 NumPy 数组。
    
    返回:
        numpy.ndarray: 处理后的图像。
    """
    # 创建一个与图像大小相同的空白掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 填充多边形区域为 1
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 1)
    
    # 将掩码应用到图像上
    result = image.copy()
    result[mask == 0] = 255  # 将掩码外部分涂成白色
    
    return result

def filter_others():
    # 加载图像
    image_path = "../data/images2/(T3) 12#楼105户型平面图（镜像）-3.jpg"
    image = imgRead(image_path)
    
    # 加载 labelme 标注文件
    json_file = "../data/labels_ArcDoor/(T3) 12#楼105户型平面图（镜像）-3_Structure.json"
    polygons = load_labelme_polygons(json_file)
    
    # 应用掩码到图像
    result_image = apply_mask_to_image(image, polygons)
    
    # 保存结果
    imgWrite("../data/tmp_res2/tmp1.jpg", result_image)
    print('----- finish -----')

def load_labelme_polygons2(json_file):
    """
    从 labelme JSON 文件中加载多边形标注。
    
    参数:
        json_file (str): labelme JSON 文件路径。
    
    返回:
        list: 多个多边形的列表，每个多边形是一个 NumPy 数组，表示顶点坐标。
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    polygons = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "polygon" and shape["label"] == "WallArea1":
            points = np.array(shape["points"], dtype=np.int32)
            polygons.append(points)
    
    return polygons

def apply_mask_to_image2(image, polygons):
    """
    将多边形区域内的部分涂成黑色，区域外的部分涂成白色。
    
    参数:
        image (numpy.ndarray): 输入图像（BGR 格式）。
        polygons (list): 多个多边形的列表，每个多边形是一个 NumPy 数组。
    
    返回:
        numpy.ndarray: 处理后的图像。
    """
    # 创建一个与图像大小相同的空白掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 填充多边形区域为 1
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 1)
    
    # 将多边形区域涂成黑色
    result = image.copy()
    result[mask == 1] = 0
    
    return result

def paint_black():
    # 加载图像
    image_path = "../data/tmp_res2/tmp1.jpg"
    image = imgRead(image_path)
    
    # 加载 labelme 标注文件
    json_file = "../data/labels_ArcDoor/(T3) 12#楼105户型平面图（镜像）-3_Structure.json"
    polygons = load_labelme_polygons2(json_file)
    
    # 应用掩码到图像
    result_image = apply_mask_to_image2(image, polygons)
    
    # 保存结果
    imgWrite("../data/tmp_res2/tmp2.jpg", result_image)
    print('----- finish -----')

def img_to_binary():
    img_path = '../data/tmp_res2/tmp2.jpg'
    img = imgRead(img_path)

    # # 提取蓝色通道
    # blue_channel = img[:, :, 0]
    # # 将蓝色通道转换为灰度图（实际上这里已经是灰度值）
    # gray_img = blue_channel

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # res = is_binary_image(gray_img)
    # print('res:', res)

    imgWrite('../data/tmp_res2/tmp30.jpg', gray_img)
    
    # 对灰度图进行二值化处理
    _, binarized_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    
    # 保存二值化后的图像
    imgWrite('../data/tmp_res2/tmp3.jpg', binarized_img)


if __name__ == "__main__":
    # filter_others()
    # paint_black()
    img_to_binary()