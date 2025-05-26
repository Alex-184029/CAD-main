# 生成掩模图矢量化尝试
import os
import cv2
import numpy as np
import json
from skimage import measure, io
import matplotlib.pyplot as plt
from PIL import Image
import shutil

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

def filter_contours(contours, min_area=100):
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)  # 周围加一圈0，保证边界轮廓闭合
    contours = measure.find_contours(padded_binary_mask, 0.5)
    # print('countours', len(contours), type(contours), contours[0], type(contours[0]))
    # contours = np.subtract(contours, 1)     # 维度不一致会报错，无法保证每个contour点数一致
    # contours = [np.array([[value - 1 for value in point] for point in con]) for con in contours]    # 三位列表生成式达到全部减一的目的
    contours = [np.subtract(contour, 1) for contour in contours]
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 4:
            continue
        # contour = np.flip(contour, axis=1)             # 翻转坐标轴，适应coco坐标格式，与opencv横纵坐标轴相反
        contour = np.round(contour)    # 四舍五入为整数
        contour = np.maximum(contour, 0)               # 负数处理
        segmentation = contour[:, ::-1].tolist()
        # segmentation = contour.ravel().tolist()
        # # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        # segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

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
    # cv2.imshow('Img', img)
    cv2.imshow('Contours', img_rgb)
    # cv2.imshow('Contours2', contour_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    imgWrite('../res/img_contours.jpg', contour_image)

def get_binary_contours2():
    imgpath = '../res/wall-gray1_opening2.jpg'
    img = imgReadGray(imgpath)
    if img is None:
        print('Read img failed.')
        return

    # 加入高斯模糊步骤
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # 查找轮廓
    # contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('contours num1:', len(contours), type(contours))
    contours = filter_contours(contours)
    print('contours num2:', len(contours), type(contours))
    # 转rgb方便绘制颜色
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cons = []
    # 遍历每个轮廓并使用 approxPolyDP 进行多边形拟合
    for cnt in contours:
        # 动态设置 epsilon 参数，通常是轮廓周长的百分比
        epsilon = 0.0008 * cv2.arcLength(cnt, True)  # 0.0002为经验值，0.0001更细节些
        # epsilon = 0.03
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 绘制原始轮廓（绿色）
        # cv2.drawContours(img_rgb, [cnt], -1, (0, 255, 0), 2)

        # 绘制多边形拟合结果（红色）
        cv2.drawContours(img_rgb, [approx], -1, (0, 0, 255), 2)

        con = []
        # 在多边形拟合的点上绘制关键点
        for point in approx:
            x, y = point.ravel()
            con.append([int(x), int(y)])
            cv2.circle(img_rgb, (x, y), 2, (255, 0, 0), -1)
        cons.append(con)

    # 绘制轮廓
    # cv2.drawContours(img_rgb, contours, -1, (0, 0, 255), 2)
    # cv2.drawContours(contour_image, contours, -1, (255), 1)

    # 显示结果图像
    # cv2.imshow('Img', img)
    cv2.imshow('Contours', img_rgb)
    # cv2.imshow('Contours2', contour_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    createLabelmeJson(cons, '../res/test/labels/test1.json')

def get_binary_contours3():
    imgpath = '../res/wall-gray1_opening2.jpg'
    img = imgReadGray(imgpath)
    if img is None:
        print('Read img failed.')
        return

    image = io.imread(imgpath, as_gray=True)
    contours = measure.find_contours(image, 0.5)     # 参数2有什么含义?，有提示0.0008

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    flag = True
    epsilon = 3.0
    for contour in contours:
        simplified_contour = measure.approximate_polygon(contour, tolerance=epsilon)
        if len(simplified_contour) < 10:
            continue
        points = simplified_contour[:, ::-1].tolist()
        if flag:
            print('points:', len(points), points[0])
            print(points)
            flag = False
        for point in points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(img_rgb, (x, y), 2, (0, 0, 255), -1)
        points = np.array(points, np.int32).reshape((-1, 1, 2))
        # 绘制多边形轮廓
        cv2.polylines(img_rgb, [points], isClosed=True, color=(0, 255, 0), thickness=1)
    imgWrite('../res/tmp2.jpg', img_rgb)
    
    cv2.imshow('contours', img_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()

def get_binary_contours4():
    imgpath = './data/masks/(T3) 12#楼105户型平面图（镜像）-2.png'
    if not os.path.exists(imgpath):
        print('img path not exist')
        return
    img = Image.open(imgpath)
    polygons = binary_mask_to_polygon(img, tolerance=5)
    print('polygons:', len(polygons), type(polygons))
    print('polygon0:', len(polygons[0]), type(polygons[0]), polygons[0])

    img_rgb = cv2.cvtColor(imgReadGray(imgpath), cv2.COLOR_GRAY2BGR)

    for polygon in polygons:
        for point in polygon:
            x, y = int(point[0]), round(point[1])
            cv2.circle(img_rgb, (x, y), 2, (0, 0, 255), -1)
        points = np.array(polygon, np.int32).reshape((-1, 1, 2))
        # 绘制多边形轮廓
        cv2.polylines(img_rgb, [points], isClosed=True, color=(0, 255, 0), thickness=1)

    cv2.imshow('contours', img_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()

    width, height = img.size

    imgname = os.path.splitext(os.path.basename(imgpath))[0]
    createLabelmeJson(polygons, os.path.join('./data/labels/', imgname + '.json'), imgWidth=width, imgHeight=height)

def get_binary_contours5(imgpath, outpath, showpath=None):
    if not os.path.exists(imgpath):
        print('imgpath or outpath not exist')
        return
    img = Image.open(imgpath)
    polygons = binary_mask_to_polygon(img, tolerance=3)
    width, height = img.size

    if not showpath is None:
        img_rgb = cv2.cvtColor(imgReadGray(imgpath), cv2.COLOR_GRAY2BGR)
        for polygon in polygons:
            for point in polygon:
                x, y = int(point[0]), round(point[1])
                cv2.circle(img_rgb, (x, y), 2, (0, 0, 255), -1)
            points = np.array(polygon, np.int32).reshape((-1, 1, 2))
            # 绘制多边形轮廓
            cv2.polylines(img_rgb, [points], isClosed=True, color=(0, 255, 0), thickness=1)
        imgWrite(showpath, img_rgb)

    if polygons is None or len(polygons) == 0:
        print('Get polygons failed.')
        return
    createLabelmeJson(polygons, outpath, imgWidth=width, imgHeight=height)

def get_contours_batch():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset32\mask_area'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset32\labels_json'
    showpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset32\labels_show'
    if not os.path.exists(imgpath):
        print('imgpath not exist')
        return
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(showpath, exist_ok=True)
    
    imgs = os.listdir(imgpath)
    num = len(imgs)
    for i, img in enumerate(imgs):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        label = os.path.splitext(img)[0] + '.json'
        get_binary_contours5(os.path.join(imgpath, img), os.path.join(outpath, label), showpath=os.path.join(showpath, img))

    print('Get contours batch finish.')

def get_canny_contours():
    imgpath = '../res/img_opening2.jpg'
    img = imgReadGray(imgpath)
    if img is None:
        print('Read img failed.')
        return

    # 加入高斯模糊步骤
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # Canny边缘检测
    edges = cv2.Canny(img, threshold1=50, threshold2=150, apertureSize=3)
    cv2.imshow('img', img)
    cv2.imshow('Edges Detected', edges)
    cv2.waitKey()
    cv2.destroyAllWindows()

    imgWrite('../res/img_canny.jpg', edges)

def get_hough_lines():
    imgpath = '../res/img_opening2.jpg'
    img = imgReadGray(imgpath)
    if img is None:
        print('Read img failed.')
        return

    edges = cv2.Canny(img, threshold1=50, threshold2=150, apertureSize=3)
    # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20)
    img_hough = np.zeros_like(img)
    if not lines is None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_hough, (x1, y1), (x2, y2), (255), 1)

    cv2.imshow('img', img)
    cv2.imshow('edges', edges)
    cv2.imshow('hough', img_hough)
    cv2.waitKey()
    cv2.destroyAllWindows()

def smooth_wall_aera():
    imgpath = '../data/masks/(T3) 12#楼105户型平面图（镜像）-2-detect2.png'
    imgout = '../data/tmp_res'
    im = imgRead(imgpath)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im_bin = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    # 进行形态学开运算
    opening = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel)

    # 进行形态学闭运算
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    imgWrite(os.path.join(imgout, 'res-opening.png'), opening)
    imgWrite(os.path.join(imgout, 'res-closing.png'), closing)
    print('----- finish -----')
    # cv2.imshow('im_bin', im_bin)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

def createLabelmeJson(contours, outpath, imgWidth=1600, imgHeight=1280):
    if contours is None or len(contours) == 0:
        print('data error for:', outpath)
        return
    shapes = []

    for contour in contours:
        # if len(contour) < 3:  # 忽略过小的轮廓
        #     continue
        
        # 转换为 LabelMe 的 points 格式
        # points = contour.squeeze().tolist()  # 转为 [(x1, y1), (x2, y2), ...]
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
    
    imgfmt = '.png'     # 图像格式，png或jpg
    imgname = os.path.splitext(os.path.basename(outpath))[0] + imgfmt
    
    # 创建 LabelMe 标注数据结构
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": f'../images/{imgname}',
        "imageData": None,
        "imageHeight": imgHeight,
        "imageWidth": imgWidth
    }

    # 将数据序列化为 JSON
    with open(outpath, 'w') as json_file:
        json.dump(labelme_data, json_file, indent=2)

def get_imgs():
    img_in = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\images'
    img_out = './data/imgs'
    maskpath = './data/masks'
    os.makedirs(img_out, exist_ok=True)

    masks = os.listdir(maskpath)
    for mask in masks:
        shutil.copy(os.path.join(img_in, mask), img_out)
    print('get imgs finish')


if __name__ == '__main__':
    # get_binary_contours4()
    # get_contours_batch()
    # get_imgs()
    smooth_wall_aera()

