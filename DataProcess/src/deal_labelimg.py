# -- 使用labelimg做简单的数据分类，生成标注框、标注筛选
import os
import shutil
import json
import numpy as np
import cv2

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)

def resize_image(image, size=(640, 640)):
    """
    将图像调整为目标尺寸。
    """
    # return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def pad_to_square(image, color=(255, 255, 255)):
    """
    将图像填充白边使其成为正方形。
    """
    h, w = image.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image, (top, left)

def isLabelRight2(labelpath):
    with open(labelpath, 'r', encoding='utf-8') as f:
        content = f.readlines()
    return True if len(content) > 0 else False

def createTempLabel():
    imgpath = r'C:\Users\DELL\Desktop\png图像数据\pngs'
    labelpath = r'C:\Users\DELL\Desktop\png图像数据\labels'
    if not os.path.exists(imgpath):
        print("Image path does not exist:", imgpath)
        return
    os.makedirs(labelpath, exist_ok=True)

    imgs = os.listdir(imgpath)
    num = len(imgs)
    for i, img in enumerate(imgs):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        label = os.path.splitext(img)[0] + '.txt'
        with open(os.path.join(labelpath, label), 'w', encoding='utf-8') as f:
            f.write('0 0.9 0.9 0.2 0.2\n')
    print('Create temp label finish.')

def doSelect1():
    imgpath = r'C:\Users\DELL\Desktop\png图像数据\pngs'
    labelpath = r'C:\Users\DELL\Desktop\png图像数据\labels'
    imgout = r'C:\Users\DELL\Desktop\png图像数据\pngs2'
    os.makedirs(imgout, exist_ok=True)

    labels = os.listdir(labelpath)
    num = len(labels)
    for i, label in enumerate(labels):
        if i % 100 == 0:
            print('%d / %d' % (i, num))
        if label == 'classes.txt':
            continue
        if not isLabelRight2(os.path.join(labelpath, label)):
            shutil.copy(os.path.join(imgpath, label.replace('.txt', '.png')), imgout)
    print('--- finish ---')


def doSelect2():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug1\images'
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug1\labels'
    imgout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug2\images'
    labelout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug2\labels'

    if not os.path.exists(imgpath) or not os.path.exists(labelpath):
        print('path does not exist')
    os.makedirs(imgout, exist_ok=True)
    os.makedirs(labelout, exist_ok=True)

    labels = os.listdir(labelpath)
    num = len(labels)
    for i, label in enumerate(labels):
        if i % 100 == 0:
            print('%d / %d' % (i, num))
        if label == 'classes.txt':
            continue
        img = label.replace('.txt', '.jpg')
        if isLabelRight2(os.path.join(labelpath, label)):
            shutil.copy(os.path.join(imgpath, img), imgout)
            shutil.copy(os.path.join(labelpath, label), labelout)
    print('--- finish ---')

def doSelect3():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\images-copy'
    pdfpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\pdfs-onlywall'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\pdfs-onlywall-error'

    if not os.path.exists(imgpath) or not os.path.exists(pdfpath):
        print('path does not exist')
    os.makedirs(outpath, exist_ok=True)

    pdfs = os.listdir(pdfpath)
    print('pdfs num:', len(pdfs))
    for pdf in pdfs:
        pdf_name = os.path.splitext(pdf)[0]
        shutil.move(os.path.join(imgpath, pdf_name + '.png'), outpath)

    print('finish')

def getIndex():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset32\images'
    imgname2 = '01-14#右单元平面-24.png'
    imgname = '01-100-b1户型平面图 2020.09.14-1.png'
    imgname3 ='02.7#01首层平面图+立面图+电图rev-01_1-1.png'
    imgname4 = '03.4#02首层平面+立面图+电图rev-01_1-1.png'
    imgname5 = '04-5#、10#、12#、14#、15#公区平面系统图-3.png'
    imgname6 = 'A（125）户型施工图20.1021_t3-4.png'
    imgname7 = "A1'公寓户型平面 20191108-1.png"
    imgs = sorted(os.listdir(imgpath))
    total = len(imgs)
    index = imgs.index(imgname6)
    print('index: %d, total %d' % (index, total))

def convert_to_yolo_format(image_width, image_height, boxes, class_id=0):
    """
    将矩形框从 [x1, y1, x2, y2] 格式转换为 YOLO 标注格式 [class_id, x_center, y_center, width, height]
    :param image_width: 图像的宽度
    :param image_height: 图像的高度
    :param boxes: 矩形框列表，每个框以 [x1, y1, x2, y2] 的形式存储
    :param class_id: 类别编号，默认为 0
    :return: YOLO 格式的标注列表
    """
    yolo_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # 计算矩形框的宽度和高度
        box_width = x2 - x1
        box_height = y2 - y1
        
        # 计算矩形框中心点的坐标
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        # 将坐标和宽高归一化到 [0, 1] 范围内
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = box_width / image_width
        height_norm = box_height / image_height
        
        # 将结果添加到 YOLO 格式的列表中
        yolo_boxes.append([class_id, x_center_norm, y_center_norm, width_norm, height_norm])
    
    return yolo_boxes

def labelme_to_yolo_boxes(json_path, yolo_dir):
    if not os.path.exists(json_path):
        print('json path does not exist:', json_path)
        return
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = data['shapes']
    boxes = []
    for shape in shapes:
        points = shape['points']
        x1, y1 = points[0]
        x2, y2 = points[0]
        for p in points[1:]:
            x1, y1, x2, y2 = min(x1, p[0]), min(y1, p[1]), max(x2, p[0]), max(y2, p[1])
        boxes.append([x1, y1, x2, y2])
    json_dir, json_name = os.path.split(json_path)
    image_path = os.path.join(json_dir, data['imagePath'])
    im = imgRead(image_path)
    h, w, _ = im.shape
    yolo_boxes = convert_to_yolo_format(w, h, boxes)
    yolo_path = os.path.join(yolo_dir, json_name.replace('.json', '.txt'))
    with open(yolo_path, "w") as f:
        for box in yolo_boxes:
            f.write(" ".join(map(str, box)) + "\n")

def labelme_to_yolo_boxes_batch():
    json_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-resize640\labels'
    yolo_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-resize640\labels_yolo'
    if not os.path.exists(json_path):
        print('json path does not exist:', json_path)
        return
    os.makedirs(yolo_path, exist_ok=True)
    jsons = os.listdir(json_path)
    for i, json in enumerate(jsons):
        if i % 50 == 0:
            print('%d / %d' % (i, len(jsons)))
        labelme_to_yolo_boxes(os.path.join(json_path, json), yolo_path)
    print('finish')

def resize_batch():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\images'
    imgout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\images-resize640'
    os.makedirs(imgout, exist_ok=True)

    imgs = os.listdir(imgpath)
    num = len(imgs)
    for i, img in enumerate(imgs):
        if i % 50 == 0:
            print('%d / %d' % (i, num))
        im = imgRead(os.path.join(imgpath, img))
        padded_image, _ = pad_to_square(im)
        im_resize = resize_image(padded_image, size=(640, 640))
        imgWrite(os.path.join(imgout, img), im_resize)
    print('finish')

def calc_label_num():
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\labels_yolo'
    if not os.path.exists(labelpath):
        print('label path does not exist:', labelpath)
        return
    labels = os.listdir(labelpath)
    num_file, num_label = len(labels), 0
    # label = labels[0]
    # with open(os.path.join(labelpath, label), 'r', encoding='utf-8') as f:
    #     data = f.readlines()
    # print('data:', data)
    for label in labels:
        if label == 'classes.txt':
            print('here is classes.txt')
            continue
        with open(os.path.join(labelpath, label), 'r', encoding='utf-8') as f:
            num_label += len(f.readlines())
    print('num_file: %d, num_label: %d' % (num_file, num_label))
    

def test():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset32\images'
    dwg_set = set()
    imgs = os.listdir(imgpath)
    for img in imgs:
        index = img.rfind('-')
        dwg_set.add(img[:index])
    print(len(dwg_set), len(imgs))


if __name__ == '__main__':
    # createTempLabel()
    doSelect1()
