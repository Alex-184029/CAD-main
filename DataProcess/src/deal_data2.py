# -- 数据处理
import os
import cv2
import numpy as np
import shutil
import random
 
W, H = 320, 320

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)

def imgResize(imgpath, outpath, target_size=(320, 320)):
    img = imgRead(imgpath)
    resized_img = cv2.resize(img, target_size)
    imgWrite(outpath, resized_img)

def dataResize():
    # 输入文件夹路径
    input_img_folder = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-attempt2\yolo-select2\Crop320Data\crop1280\imgs'
    input_label_folder = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-attempt2\yolo-select2\Crop320Data\crop1280\labels'
    
    # 输出文件夹路径
    output_img_folder = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-attempt2\yolo-select2\Crop320Data\crop1280\imgs-320'
    output_label_folder = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-attempt2\yolo-select2\Crop320Data\crop1280\labels-320'

    if not os.path.exists(input_img_folder) or not os.path.exists(input_label_folder):
        print('input folder not exist')
        return
    
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)
    
    # 目标尺寸
    target_size = (320, 320)
    
    # 确保输出文件夹存在
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)
    
    # 遍历文件夹a中的每张图片
    imgs = os.listdir(input_img_folder)
    total = len(imgs)
    for i, file_name in enumerate(imgs):
        if i % 100 == 0:
            print('%d for %d is doing' % (i, total))
        if os.path.splitext(file_name)[1] == '.jpg':
            # 读取图片
            img_path = os.path.join(input_img_folder, file_name)
            img = imgRead(img_path)
    
            # 缩放图片
            resized_img = cv2.resize(img, target_size)
    
            # 保存缩放后的图片到文件夹c
            output_img_path = os.path.join(output_img_folder, os.path.splitext(file_name)[0] + '-aug.jpg')
            imgWrite(output_img_path, resized_img)
    
            # 处理对应的标签文件
            label_file_name = os.path.splitext(file_name)[0] + '.txt'
            label_file_name_out = os.path.splitext(file_name)[0] + '-aug.txt'
            label_file_path = os.path.join(input_label_folder, label_file_name)
            label_file_path_out = os.path.join(output_label_folder, label_file_name_out)
            shutil.copy(label_file_path, label_file_path_out)
    
def findError1():
    imgspath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-select\imgs'
    labelspath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-select\labels'
    imgs = os.listdir(imgspath)
    labels = os.listdir(labelspath)
    imgs_ext = [os.path.splitext(img)[1] for img in imgs]
    imgs = [os.path.splitext(img)[0] for img in imgs]
    labels = [os.path.splitext(label)[0] for label in labels]

    print(len(imgs), len(labels))

    for img in imgs:
        if not img in labels:
            print('error label:', img)

    for label in labels:
        if not label in imgs:
            print('error img:', label)

    for ext in imgs_ext:
        if ext != '.jpg':
            print(ext)

def readLabelMerge(labelpath):
    with open(labelpath, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [list(map(float, i.strip().split(' '))) for i in labels]

    return labels

def isLabelRight(labelpath):
    if not os.path.exists(labelpath):
        print('label path not exist,', labelpath)
        return False
    with open(labelpath, 'r', encoding='utf-8') as f:
        label = f.readlines()
    label = [list(map(float, i.strip().split(' '))) for i in label]
    if len(label) == 0:
        return False
    for l in label:
        if any(i <= 0 or i >=1 for i in l[1:]):
            return False
    return True

def isLabelRight2(labelpath):
    with open(labelpath, 'r', encoding='utf-8') as f:
        content = f.readlines()
    return True if len(content) > 0 else False

def readLabel(labelpath, w, h):
    with open(labelpath, 'r', encoding='utf-8') as f:
        label = f.readlines()
    if len(label) == 0:
        return []
    label = [list(map(float, i.strip().split())) for i in label]
    boxes = []
    for l in label:
        if len(l) < 5:
            continue
        x_center, y_center, width, height = l[1], l[2], l[3], l[4]
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        x_min, x_max, y_min, y_max = int(x_center - width / 2), int(x_center + width / 2), int(y_center - height / 2), int(y_center + height / 2)
        boxes.append([x_min, y_min, x_max, y_max, l[0]])

    return boxes

def isLabelRepeat(label1, label2):
    x11 = label1[1] - label1[3] / 2
    y11 = label1[2] - label1[4] / 2
    x12 = label1[1] + label1[3] / 2
    y12 = label1[2] + label1[4] / 2

    x21 = label2[1] - label2[3] / 2
    y21 = label2[2] - label2[4] / 2
    x22 = label2[1] + label2[3] / 2
    y22 = label2[2] + label2[4] / 2
    
    rect1 = [x11, y11, x12, y12]
    rect2 = [x21, y21, x22, y22]
    return isIouRight(rect1, rect2, iou=0.8)

    # if x11 <= x21 and y11 <= y21 and x12 >= x22 and y12 >= y22:
    #     return True
    # elif x11 >= x21 and y11 >= y21 and x12 <= x22 and y12 <= y22:
    #     return True
    # else:
        # return False

def removeRepeatLabel(labelpath):
    with open(labelpath, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [list(map(float, i.strip().split(' '))) for i in labels]
    signs = [False for i in labels]
    for i in range(len(labels)):
        if signs[i]:
            continue
        for j in range(i + 1, len(labels)):
            if isLabelRepeat(labels[i], labels[j]):
                signs[j] = True

    labels = [labels[i] for i in range(len(labels)) if not signs[i]]
    with open(labelpath, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write('0 %f %f %f %f\n' % (label[1], label[2], label[3], label[4]))
            # f.write(' '.join(map(str, label)) + '\n')

def isLabelError(label):
    if any(i <= 0 or i >= 1 for i in label[1:]):
        return True
    x1 = label[1] - label[3] / 2
    y1 = label[2] - label[4] / 2
    x2 = label[1] + label[3] / 2 
    y2 = label[2] + label[4] / 2
    if any(i <= 0 or i >= 1 for i in [x1, y1, x2, y2]):
        return True
    return False

def removeLabelError(labelpath):
    with open(labelpath, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [list(map(float, i.strip().split(' '))) for i in labels]
    signs = [False for i in labels]
    for i in range(len(labels)):
        if isLabelError(labels[i]):
            signs[i] = True

    labels = [labels[i] for i in range(len(labels)) if not signs[i]]
    with open(labelpath, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write('0 %f %f %f %f\n' % (label[1], label[2], label[3], label[4]))
            # f.write(' '.join(map(str, label)) + '\n')

def writeLabel(labelpath, bboxes):
    with open(labelpath, 'w', encoding='utf-8') as f:
        for bbox in bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')

def mergeClass():
    in_labels = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-select\labels'
    out_labels = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-select2\labels'

    if not os.path.exists(in_labels):
        print('in_labels not exist')
        return

    if not os.path.exists(out_labels):
        os.makedirs(out_labels)

    labels = os.listdir(in_labels)

    for label in labels:
        bboxes = readLabelMerge(os.path.join(in_labels, label))
        writeLabel(os.path.join(out_labels, label), bboxes)

    print('merge finish')

def splitBox(labelpath):
    with open(labelpath, 'r', encoding='utf-8') as f:
        label = f.readlines()
    label = [list(map(float, i.strip().split(' '))) for i in label]
    label = [[int(l[0]), l[1], l[2], l[3], l[4]] for l in label]

    label_add = []
    for l in label:
        if l[4] / l[3] > 1.5:
            # 竖着分成两份
            w = l[3]
            h = l[4] / 2
            x = l[1]
            y = l[2] + h / 2
            l[2] = l[2] - h / 2
            l[4] = h
            label_add.append([0, x, y, w, h])
        elif l[4] / l[3] < 0.75:
            # 横着分成两份
            w = l[3] / 2
            h = l[4]
            x = l[1] + w / 2
            y = l[2]
            l[1] = l[1] - w / 2
            l[3] = w
            label_add.append([0, x, y, w, h])

    label = label + label_add
    return label

def splitBoxes():
    in_labels = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-select2\labels2'
    out_labels = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-select2\labels3'

    if not os.path.exists(in_labels):
        print('in_labels not exist')
        return

    if not os.path.exists(out_labels):
        os.makedirs(out_labels)

    labels = os.listdir(in_labels)
    totol = len(labels)
    for i, label in enumerate(labels):
        if i % 50 == 0:
            print('%d for %d is doing' % (i, totol))
        try:
            boxes = splitBox(os.path.join(in_labels, label))
            writeLabel(os.path.join(out_labels, label), boxes)
        except:
            print('error label:', label)
    print('split box finish')

def isIouRight(rect1, rect2, iou=0.8):
    if rect1[4] != rect2[4]:
        return False
    # 计算交集区域的坐标范围
    x_left = max(rect1[0], rect2[0])
    y_bottom = max(rect1[1], rect2[1])
    x_right = min(rect1[2], rect2[2])
    y_top = min(rect1[3], rect2[3])

    # 如果交集不存在，则返回0
    if x_right < x_left or y_top < y_bottom:
        return False

    # 计算交集区域的面积
    intersection_area = (x_right - x_left) * (y_top - y_bottom)

    # 计算两个矩形的面积
    rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    # 计算并集区域的面积
    union_area = rect1_area + rect2_area - intersection_area

    # 计算交并比
    iou_calc = intersection_area / union_area

    return True if iou_calc > iou else False


def getLou(boxes1, boxes2):    # boxes1为标准框，boxes2为检测框，找出漏检的情况
    if len(boxes1) == 0 or len(boxes2) == 0:
        return max(len(boxes1), len(boxes2))
    iou = 0.5
    lou_cnt = 0
    for box1 in boxes1:
        sign = True
        for box2 in boxes2:
            if isIouRight(box1, box2, iou=iou):
                sign = False
                break
        if sign:
            lou_cnt += 1
    return lou_cnt

def findError2(imgpath, labelpath1, labelpath2, outpath):
    if not os.path.exists(imgpath) or not os.path.exists(labelpath1) or not os.path.exists(labelpath2):
        print('some input path not exist')
        return
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    imgs = os.listdir(imgpath)
    imgs_err = []
    total, cnt = len(imgs), 0
    box_total, lou_total = 0, 0

    for i, img in enumerate(imgs):
        if i % 100 == 0:
            print('%d for %d is doing' % (i, total))
        image = imgRead(os.path.join(imgpath, img))
        h, w, _ = image.shape
        img = os.path.splitext(img)[0]
        boxes1 = readLabel(os.path.join(labelpath1, img + '.txt'), w, h)
        boxes2 = readLabel(os.path.join(labelpath2, img + '.txt'), w, h)
        cnt_lou = getLou(boxes1, boxes2)
        box_total += len(boxes1)
        lou_total += cnt_lou
        if cnt_lou > 0:
            imgs_err.append(img)
            cnt += 1
    print('total: %d, cnt: %d' % (total, cnt))
    print('box_total: %d, lou_total: %d, acc: %.4f' % (box_total, lou_total, (1 - lou_total / box_total)))

    print('imgs_err', imgs_err)

def convertToBlack(imgpath, outpath):
    imgs = os.listdir(imgpath)
    for img in imgs:
        img_color = imgRead(os.path.join(imgpath, img))
        # 转换为灰度图像
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        # 阈值处理（二值化）
        _, img_binary = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)
        # 保存二值图像
        imgWrite(os.path.join(outpath, img), img_binary)

def selectSvg(inpath, outpath):
    if not os.path.exists(inpath):
        print('inpath not exist')
        return
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    files = os.listdir(inpath)
    total = len(files)
    for i, f in enumerate(files):
        if i % 1000 == 0:
            print('%d for %d is doing' % (i, total))
        if f.endswith('.svg'):
            shutil.copy(os.path.join(inpath, f), outpath)

def splitImgLabel(imgpath, labelpath, imgout, labelout):
    if not os.path.exists(imgpath) or not os.path.exists(labelpath):
        print('datapath not exist')
        return
    if not os.path.exists(imgout):
        os.makedirs(imgout)
    if not os.path.exists(labelout):
        os.makedirs(labelout)
    
    imgs = os.listdir(imgpath)
    for img in imgs:
        labeltmp = os.path.join(labelpath, img.replace('.jpg', '.txt'))
        if not isLabelRight(labeltmp):
            shutil.move(os.path.join(imgpath, img), imgout)
            shutil.move(labeltmp, labelout)
    print('finish')

def colorToGray(colorpath, graypath):
    if not os.path.exists(graypath):
        os.makedirs(graypath)
    imgs = os.listdir(colorpath)
    # num = 50
    # imgs = random.sample(imgs, num)
    print('num:', len(imgs))
    for img in imgs:
        img_color = imgRead(os.path.join(colorpath, img))
        # 转换为灰度图像
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        # 转换为二值图像
        # _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        # 结果保存
        # imgWrite(os.path.join(graypath, img), img_bin)
        imgWrite(os.path.join(graypath, img), img_gray)
    print('to gray finish')

def copyViewports():
    imgpath = r'E:\School\Grad1\CAD\MyCAD\DataProcess\tests\test4.26\data-test4.26\images'
    viewpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\AllDwgFiles'
    outpath = r'E:\School\Grad1\CAD\MyCAD\DataProcess\tests\test4.26\data-test4.26\OriginDwgFiles'
    if not os.path.exists(imgpath) or not os.path.exists(viewpath):
        print('src path not exist')
    os.makedirs(outpath, exist_ok=True)

    imgs = os.listdir(imgpath)

    for img in imgs:
        # index = img.find('-aug1-aug')
        # if index == -1:
        #     print('error in:', img)
        #     continue
        # img = img[:index]
        # shutil.copy(os.path.join(viewpath, img + '.jpg'), outpath)
        data = img.split('-')
        index = len(data) - 3
        dwgName = ""
        for i in range(index):
            dwgName += data[i] + '-'
        dwgName = dwgName[:-1]
        print(dwgName)
        shutil.copy(os.path.join(viewpath, dwgName + '.dwg'), outpath)

    print('copy finish')

def removeEnd():
    labelpath = r'E:\School\Grad1\CAD\MyCAD\DataProcess\tests\test4.26\net-test4.26\labels'
    labels = os.listdir(labelpath)

    for label in labels:
        if label.find('-ArcDoor.txt') != -1:
            label2 = label[:-12]
            shutil.move(os.path.join(labelpath, label), os.path.join(labelpath, label2 + '.txt'))
    print('remove finish')

def apply_green_tint(region, color='g'):
    """
    给输入区域加上淡淡绿色滤镜。
    """
    # 创建一个淡淡绿色的滤镜矩阵，这里通过增加绿色分量并稍微减少其他分量来模拟
    green_tint = np.array([1, 1, 1]).reshape((1, 1, 3))  # 调整系数以控制绿色强度
    if color == 'g':
        green_tint = np.array([0.8, 1.1, 0.8]).reshape((1, 1, 3))  # 调整系数以控制绿色强度
    elif color == 'r':
        green_tint = np.array([0.8, 0.8, 1.1]).reshape((1, 1, 3))  # 调整系数以控制绿色强度
    elif color == 'b':
        green_tint = np.array([1.1, 0.8, 0.8]).reshape((1, 1, 3))  # 调整系数以控制绿色强度
    elif color == 'y':
        green_tint = np.array([0.8, 0.7, 0.2]).reshape((1, 1, 3))  # 调整系数以控制绿色强度
    tinted_region = region * green_tint
    # 确保像素值在0-255之间
    tinted_region = np.clip(tinted_region, 0, 255).astype(np.uint8)
    return tinted_region

def drawSplitRect():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-attempt2\yolo-select2\ReadyData\imgs\(1)25#楼一层平面图-1.jpg'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\yolo-attempt2\yolo-select2\ReadyData\tmp\(1)25#楼一层平面图-1.jpg'
    img = imgRead(imgpath)
    rect_w, rect_h = 900, 900
    h, w, _ = img.shape
    print(w, h)

    # 定义矩形区域
    x, y, width, height = 0, 0, rect_w, rect_h  # 左上角坐标(x, y)，宽度width，高度height
    region_of_interest = img[y:y+height, x:x+width].copy()
    tinted_roi = apply_green_tint(region_of_interest, color='r')
    # 将处理过的区域放回原图
    img[y:y+height, x:x+width] = tinted_roi

    x, y, width, height = 0, h - rect_h, rect_w, rect_h
    region_of_interest = img[y:y+height, x:x+width].copy()
    tinted_roi = apply_green_tint(region_of_interest, color='g')
    # 将处理过的区域放回原图
    img[y:y+height, x:x+width] = tinted_roi

    x, y, width, height = w - rect_w, 0, rect_w, rect_h
    region_of_interest = img[y:y+height, x:x+width].copy()
    tinted_roi = apply_green_tint(region_of_interest, color='b')
    # 将处理过的区域放回原图
    img[y:y+height, x:x+width] = tinted_roi

    x, y, width, height = w - rect_w, h - rect_h, rect_w, rect_h
    region_of_interest = img[y:y+height, x:x+width].copy()
    tinted_roi = apply_green_tint(region_of_interest, color='y')
    # 将处理过的区域放回原图
    img[y:y+height, x:x+width] = tinted_roi

    # cv2.rectangle(img, (0, 0), (rect_w, rect_h), color=(0, 0, 255), thickness=2)
    # cv2.rectangle(img, (0, h - rect_h), (rect_w, h), color=(0, 0, 255), thickness=2)
    # cv2.rectangle(img, (w - rect_w, 0), (w, rect_h), color=(0, 0, 255), thickness=2)
    # cv2.rectangle(img, (w - rect_w, h - rect_h), (w, h), color=(0, 0, 255), thickness=2)
    imgWrite(outpath, img)
    print('split finish')

def getDwgName(dwg):
    dwg_split = dwg.split('-')
    end = len(dwg) - len(dwg_split[-1]) - len(dwg_split[-2]) - 2
    return dwg[:end] + '.dwg'

def getSelctDwgs():
    dwgpath1 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsParallelWindow\LabelsSelect1'
    dwgpath2 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsCircleLight\LabelsSelect1'
    dwgpath3 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsArcDoor\LabelsSelect1'
    dwgOrigin = r'E:\School\Grad1\CAD\Datasets\DwgFiles\AllDwgFiles'

    dwgs1 = os.listdir(dwgpath1)
    dwgs1 = [getDwgName(dwg) for dwg in dwgs1]
    dwgs2 = os.listdir(dwgpath2)
    dwgs2 = [getDwgName(dwg) for dwg in dwgs2]
    dwgs3 = os.listdir(dwgpath3)
    dwgs3 = [getDwgName(dwg) for dwg in dwgs3]

    print(len(dwgs1), len(dwgs2), len(dwgs3))

    dwgs1.extend(dwgs2)
    dwgs1.extend(dwgs3)
    print(len(dwgs1))

    dwgs1 = list(set(dwgs1))
    print(len(dwgs1))

    cntErr, total = 0, len(dwgs1)
    for dwg in dwgs1:
        dwgpath = os.path.join(dwgOrigin, dwg)
        if not os.path.exists(dwgpath):
            cntErr += 1
            print('Error %d, %s' % (cntErr, dwgpath))
    print('total: %d, error: %d' % (total, cntErr))
    print('finish')

def imgToRect():
    imgpath1 = r'C:\Users\DELL\Desktop\01. 1#2#9#10#、12-14#楼标准层平面图-2.jpg'
    imgpath2 = r'C:\Users\DELL\Desktop\01.7#封面+目录+施工说明+物料表rev-01_1-1.jpg'
    imgpath3 = r'C:\Users\DELL\Desktop\03 碧云湾A户型立面图-10.jpg'
    imgout = './DataProcess/src/img_tmp/'

    image = imgRead(imgpath3)
    
    # 获取图像尺寸
    height, width, channels = image.shape
    # 计算需要填充的高度
    padding_height = (1600 - height) // 2
    # 创建一个与原图像相同尺寸的全白图像
    white_image = np.ones((padding_height, width, channels), dtype=np.uint8) * 255
    # 将原图像和白色图像拼接在一起
    padded_image = np.concatenate((white_image, image, white_image), axis=0)
    # resized_image = cv2.resize(padded_image, (320, 320))
    resized_image = cv2.resize(image, (320, 256))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # 保存填充后的图像
    cv2.imwrite(os.path.join(imgout, 'tmp2.jpg'), gray_image)

def createViewportDataset():
    labelOrigin = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\OriginLabels'
    imgOrigin = r'E:\School\Grad1\CAD\Datasets\DwgFiles\AllViewportImgs'
    imgout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset0\images'
    labelout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset0\labels'

    if not os.path.exists(labelOrigin) or not os.path.exists(imgOrigin):
        print('source not exist')
        return
    os.makedirs(imgout, exist_ok=True)
    os.makedirs(labelout, exist_ok=True)

    datas = os.listdir(labelOrigin)
    total = len(datas)
    for i, data in enumerate(datas):
        if i % 50 == 0:
            print('%d for %d is doing' % (i, total))
        with open(os.path.join(labelOrigin, data), 'r', encoding='utf-8') as f:
            labels = f.readlines()
        labels = [label.strip().split() for label in labels]
        for label in labels:
            tmp_img = os.path.join(imgOrigin, os.path.splitext(data)[0] + '-' + label[0] + '.jpg')
            tmp_label = os.path.join(labelout, os.path.splitext(data)[0] + '-' + label[0] + '.txt')
            if not os.path.exists(tmp_img):
                print('tmp_img not exist,', tmp_img)
                continue
            shutil.copy(tmp_img, imgout)
            value_label = label[1] + label[2] + label[3] + label[4] + label[5]
            with open(tmp_label, 'w', encoding='utf-8') as f:
                f.write('%s %d %d %d %d\n' % (value_label, 0, 0, 1, 1))

def createViewportDataset2():
    imgpath1 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset0\images-clean'
    labelpath1 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset0\labels'
    imgpath2 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset5-furn\images'
    labelpath2 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset5-furn\labels'

    os.makedirs(imgpath2, exist_ok=True)
    os.makedirs(labelpath2, exist_ok=True)

    datas = os.listdir(imgpath1)
    total = len(datas)
    err_datas = []
    for i, data in enumerate(datas):
        if i % 200 == 0:
            print('%d for %d is doing' % (i, total))
        try:
            with open(os.path.join(labelpath1, data.replace('.jpg', '.txt')), 'r', encoding='utf-8') as f:
                label = f.readlines()[0].split()[0]
            label = label[4]     # 顺序：窗、门、墙、灯、家具（从索引0开始）
            tmp_img = os.path.join(imgpath2, label)
            tmp_label = os.path.join(labelpath2, label)
            if not os.path.exists(tmp_img):
                os.makedirs(tmp_img, exist_ok=True)
            if not os.path.exists(tmp_label):
                os.makedirs(tmp_label, exist_ok=True)
            shutil.copy(os.path.join(imgpath1, data), tmp_img)
            with open(os.path.join(tmp_label, data.replace('.jpg', '.txt')), 'w', encoding='utf-8') as f:
                f.write('%s %f %f %f %f\n' % (label, 0.25, 0.25, 0.5, 0.5))
            # shutil.copy(os.path.join(labelpath1, data), tmp_label)
        except:
            err_datas.append(data)

    print('err_datas:', len(err_datas))
    print(err_datas)
    print('create finish')

def createViewportDataset3():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset5-furn\OriginData\images\0'
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset5-furn\OriginData\labels\0'
    imgout0 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset5-furn\SelectData\0'
    imgout1 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset5-furn\SelectData\1'
    imgerr = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset5-furn\SelectData\err'

    if not os.path.exists(imgpath) or not os.path.exists(labelpath):
        print('source not exist')
        return
    os.makedirs(imgout0, exist_ok=True)
    os.makedirs(imgout1, exist_ok=True)
    os.makedirs(imgerr, exist_ok=True)

    labels = os.listdir(labelpath)
    total = len(labels)
    cnt0, cnt1, cntErr = 0, 0, 0
    for i, label in enumerate(labels):
        if i % 200 == 0:
            print('%d for %d is doing' % (i, total))
        if label == 'classes.txt':
            continue
        with open(os.path.join(labelpath, label), 'r', encoding='utf-8') as f:
            content = f.readlines()
        if len(content) == 0:
            content = '1'
        else:
            content = content[0].split()[0]
            content = content if content == '0' else 'error'
        if content == '0':
            shutil.move(os.path.join(imgpath, label.replace('.txt', '.jpg')), imgout0)
            cnt0 += 1
        elif content == '1':
            shutil.move(os.path.join(imgpath, label.replace('.txt', '.jpg')), imgout1)
            cnt1 += 1
        else:
            shutil.move(os.path.join(imgpath, label.replace('.txt', '.jpg')), imgerr)
            cntErr += 1

    print('total: %d, 0: %d, 1: %d, error: %d' % (total, cnt0, cnt1, cntErr))

def mergeViewportDataset4(mode=0):
    if mode == 0:
        datapath1 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset3'
        datapath2 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset4-light'
        datapath3 = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset5-furn'
        datapaths = [datapath1, datapath2, datapath3]

        img_0 = r'SelectData\0'
        img_1 = r'SelectData\1'
        img_err = r'SelectData\err'

        outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset6_merge\labels'
        os.makedirs(outpath, exist_ok=True)

        err_list = os.listdir(os.path.join(datapath1, img_err))
        err_list += os.listdir(os.path.join(datapath2, img_err))
        err_list += os.listdir(os.path.join(datapath3, img_err))
        err_list = list(set(err_list))
        total_num = len(os.listdir(os.path.join(datapath1, img_0))) + len(os.listdir(os.path.join(datapath1, img_1)))
        print('total: %d, err: %d, valid: %d' % (total_num, len(err_list), total_num - len(err_list)))

        # for i, datapath in enumerate(datapaths):
        #     print('merge dataset', i + 1)
        #     imgs0 = os.listdir(os.path.join(datapath, img_0))
        #     for img in imgs0:
        #         if img in err_list:
        #             continue
        #         with open(os.path.join(outpath, img.replace('.jpg', '.txt')), 'a', encoding='utf-8') as f:
        #             f.write('0')
        #     imgs1 = os.listdir(os.path.join(datapath, img_1))
        #     for img in imgs1:
        #         if img in err_list:
        #             continue
        #         with open(os.path.join(outpath, img.replace('.jpg', '.txt')), 'a', encoding='utf-8') as f:
        #             f.write('1')
    elif mode == 1:
        imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset0\images-clean'
        labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset6_merge\labels'
        datapath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\ViewportClassLabels\dataset6_merge\data'

        datas = os.listdir(labelpath)
        total = len(datas)
        for i, data in enumerate(datas):
            if i % 200 == 0:
                print('%d for %d is doing' % (i, total))
            with open(os.path.join(labelpath, data), 'r', encoding='utf-8') as f:
                labelname = f.readlines()
            if len(labelname) == 0:
                print('error label:', data)
                continue
            labelname = labelname[0].strip()
            tmppath = os.path.join(datapath, labelname)
            if not os.path.exists(tmppath):
                os.makedirs(tmppath)
            shutil.copy(os.path.join(imgpath, data.replace('.txt', '.jpg')), tmppath)

        print('----- finish -----')

def copyRandom():
    inpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\pdfs'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\PdfScaleTest\pdfs'
    if not os.path.exists(inpath):
        print('inpath not exist')
    os.makedirs(outpath, exist_ok=True)

    num = 200
    datas = random.sample(os.listdir(inpath), num)
    for data in datas:
        shutil.copy(os.path.join(inpath, data), outpath)
    print('finish')

def createList():
    inpath = r'E:\School\Grad1\restful_car3_deploy\Data\DataFromServer\Data_11.12\origin_pic\origin_car_pic'
    txtpath = r'E:\School\Grad1\restful_car3_deploy\Data\DataFromServer\Data_11.12\origin_pic\list.txt'
    if not os.path.exists(inpath):
        print('inpath not exist')
        return

    datas = os.listdir(inpath)
    print('data num:', len(datas))
    with open(txtpath, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write(data + '\n')
    print('----- finish -----')

def selectList():
    txtpath = r'E:\School\Grad1\restful_car3_deploy\Data\DataFromServer\Data_11.12\origin_pic\list.txt'
    inpath = r'E:\School\Grad1\restful_car3_deploy\Data\DataFromServer\Data_11.12\origin_pic'
    outpath = r'E:\School\Grad1\restful_car3_deploy\Data\DataFromServer\Data_11.12\origin_pic'

    if not os.path.exists(txtpath) or not os.path.exists(inpath):
        print('src path not exist')
        return
    os.makedirs(outpath, exist_ok=True)

    with open(txtpath, 'r', encoding='utf-8') as f:
        datas_list = f.readlines()
    datas_list = [data.strip() for data in datas_list]
    datas_in = os.listdir(inpath)
    datas_out = [data for data in datas_in if not data in datas_list]

    print(len(datas_in, datas_list, datas_out))

def doSelect1():
    dwgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\AllDwgFiles'
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\labels'
    dwgout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\AllDwgFiles3'
    os.makedirs(dwgout, exist_ok=True)
    
    labels = os.listdir(labelpath)
    num = len(labels)
    dwg_set = set()
    for i, label in enumerate(labels):
        if i % 1000 == 0:
            print('%d / %d' % (i, num))
        index = label.rfind('-')
        if index == -1:
            print('label error:', label)
            continue
        # index = label.rfind('-', 0, index)
        # if index == -1:
        #     print('label error2:', label)
        #     continue
        dwg = label[:index] + '.dwg'
        # print('dwg:', dwg)
        dwg_set.add(dwg)
    print('dwg num:', len(dwg_set))
    for dwg in dwg_set:
        shutil.copy(os.path.join(dwgpath, dwg), dwgout)
    print('----- finish -----')


def statisticLabels():
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsParallelWindow\yolo5\DataAug\dataset-aug\labels'

    if not os.path.exists(labelpath):
        print('labelpath not exist,', labelpath)
        return

    labels = os.listdir(labelpath)
    total, num = len(labels), 0
    for i, label in enumerate(labels):
        if i % 500 == 0:
            print('%d / %d' % (i, total))
        with open(os.path.join(labelpath, label), 'r', encoding='utf-8') as f:
            num += len(f.readlines())

    print('total: %d, num: %d' % (total, num))

def selectImages():
    imgpath = r'C:\Users\DELL\Desktop\Console\Windows4'
    imgout = r'C:\Users\DELL\Desktop\Console\Windows2'

    imgs = os.listdir(imgpath)
    for img in imgs:
        if img.endswith('_ParallelWindow.jpg'):
            img2 = os.path.splitext(img)[0]
            shutil.copy(os.path.join(imgpath, img), os.path.join(imgout, img2 + '_2.jpg'))
    print('select images finish')

def selectEndwithDwg():
    dwgpath = r'E:\School\Grad1\CAD\MyCAD\dwgs\plans'
    dwgs = os.listdir(dwgpath)
    
    cnt, num = 0, len(dwgs)
    for dwg in dwgs:
        if not dwg.endswith('.dwg'):
            print('remove file:', dwg)
            os.remove(os.path.join(dwgpath, dwg))
            cnt += 1

    print('cnt: %d, num: %d' % (cnt, num))


if __name__ == '__main__':
    # copyRandom()
    doSelect1()

