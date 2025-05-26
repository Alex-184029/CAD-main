# -- labelme格式处理：txt到labelme转换、可视化、转掩膜
import os
import json
import cv2
import numpy as np
from PIL import Image
import shutil

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)

def parseResult(inpath):
    if not os.path.exists(inpath):
        print('inpath not exists')
        return None
    with open(inpath, 'r', encoding='utf-8') as f:
        content = f.readlines()
    try:
        content = [con.strip() for con in content]
        box = [float(i) for i in content[0][7:].split(', ')]    # 这里的类型转换可能没必要，传入前端又会变成字串
        lines = []
        # print(type(content[2]), content[2])
        index = content[2].find(': ') + 2
        # print('index:', index)
        if index == 1:     # 没找到:
            print('Parse colon info failed.')
            return None
        for i in range(2, len(content)):
            index = content[i].find(': ') + 2
            arr = [float(i) for i in content[i][index:].split(', ')]
            line = {'x1': arr[0], 'y1': arr[1], 'x2': arr[2], 'y2': arr[3]}
            lines.append(line)
        return {'box': box, 'total': len(lines), 'lines': lines}
    except:
        return None

def parseResultCeiling(inpath):
    if not os.path.exists(inpath):
        print('inpath not exists')
        return None
    with open(inpath, 'r', encoding='utf-8') as f:
        content = f.readlines()
    try:
        content = [con.strip() for con in content]
        box = [float(i) for i in content[1][7:].split(', ')]    # 这里的类型转换可能没必要，传入前端又会变成字串
        rects = []
        for i in range(3, len(content)):
            index = content[i].find(': ')
            label = content[i][:index]
            index += 2
            arr = [float(i) for i in content[i][index:].split(', ')]
            rect = {'label': label, 'x1': arr[0], 'y1': arr[1], 'x2': arr[2], 'y2': arr[3]}
            rects.append(rect)
        return {'box': box, 'total': len(rects), 'rects': rects}
    except:
        return None

def parseResultArcDoor(inpath):
    if not os.path.exists(inpath):
        print('inpath not exists')
        return None
    with open(inpath, 'r', encoding='utf-8') as f:
        content = f.readlines()
    try:
        content = [con.strip() for con in content]
        box = [float(i) for i in content[0][7:].split(', ')]    # 这里的类型转换可能没必要，传入前端又会变成字串
        arcs = []
        # print(type(content[2]), content[2])
        index = content[2].find(': ') + 2
        # print('index:', index)
        if index == 1:     # 没找到:
            print('Parse colon info failed.')
            return None
        for i in range(2, len(content)):
            index = content[i].find(': ') + 2
            arr = [float(i) for i in content[i][index:].split(', ')]
            # 后四个记录圆弧端点坐标，也需要转换
            line = {'x1': arr[0], 'y1': arr[1], 'x2': arr[2], 'y2': arr[3], 'xx1': arr[4], 'yy1': arr[5], 'xx2': arr[6], 'yy2': arr[7]}
            arcs.append(line)
        return {'box': box, 'total': len(arcs), 'arcs': arcs}
    except:
        return None

def doMapRange(data, imgWidth=1600, imgHeight=1280):
    if (data['box'] is None or len(data['box']) != 4 or len(data['lines']) == 0):
        return
    imgCenterX = imgWidth / 2
    imgCenterY = imgHeight / 2
    rangeWidth = data['box'][2] - data['box'][0]
    rangeHeight = data['box'][3] - data['box'][1]
    rangeCenterX = (data['box'][2] + data['box'][0]) / 2
    rangeCenterY = (data['box'][3] + data['box'][1]) / 2

    k1 = imgHeight * 1. / imgWidth
    k2 = rangeHeight * 1. / rangeWidth 
    scale = (imgWidth * 1. / rangeWidth) if k1 > k2 else (imgHeight * 1. / rangeHeight)
    
    lines_int = []
    for i in range(data['total']):
        line = data['lines'][i]
        x1 = round(imgCenterX + (line['x1'] - rangeCenterX) * scale)
        y1 = imgHeight - round(imgCenterY + (line['y1'] - rangeCenterY) * scale)
        x2 = round(imgCenterX + (line['x2'] - rangeCenterX) * scale)
        y2 = imgHeight - round(imgCenterY + (line['y2'] - rangeCenterY) * scale)
        lines_int.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    data['lines'] = lines_int

def doMapRangeCeiling(data, imgWidth=1600, imgHeight=1280):
    if (data['box'] is None or len(data['box']) != 4 or len(data['rects']) == 0):
        return
    imgCenterX = imgWidth / 2
    imgCenterY = imgHeight / 2
    rangeWidth = data['box'][2] - data['box'][0]
    rangeHeight = data['box'][3] - data['box'][1]
    rangeCenterX = (data['box'][2] + data['box'][0]) / 2
    rangeCenterY = (data['box'][3] + data['box'][1]) / 2

    k1 = imgHeight * 1. / imgWidth
    k2 = rangeHeight * 1. / rangeWidth 
    scale = (imgWidth * 1. / rangeWidth) if k1 > k2 else (imgHeight * 1. / rangeHeight)
    
    rects_int = []
    for i in range(data['total']):
        rect = data['rects'][i]
        x1 = round(imgCenterX + (rect['x1'] - rangeCenterX) * scale)
        y1 = imgHeight - round(imgCenterY + (rect['y1'] - rangeCenterY) * scale)
        x2 = round(imgCenterX + (rect['x2'] - rangeCenterX) * scale)
        y2 = imgHeight - round(imgCenterY + (rect['y2'] - rangeCenterY) * scale)
        rects_int.append({'label': rect['label'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    data['rects'] = rects_int

def doMapRangeArcDoor(data, imgWidth=1600, imgHeight=1280):
    if (data['box'] is None or len(data['box']) != 4 or len(data['arcs']) == 0):
        return
    imgCenterX = imgWidth / 2
    imgCenterY = imgHeight / 2
    rangeWidth = data['box'][2] - data['box'][0]
    rangeHeight = data['box'][3] - data['box'][1]
    rangeCenterX = (data['box'][2] + data['box'][0]) / 2
    rangeCenterY = (data['box'][3] + data['box'][1]) / 2

    k1 = imgHeight * 1. / imgWidth
    k2 = rangeHeight * 1. / rangeWidth 
    scale = (imgWidth * 1. / rangeWidth) if k1 > k2 else (imgHeight * 1. / rangeHeight)
    
    arcs_int = []
    for i in range(data['total']):
        arc = data['arcs'][i]
        x1 = round(imgCenterX + (arc['x1'] - rangeCenterX) * scale)
        y1 = imgHeight - round(imgCenterY + (arc['y1'] - rangeCenterY) * scale)
        x2 = round(imgCenterX + (arc['x2'] - rangeCenterX) * scale)
        y2 = imgHeight - round(imgCenterY + (arc['y2'] - rangeCenterY) * scale)
        xx1 = round(imgCenterX + (arc['xx1'] - rangeCenterX) * scale)
        yy1 = imgHeight - round(imgCenterY + (arc['yy1'] - rangeCenterY) * scale)
        xx2 = round(imgCenterX + (arc['xx2'] - rangeCenterX) * scale)
        yy2 = imgHeight - round(imgCenterY + (arc['yy2'] - rangeCenterY) * scale)
        arcs_int.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'xx1': xx1, 'yy1': yy1, 'xx2': xx2, 'yy2': yy2})

    data['arcs'] = arcs_int

def createLabelmeJson(data, outpath, imgWidth=1600, imgHeight=1280, label='WallLine1'):
    if data is None or data['total'] == 0:
        print('data error for:', outpath)
        return
    
    shapes = []
    for line in data['lines']:
        shape = {
            'label': label, 
            'points': [[line['x1'], line['y1']], [line['x2'], line['y2']]],
            "group_id": None,
            "description": "",
            "shape_type": "line",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
    
    imgname = os.path.splitext(os.path.basename(outpath))[0]
    suffix = '.jpg'

    # 创建 LabelMe 标注数据结构
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": f'../images/{imgname}' + suffix,
        "imageData": None,
        "imageHeight": imgHeight,
        "imageWidth": imgWidth
    }

    # 将数据序列化为 JSON
    with open(outpath, 'w', encoding='utf-8') as json_file:
        json.dump(labelme_data, json_file, indent=2)

    # print(f"LabelMe JSON 文件已生成: {outpath}")

def createLabelmeJsonRect(data, outpath, imgWidth=1600, imgHeight=1280, label='ArcDoor1'):
    if data is None or data['total'] == 0:
        print('data error for:', outpath)
        return
    
    shapes = []
    print('rect:', len(data['lines']), type(data['lines']), data['lines'][0])
    for rect in data['lines']:
        x1, y1, x2, y2 = rect['x1'], rect['y1'], rect['x2'], rect['y2']
        shape = {
            'label': label, 
            'points': [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
    
    imgname = os.path.splitext(os.path.basename(outpath))[0]
    suffix = '.jpg'

    # 创建 LabelMe 标注数据结构
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": f'../images/{imgname}' + suffix,
        "imageData": None,
        "imageHeight": imgHeight,
        "imageWidth": imgWidth
    }

    # 将数据序列化为 JSON
    with open(outpath, 'w', encoding='utf-8') as json_file:
        json.dump(labelme_data, json_file, indent=2)

    # print(f"LabelMe JSON 文件已生成: {outpath}")

def createLabelmeJsonCeiling(data, outpath, imgWidth=1600, imgHeight=1280):
    if data is None or data['total'] == 0:
        print('data error for:', outpath)
        return
    
    shapes = []
    print('rect:', len(data['rects']), type(data['rects']), data['rects'][0])
    for rect in data['rects']:
        label, x1, y1, x2, y2 = rect['label'], rect['x1'], rect['y1'], rect['x2'], rect['y2']
        shape = {
            'label': label, 
            'points': [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
    
    imgname = os.path.splitext(os.path.basename(outpath))[0]
    suffix = '.jpg'

    # 创建 LabelMe 标注数据结构
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": f'../images/{imgname}' + suffix,
        "imageData": None,
        "imageHeight": imgHeight,
        "imageWidth": imgWidth
    }

    # 将数据序列化为 JSON
    with open(outpath, 'w', encoding='utf-8') as json_file:
        json.dump(labelme_data, json_file, indent=2)

    # print(f"LabelMe JSON 文件已生成: {outpath}")

def createLabelmeJsonArcDoor(data, outpath, imgWidth=1600, imgHeight=1280, label='ArcDoor1'):
    if data is None or data['total'] == 0:
        print('data error for:', outpath)
        return
    shapes = []
    print('arcs:', len(data['arcs']), type(data['arcs']))
    for arc in data['arcs']:
        x1, y1, x2, y2 = arc['x1'], arc['y1'], arc['x2'], arc['y2']
        xx1, yy1, xx2, yy2 = arc['xx1'], arc['yy1'], arc['xx2'], arc['yy2']
        shape = {
            'label': label, 
            'points': [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            "group_id": None,
            "description": "%d %d %d %d" % (xx1, yy1, xx2, yy2),
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
    
    imgname = os.path.splitext(os.path.basename(outpath))[0]
    suffix = '.jpg'

    # 创建 LabelMe 标注数据结构
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": f'../images/{imgname}' + suffix,
        "imageData": None,
        "imageHeight": imgHeight,
        "imageWidth": imgWidth
    }

    # 将数据序列化为 JSON
    with open(outpath, 'w', encoding='utf-8') as json_file:
        json.dump(labelme_data, json_file, indent=2)

    # print(f"LabelMe JSON 文件已生成: {outpath}")

def txtToJson(inpath, outpath, imgWidth=1600, imgHeight=1280, label='WallLine1'):
    data = parseResult(inpath)
    if data is None:
        print('data error for:', outpath)
        return
    doMapRange(data, imgWidth, imgHeight)
    createLabelmeJson(data, outpath, imgWidth, imgHeight, label)

def txtToJsonRect(inpath, outpath, imgWidth=1600, imgHeight=1280, label='ArcDoor1'):
    data = parseResult(inpath)
    if data is None:
        print('data error for:', outpath)
        return
    doMapRange(data, imgWidth, imgHeight)
    createLabelmeJsonRect(data, outpath, imgWidth, imgHeight, label=label)

def txtToJsonArcDoor(inpath, outpath, imgWidth=1600, imgHeight=1280, label='ArcDoor1'):
    data = parseResultArcDoor(inpath)
    if data is None:
        print('data error for:', outpath)
        return
    doMapRangeArcDoor(data, imgWidth, imgHeight)
    createLabelmeJsonArcDoor(data, outpath, imgWidth, imgHeight, label=label)

def txtToJsonCeiling(inpath, outpath, imgWidth=1600, imgHeight=1280):
    data = parseResultCeiling(inpath)
    if data is None:
        print('data error for:', outpath)
        return
    doMapRangeCeiling(data, imgWidth, imgHeight)
    createLabelmeJsonCeiling(data, outpath, imgWidth, imgHeight)
    
def txtToJsonBatch():
    inpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\Tests\labels'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\Tests\labels_json'
    os.makedirs(outpath, exist_ok=True)

    txts = os.listdir(inpath)
    total = len(txts)
    for i, txt in enumerate(txts):
        if i % 200 == 0:
            print('%d / %d' % (i, total))
        txtpath = os.path.join(inpath, txt)
        jsonpath = os.path.join(outpath, txt.replace('.txt', '.json'))
        txtToJson(txtpath, jsonpath)

    print('finish')

def txtToJsonBatch2():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset2-pdf\images'
    inpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset2-pdf\labels'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset2-pdf\labels_line'
    if not os.path.exists(imgpath) or not os.path.exists(inpath):
        print('input path not exist')
    os.makedirs(outpath, exist_ok=True)

    imgs = os.listdir(imgpath)
    num = len(imgs)
    for i, img in enumerate(imgs):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        imgname = os.path.splitext(img)[0]
        txtpath = os.path.join(inpath, imgname + '.txt')
        jsonpath = os.path.join(outpath, imgname + '.json')
        im = imgRead(os.path.join(imgpath, img))
        imgHeight, imgWidth, _ = im.shape
        txtToJson(txtpath, jsonpath, imgWidth, imgHeight)

    print('Txt to json batch finish.')


def visualizeSingle(jsonpath, imgout, color=(0, 0, 255), thickness=2):    # 默认红色，厚度为2
    if not os.path.exists(jsonpath):
        print('json path not exist')
        return
    with open(jsonpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    imgpath = os.path.join(os.path.dirname(jsonpath), data['imagePath'])
    # print('imgpath:', imgpath)
    shapes = data['shapes']
    # 开始绘制
    im = imgRead(imgpath)
    for shape in shapes:
        cv2.line(im, (shape['points'][0][0], shape['points'][0][1]), (shape['points'][1][0], shape['points'][1][1]), color, thickness)
    imgWrite(imgout, im)

def visualizeBatch():
    inpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsParallelWindow\yolo3\WallLineData\dataset1\labels_json'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsParallelWindow\yolo3\WallLineData\dataset1\visualize1\images'
    if not os.path.exists(inpath):
        print('inpath not exist: ', inpath)
        return
    os.makedirs(outpath, exist_ok=True)

    jsons = os.listdir(inpath)
    num = len(jsons)
    for i, json in enumerate(jsons):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        jsonpath = os.path.join(inpath, json)
        imgout = os.path.join(outpath, json.replace('.json', '.jpg'))
        visualizeSingle(jsonpath, imgout)
    print('finish')

def createWallMask(json_path, out_path):
    if not os.path.exists(json_path):
        print('json path not exist')
        return
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = data['shapes']
    if len(shapes) == 0:
        return

    w = data['imageWidth']
    h = data['imageHeight']
    image = np.zeros((h, w), dtype=np.uint8)
    for shape in shapes:
        cv2.line(image, (shape['points'][0][0], shape['points'][0][1]), (shape['points'][1][0], shape['points'][1][1]), 255, 1)     # 255更易观察到，实际应该为1

    png_path = os.path.join(out_path, os.path.splitext(os.path.basename(json_path))[0] + '.png')
    imgWrite(png_path, image)

def createWallMaskBatch():
    json_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset31\labels_line'
    out_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset31\mask_line'
    if not os.path.exists(json_path):
        print('label path not exist')
        return
    os.makedirs(out_path, exist_ok=True)

    jsons = os.listdir(json_path)
    total = len(jsons)
    for i, json in enumerate(jsons):
        if i % 200 == 0:
            print('%d for %d is doing' % (i, total))
        createWallMask(os.path.join(json_path, json), out_path)
    print('------ finish ------')

def displayMask(imgpath, outpath):
    with Image.open(imgpath) as img:
        assert img.mode == 'L', "The input image must be in grayscale mode ('L')"
        # 将非零像素值置为 255
        pixels = img.load()
        width, height = img.size
        for x in range(width):
            for y in range(height):
                if pixels[x, y] != 0:
                    pixels[x, y] = 1     # origin is 255

        # 保存更新后的灰度图
        img.save(outpath)

def displayMaskBatch(imgpath, outpath):
    if not os.path.exists(imgpath):
        print('img_path not exist')
        return
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    imgs = os.listdir(imgpath)
    num = len(imgs)
    for i, img in enumerate(imgs):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        displayMask(os.path.join(imgpath, img), os.path.join(outpath, img))
    print('display mask batch finish')

def isLabelRight(label: str) -> bool:
    if label is None or not os.path.exists(label):
        return False
    with open(label, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data is None:
        return False
    shapes = data['shapes']
    if shapes is None or len(shapes) == 0:
        return False
    if shapes[-1].get('label') != 'WallArea1':
        return False
    return True

def doSelect1():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\images'
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\labels_slide_door5'
    imgout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\images'
    labelout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\labels'
    os.makedirs(imgout, exist_ok=True)
    os.makedirs(labelout, exist_ok=True)

    labels = os.listdir(labelpath)
    num, cnt = len(labels), 0
    for i, label in enumerate(labels):
        if i % 50 == 0:
            print('%d / %d' % (i, num))
        labelname = os.path.splitext(label)[0]
        with open(os.path.join(labelpath, label), 'r', encoding='utf-8') as f:
            data = json.load(f)
        shapes = data['shapes']
        if len(shapes) == 0 or shapes[-1]['label'] == 'Delete':
            cnt += 1
            continue
        shutil.copy(os.path.join(labelpath, label), labelout)
        shutil.copy(os.path.join(imgpath, labelname + '.jpg'), imgout)
    print('num:', num, ", delete cnt:", cnt)

def doSelect2():
    labelpath = r'C:\Users\DELL\Desktop\test1\labels'
    imgout = r'C:\Users\DELL\Desktop\test1\images'
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug-ori\images'
    os.makedirs(imgout, exist_ok=True)

    labels = os.listdir(labelpath)
    for label in labels:
        labelname = os.path.splitext(label)[0]
        shutil.copy(os.path.join(imgpath, labelname + '.jpg'), imgout)
    print('finish')

def test():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\images'
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\labels_line'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\images-err'
    os.makedirs(outpath, exist_ok=True)
    imgs = [os.path.splitext(img)[0] for img in os.listdir(imgpath)]
    labels = [os.path.splitext(label)[0] for label in os.listdir(labelpath)]
    imgs_err = [img for img in imgs if not img in labels]

    print('error num:', len(imgs_err))
    for img in imgs_err:
        shutil.move(os.path.join(imgpath, img + '.jpg'), outpath)
    print('finish')

def test2():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\PdfScaleTest\data_scale_5\images'
    imgname = os.listdir(imgpath)[0]
    print('imgname:', imgname)
    im = imgRead(os.path.join(imgpath, imgname))
    h, w, x = im.shape
    print(w, h, x)

def test3():
   inpath = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\DataProcess\ParseDoorLine\data\labels_Balcony\01 1-6号住宅楼标准层A户型平面图-2_DoorLine.txt' 
   outpath = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\DataProcess\ParseDoorLine\data\labels_Balcony\01 1-6号住宅楼标准层A户型平面图-2_DoorLine.json' 
   imgpath = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\DataProcess\ParseDoorLine\data\images\01 1-6号住宅楼标准层A户型平面图-2.jpg'
   im = imgRead(imgpath)
   h, w, _ = im.shape
   txtToJson(inpath, outpath, w, h, label='DoorLine')
   # txtToJsonArcDoor(inpath, outpath, w, h)
   # txtToJsonRect(inpath, outpath, w, h, label='ParallelWindow')

def test4():
    inpath = r'C:\Users\DELL\Desktop\tmp1.txt'
    outpath = r'C:\Users\DELL\Desktop\tmp1.json'
    imgpath = r'C:\Users\DELL\Desktop\tmp1.jpg'

    im = imgRead(imgpath)
    h, w, _ = im.shape
    txtToJsonCeiling(inpath, outpath, w, h)


if __name__ == '__main__':
    # createWallMaskBatch()
    # txtToJsonBatch2()
    # test()
    # doSelect2()
    # test3()
    test4()
