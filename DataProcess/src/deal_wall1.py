import cv2
import numpy as np
import os
import random
import shutil
from PIL import Image

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

def grayToRgb(graypath, rgbpath):
    img_gray = imgReadGray(graypath)
    img_rbg = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    imgWrite(rgbpath, img_rbg)
    
def grayToRgbBatch():
    graypath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\CloseWallTest\CloseWall2\dataset2\JPEGImages'
    rgbpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\CloseWallTest\CloseWall2\dataset2\JPEGImages2'

    if not os.path.exists(graypath):
        print('gray path not exist: ', graypath)
        return
    os.makedirs(rgbpath, exist_ok=True)
    
    grays = os.listdir(graypath)
    num = len(grays)
    for i, gray in enumerate(grays):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        grayToRgb(os.path.join(graypath, gray), os.path.join(rgbpath, gray))
    print('gray to rgb batch finish')

def copyRandom():
    inpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\SegmentationClass'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset2\CloseWallTest\segments2'
    if not os.path.exists(inpath):
        print('inpath not exist')
    os.makedirs(outpath, exist_ok=True)

    num = 100
    datas = random.sample(os.listdir(inpath), num)
    for data in datas:
        shutil.copy(os.path.join(inpath, data), outpath)
    print('copy random finish:', num)

def getScale(imgname, labelpath, imgWidth=1600, imgHeight=1280):
    label = os.path.join(labelpath, os.path.splitext(imgname)[0] + '.txt')
    if not os.path.exists(label):
        print('label path not exist.', label)
        return None
    with open(label, 'r', encoding='utf-8') as f:
        content = f.readlines()[0].strip()
    box = [float(i) for i in content[7:].split(', ')]

    rangeWidth = box[2] - box[0]
    rangeHeight = box[3] - box[1]

    k1 = imgHeight * 1. / imgWidth
    k2 = rangeHeight * 1. / rangeWidth 
    scale = (rangeWidth * 1. / imgWidth) if k1 > k2 else (rangeHeight * 1. / imgHeight)
    # print('rangeWidth: %.2f, rangeHeight: %.2f' % (rangeWidth, rangeHeight))
    # print('k1: %.2f, k2: %.2f' % (k1, k2))

    return scale

def closeWall1():
    # imgpath = '../imgs/wall-gray1.png'
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\PdfScaleTest\data_scale_5\mask_line\(T3) 12#楼105户型平面图（镜像）-6.png'
    img = imgReadGray(imgpath)
    
    size = 100
    kernel = np.ones((size, size), np.uint8)     # origin is 40
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算结果
    # imgWrite('../res/wall-gray1_closing.jpg', closing)
    # 漫水填充
    # h, w = closing.shape[:2]
    # mask = np.zeros([h + 2, w + 2], np.uint8)  # 这里必须为 h+2,w+2
    # # line = np.zeros([h + 2, w + 2], np.uint8)
    # cv2.floodFill(closing, mask, (2, 2), (0, 0, 0), (100, 100, 100), (50, 50, 50),
    #             flags=4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)
    # opening = ~mask
    # imgWrite('../res/img_opening.jpg', opening)
    kernel = np.ones((5, 5), np.uint8)  # 去除外部游离线
    opening2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    imgWrite('../res/tmp3.png', opening2)
    print('--- finish ---')
    # cv2.imshow('img', img)
    # cv2.imshow('opening2', opening2)

    # cv2.waitKey()
    # cv2.destroyAllWindows()


def closeWall2(imgpath):
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset2\labels'
    
    imgname = os.path.basename(imgpath)
    scale = getScale(imgname, labelpath)
    if scale is None:
        return
    print('scale:', scale)
    k = 60 * 1. / 16      # origin is 20
    size = round(scale * k)
    if size % 2 == 0:
        size += 1
    print('size:', size)
    
    img = imgReadGray(imgpath)
    imgWrite('../res/img_0.jpg', img)
    kernel = np.ones((size, size), np.uint8)     
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算结果
    imgWrite('../res/img_closing.jpg', closing)
    kernel = np.ones((5, 5), np.uint8)  # 去除外部游离线
    opening2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    imgWrite('../res/img_opening2.jpg', opening2)
    print('----- finish -----')

def closeWall3(imgpath, outpath):
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\labels'
    
    imgname = os.path.basename(imgpath)
    scale = getScale(imgname, labelpath)
    if scale is None:
        print('get scale fail:', imgpath)
        return
    # print('scale:', scale)
    k = 40 * 1. / 16      # origin is 20
    size = round(scale * k)
    if size % 2 == 0:
        size += 1
    # print('size:', size)
    
    imgname = os.path.splitext(imgname)[0]
    img = imgReadGray(imgpath)
    # imgWrite(os.path.join(outpath, imgname + '.jpg'), img)
    kernel = np.ones((size, size), np.uint8)     
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算结果
    # imgWrite(os.path.join(outpath, imgname + '_closing.png'), closing)
    # imgWrite(os.path.join(outpath, imgname + '_seg.png'), closing)
    kernel = np.ones((5, 5), np.uint8)  # 去除外部游离线
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    imgWrite(os.path.join(outpath, imgname + '.png'), opening)

def closeWall4(imgpath, outpath, size=70):
    if not os.path.exists(imgpath) or not os.path.exists(outpath):
        print('path not exist.')
        return
    imgname = os.path.splitext(os.path.basename(imgpath))[0]
    img = imgReadGray(imgpath)
    kernel = np.ones((size, size), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)      # 闭运算，封闭墙体区域
    kernel = np.ones((5, 5), np.uint8)                        
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)   # 开运算，去除游离线
    imgWrite(os.path.join(outpath, imgname + '.png'), opening)

def closeWallBatch():
    inpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset31\mask_line'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset31\mask_area01'
    if not os.path.exists(inpath):
        print('inpath not exist:', inpath)
    os.makedirs(outpath, exist_ok=True)

    segs = os.listdir(inpath)
    num = len(segs)
    for i, seg in enumerate(segs):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        closeWall4(os.path.join(inpath, seg), outpath, size=55)
    print('finish')
    
def test():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset2\SegmentationClass'
    imgname1 = '01 1-6号住宅楼标准层A户型平面图-3.png'
    imgname2 = '(T3) 12#楼105户型平面图（镜像）-5.png'   # 这个墙比较薄
    imgname3 = '01 1-6号住宅楼标准层电梯间平面图-1.png'
    imgname4 = '02 碧云湾A户型平面图-9.png'             # 这个墙比较厚，还需计算绘图中墙体厚度？
    imgname5 = '01、C户型平面图-12.png'                # 有厚有薄，典型，40正好
    imgname6 = '02-PM.01_8-4.png'                    # 超厚
    closeWall2(os.path.join(imgpath, imgname6))

def test2():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\CloseWallTest\CloseWall2\dataset2\JPEGImages'
    imgname = '(T3) 12#楼105户型平面图（镜像）-2.jpg'
    img = imgRead(os.path.join(imgpath, imgname))
    print(img.shape)

def displayMask(imgpath, outpath):
    with Image.open(imgpath) as img:
        assert img.mode == 'L', "The input image must be in grayscale mode ('L')"
        # 2. 将非零像素值置为 255
        pixels = img.load()
        width, height = img.size
        for x in range(width):
            for y in range(height):
                if pixels[x, y] != 0:
                    pixels[x, y] = 255

        # 3. 保存更新后的灰度图
        img.save(outpath)

def displayMaskBatch(imgpath, outpath):
    if not os.path.exists(imgpath):
        print('img_path not exist')
        return
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    imgs = os.listdir(imgpath)
    for i, img in enumerate(imgs):
        displayMask(os.path.join(imgpath, img), os.path.join(outpath, img))
    print('display mask batch finish')
    

if __name__ == '__main__':
    # test()
    # copyRandom()
    closeWallBatch()
    # test2()
    # grayToRgbBatch()
    # closeWall1()

