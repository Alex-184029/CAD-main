# -- 数据增强
import random
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import albumentations as A
import shutil

BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 255, 0)

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)

def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=1):
    """Visualizes a single bounding box on the image"""
    h, w, _ = img.shape
    x_center, y_center, width, height = bbox[:-1]
    x_center *= w
    y_center *= h
    width *= w
    height *= h
    x_min, x_max, y_min, y_max = int(x_center - width / 2), int(x_center + width / 2), int(y_center - height / 2), int(y_center + height / 2)
    print(x_min, y_min, x_max, y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(bbox[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=bbox[-1],
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes):
    img = image.copy()
    for bbox in bboxes:
        img = visualize_bbox(img, bbox)
    
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def doTransform():
    transform = A.Compose(
        [A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2)],   # 随机裁剪为448 * 448，但保证标注框安全
        bbox_params=A.BboxParams(format='yolo'),
    )
    random.seed(10)
    im, bboxes = getData()

    transformed = transform(image=im, bboxes=bboxes)
    visualize(
        transformed['image'],
        transformed['bboxes'],
    )

def doTransform2(image, bboxes):
    random.seed()
    np.random.seed()
    scale = random.random() * 0.2
    transform = A.Compose([
        A.RandomCrop(height=640, width=640, p=1),
        A.HorizontalFlip(p=0.3),  # 随机水平翻转概率为50%
        A.Rotate(limit=(90, 90), p=0.3),
        A.Rotate(limit=(45, 45), p=0.3),
        # A.Rotate(limit=(-20, 20), p=0.2),
        A.RandomScale(scale_limit=(scale, scale), p=0.3),

        # A.RandomSizedBBoxSafeCrop(height=680, width=680, p=1),
        # a.elastictransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        # 色调
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        # A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, always_apply=False, p=0.3),
        # 噪声
        # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5)
        # 添加其他需要的操作
    ], bbox_params=A.BboxParams(format='yolo'))

    transformed = transform(image=image, bboxes=bboxes)
    return transformed['image'], transformed['bboxes']

def doTransform3(image, bboxes):
    random.seed()
    np.random.seed()
    size_w, size_h = 640, 640
    transform = A.Compose([
        A.RandomCrop(height=size_h, width=size_w, p=1),
        A.HorizontalFlip(p=0.3),  # 随机水平翻转概率为50%
        A.Rotate(limit=(90, 90), p=0.3),
        A.Rotate(limit=(180, 180), p=0.3),
        A.Rotate(limit=(270, 270), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo'))

    transformed = transform(image=image, bboxes=bboxes)
    return transformed['image'], transformed['bboxes']

def getData(imgpath, labelpath):
    im = imgRead(imgpath)
    with open(labelpath, 'r', encoding='utf-8') as f:
        label = f.readlines()
    label = [list(map(float, i.strip().split(' '))) for i in label]
    bboxes = [[l[1], l[2], l[3], l[4], str(int(l[0]))] for l in label]

    return im, bboxes

def doTransBatch():
    imgspath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\images'
    labelspath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\labels_yolo'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug6'
    out_images = os.path.join(outpath, 'images')
    out_labels = os.path.join(outpath, 'labels')

    if not os.path.exists(imgspath) or not os.path.exists(labelspath):
        print('input path not exist')
        return
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    suffix = '-aug8'

    imgnames = os.listdir(imgspath)
    cnt_valid = 0
    total = len(imgnames)
    for i, imgname in enumerate(imgnames):
        if i % 50 == 0:
            print('%d for %d is doing, %s' % (i, total, imgname))
        imgpath = os.path.join(imgspath, imgname)
        labelpath = os.path.join(labelspath, os.path.splitext(imgname)[0] + '.txt')
        img, label = getData(imgpath, labelpath)
        transformed_img, transformed_label = doTransform3(img, label)
        if len(transformed_label) == 0:
            continue
        imgWrite(os.path.join(out_images, os.path.splitext(imgname)[0] + suffix + '.jpg'), transformed_img)
        with open(os.path.join(out_labels, os.path.splitext(imgname)[0] + suffix + '.txt'), 'w', encoding='utf-8') as f:
            for label in transformed_label:
                label2 = [float(i) for i in label[:-1]]
                if any(i <= 0 for i in label2) or any(i >= 1 for i in label2):   # 超范围标注框
                    continue
                f.write('%s %.5f %.5f %.5f %.5f\n' % (label[4], label[0], label[1], label[2], label[3]))
        cnt_valid += 1
        # print('Err: %d, %s' % (i, imgname))

    print('total: %d, valid: %d' % (total, cnt_valid))

def doSelect1():
    inpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug2'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug1'
    img_in = os.path.join(inpath, 'images')
    label_in = os.path.join(inpath, 'labels')
    img_out = os.path.join(outpath, 'images')
    label_out = os.path.join(outpath, 'labels')

    imgs = os.listdir(img_in)
    total = len(imgs)
    for i, img in enumerate(imgs):
        label = img.replace('.jpg', '.txt') 
        shutil.move(os.path.join(img_in, img), img_out)
        shutil.move(os.path.join(label_in, label), label_out)
    print('finish')


def main():
    # doTransBatch()
    doSelect1()


if __name__ == '__main__':
    main()