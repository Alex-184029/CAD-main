# -- 数据增强，随机裁剪
import os
import cv2
import numpy as np
import shutil

from albumentations import (
    HorizontalFlip,
    Rotate,
    RandomScale,
    RandomCrop,
    Compose,
    OneOf,
)

# 定义数据增强管道
target_size = 640
augmentation_pipeline = Compose([
    HorizontalFlip(p=0.3),  # 30% 概率水平翻转
    OneOf([  # 50% 概率旋转
        Rotate(limit=(90, 90), p=0.5),  # 旋转 90°
        Rotate(limit=(180, 180), p=0.5),  # 旋转 180°
    ], p=0.5),
    RandomScale(scale_limit=(0.9, 1.1), p=0.3),  # 30% 概率随机缩放
    RandomCrop(height=target_size, width=target_size, always_apply=True)  # 随机裁剪到 640x640
])

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

# 数据增强函数
def augment_image_and_mask(image, mask, pipeline):
    augmented = pipeline(image=image, mask=mask)
    return augmented['image'], augmented['mask']

def augment_batch():
    # 输入和输出路径
    input_images_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\images"
    input_masks_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\masks2"
    output_images_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug1\data-aug5\images"
    output_masks_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug1\data-aug5\masks"

    if not os.path.exists(input_images_dir) or not os.path.exists(input_masks_dir):
        print('Input directories not found.')
        return
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # 处理所有图片和掩码
    imgs = os.listdir(input_images_dir)
    num = len(imgs)
    for i, image_filename in enumerate(imgs):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        # 加载图片和掩码
        image_path = os.path.join(input_images_dir, image_filename)
        mask_path = os.path.join(input_masks_dir, image_filename)
        image = imgRead(image_path)
        mask = imgReadGray(mask_path)

        h, w, _ = image.shape
        if h < target_size or w < target_size:    # 尺寸过小则先放大
            # print('image %s is to small, size: (%d, %d)' % (image_filename, w, h))
            scale = 1.
            if w < h:
                scale = target_size / w
            else:
                scale = target_size / h
            w_new = int(w * scale) + 10
            h_new = int(w * scale) + 10
            image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_AREA)

        # 执行数据增强
        augmented_image, augmented_mask = augment_image_and_mask(image, mask, augmentation_pipeline)

        # 保存增强后的图片和掩码
        output_image_path = os.path.join(output_images_dir, image_filename)
        output_mask_path = os.path.join(output_masks_dir, image_filename)
        imgWrite(output_image_path, augmented_image)
        imgWrite(output_mask_path, augmented_mask)

    print('----- finish -----')

def is_image_all_zeros(input_image_path):
    # 读取输入图片为灰度图像
    image = imgReadGray(input_image_path)
    
    if image is None:
        print("Error: Could not open or find the image.")
        return False
    
    # 判断图像中的所有像素值是否全部为0
    if cv2.countNonZero(image) == 0:
        return True
    else:
        return False

def select_mask():
    image_in = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug1\images'
    mask_in = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug1\masks'
    image_out = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug2\images'
    mask_out = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug2\masks'

    if not os.path.exists(image_in) or not os.path.exists(mask_in):
        print('Input directories not found.')
        return
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    masks = os.listdir(mask_in)
    num = len(masks)
    cnt = 0
    for i, mask in enumerate(masks):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        mask_path = os.path.join(mask_in, mask)
        if not is_image_all_zeros(mask_path):
            shutil.move(mask_path, mask_out)
            shutil.move(os.path.join(image_in, mask), image_out)
            cnt += 1
    
    print('total: %d, valid: %d, all zero %d' % (num, cnt, num - cnt))

def png_to_jpg():
    input_dir = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug1\VOCdevkit7\images'
    output_dir = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug1\VOCdevkit7\JPEGImages'
    os.makedirs(output_dir, exist_ok=True)

    imgs = os.listdir(input_dir)
    num = len(imgs)
    for i, img in enumerate(imgs):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        im = imgRead(os.path.join(input_dir, img))
        imgout = os.path.splitext(img)[0] + '.jpg'
        imgWrite(os.path.join(output_dir, imgout), im)

    print('----- finish -----')


if __name__ == '__main__':
    # augment_batch()
    # select_mask()
    png_to_jpg()
