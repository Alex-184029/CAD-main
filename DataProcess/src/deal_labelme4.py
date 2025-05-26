# -- labelme格式数据增强，不随机裁剪
import albumentations as A
import cv2
import os
import json
from glob import glob
import numpy as np
import shutil

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)

# 定义增强管道
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.7),  # 水平翻转
        A.OneOf(
            [
                A.Rotate(limit=(90, 90), p=0.7),  # 概率旋转
                A.Rotate(limit=(180, 180), p=0.7),  
                A.Rotate(limit=(270, 270), p=0.7),  
            ],
            p=1.0,
        ),
        # A.RandomScale(scale_limit=(0.1, 0.1), p=0.3),  # 概率随机缩放，0.9到1.1比例，会改变图像尺寸
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

# 处理单个图像和对应的标注文件
def augment_labelme(image_path, labelme_json_path, output_dir):
    # 加载图像
    image = imgRead(image_path)
    # height, width = image.shape[:2]

    # 加载 LabelMe 标注
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        labelme_data = json.load(f)

    shapes = labelme_data['shapes']
    keypoints = []
    shape_labels = []
    for shape in shapes:
        if shape['shape_type'] == 'polygon':
            keypoints.extend(shape['points'])
            shape_labels.append((len(shape['points']), shape['label']))

    # 应用增强
    transformed = transform(image=image, keypoints=keypoints)
    aug_image = transformed['image']
    aug_keypoints = transformed['keypoints']

    aug_keypoints = [[int(round(x)), int(round(y))] for x, y in aug_keypoints]

    # 转换增强后的关键点为多边形格式
    augmented_shapes = []
    idx = 0
    for num_points, label in shape_labels:
        points = aug_keypoints[idx: idx + num_points]
        augmented_shapes.append({
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })
        idx += num_points

    # 保存增强后的图像
    suffix = '-aug4'
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    augmented_image_path = os.path.join(output_dir, 'images', image_name + suffix + '.png')
    imgWrite(augmented_image_path, aug_image)

    # 保存增强后的标注
    labelme_data['shapes'] = augmented_shapes
    augmented_json_path = os.path.join(output_dir, 'labels', image_name + suffix + '.json')
    labelme_data['imagePath'] = os.path.join('..\\images\\', image_name + suffix + '.png')
    with open(augmented_json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, indent=2)

# 批量处理数据集
def augment_dataset(image_dir, labelme_dir, output_dir):
    if not os.path.exists(image_dir) or not os.path.exists(labelme_dir):
        print('image or labelme dir not exist')
        return
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    image_paths = glob(os.path.join(image_dir, "*.png"))  # 根据你的图片格式调整扩展名
    num = len(image_paths)
    for i, image_path in enumerate(image_paths):
        if i % 100 == 0:
            print('%d / %d' % (i, num))
        labelme_json_path = os.path.join(labelme_dir, os.path.basename(image_path).replace('.png', '.json'))
        if os.path.exists(labelme_json_path):
            augment_labelme(image_path, labelme_json_path, output_dir)
        else:
            print('labelme json not exist:', labelme_json_path)

def doSelect():
    inpath = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-resize"
    outpath = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-aug\data-aug1"

    imgpath = os.path.join(inpath, 'images')
    labelpath = os.path.join(inpath, 'labels')
    imgs = os.listdir(imgpath)

    for img in imgs:
        label = img.replace('.png', '.json')
        # label = img    # 掩码格式
        shutil.copy(os.path.join(imgpath, img), os.path.join(outpath, 'images'))
        shutil.copy(os.path.join(labelpath, label), os.path.join(outpath, 'labels'))
    print('----- finish -----')

def doSelect2():
    inpath = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug1\data-aug5"
    outpath = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\data-aug1\data-aug1"

    imgpath = os.path.join(inpath, 'images')
    labelpath = os.path.join(inpath, 'masks')
    imgs = os.listdir(imgpath)

    for img in imgs:
        # label = img.replace('.png', '.json')
        label = img    # 掩码格式
        img_new = os.path.splitext(img)[0] + '-aug5.png'
        label_new = img_new
        shutil.move(os.path.join(imgpath, img), os.path.join(outpath, 'images', img_new))
        shutil.move(os.path.join(labelpath, label), os.path.join(outpath, 'masks', label_new))

    print('----- finish -----')

def test_augment_dataset():
    image_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-resize\images"  
    labelme_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-resize\labels"
    output_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-aug\data-aug4"

    augment_dataset(image_dir, labelme_dir, output_dir)
    

if __name__ == '__main__':
    # test_augment_dataset()
    # doSelect2()
    doSelect()
    
