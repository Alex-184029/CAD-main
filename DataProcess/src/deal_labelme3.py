# -- 从labelme构建VOCdevkit掩膜标注数据集
import argparse
import glob
import os
import os.path as osp
import imgviz
import numpy as np
import labelme
import base64
from PIL import Image 
import cv2
import json

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)
 
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default=r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-aug\data-aug1\labels", help="input annotated directory") #json路径
    parser.add_argument("--output_dir", default=r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-aug\VOCdevkit8", help="output dataset directory")  #输出地址
    parser.add_argument("--labels", default=r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\resize_data_640\labels.txt", help="labels file")  #标签txt
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()
 
    # if osp.exists(args.output_dir):
    #     print("Output directory already exists:", args.output_dir)
    #     sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationClassVisualization")
        )
    print("Creating dataset:", args.output_dir)
 
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        # class_id = i - 1  # starts with -1
        class_id = i
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            # assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w", encoding='utf-8') as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)
 
    num = len(os.listdir(args.input_dir))
    for i, filename in enumerate(glob.glob(osp.join(args.input_dir, "*.json"))):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
 
        label_file = labelme.LabelFile(filename=filename)
 
        # base = osp.splitext(osp.basename(filename))[0]    # 图像原始名称
        base = 'img_' + str(i + 1)                                   # 数字字串名称（应对非中文路径）
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_png_file = osp.join(args.output_dir, "SegmentationClass", base + ".png")
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )
 
        imageData = label_file.imageData
        if imageData is None:
            imagePath = osp.join(args.input_dir, label_file["imagePath"])
            try:
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")
            except FileNotFoundError:
                print(f"File not found: {imagePath}")
                continue  # Skip to the next JSON file
            except Exception as e:
                print(f"Error reading image file {imagePath}: {e}")
                continue
                        
        with open(out_img_file, "wb") as f:
            f.write(imageData)
        img = labelme.utils.img_data_to_arr(imageData)
 
        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        # labelme.utils.lblsave(out_png_file, lbl)

        binary_mask = (lbl > 0).astype(np.uint8)
        mask_image = Image.fromarray(binary_mask * 1)            # 目标区域为1，训练常用
        # mask_image = PIL.Image.fromarray(binary_mask * 255)    # 目标区域为255
        mask_image.save(out_png_file)
 
        if not args.noviz:
            viz = imgviz.label2rgb(
                label=lbl,
                #img改成image，labelme接口的问题不然会报错
                #img=imgviz.rgb2gray(img),
                image=imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            imgviz.io.imsave(out_viz_file, viz)

def is_binary_image(image):
    """
    判断一个灰度图像是否是二值图。

    参数:
    image (numpy.ndarray): 输入的灰度图像。

    返回:
    bool: 如果图像是二值图，返回True；否则返回False。
    """
    # 检查图像是否为单通道
    # if len(image.shape) != 2:
    #     return False
    image = np.array(image)
    
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

def json_to_mask(json_path, mask_path):
    label_file = labelme.LabelFile(filename=json_path)
    class_name_to_id = {'_background_': 0, 'SlideDoor': 1, 'ArcDoor': 2, 'Window1': 3, 'WallArea1': 4}

    tmp_path = os.path.dirname(json_path)
    img_path = os.path.join(tmp_path, label_file.imagePath)
    try:
        with open(img_path, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    except FileNotFoundError:
        print(f"File not found: {img_path}")
    except Exception as e:
        print(f"Error reading image file {img_path}: {e}")

    # img = labelme.utils.img_data_to_arr(imageData)
    img = imgRead(img_path)
    # img = Image.open(img_path)
    lbl, _ = labelme.utils.shapes_to_label(
        img_shape=img.shape,
        shapes=label_file.shapes,
        label_name_to_value=class_name_to_id,
    )
    binary_mask = (lbl > 0).astype(np.uint8)
    mask_image = Image.fromarray(binary_mask * 255)            # 255方便观察，1训练常用
    res = is_binary_image(mask_image)
    print('res:', res)

    # mask_image.save(mask_path)
    mask_image_opencv = np.array(mask_image)
    imgWrite(mask_path, mask_image_opencv)     # 使用opencv的保存方法效果更好一些

    image2 = Image.open(mask_path).convert('L')
    res2 = is_binary_image(image2)
    print('res2:', res2)

def test1():
    json_path = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\DataProcess\ParseDoorLine\data\labels_ArcDoor\(T3) 12#楼105户型平面图（镜像）-3_Structure2.json'
    mask_path = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\DataProcess\ParseDoorLine\data\masks\(T3) 12#楼105户型平面图（镜像）-3_Structure2.jpg'
    json_to_mask(json_path, mask_path)

def doSplit():
    root = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-aug\VOCdevkit8\JPEGImages"
    output = r"E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-aug\VOCdevkit8\ImageSets\Segmentation"
    os.makedirs(output, exist_ok=True)
    filename = []
    #从存放原图的目录中遍历所有图像文件
    # dirs = os.listdir(root)
    for root, dir, files in os.walk(root):
        for file in files:
            # print(file)
            filename.append(file[:-4])  # 去除后缀，存储
    
    #打乱文件名列表
    np.random.shuffle(filename)
    #划分训练集、测试集，默认比例，8:2
    train_ratio = 0.8
    train = filename[:int(len(filename) * train_ratio)]
    val = filename[int(len(filename) * train_ratio):]
    
    #分别写入train.txt, test.txt
    with open(os.path.join(output, 'train.txt'), 'w', encoding='utf-8') as f1, open(os.path.join(output, 'val.txt'), 'w', encoding='utf-8') as f3:
        for i in train:
            f1.write(i + '\n')
        for i in val:
            f3.write(i + '\n')
    
    print('----- split finish -----')

def png_to_jpg():
    input_dir = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-aug\VOCdevkit8\JPEGImages'
    output_dir = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\onlywall2\data-aug\VOCdevkit8\JPEGImages2'
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
 
 
if __name__ == "__main__":
    # main()
    # doSplit()
    # png_to_jpg()
    test1()
