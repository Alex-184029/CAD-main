import os
import json
import random
from PIL import Image
import shutil

def yolo_to_coco(images_dir, labels_dir, output_dir, train_ratio=0.8):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train2017"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val2017"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)  # 随机打乱

    # 分割训练集和验证集
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # COCO格式的基本结构
    coco_format = {
        "info": {
            "description": "SlideDoor Augment Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Alex",
            "date_created": "2025/3/3"
        },
        "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "http://creativecommons.org/licenses/by/4.0/"}],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "SlideDoor", "supercategory": "none"}]  # 假设只有一个类别
    }

    # 辅助变量
    image_id = 1
    annotation_id = 1

    # 处理训练集
    for file_name in train_files:
        image_path = os.path.join(images_dir, file_name)
        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt'))

        # 读取图片尺寸
        with Image.open(image_path) as img:
            width, height = img.size

        # 添加图片信息
        coco_format["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # 读取标注文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

                # 将YOLO格式转换为COCO格式
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                bbox_width *= width
                bbox_height *= height

                # 添加标注信息
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,  # YOLO类别从0开始，COCO从1开始
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotation_id += 1

        # 复制图片到训练集目录
        # os.system(f"cp {image_path} {os.path.join(output_dir, 'train2017', file_name)}")
        shutil.copy(image_path, os.path.join(output_dir, 'train2017', file_name))
        image_id += 1
    print('Train dataset deal finish, image_id = %d, annotation_id = %d' % (image_id, annotation_id))

    # 处理验证集
    for file_name in val_files:
        image_path = os.path.join(images_dir, file_name)
        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt'))

        # 读取图片尺寸
        with Image.open(image_path) as img:
            width, height = img.size

        # 添加图片信息
        coco_format["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # 读取标注文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

                # 将YOLO格式转换为COCO格式
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                bbox_width *= width
                bbox_height *= height

                # 添加标注信息
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,  # YOLO类别从0开始，COCO从1开始
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotation_id += 1

        # 复制图片到验证集目录
        # os.system(f"cp {image_path} {os.path.join(output_dir, 'val2017', file_name)}")
        shutil.copy(image_path, os.path.join(output_dir, 'val2017', file_name))
        image_id += 1
    print('Val dataset deal finish, image_id = %d, annotation_id = %d' % (image_id, annotation_id))

    # 保存COCO格式的标注文件
    # with open(os.path.join(output_dir, "annotations", "instances_train2017.json"), 'w', encoding='utf-8') as f:
    #     json.dump({k: v for k, v in coco_format.items() if k != "images" or v["id"] <= split_idx}, f, indent=2)
    # with open(os.path.join(output_dir, "annotations", "instances_val2017.json"), 'w', encoding='utf-8') as f:
    #     json.dump({k: v for k, v in coco_format.items() if k != "images" or v["id"] > split_idx}, f, indent=2)

    # 保存训练集的标注文件
    train_images = [img for img in coco_format["images"] if img["id"] <= split_idx]
    train_annotations = [ann for ann in coco_format["annotations"] if ann["image_id"] <= split_idx]

    train_data = {
        "info": coco_format["info"],
        "licenses": coco_format["licenses"],
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_format["categories"]
    }

    with open(os.path.join(output_dir, "annotations", "instances_train2017.json"), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)

    # 保存验证集的标注文件
    val_images = [img for img in coco_format["images"] if img["id"] > split_idx]
    val_annotations = [ann for ann in coco_format["annotations"] if ann["image_id"] > split_idx]

    val_data = {
        "info": coco_format["info"],
        "licenses": coco_format["licenses"],
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_format["categories"]
    }

    with open(os.path.join(output_dir, "annotations", "instances_val2017.json"), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)

    print("转换完成！")


def main():
    # 使用示例
    images_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug-rename\images"
    labels_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug-rename\labels"
    output_dir = r"E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\coco_dataset"
    yolo_to_coco(images_dir, labels_dir, output_dir)


if __name__ == "__main__":
    main()
