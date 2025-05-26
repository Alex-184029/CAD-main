# -- 填充调整为正方形尺寸，转yolo分割格式
import glob
import numpy as np
import json
import os
import cv2
import tqdm

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)

def jsonToYolo():
    '''
        LabelMe标注json格式转换为yolo格式
        根据原图和JSON格式的标签文件生成对应的YOLO的TXT标签文件保存到json_path路径下（保存文件名字和原来文件的名字一样，后缀换成txt）
    '''
    json_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-aug\data-aug1\labels'
    TXT_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-aug\data-yolo\labels'
    os.makedirs(TXT_path, exist_ok=True)

    label_dict = {'WallArea1': 0}                 # 类别情况
    json_files = glob.glob(json_path + "/*.json")
    num = len(json_files)
    for i, json_file in enumerate(json_files):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        with open(json_file, 'r', encoding='utf-8') as f:
            json_info = json.load(f)
        height, width = json_info["imageHeight"], json_info["imageWidth"]
        np_w_h = np.array([[width, height]], np.int32)

        txt_file = os.path.basename(json_file).replace(".json", ".txt")
        txt_file = os.path.join(TXT_path, txt_file)
        with open(txt_file, "w", encoding='utf-8') as f:
            for point_json in json_info["shapes"]:
                txt_content = ""
                label = point_json["label"]
                label_index = label_dict.get(label, None)
                np_points = np.array(point_json["points"], np.int32)
                norm_points = np_points / np_w_h
                norm_points_list = norm_points.tolist()
                txt_content += f"{label_index} " + " ".join([" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
                f.write(txt_content)
    print('---- finish ----')

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

def resize_image(image, size=(640, 640)):
    """
    将图像调整为目标尺寸。
    """
    # return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def adjust_annotations(data, scale, padding):
    """
    根据图像缩放和填充调整标注。
    """
    top, left = padding
    for shape in data['shapes']:
        points = shape['points']
        adjusted_points = []
        for x, y in points:
            # Apply padding and scaling
            new_x = round((x + left) * scale)
            new_y = round((y + top) * scale)
            adjusted_points.append([new_x, new_y])
        shape['points'] = adjusted_points
    return data

def process_labelme_dataset():
    """
    处理Labelme格式数据集。
    - input_dir: 输入目录，包含图片和对应的JSON文件。
    - output_dir: 输出目录。
    """
    label_in_dir = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\labels'
    output_dir = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-resize640'
    if not os.path.exists(label_in_dir):
        print('input path error')
        return
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    labels = os.listdir(label_in_dir)
    num = len(labels)
    target_size = 640
    for i, file in enumerate(labels):
        if i % 50 == 0:
            print('%d / %d' % (i, num))
        if file.endswith('.json'):
            json_path = os.path.join(label_in_dir, file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load the corresponding image
            image_path = os.path.join(label_in_dir, data['imagePath'])
            if not os.path.exists(image_path):
                print(f"Image {data['imagePath']} not found for {file}")
                continue

            image = imgRead(image_path)
            if image is None:
                print(f"Failed to load image {data['imagePath']}")
                continue

            # Pad the image to square
            padded_image, padding = pad_to_square(image)

            # Resize the image to target size
            resized_image = resize_image(padded_image, size=(target_size, target_size))

            # Save the processed image
            output_image_path = os.path.join(image_dir, os.path.basename(data['imagePath']))
            imgWrite(output_image_path, resized_image)

            # Adjust annotations
            original_size = max(image.shape[:2])
            scale = target_size / original_size
            adjusted_data = adjust_annotations(data, scale, padding)

            # Update imagePath in JSON
            adjusted_data['imagePath'] = '../images/' + os.path.basename(output_image_path)
            adjusted_data['imageHeight'] = target_size
            adjusted_data['imageWidth'] = target_size

            # Save the adjusted JSON
            output_json_path = os.path.join(label_dir, file)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(adjusted_data, f, indent=2, ensure_ascii=False)

def test():
    labels = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\resize_data_640\labels.txt'
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            class_name == "_background_"
        class_names.append(class_name)
    print('class_names:', class_names)
    print('class_name_to_id:', class_name_to_id)


if __name__ == '__main__':
    # jsonToYolo()
    process_labelme_dataset()
    # test()
 
