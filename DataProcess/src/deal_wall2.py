import cv2
import re
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os

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

# 解析SVG文件
def parse_svg(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = {'svg': 'http://www.w3.org/2000/svg', 'inkscape': 'http://www.inkscape.org/namespaces/inkscape'}
    all_segments = []

    for group in root.findall('.//svg:g', ns):
        group_id = group.attrib.get('id', '')

        if 'WALL' in group_id:
            print(f"找到墙体组: {group_id}")

            for path in group.findall('.//svg:path', ns):
                path_d = path.attrib.get('d', '')
                if path_d:
                    print(f"处理路径，d属性为: {path_d}")
                    segments = parse_path_d(path_d)
                    all_segments.extend(segments)

    return all_segments

# 解析路径的 d 属性，提取线段端点
def parse_path_d(path_d):
    commands = re.findall(r'([ML])\s*([-0-9.]+),([-0-9.]+)', path_d)
    segments = []
    current_point = [0, 0]
    segment_id = 1

    for command, x, y in commands:
        x, y = float(x), float(y)
        if command == 'M':
            current_point = [x, y]
        elif command == 'L':
            segments.append({
                "id": f"{segment_id}",
                "start": {"x": current_point[0], "y": current_point[1]},
                "end": {"x": x, "y": y}
            })
            current_point = [x, y]
            segment_id += 1

    return segments

# 将线段绘制到二值图像，填充延伸线段的中间区域
def draw_segments_to_binary_image_with_extension(segments, scale_factor=10, padding=10):
    max_x, max_y = 0, 0
    for segment in segments:
        max_x = max(max_x, segment['start']['x'], segment['end']['x'])
        max_y = max(max_y, segment['start']['y'], segment['end']['y'])

    max_x = int(max_x * scale_factor + padding)
    max_y = int(max_y * scale_factor + padding)

    # 创建空白图像
    binary_image = np.zeros((max_y, max_x), dtype=np.uint8)

    for segment in segments:
        start = segment['start']
        end = segment['end']
        start_scaled = (int(start['x'] * scale_factor), int(start['y'] * scale_factor))
        end_scaled = (int(end['x'] * scale_factor), int(end['y'] * scale_factor))

        # 绘制线段
        cv2.line(binary_image, start_scaled, end_scaled, color=255, thickness=1)

    # 处理延伸线段的中间部分
    height, width = binary_image.shape
    for x in range(width):
        # 如果某一列的顶部和底部有白色像素（线延伸出边界）
        if binary_image[0, x] == 255 and binary_image[-1, x] == 255:
            # 将该列中间区域填充为白色
            binary_image[:, x] = 255

    return binary_image

# 使用cv2.connectedComponents计算连通块
def find_connected_components_with_opencv(binary_image):
    num_labels, labels = cv2.connectedComponents(binary_image)
    return num_labels, labels

# 绘制并统一填充所有连通块
def fill_connected_components(labels, output_image_path, fill_color=(255, 255, 255)):
    """
    用指定颜色填充所有连通块并保存为图像文件。
    """
    height, width = labels.shape
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)

    # 获取所有唯一连通块标签
    unique_labels = np.unique(labels)

    for label_id in unique_labels:
        if label_id == 0:
            continue  # 跳过背景
        mask = (labels == label_id)

        # 提取边界并填充
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            try:
                points = [(int(point[0][0]), int(point[0][1])) for point in contour]
                draw.polygon(points, fill=fill_color)
                # draw.polygon(points)    # 不填充会怎样
                # print('step1')
            except:
                print('Error points:', contour)
                print('step2')
            # points = [(int(point[0][0]), int(point[0][1])) for point in contour]
            # print('points:', len(points), points)
            # draw.polygon(points, fill=fill_color)

    img.save(output_image_path)
    print(f"所有连通块已填充为统一颜色，保存到: {output_image_path}")
    return img

# 主程序
def main_old():
    svg_input_path = 'svgs/0001-0023.svg'   # 输入SVG文件路径
    binary_image_path = 'binary_image.png'  # 二值图像保存路径
    filled_image_path = 'filled_connected_components.png'  # 连通块填充图像保存路径

    # 解析SVG提取线段
    wall_segments = parse_svg(svg_input_path)

    # 将线段绘制到二值图像，并处理延伸线段的中间部分
    scale_factor = 10
    binary_image = draw_segments_to_binary_image_with_extension(wall_segments, scale_factor=scale_factor)

    # 保存二值图像（可选）
    cv2.imwrite(binary_image_path, binary_image)
    print(f"二值图像已保存到: {binary_image_path}")

    # 使用cv2.connectedComponents计算连通块
    num_labels, labels = find_connected_components_with_opencv(binary_image)
    print(f"检测到 {num_labels - 1} 个连通块（不含背景）。")

    # 填充所有连通块
    fill_connected_components(labels, filled_image_path)

def test():
    binary_image_path = '../ParseDoorLine/data/tmp_res/img_contours2.jpg'  # 二值图像保存路径
    filled_image_path = '../ParseDoorLine/data/tmp_res/img_filled.png'  # 连通块填充图像保存路径

    binary_image = imgReadGray(binary_image_path)
    # 使用cv2.connectedComponents计算连通块
    num_labels, labels = find_connected_components_with_opencv(binary_image)
    print(f"检测到 {num_labels - 1} 个连通块（不含背景）。")

    # 填充所有连通块
    fill_connected_components(labels, filled_image_path)


if __name__ == '__main__':
    test()
