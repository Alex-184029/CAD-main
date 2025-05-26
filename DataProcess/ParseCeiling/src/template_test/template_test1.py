import os
import cv2
import numpy as np
from skimage.feature import match_template
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import img_as_float

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

def apply_nms(boxes, scores, iou_threshold=0.3):
    """ 非极大值抑制 (NMS) """
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    return indices

def multiscale_template_match_rgb_skimage(image, template, scales, threshold):
    # skimage 要求浮点格式的 RGB 图像（范围 0.0 - 1.0）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    img_proc = img_as_float(image)
    template_proc = img_as_float(template)

    h0, w0 = template_proc.shape[:2]
    matched_boxes = []
    matched_scores = []

    for scale in scales:
        resized_template = resize(template_proc, (int(h0 * scale), int(w0 * scale)), anti_aliasing=True, preserve_range=True)

        if resized_template.shape[0] > img_proc.shape[0] or resized_template.shape[1] > img_proc.shape[1]:
            continue

        # skimage RGB 模板匹配（每通道匹配 + 平均）
        result = match_template(img_proc, resized_template, pad_input=False)

        # 匹配得分大于阈值的区域
        match_locations = np.where(result >= threshold)
        for (y, x) in zip(*match_locations):
            h, w = resized_template.shape[:2]
            box = [x, y, w, h]
            score = result[y, x]
            matched_boxes.append(box)
            matched_scores.append(float(score))

    return matched_boxes, matched_scores

def multiscale_template_match_skimage(image, template, scales, threshold, edge=False):
    # 转灰度 & 边缘处理（可选）
    if edge:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        img_proc = cv2.Canny(img_gray, 50, 150)
        template_proc = cv2.Canny(template_gray, 50, 150)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_proc = rgb2gray(img_as_float(image))     # 使用 skimage 的浮点灰度图
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        template_proc = rgb2gray(img_as_float(template))

    h0, w0 = template_proc.shape
    matched_boxes = []
    matched_scores = []

    for scale in scales:
        resized_template = resize(template_proc, (int(h0 * scale), int(w0 * scale)), anti_aliasing=True)

        if resized_template.shape[0] > img_proc.shape[0] or resized_template.shape[1] > img_proc.shape[1]:
            continue

        # 使用 skimage 进行匹配
        # result = match_template(img_proc, resized_template, pad_input=True)
        result = match_template(img_proc, resized_template, pad_input=False)

        # 匹配分数满足阈值的坐标
        match_locations = np.where(result >= threshold)
        for (y, x) in zip(*match_locations):
            h, w = resized_template.shape
            box = [x, y, w, h]
            score = result[y, x]
            matched_boxes.append(box)
            matched_scores.append(float(score))

    return matched_boxes, matched_scores

def multiscale_template_match(image, template, scales, threshold, edge: bool):
    if edge:
        # 使用Canny边缘检测增强几何特征
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 1.0)  # 去噪
        # img_gray = cv2.Canny(img_gray, 50, 150)
        template_gray = cv2.Canny(template_gray, 50, 150)
    else:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h0, w0 = template_gray.shape

    matched_boxes = []
    matched_scores = []

    for scale in scales:
        resized_template = cv2.resize(template_gray, (0, 0), fx=scale, fy=scale)
        h, w = resized_template.shape[:2]

        if h > img_gray.shape[0] or w > img_gray.shape[1]:
            continue

        result = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        # result = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCORR_NORMED)
        # cv2.TM_SQDIFF_NORMED
        # result = cv2.matchTemplate(img_gray, resized_template, cv2.TM_SQDIFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            box = [pt[0], pt[1], w, h]
            score = result[pt[1], pt[0]]
            matched_boxes.append(box)
            matched_scores.append(float(score))

    return matched_boxes, matched_scores


def main():
    # === 主程序 ===

    # 图纸与模板路径
    drawing_path = '../imgs/tmp1.jpg'
    template_path = '../imgs/templates/template2.jpg'
    out_path = '../res/res1.jpg'

    # 加载图像
    drawing_img = imgRead(drawing_path)
    template_img = imgRead(template_path)

    # 多尺度匹配
    # scales = np.linspace(0.8, 1.2, 9)  # 从0.8倍到1.2倍，共9个尺度
    scales = np.linspace(0.2, 4, 20)  # 从0.8倍到1.2倍，共9个尺度
    # scales = np.linspace(0.1, 10, 100)  # 从0.8倍到1.2倍，共9个尺度
    # threshold = 0.75
    threshold = 0.7

    # boxes, scores = multiscale_template_match(drawing_img, template_img, scales, threshold, edge=True)
    boxes, scores = multiscale_template_match_rgb_skimage(drawing_img, template_img, scales, threshold)

    print('boxes num:', len(boxes), type(boxes))
    # NMS处理
    indices = apply_nms(boxes, scores, iou_threshold=0.3)
    print('indices:', indices, type(indices), len(indices))

    # 绘制并导出结果
    output_img = drawing_img.copy()
    matched_points = []

    for i in indices:
        # i = i[0]
        x, y, w, h = boxes[i]
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        center = (x + w // 2, y + h // 2)
        matched_points.append(center)

    # 输出匹配数量
    print(f"匹配到的灯图元数量（去重后）：{len(matched_points)}")

    # 保存结果图
    imgWrite(out_path, output_img)

    # 保存匹配中心点坐标
    # with open("lamp_coordinates.csv", "w") as f:
    #     f.write("x,y\n")
    #     for (x, y) in matched_points:
    #         f.write(f"{x},{y}\n")

    # 可视化（可选）
    cv2.imshow("Matched Lamps", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 
