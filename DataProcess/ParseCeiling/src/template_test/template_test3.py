import cv2
import numpy as np

def apply_nms(boxes, scores, iou_threshold=0.3):
    """ 非极大值抑制 (NMS) """
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    return indices

def multiscale_template_match(image, template, scales, threshold, edge=True):
    if edge:
        # 使用Canny边缘检测增强几何特征
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        img_proc = cv2.Canny(img_gray, 50, 150)
        template_proc = cv2.Canny(template_gray, 50, 150)
    else:
        img_proc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_proc = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h0, w0 = template_proc.shape
    matched_boxes = []
    matched_scores = []

    for scale in scales:
        resized_template = cv2.resize(template_proc, (0, 0), fx=scale, fy=scale)
        h, w = resized_template.shape[:2]

        if h > img_proc.shape[0] or w > img_proc.shape[1]:
            continue

        result = cv2.matchTemplate(img_proc, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            box = [pt[0], pt[1], w, h]
            score = result[pt[1], pt[0]]
            matched_boxes.append(box)
            matched_scores.append(float(score))

    return matched_boxes, matched_scores

# === 主程序 ===

# 图纸与模板路径
drawing_path = 'drawing.jpg'
template_path = 'lamp_template.jpg'

# 加载图像
drawing_img = cv2.imread(drawing_path)
template_img = cv2.imread(template_path)

# 多尺度匹配 + 边缘处理
scales = np.linspace(0.8, 1.2, 9)
threshold = 0.75

boxes, scores = multiscale_template_match(
    drawing_img, template_img,
    scales, threshold,
    edge=True  # 开启边缘增强
)

# NMS
indices = apply_nms(boxes, scores, iou_threshold=0.3)

# 绘制结果
output_img = drawing_img.copy()
matched_points = []

for i in indices:
    i = i[0]
    x, y, w, h = boxes[i]
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    matched_points.append((x + w // 2, y + h // 2))

print(f"匹配到的灯图元数量（边缘匹配 + NMS）：{len(matched_points)}")

# 保存输出
cv2.imwrite("matched_result_edges.jpg", output_img)

with open("lamp_coordinates_edges.csv", "w") as f:
    f.write("x,y\n")
    for (x, y) in matched_points:
        f.write(f"{x},{y}\n")

# 显示结果
cv2.imshow("Matched Lamps (Edge)", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
