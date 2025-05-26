import cv2
import numpy as np
from matplotlib import pyplot as plt
from template_test1 import imgRead, imgWrite

def multi_scale_template_matching(img, template, scales=[0.8, 0.9, 1.0, 1.1, 1.2], 
                                 angles=[0, 90, 180, 270], threshold=0.8):
    """
    多尺度+旋转不变的模板匹配
    参数:
        img: 待搜索图像(BGR格式)
        template: 模板图像(BGR格式)
        scales: 缩放比例列表
        angles: 旋转角度列表(度)
        threshold: 匹配阈值(0-1)
    返回:
        results: 匹配结果列表[(x, y, w, h, score), ...]
    """
    # 转换为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape
    
    results = []
    
    for scale in scales:
        # 缩放模板
        resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))
        new_h, new_w = resized_template.shape
        
        for angle in angles:
            # 旋转模板
            if angle != 0:
                M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1)
                rotated_template = cv2.warpAffine(resized_template, M, (new_w, new_h))
            else:
                rotated_template = resized_template
            
            # 执行模板匹配
            res = cv2.matchTemplate(img_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
            
            # 获取高于阈值的位置
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                results.append((pt[0], pt[1], new_w, new_h, res[pt[1], pt[0]]))
    
    # 非极大值抑制(NMS)去除重复匹配
    results = non_max_suppression(results)
    
    return results

def non_max_suppression(boxes, overlap_thresh=0.3):
    """
    非极大值抑制(NMS)过滤重复框
    参数:
        boxes: [(x, y, w, h, score), ...]
        overlap_thresh: 重叠阈值
    返回:
        过滤后的结果
    """
    if len(boxes) == 0:
        return []
    
    # 转换为float方便计算
    boxes = np.array(boxes, dtype="float")
    
    # 初始化选择索引
    pick = []
    
    # 获取坐标和得分
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    scores = boxes[:, 4]
    
    # 按得分排序
    idxs = np.argsort(scores)[::-1]
    
    while len(idxs) > 0:
        # 选择得分最高的加入结果
        i = idxs[0]
        pick.append(i)
        
        # 计算与其他框的交并比(IOU)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersection = w * h
        area_i = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1)
        area_others = (x2[idxs[1:]] - x1[idxs[1:]] + 1) * (y2[idxs[1:]] - y1[idxs[1:]] + 1)
        iou = intersection / (area_i + area_others - intersection)
        
        # 删除重叠过高的框
        idxs = np.delete(idxs, np.where(iou > overlap_thresh)[0] + 1)
    
    return boxes[pick].astype("int")

def visualize_results(img, results, outpath):
    """可视化匹配结果"""
    output = img.copy()
    for (x, y, w, h, _) in results:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Matched Lamps", output)
    cv2.waitKey()
    cv2.destroyAllWindows()

    imgWrite(outpath, output)
    
    # plt.figure(figsize=(15, 10))
    # plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    # plt.title(f"Found {len(results)} matches")
    # plt.axis('off')
    # plt.show()


# 使用示例
if __name__ == "__main__":

    # 图纸与模板路径
    drawing_path = '../imgs/tmp1.jpg'
    template_path = '../imgs/templates/template2.jpg'
    out_path = '../res/res2.jpg'

    # 加载图像
    img = imgRead(drawing_path)
    template = imgRead(template_path)

    # 多尺度匹配
    scales = np.linspace(0.8, 1.2, 9)  # 从0.8倍到1.2倍，共9个尺度
    # scales = np.linspace(0.3, 2, 18)  # 从0.8倍到1.2倍，共9个尺度
    print('scales:', scales)
    threshold = 0.75

    # 执行匹配
    results = multi_scale_template_matching(img, template, 
                                          scales=scales,
                                          threshold=0.75)  # 调整阈值
    
    # 输出结果
    print(f"共检测到 {len(results)} 个灯图元")
    visualize_results(img, results, out_path)