# 测试程序1：labelme标注到标准掩模图比对测试
import json
import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict
from pycocotools import mask as maskUtils

# 计算IoU
def compute_iou(gt_polygon, pred_polygon):
    # 将多边形转换为Shapely Polygon对象
    gt = Polygon(gt_polygon)
    pred = Polygon(pred_polygon)
    # 计算交并比（IoU）
    if gt.is_valid and pred.is_valid:
        intersection = gt.intersection(pred).area
        union = gt.union(pred).area
        return intersection / union if union != 0 else 0
    else:
        return 0

# 计算准确率和召回率
def compute_precision_recall(gt, pred, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    
    # 对每个真实标注实例计算IoU
    for gt_instance in gt:
        best_iou = 0
        for pred_instance in pred:
            iou = compute_iou(gt_instance['points'], pred_instance['points'])
            best_iou = max(best_iou, iou)
        if best_iou >= iou_threshold:
            tp += 1
        else:
            fn += 1

    # 对每个预测实例计算IoU
    for pred_instance in pred:
        best_iou = 0
        for gt_instance in gt:
            iou = compute_iou(gt_instance['points'], pred_instance['points'])
            best_iou = max(best_iou, iou)
        if best_iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    return precision, recall

# 计算Mask mAP
def compute_mask_map(gt, pred, iou_thresholds=[0.5, 0.75]):
    aps = defaultdict(list)
    for iou_threshold in iou_thresholds:
        tp, fp, fn = 0, 0, 0
        for gt_instance in gt:
            best_iou = 0
            for pred_instance in pred:
                iou = compute_iou(gt_instance['points'], pred_instance['points'])
                best_iou = max(best_iou, iou)
            if best_iou >= iou_threshold:
                tp += 1
            else:
                fn += 1
        for pred_instance in pred:
            best_iou = 0
            for gt_instance in gt:
                iou = compute_iou(gt_instance['points'], pred_instance['points'])
                best_iou = max(best_iou, iou)
            if best_iou >= iou_threshold:
                tp += 1
            else:
                fp += 1
        ap = tp / (tp + fp) if tp + fp > 0 else 0
        aps[iou_threshold].append(ap)
    
    return aps

# 计算Boundary F1 Score
def compute_boundary_f1(gt, pred, boundary_threshold=5):
    tp, fp, fn = 0, 0, 0
    for gt_instance in gt:
        for pred_instance in pred:
            boundary_distance = compute_boundary_distance(gt_instance['points'], pred_instance['points'])
            if boundary_distance < boundary_threshold:
                tp += 1
            else:
                fn += 1
    for pred_instance in pred:
        if pred_instance not in gt:
            fp += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1

# 计算边界的距离
def compute_boundary_distance(gt_points, pred_points):
    gt_polygon = Polygon(gt_points)
    pred_polygon = Polygon(pred_points)
    if gt_polygon.is_valid and pred_polygon.is_valid:
        return gt_polygon.distance(pred_polygon)
    return float('inf')

# 加载LabelMe JSON文件
def load_labelme_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 计算模型的各种指标
def compute_metrics(gt_json, pred_json):
    gt_data = load_labelme_json(gt_json)
    pred_data = load_labelme_json(pred_json)

    categories = set()
    for shape in gt_data['shapes']:
        categories.add(shape['label'])
    for shape in pred_data['shapes']:
        categories.add(shape['label'])
    
    results = {}
    for category in categories:
        gt_category = [shape for shape in gt_data['shapes'] if shape['label'] == category]
        pred_category = [shape for shape in pred_data['shapes'] if shape['label'] == category]

        # 计算IoU、准确率、召回率
        precision, recall = compute_precision_recall(gt_category, pred_category)
        iou = compute_iou(gt_category[0]['points'], pred_category[0]['points']) if gt_category and pred_category else 0

        # Mask mAP
        mask_map = compute_mask_map(gt_category, pred_category)

        # Boundary F1 Score
        boundary_f1 = compute_boundary_f1(gt_category, pred_category)

        results[category] = {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'mask_map': mask_map,
            'boundary_f1': boundary_f1
        }
    
    return results

def main():
    # 示例用法
    pred_json = '../data/tmp_res/tmp15-cycles.json'
    gt_json = '../data/labels/01 1-6号住宅楼标准层A户型平面图-5.json'

    metrics = compute_metrics(gt_json, pred_json)
    for category, metrics_value in metrics.items():
        print('----- begin -----')
        print(f"Category: {category}")
        print(f"Precision: {metrics_value['precision']:.4f}")
        print(f"Recall: {metrics_value['recall']:.4f}")
        print(f"IoU: {metrics_value['iou']:.4f}")
        print(f"Mask mAP: {metrics_value['mask_map']}")
        print(f"Boundary F1 Score: {metrics_value['boundary_f1']:.4f}")


if __name__ == '__main__':
    main()
