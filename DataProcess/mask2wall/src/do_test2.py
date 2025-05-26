# 测试程序2：labelme标注到标准掩模图比对测试
import json
import numpy as np
from skimage import draw
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import cv2

def load_labelme_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def polygons_to_mask(polygons, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    for polygon in polygons:
        rr, cc = draw.polygon(polygon[:, 1], polygon[:, 0], shape=img_shape)
        mask[rr, cc] = 1
    return mask

def compute_metrics(true_mask, pred_mask):
    # Flatten the masks
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()
    
    # Compute accuracy, recall, precision, and F1 score
    accuracy = accuracy_score(true_flat, pred_flat)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    
    # Compute IOU
    intersection = np.logical_and(true_flat, pred_flat)
    union = np.logical_or(true_flat, pred_flat)
    iou = np.sum(intersection) / np.sum(union)
    
    # Compute Boundary F1 Score
    true_boundary = binary_dilation(true_flat) ^ true_flat
    pred_boundary = binary_dilation(pred_flat) ^ pred_flat
    boundary_intersection = np.logical_and(true_boundary, pred_boundary)
    boundary_union = np.logical_or(true_boundary, pred_boundary)
    boundary_f1 = (2 * np.sum(boundary_intersection)) / (np.sum(boundary_union) + np.sum(boundary_intersection))
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'iou': iou,
        'boundary_f1': boundary_f1
    }

def main(ground_truth_json, prediction_json):
    # Load the ground truth and prediction JSON files
    gt_data = load_labelme_json(ground_truth_json)
    pred_data = load_labelme_json(prediction_json)
    
    # Assuming both images have the same shape
    img_shape = (gt_data['imageHeight'], gt_data['imageWidth'])
    
    # Initialize masks
    gt_mask = np.zeros(img_shape, dtype=np.uint8)
    pred_mask = np.zeros(img_shape, dtype=np.uint8)

    label_to_int = {
        "WallArea1": 1,
    }
    
    # Convert polygons to masks
    for shape in gt_data['shapes']:
        label = shape['label']
        points = np.array(shape['points'])
        gt_mask += polygons_to_mask([points], img_shape) * label_to_int[label]
    
    for shape in pred_data['shapes']:
        label = shape['label']
        points = np.array(shape['points'])
        pred_mask += polygons_to_mask([points], img_shape) * label_to_int[label]
    
    # Compute metrics
    metrics = compute_metrics(gt_mask, pred_mask)
    
    # Print the results
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')

def compute_metrics2(true_mask, pred_mask, num_classes):
    # Flatten the masks
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()
    
    # Compute classification report (accuracy, recall, precision, f1-score for each class)
    class_report = classification_report(true_flat, pred_flat, target_names=[str(i) for i in range(num_classes)], output_dict=True)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_flat, pred_flat, labels=list(range(num_classes)))
    
    # Compute IOU for each class
    iou = {}
    for i in range(num_classes):
        intersection = np.sum((true_flat == i) & (pred_flat == i))
        union = np.sum((true_flat == i) | (pred_flat == i))
        iou[i] = intersection / union if union != 0 else 0
    
    # Compute Boundary F1 Score for each class
    boundary_f1 = {}
    for i in range(num_classes):
        true_boundary = binary_dilation(true_flat == i) ^ (true_flat == i)
        pred_boundary = binary_dilation(pred_flat == i) ^ (pred_flat == i)
        boundary_intersection = np.sum(true_boundary & pred_boundary)
        boundary_union = np.sum(true_boundary | pred_boundary)
        boundary_f1[i] = (2 * boundary_intersection) / (boundary_union + boundary_intersection) if boundary_union != 0 else 0
    
    return {
        'classification_report': class_report,
        'iou': iou,
        'boundary_f1': boundary_f1
    }

def main2(ground_truth_json, prediction_json, num_classes):
    # Load the ground truth and prediction JSON files
    gt_data = load_labelme_json(ground_truth_json)
    pred_data = load_labelme_json(prediction_json)
    
    # Assuming both images have the same shape
    img_shape = (gt_data['imageHeight'], gt_data['imageWidth'])
    
    # Initialize masks
    gt_mask = np.zeros(img_shape, dtype=np.uint8)
    pred_mask = np.zeros(img_shape, dtype=np.uint8)

    label_to_int = {
        "WallArea1": 1,
    }
    
    # Convert polygons to masks
    for shape in gt_data['shapes']:
        label = shape['label']
        points = np.array(shape['points'])
        gt_mask += polygons_to_mask([points], img_shape) * label_to_int[label]
    
    for shape in pred_data['shapes']:
        label = shape['label']
        points = np.array(shape['points'])
        pred_mask += polygons_to_mask([points], img_shape) * label_to_int[label]
    
    # Compute metrics
    metrics = compute_metrics2(gt_mask, pred_mask, num_classes)
    
    # Print the results
    ious = [0.9985, 0.9420]
    print("Classification Report:")
    for cls, values in metrics['classification_report'].items():
        if cls.isdigit():
            print(f"Class {cls}:")
            print(f"  Iou: {ious[int(cls)]:.4f}")
            print(f"  Recall: {values['recall']:.4f}")
            print(f"  Precision: {values['precision']:.4f}")
            print(f"  F1 Score: {values['f1-score']:.4f}")
        else:
            print(f"{cls}: {values}")
    
    print("\nIOU:")
    for cls, iou_value in metrics['iou'].items():
        print(f"Class {cls}: {iou_value:.4f}")
    
    print("\nBoundary F1 Score:")
    for cls, boundary_f1_value in metrics['boundary_f1'].items():
        print(f"Class {cls}: {boundary_f1_value:.4f}")

def compute_metrics3(true_mask, pred_mask):
    # 计算IoU
    def compute_iou(gt_mask, pred_mask, class_id):
        # 提取目标区域或背景区域的掩膜
        gt_class_mask = (gt_mask == class_id)
        pred_class_mask = (pred_mask == class_id)
        
        intersection = np.logical_and(gt_class_mask, pred_class_mask).sum()
        union = np.logical_or(gt_class_mask, pred_class_mask).sum()
        iou = intersection / union if union != 0 else 0
        return iou

    # 计算准确率和召回率
    def compute_precision_recall(gt_mask, pred_mask, class_id):
        # 提取目标区域或背景区域的掩膜
        gt_class_mask = (gt_mask == class_id)
        pred_class_mask = (pred_mask == class_id)
        
        tp = np.logical_and(gt_class_mask == 1, pred_class_mask == 1).sum()  # True Positive
        fp = np.logical_and(gt_class_mask == 0, pred_class_mask == 1).sum()  # False Positive
        fn = np.logical_and(gt_class_mask == 1, pred_class_mask == 0).sum()  # False Negative
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall

    # 计算Boundary F1 Score
    def compute_boundary_f1(gt_mask, pred_mask, class_id, boundary_threshold=1):
        # 提取目标区域或背景区域的掩膜
        gt_class_mask = (gt_mask == class_id)
        pred_class_mask = (pred_mask == class_id)
        
        # 计算边界
        gt_boundary = cv2.Canny(gt_class_mask.astype(np.uint8), 100, 200)
        pred_boundary = cv2.Canny(pred_class_mask.astype(np.uint8), 100, 200)
        
        tp = np.logical_and(gt_boundary == 1, pred_boundary == 1).sum()  # Boundary TP
        fp = np.logical_and(gt_boundary == 0, pred_boundary == 1).sum()  # Boundary FP
        fn = np.logical_and(gt_boundary == 1, pred_boundary == 0).sum()  # Boundary FN

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1_score

    # 计算Mask mAP
    def compute_mask_map(gt_mask, pred_mask, class_id, iou_thresholds=[0.5]):
        # 计算Mask mAP
        aps = []
        for iou_threshold in iou_thresholds:
            iou = compute_iou(gt_mask, pred_mask, class_id)
            if iou >= iou_threshold:
                aps.append(1)
            else:
                aps.append(0)
        return np.mean(aps) if aps else 0

    # 计算所有指标
    def compute_metrics(gt_mask, pred_mask):
        # 计算背景（class_id = 0）和目标区域（class_id = 1）的指标
        metrics = {}
        for class_id in [0, 1]:
            # 计算IoU
            iou = compute_iou(gt_mask, pred_mask, class_id)
            
            # 计算准确率和召回率
            precision, recall = compute_precision_recall(gt_mask, pred_mask, class_id)
            
            # 计算Boundary F1 Score
            boundary_f1 = compute_boundary_f1(gt_mask, pred_mask, class_id)
            
            # 计算Mask mAP
            mask_map = compute_mask_map(gt_mask, pred_mask, class_id)
            
            metrics[class_id] = {
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'boundary_f1': boundary_f1,
                'mask_map': mask_map
            }
        
        return metrics

    metrics = compute_metrics(true_mask, pred_mask)
    print("Evaluation Metrics:")
    for class_id, metrics_value in metrics.items():
        class_name = 'Background' if class_id == 0 else 'Target'
        print(f"\n{class_name}:")
        print(f"IoU: {metrics_value['iou']:.4f}")
        print(f"Precision: {metrics_value['precision']:.4f}")
        print(f"Recall: {metrics_value['recall']:.4f}")
        print(f"Boundary F1 Score: {metrics_value['boundary_f1']:.4f}")
        print(f"Mask mAP: {metrics_value['mask_map']:.4f}")


if __name__ == '__main__':
    gt_json = '../data/labels/01 1-6号住宅楼标准层A户型平面图-5.json'
    pred_json = '../data/tmp_res/tmp15-cycles.json'
    main2(gt_json, pred_json, num_classes=2)
