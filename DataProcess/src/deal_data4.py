# -- 绘制标注框
import os
import cv2
import numpy as np

BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 255, 0)

def imgRead(imgpath):
    if not os.path.exists(imgpath):
        print('img path not exist')
        return None
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)

def imgWrite(imgpath, img):
    cv2.imencode(os.path.splitext(imgpath)[1], img)[1].tofile(imgpath)

def visualize_bbox(img, bboxes, color=BOX_COLOR, thickness=2):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox[:-1]
        label = bbox[-1]
        color = (0, 0, 255) if label == '0' else (0, 255, 0)
        
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    #     # 添加文字
    #     ((text_width, text_height), _) = cv2.getTextSize(bbox[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    #     cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    #     cv2.putText(
    #         img,
    #         text=bbox[-1],
    #         org=(x_min, y_min - int(0.3 * text_height)),
    #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #         fontScale=0.35,
    #         color=TEXT_COLOR,
    #         lineType=cv2.LINE_AA,
    #     )

    return img

def readLabel(labelpath, w, h):
    with open(labelpath, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [list(map(float, i.strip().split())) for i in labels]
    labels = [[l[1], l[2], l[3], l[4], str(int(l[0]))] for l in labels]
    # print('labels0:', labels)
    bboxes = []
    
    for l in labels:
        x_center, y_center, width, height = l[:-1]
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        x_min, x_max, y_min, y_max = int(x_center - width / 2), int(x_center + width / 2), int(y_center - height / 2), int(y_center + height / 2)

        bboxes.append([x_min, y_min, x_max, y_max, l[-1]])

    return bboxes

def drawLabels(imgpath, labelpath, outpath):
    if not os.path.exists(imgpath) or not os.path.exists(labelpath):
        print('img or label path not exist')
        return

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    labels = os.listdir(labelpath)
    total = len(labels)
    labels = [os.path.splitext(name)[0] for name in labels]
    # w, h = 640, 640

    for i, label in enumerate(labels):
        if i % 50 == 0:
            print('%d for %d is doing' % (i, total))
        label1 = os.path.join(labelpath, label + '.txt')
        img1 = os.path.join(imgpath, label + '.jpg')
        if not os.path.exists(img1):
            print('img not exist:', img1)
            continue
        try:
            im = imgRead(img1)
            h, w, _ = im.shape
            labels1 = readLabel(label1, w, h)
            im = visualize_bbox(im, labels1)
            imgWrite(os.path.join(outpath, label + '.jpg'), im)
        except:
            print('draw labels err:', label)

    print('draw labels finish')

def doTestOne():
    imgpath = './alex/valdata-4.15/images/'
    labelpath1 = './alex/valdata-4.15/labels/'
    labelpath2 = './alex/valdata-4.15/labels-test/'
    img = '!@ 01 PL 平面图 1#-1'
    w, h = 320, 320

    label1 = os.path.join(labelpath1, img + '.txt')
    label2 = os.path.join(labelpath2, img + '.txt')
    labels1 = readLabel(label1, w, h)
    labels2 = readLabel(label2, w, h)
    im = imgRead(os.path.join(imgpath, img + '.jpg'))
    im = visualize_bbox(im, labels1, labels2)

    cv2.imshow('im', im)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\SelectDwgs1\labelsParallelWindow\yolo5\DataAug\dataset-aug\images'
    labelpath = r'C:\Users\DELL\Desktop\Train10.9\test_windows12\labels'
    outpath = r'C:\Users\DELL\Desktop\Train10.9\test_windows12\outs2'

    drawLabels(imgpath, labelpath, outpath)


if __name__ == '__main__':
    main()