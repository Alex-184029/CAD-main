import json
import os
import shutil
import fitz
import cv2
import numpy as np

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

def copy_data1():
    dwgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\AllDwgFiles3'
    validpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dwgs-valid'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dwgs-out1'
    os.makedirs(outpath, exist_ok=True)

    dwgs_all = os.listdir(dwgpath)
    dwgs_valid = os.listdir(validpath)
    dwgs_out = [dwg for dwg in dwgs_all if not dwg in dwgs_valid]
    for dwg in dwgs_out:
        shutil.copy(os.path.join(dwgpath, dwg), outpath)
    print('copy finish')

# pdf打印图像
def pdf_to_image2(pdfpath, outpath, labelpath, suffix='.jpg'):

    # 调用转换库
    def pdf_to_jpg_with_zoom(pdf_path, output_path, page_number=0, zoom_x=2.0, zoom_y=2.0):
        """
        将 PDF 文档的指定页面放大并转换为 JPG 图像。

        :param pdf_path: PDF 文件路径
        :param output_path: 输出 JPG 文件路径
        :param page_number: 要转换的页面编号（从 0 开始）
        :param zoom_x: 水平放大倍数
        :param zoom_y: 垂直放大倍数
        """
        # 打开 PDF 文件
        pdf_document = fitz.open(pdf_path)
        # 获取指定页面
        page = pdf_document.load_page(page_number)
        # 创建缩放矩阵
        matrix = fitz.Matrix(zoom_x, zoom_y)
        # 将页面转换为图像
        pix = page.get_pixmap(matrix=matrix)
        # 保存图像为 JPG 格式
        pix.save(output_path)

    # 计算缩放比例
    def parse_zoom_ratio(box, scale0=10):
        w0, h0 = 1684, 1191
        boxWidth = box[2] - box[0]
        boxHeight = box[3] - box[1]

        # k1 = 420 / 594     # pdf打印设置比例
        k1 = h0 * 1. / w0
        k2 = boxHeight * 1. / boxWidth
        scale = (boxWidth * 1. / w0) if k1 > k2 else (boxHeight * 1. / h0)   # 不缩放单位像素毫米数
        # print('scale:', scale, 'scale0:', scale0, 'boxWidth:', boxWidth, 'boxHeight:', boxHeight)
        return scale * 1. / scale0

    # 获取range框
    def get_box(labelpath):
        try:
            with open(labelpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data['range']
        except Exception as e:
            print(f'Error: {e}')
            return None

    if not os.path.exists(pdfpath) or not os.path.exists(labelpath):
        print('Input path not exist.')
        return
    os.makedirs(outpath, exist_ok=True)
    # label = os.path.join(labelpath, os.path.splitext(os.path.basename(pdfpath))[0] + '.txt')
    imgout = os.path.join(outpath, os.path.splitext(os.path.basename(pdfpath))[0] + suffix)
    box = get_box(labelpath)
    if box is None:
        print('box is None, skip.')
        return
    zoom = parse_zoom_ratio(box, scale0=10)
    pdf_to_jpg_with_zoom(pdfpath, imgout, zoom_x=zoom, zoom_y=zoom)

# json批量处理
def deal_json1_batch():
    json_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset1\json'
    pdf_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset1\pdfs'
    out_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset1\dataset11'
    if not os.path.exists(json_path) or not os.path.exists(pdf_path):
        print('Path not exist.')
        return
    os.makedirs(out_path, exist_ok=True)

    jsons = os.listdir(json_path)
    for json_item in jsons:
        json_path2 = os.path.join(json_path, json_item)
        with open(json_path2, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vp_id = data['vp_id']
        if not vp_id is None:
            pdf_name = os.path.splitext(json_item)[0] + '-' + str(vp_id) + '.pdf'
            deal_json1(json_path2, os.path.join(pdf_path, pdf_name), out_path)
    print('----- finish -----')

# json处理
def deal_json1(json_path, pdf_path, out_path):
    if not os.path.exists(json_path) or not os.path.exists(pdf_path):
        print('Error: Input path not exists.')
        return
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dwg_name = os.path.splitext(os.path.basename(pdf_path))[0]
        if not 'legend' in data:
            print(f'{dwg_name}, legend not exist.')
            return
        out_path = os.path.join(out_path, dwg_name)
        legend_path = os.path.join(out_path, 'legends')
        # print('out_path:', out_path)
        # print('legend_path:', legend_path)
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(legend_path, exist_ok=True)
        # 图像打印
        suffix = '.jpg'
        pdf_to_image2(pdf_path, out_path, json_path, suffix=suffix)
        img_out = os.path.join(out_path, dwg_name + suffix)
        if not os.path.exists(img_out):
            print('Error: print img,', img_out)
        im = imgRead(img_out)
        h, w, _ = im.shape

        # 图例筛选转换
        box = data['range']
        legends = data['legend']
        # print('legends len 1:', len(legends))
        do_map_legends(legends, box, w, h)
        # print('legends len 2:', len(legends))
        legends = [legend for legend in legends if len(legend['items']) > 0]
        # print('legends len 3:', len(legends))

        # 数据生成
        for i, legend in enumerate(legends):
            # os.makedirs(os.path.join(out_path, legend['label']), exist_ok=True)      # 不用block_name，因为block_name中可能包含不规范字符
            legend['id'] = i + 1
            # 截取图例
            x1, y1, x2, y2 = legend['items'][0]
            # print('legend:', x1, y1, x2, y2)
            im_legend = im[y1:y2, x1:x2]
            # print('step21')
            imgWrite(os.path.join(legend_path, str(legend['id']) + suffix), im_legend)
            # print('step22')
        # 更新数据保存
        data['legend'] = legends
        json_out = os.path.join(out_path, dwg_name + '.json')
        data['outPath'] = json_out
        with open(json_out, 'w', encoding='utf-8') as f:    # 支持中文字符保存
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f'Create data {dwg_name} finish.')

    except Exception as e:
        print(f'Error: {e}')
        return None

def do_map_legends(legends, box, imgWidth=1600, imgHeight=1280, extend=1):
    if legends is None or len(legends) == 0 or box is None or len(box) != 4:
        print('Error: do_map_legends input error.')
        return
    imgCenterX = imgWidth / 2
    imgCenterY = imgHeight / 2
    rangeWidth = box[2] - box[0]
    rangeHeight = box[3] - box[1]
    rangeCenterX = (box[2] + box[0]) / 2
    rangeCenterY = (box[3] + box[1]) / 2

    k1 = imgHeight * 1. / imgWidth
    k2 = rangeHeight * 1. / rangeWidth 
    scale = (imgWidth * 1. / rangeWidth) if k1 > k2 else (imgHeight * 1. / rangeHeight)
    num = len(legends)
    for i in range(num):
        items = legends[i]['items']
        items_new = []
        for item in items:
            x1, y1, x2, y2 = item
            xx1 = round(imgCenterX + (x1 - rangeCenterX) * scale)
            yy1 = imgHeight - round(imgCenterY + (y1 - rangeCenterY) * scale)
            xx2 = round(imgCenterX + (x2 - rangeCenterX) * scale)
            yy2 = imgHeight - round(imgCenterY + (y2 - rangeCenterY) * scale)
            if xx1 < 0 or xx1 > imgWidth or xx2 < 0 or xx2 > imgWidth or yy1 < 0 or yy1 > imgHeight or yy2 < 0 or yy2 > imgHeight:
                # print('Out of range: (%.2f, %.2f, %.2f, %.2f) -> (%d, %d, %d, %d)' % (x1, y1, x2, y2, xx1, xx2, yy1, yy2))
                continue
            items_new.append([xx1 - extend, min(yy1, yy2) - extend, xx2 + extend, max(yy1, yy2) + extend])
            # print('Right rect: (%.2f, %.2f, %.2f, %.2f) -> (%d, %d, %d, %d)' % (x1, y1, x2, y2, xx1, xx2, yy1, yy2))
        legends[i]['items'] = items_new

def check_finish():
    log_file = r'C:\Users\DELL\Desktop\record2.txt'
    if not os.path.exists(log_file):
        print('Error: log file not exist.')
        return
    with open(log_file, 'r', encoding='utf-8') as f:
        logs = f.readlines()
    logs = [log.strip() for log in logs]
    total, valid, legend_err, vp_err = len(logs), 0, 0, 0
    for log in logs:
        if log.endswith('finish'):
            valid += 1
        elif log.endswith('Get legend failed.'):
            legend_err += 1
        elif log.endswith('Get viewport failed.'):
            vp_err += 1
        else:
            print('Error log:', log)
    print('total: %d, valid: %d, legend_err: %d, vp_err: %d' % (total, valid, legend_err, vp_err))

def clean_dwgs():
    dwg_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\AllDwgFiles3'
    if not os.path.exists(dwg_path):
        print('Error: dwg_path not exist.')
        return
    dwgs = os.listdir(dwg_path)
    cnt_err = 0
    for dwg in dwgs:
        if not dwg.endswith('.dwg'):
            os.remove(os.path.join(dwg_path, dwg))
            cnt_err += 1
        
    print('Remove dwgs:', cnt_err)

def dataset_statistic():
    dataset_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset1\dataset11'
    dataset_err = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset1\dataset11-err'
    if not os.path.exists(dataset_path):
        print('Dataset path not exist:', dataset_path)
        return
    os.makedirs(dataset_err, exist_ok=True)
    items = os.listdir(dataset_path)
    cnt_legend, cnt_rect = 0, 0
    for item in items:
        # item_path = os.path.join(dataset_path, item)
        json_path = os.path.join(dataset_path, item, item + '.json')
        if not os.path.exists(json_path):
            print('Blank json for', item)
            shutil.move(os.path.join(dataset_path, item), dataset_err)
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not 'legend' in data or len(data['legend']) == 0:
            print('Blank legends for', item)
            shutil.move(os.path.join(dataset_path, item), dataset_err)
            continue
        cnt_legend += len(data['legend'])
        for legend in data['legend']:
            cnt_rect += len(legend['items'])
    print('item: %d, legend: %d, rect: %d' % (len(items), cnt_legend, cnt_rect))
        

def test1():
    json_path = r"E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset1\json\04 碧云湾C.C'户型水系统施工图.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vp_id = data['vp_id']
    print(vp_id, type(vp_id))
    if not vp_id is None:
        print('yes')
    else:
        print('no')
        
def test_print():
    pdfpath = r'C:\Users\DELL\Desktop\test3\pdfs\plan_2-4.pdf'
    outpath = r'C:\Users\DELL\Desktop\test3\imgs'
    labelpath = r'C:\Users\DELL\Desktop\test3\jsons\plan_2-ceiling.json'

    pdf_to_image2(pdfpath, outpath, labelpath)
    print('----- finish -----')


if __name__ == '__main__':
    # copy_data1()
    # json_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset1\json\(T3) 12#楼105户型平面图（镜像）.json'
    # pdf_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset1\pdfs\(T3) 12#楼105户型平面图（镜像）-4.pdf'
    # out_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset1\dataset11'

    # deal_json1(json_path, pdf_path, out_path)
    # check_finish()
    # clean_dwgs()
    # deal_json1_batch()
    # test1()

    # dataset_statistic()
    test_print()
