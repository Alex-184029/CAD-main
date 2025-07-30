'''
@ Function: cad服务接口
    提供墙、门、窗等图元解析
    打印图像获取
'''
from flask import *
from flask_cors import CORS
import os
import time
import base64
import socket
import fitz
import json

app = Flask(__name__)
CORS(app)
# dwgpath = r'C:\Users\Administrator\Desktop\MyCAD\public\dwgs1'
public_path = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public3\dwgs1'

def send_msg(message):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8088))   # 连接本地socket端口

    # 发送字符串数据到服务器端
    client_socket.send(message.encode('utf-8'))

    data = client_socket.recv(1024).decode('utf-8')
    print('res:', data)
    client_socket.close()
    return data

def closeSocket():
    send_msg('Terminate')

def readBase64(imgpath):
    if not os.path.exists(imgpath):
        return None
    with open(imgpath, 'rb') as img_file:
        img_data = img_file.read()
        base64_img = base64.b64encode(img_data)
        base64_img_str = base64_img.decode('utf-8')
    return base64_img_str

def save_base64_as_jpg(base64_data, output_path):
    """
    将 Base64 编码的图片转换为 jpg 格式并保存
    
    参数:
    base64_data (str): 包含 Base64 数据的字符串（可能包含前缀，如 'data:image/png;base64,'）
    output_path (str): 保存 jpg 文件的路径
    """
    # 去除可能存在的前缀（如 'data:image/png;base64,'）
    if base64_data.startswith('data:'):
        base64_data = base64_data.split(',', 1)[1]
    
    # 解码 Base64 数据
    image_data = base64.b64decode(base64_data)
    
    # 保存为 PNG 文件
    with open(output_path, 'wb') as f:
        f.write(image_data)
    
    # print(f"jpg 文件已保存至: {output_path}")

# pdf打印图像
def pdf_to_image2(pdfpath, imgout, labelpath):

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
    # os.makedirs(outpath, exist_ok=True)
    # label = os.path.join(labelpath, os.path.splitext(os.path.basename(pdfpath))[0] + '.txt')
    # imgout = os.path.join(outpath, os.path.splitext(os.path.basename(pdfpath))[0] + suffix)
    box = get_box(labelpath)
    if box is None:
        print('box is None, skip.')
        return
    zoom = parse_zoom_ratio(box, scale0=10)
    pdf_to_jpg_with_zoom(pdfpath, imgout, zoom_x=zoom, zoom_y=zoom)

def parseResult(inpath):
    if not os.path.exists(inpath):
        print('inpath not exists')
        return None
    with open(inpath, 'r', encoding='utf-8') as f:
        content = f.readlines()
    content = [con.strip() for con in content]
    try:
        box = [float(i) for i in content[0][7:].split(', ')]    # 这里的类型转换可能没必要，传入前端又会变成字串
        rects = []
        # print(type(content[2]), content[2])
        index = content[2].find(': ') + 2
        # print('index:', index)
        if index == 1:     # 没找到:
            print('Parse colon info failed.')
            return None
        for i in range(2, len(content)):
            index = content[i].find(': ') + 2
            arr = [float(i) for i in content[i][index:].split(', ')]
            rect = {'item_order': len(rects) + 1, 'x1': arr[0], 'y1': arr[1], 'x2': arr[2], 'y2': arr[3], 'item_type': arr[4], 'probability': arr[5]}
            # if isInRange(box, rect):
            #     rects.append(rect)
            rects.append(rect)
        return {'box': box, 'total': len(rects), 'rects': rects}
    except:
        print('parse %s error' % inpath)
        return None

def parseResult2(inpath, item='ArcDoor'):
    with open(inpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    att = None
    if item == 'ArcDoor':
        att = 'arc_items'
    elif item == 'Balcony':
        att = 'balcony_items'
    elif item == 'DoorLine':
        att = 'door_line_items'
    elif item == 'ParallelWindow':
        att = 'window_items'
    elif item == 'WallLine':
        att = 'wall_line_items'
    elif item == 'Text':
        att = 'text_items'
    else:
        print('Error item type:', item)
        return None
    if not att in data:
        print('Error %s in dict.' % att)
        return None
    return att, data[att]

def parseResultBasic(inpath):
    basic_atts = ['dwg_name', 'range', 'box']
    res = dict()
    with open(inpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for att in basic_atts:
        if not att in data:
            print('Error %s in dict.' % att)
            return None
        res[att] = data[att]
    return res

def parseResult3(work_dir, dwgname, items, task_name):
    print('here is parseResult3, task_name: %s, dwgname: %s' % (task_name, dwgname))
    data = dict()
    to_parse = False
    dwg = os.path.splitext(dwgname)[0]
    for item in items:
        logfile = os.path.join(work_dir, dwg + '_' + item + '.json')
        if not os.path.exists(logfile):
            to_parse = True
            break
    if to_parse:
        res0 = send_msg(f'{task_name} {dwgname}')
    if not to_parse or res0 == 'Succeed':
        # 读取元信息
        item0 = items[0]
        logfile = os.path.join(work_dir, dwg + '_' + item0 + '.json')
        res = parseResultBasic(logfile)
        if res is None:
            return None
        data.update(res)

        # 依次读取结果
        for item in items:
            logfile = os.path.join(work_dir, dwg + '_' + item + '.json')
            res = parseResult2(logfile, item=item)
            if res is None:
                return None
            att, value = res
            data[att] = value

        if to_parse:    # 等待打印完成
            # 打印存在延迟，需要定期监听并设置timeout，一般会经过30s以上才能监听到图像打印完成
            pdfpath = os.path.join(work_dir, dwg + '_PlaneLayout.pdf')
            timeout = 100
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                if os.path.exists(pdfpath):
                    print(f"经过 %.3f s，监听到 %s 打印完成。" % (time.time() - start_time, pdfpath))
                    break
                time.sleep(0.1)  # 每隔0.1秒检查一次
        return data
    else:
        return None

@app.route('/hello', methods=['POST'])
def hello_world():
    return 'Hello World !'

@app.route('/get_plane_layout_img', methods=['POST'])
def get_plane_layout_img():
    dwg_name = request.form['dwg_name']
    dwg_name = os.path.splitext(dwg_name)[0]
    img_path = os.path.join(public_path, dwg_name, dwg_name + '_PlaneLayout.jpg')
    if not os.path.exists(img_path):
        pdf_path = os.path.join(public_path, dwg_name, dwg_name + '_PlaneLayout.pdf')
        if not os.path.exists(pdf_path):
            print('Error: pdf path not exist,', pdf_path)
            return jsonify({'status': 'error', 'error info': 'Get pdf file fail.'}), 400
        else:
            # 执行pdf到jpg转换
            label_dir = os.path.join(public_path, dwg_name)
            labels = os.listdir(label_dir)
            label_path = None
            for label in labels:
                if label.endswith('.json'):
                    label_path = os.path.join(label_dir, label)
                    break
            if label_path:
                pdf_to_image2(pdf_path, img_path, label_path)

    # 图片转换为base64格式传出
    imgbase64 = readBase64(img_path)
    if imgbase64 is None:
        print('imgbase64 is None, ', img_path)
        return jsonify({'status': 'error', 'error info': 'Get image base64 fail.'}), 400
    return jsonify({'status': 'success', 'img': imgbase64})


@app.route('/parse_door', methods=['POST'])
def parse_door():
    # 接收传入图纸文件并存储
    f = request.files['file']
    dwgname = f.filename
    if dwgname == '':
        return jsonify({'status': 'error', 'error info': 'No selected file.'}), 400
    work_name = os.path.splitext(dwgname)[0]
    work_dir = os.path.join(public_path, work_name)
    os.makedirs(work_dir, exist_ok=True)
    f.save(os.path.join(work_dir, dwgname))

    # 图元解析
    print('图纸 %s 开始解析' % dwgname)
    items = ['ArcDoor', 'DoorLine']
    data = parseResult3(work_dir, dwgname, items, 'Door')
    if not data is None:
        return jsonify({'status': 'success', 'res': data})
    return jsonify({'status': 'error', 'error info': 'Parse dwg fail.'}), 400


@app.route('/parse_window', methods=['POST'])
def parse_window():
    # 接收传入图纸文件并存储
    f = request.files['file']
    dwgname = f.filename
    if dwgname == '':
        return jsonify({'status': 'error', 'error info': 'No selected file.'}), 400
    work_name = os.path.splitext(dwgname)[0]
    work_dir = os.path.join(public_path, work_name)
    f.save(os.path.join(work_dir, dwgname))

    # 图元解析
    print('图纸 %s 开始解析' % dwgname)
    items = ['ParallelWindow']
    data = parseResult3(work_dir, dwgname, items, 'ParallelWindow')
    if not data is None:
        return jsonify({'status': 'success', 'res': data})
    return jsonify({'status': 'error', 'error info': 'Parse dwg fail.'}), 400

@app.route('/parse_wall', methods=['POST'])
def parse_wall():
    # 接收传入图纸文件并存储
    f = request.files['file']
    dwgname = f.filename
    if dwgname == '':
        return jsonify({'status': 'error', 'error info': 'No selected file.'}), 400
    work_name = os.path.splitext(dwgname)[0]
    work_dir = os.path.join(public_path, work_name)
    f.save(os.path.join(work_dir, dwgname))

    # 图元解析
    print('图纸 %s 开始解析' % dwgname)
    items = ['WallLine']
    data = parseResult3(work_dir, dwgname, items, 'Wall')
    if not data is None:
        return jsonify({'status': 'success', 'res': data})
    return jsonify({'status': 'error', 'error info': 'Parse dwg fail.'}), 400

@app.route('/parse_area', methods=['POST'])
def parse_area():
    # 接收传入图纸文件并存储
    f = request.files['file']
    dwgname = f.filename
    if dwgname == '':
        return jsonify({'status': 'error', 'error info': 'No selected file.'}), 400
    work_name = os.path.splitext(dwgname)[0]
    work_dir = os.path.join(public_path, work_name)
    os.makedirs(work_dir, exist_ok=True)
    f.save(os.path.join(work_dir, dwgname))

    # 图元解析
    print('图纸 %s 开始解析' % dwgname)
    items = ['ArcDoor', 'DoorLine', 'WallLine', 'Balcony', 'ParallelWindow', 'Text']
    data = parseResult3(work_dir, dwgname, items, 'Area')
    if not data is None:
        return jsonify({'status': 'success', 'res': data})
    return jsonify({'status': 'error', 'error info': 'Parse dwg fail.'}), 400

def test():
    pdf_path = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public3\dwgs2\2c34a2b5-88c3-4d78-a42b-5623cf225044\legend_data\pdfs'
    label_path = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public3\dwgs2\2c34a2b5-88c3-4d78-a42b-5623cf225044\legend_data\legends'
    img_path = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public3\dwgs2\2c34a2b5-88c3-4d78-a42b-5623cf225044\legend_data\imgs'
    os.makedirs(img_path, exist_ok=True)
    labels = os.listdir(label_path)
    labels = [label for label in labels if label.endswith('.json')]
    pdfs = os.listdir(pdf_path)
    for pdf in pdfs:
        pdf_name = os.path.splitext(pdf)[0]
        for label in labels:
            if pdf_name in label:
                pdf_to_image2(os.path.join(pdf_path, pdf), os.path.join(img_path, pdf_name + '.jpg'), os.path.join(label_path, label))
                break


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5005)
    test()