from datetime import datetime
import pymysql
import uuid
import os
import hashlib
import time
import json
import requests
import base64
from PIL import Image
import io
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

def imgShape(imgpath):
    if not os.path.exists(imgpath):
        return None, None
    im = imgRead(imgpath)
    h, w, _ = im.shape
    return w, h

def getUUID():
    return str(uuid.uuid4())

def parseTimeStr(time_str):
    time_format = "%a %b %d %Y %H:%M:%S %Z"
    time_part = time_str.split(' ')[:6]
    time_part[-1] = time_part[-1][:3]
    time_part = ' '.join(time_part)
    parsed_time = datetime.strptime(time_part, time_format)
    return parsed_time

def parseTime():
    # 示例时间字符串列表
    time_strings = [
        "Wed Jul 31 2024 20:39:58 GMT+0800 (中国标准时间)",
        "Tue Aug 01 2024 09:45:30 GMT+0800 (中国标准时间)",
        "Mon Jul 29 2024 15:12:45 GMT+0800 (中国标准时间)",
        "Thu Aug 02 2024 11:23:50 GMT+0800 (中国标准时间)",
        "Fri Aug 02 2024 17:30:15 GMT+0800 (中国标准时间)"
    ]

    # 定义时间字符串的格式
    time_format = "%a %b %d %Y %H:%M:%S %Z"

    # 解析时间字符串并转换为datetime对象
    parsed_times = []
    for time_str in time_strings:
        # 提取时间部分
        time_part = time_str.split(" ")[0:6]
        time_part[-1] = time_part[-1][:3]
        time_part = " ".join(time_part)
        
        # 解析时间字符串
        parsed_time = datetime.strptime(time_part, time_format)
        parsed_times.append(parsed_time)

    # 按时间先后顺序排序
    sorted_times = sorted(parsed_times, reverse=True)    # 降序，reverse为False时默认升序

    # 打印排序后的结果
    for parsed_time in sorted_times:
        print(parsed_time.strftime("%Y-%m-%d %H:%M:%S"))

def changeMySql():
    # 连接数据库
    conn = pymysql.connect(
        host='localhost',    # 数据库主机地址
        port=3306,
        user='root',         # 数据库用户名
        password='123456',   # 数据库密码
        database='cad',      # 数据库名
        charset='utf8mb4',   # 编码，确保支持全字符集
        cursorclass=pymysql.cursors.DictCursor  # 使用字典游标
    )

    try:
        # 创建游标对象
        with conn.cursor() as cursor:
            # 要更新的数据
            # new_value = str(uuid.uuid4())
            new_value = '图纸plan_3的门识别'
            condition_value = 'plan_3.dwg'  # 条件字段值

            # 准备SQL语句
            sql = "UPDATE task2 SET task_name = %s WHERE drawing_name = %s"
            print('sql:', sql)

            # 执行SQL语句
            cursor.execute(sql, (new_value, condition_value))

            # 提交事务
            conn.commit()

            # 检查是否有更新的行
            if cursor.rowcount > 0:
                print(f"成功更新了 {cursor.rowcount} 行数据。")
            else:
                print("没有更新任何数据。")

    except pymysql.MySQLError as e:
        print(f"发生错误：{e}")
        conn.rollback()  # 发生错误时回滚

    finally:
        # 关闭数据库连接
        conn.close()

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
        print(type(content[2]), content[2])
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

def isInRange(box, rect):
    if box[0] <= rect['x1'] and box[1] <= rect['y1'] and box[2] >= rect['x2'] and box[3] >= rect['y2']:
        return True
    return False

def generate_sha256_hash(file_path):     # 二进制文件生成哈希校验字串
    # 创建一个SHA-256的哈希对象
    sha256 = hashlib.sha256()
    
    # 以二进制模式打开文件
    with open(file_path, 'rb') as f:
        # 逐块读取文件内容，防止大文件一次性读取导致内存不足
        while chunk := f.read(8192):  # 每次读取 8192 字节（8 KB）  :=为海象运算符，在表达式中赋值并返回该值
            sha256.update(chunk)
    
    # 返回文件的SHA-256哈希值（十六进制字符串）
    return sha256.hexdigest()

def write_dict_to_json(data, filename):
    """
    将字典对象写入JSON文件。
    
    :param data: 字典对象
    :param filename: JSON文件名
    """
    with open(filename, 'w') as file:
        json.dump(data, file)

def read_json_to_dict(filename):
    """
    从JSON文件读取字典对象。
    
    :param filename: JSON文件名
    :return: 字典对象
    """
    with open(filename, 'r') as file:
        return json.load(file)

def post_url(url):
    response = requests.post(url, verify=False)
    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code, response.text

def readBase64(imgpath):
    if not os.path.exists(imgpath):
        return None
    with open(imgpath, 'rb') as img_file:
        img_data = img_file.read()
        base64_img = base64.b64encode(img_data)
        base64_img_str = base64_img.decode('utf-8')
    return base64_img_str

def writeBase64(base64str, outpath):
    image_data = base64.b64decode(base64str)
    image = Image.open(io.BytesIO(image_data))
    image.save(outpath, format='JPEG')

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

def do_map_data(data: dict, box: list, imgWidth: float, imgHeight: float):
    atts = list(data.keys())
    for att in atts:
        if 'items' in att:
            do_map_data_item(data[att], box, imgWidth, imgHeight)
            if not data[att] is None and len(data[att]) > 0:
                print('Convert index 0:', data[att][0])
            else:
                print('Convert %s is none.' % att)
                print(data[att])

    # return data        # dict类型可以直接传参修改，无需返回

def do_map_data_item(items: dict, box: list, imgWidth=1600, imgHeight=1280):
    print('step3')
    if items is None or len(items) == 0 or box is None or len(box) != 4:
        return []
    imgCenterX = imgWidth / 2
    imgCenterY = imgHeight / 2
    rangeWidth = box[2] - box[0]
    rangeHeight = box[3] - box[1]
    rangeCenterX = (box[2] + box[0]) / 2
    rangeCenterY = (box[3] + box[1]) / 2

    k1 = imgHeight * 1. / imgWidth
    k2 = rangeHeight * 1. / rangeWidth 
    scale = (imgWidth * 1. / rangeWidth) if k1 > k2 else (imgHeight * 1. / rangeHeight)
    num = len(items)

    for i in range(num):
        item = items[i]
        atts = list(item.keys())
        item_ans = dict()
        for att in atts:
            if att == 'rect':
                x1, y1, x2, y2 = item[att]
                xx1 = round(imgCenterX + (x1 - rangeCenterX) * scale)
                yy1 = imgHeight - round(imgCenterY + (y1 - rangeCenterY) * scale)
                xx2 = round(imgCenterX + (x2 - rangeCenterX) * scale)
                yy2 = imgHeight - round(imgCenterY + (y2 - rangeCenterY) * scale)
                if xx1 < 0 or xx1 > imgWidth or xx2 < 0 or xx2 > imgWidth or yy1 < 0 or yy1 > imgHeight or yy2 < 0 or yy2 > imgHeight:
                    continue
                item_ans[att] = [xx1, min(yy1, yy2), xx2, max(yy1, yy2)]
            elif att == 'point':
                x1, y1, x2, y2 = item[att]
                xx1 = round(imgCenterX + (x1 - rangeCenterX) * scale)
                yy1 = imgHeight - round(imgCenterY + (y1 - rangeCenterY) * scale)
                xx2 = round(imgCenterX + (x2 - rangeCenterX) * scale)
                yy2 = imgHeight - round(imgCenterY + (y2 - rangeCenterY) * scale)
                if xx1 < 0 or xx1 > imgWidth or xx2 < 0 or xx2 > imgWidth or yy1 < 0 or yy1 > imgHeight or yy2 < 0 or yy2 > imgHeight:
                    continue
                item_ans[att] = [xx1, yy1, xx2, yy2]
            elif att == 'x':
                x1 = item[att]
                xx1 = round(imgCenterX + (x1 - rangeCenterX) * scale)
                if xx1 < 0 or xx1 > imgWidth:
                    continue
                item_ans[att] = xx1
            elif att == 'y':
                y1 = item[att]
                yy1 = imgHeight - round(imgCenterY + (y1 - rangeCenterY) * scale)
                if yy1 < 0 or yy1 > imgHeight:
                    continue
                item_ans[att] = yy1
            elif att == 'text':
                item_ans[att] = item[att]
        items[i] = item_ans


if __name__ == '__main__':
    t0 = time.time()
    dwgpath = r'E:\School\Grad1\CAD\CAD_ltl\CAD-ltl\CAD-main\dwg_file\plan_9.dwg'
    code = generate_sha256_hash(dwgpath)
    print('code:', code, type(code))
    print('Elapse time: %.5f s' % (time.time() - t0))
