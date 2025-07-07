import json
import os
import shutil
import re

def clear_file(file_path):
    """
    清空现有文件或创建新的空文件
    
    参数:
    file_path (str): 文件路径
    
    返回:
    bool: 操作成功返回True，失败返回False
    """
    try:
        # 确保文件所在目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # 以写入模式打开文件（如果文件存在会被清空，不存在则创建）
        with open(file_path, 'w', encoding='utf-8') as f:
            pass  # 不需要写入任何内容，打开即清空
        
        # print(f"操作成功：{'清空' if os.path.exists(file_path) else '创建'}文件 {file_path}")
        return True
    
    except Exception as e:
        # print(f"操作失败：{e}")
        return False

def is_legend_label(label: str):
    err_substr = ['注：', '图例', '图 例', '图 例', '说明', '说 明', '说 明', '居中', '备注']
    if len(label) < 2:
        return False
    if any(substr in label for substr in err_substr):
        return False
    if not contains_chinese(label):
        return False
    return True

def contains_chinese(text):
    """
    检查字符串中是否包含中文汉字
    
    参数:
    text (str): 需要检查的字符串
    
    返回:
    bool: 包含中文返回True，否则返回False
    """
    # 使用正则表达式匹配中文字符
    # \u4e00-\u9fff 是常用汉字的Unicode范围
    # \u3400-\u4dbf 是扩展A区
    # \u4dc0-\u9fbb 是扩展B区
    # \uf900-\ufa2d 是兼容汉字
    pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u4dc0-\u9fbb\uf900-\ufa2d]')
    
    return bool(pattern.search(text))

def get_legend_label_batch(json_dir, out_path, err_path):
    if not os.path.exists(json_dir):
        print('json_dir not exist:', json_dir)
        return
    clear_file(out_path)
    clear_file(err_path)
    jsons = os.listdir(json_dir)
    total = len(jsons)
    for i, json_item in enumerate(jsons):
        if i % 100 == 0:
            print('%d / %d' % (i, total))
        get_legend_label(os.path.join(json_dir, json_item), out_path, err_path)
    print('----- finish -----')

def get_legend_label(json_path, out_path, err_path):
    if not os.path.exists(json_path) or not os.path.exists(out_path) or not os.path.exists(err_path):
        print('json_path not exist:', json_path, out_path, err_path)
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not 'legend_item' in data:
        return None
    legend_data = data['legend_item']
    legend_label = []
    legend_label_err = []
    for item in legend_data:
        if 'Type' in item and item['Type'] == 'Text' and 'Text' in item:
            text = item['Text']
            if is_legend_label(text):
                legend_label.append(text)
            else:
                legend_label_err.append(text)
    with open(out_path, 'a', encoding='utf-8') as f:
        for text in legend_label:
            f.write(text + '\n')
    with open(err_path, 'a', encoding='utf-8') as f:
        for text in legend_label_err:
            f.write(text + '\n')
    # print('json %s, valid num: %d, err num: %d' % (os.path.splitext(json_path)[0], len(legend_label), len(legend_label_err)))


def copy_data():
    dwg_path =  r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\AllDwgFiles3'
    out_path =  r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\tmp_dwg'
    os.makedirs(out_path, exist_ok=True)
    log_file = r'C:\Users\DELL\Desktop\record2.txt'
    with open(log_file, 'r', encoding='utf-8') as f:
        logs = f.readlines()

    dwgs_copy = []
    for log in logs:
        if '.dwg' in log:
            start = log.index(':') + 2
            end = log.index('.dwg') + 4
            dwg = log[start:end]
            dwgs_copy.append(dwg)

    dwgs = os.listdir(dwg_path)
    for dwg in dwgs_copy:
        if dwg in dwgs:
            shutil.move(os.path.join(dwg_path, dwg), out_path)
        else:
            print('error:', dwg)

    for dwg in dwgs:
        if not dwg.endswith('.dwg'):
            os.remove(os.path.join(dwg_path, dwg))
    print('----- finish -----')

def read_classify_txt():
    txt_path = '../data/classify1.txt'
    # out_path = '../data/classify_catelog.json'
    out_path = '../data/classify1.json'
    catelog = False
    clear_file(out_path)
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if catelog:      # 是否生成总体目录
        catelog_dict = {}
        catelog_key = ''
        for line in lines:
            if '、' in line:
                catelog_key = line.split('、')[1].strip()
                catelog_dict[catelog_key] = []
            if ': ' in line:
                catelog_value = line.split(': ')[0]
                catelog_dict[catelog_key].append(catelog_value)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(catelog_dict, f, ensure_ascii=False, indent=4)
    else:
        data_dict = {}
        for line in lines:
            line = line.strip()
            if ': ' in line:
                data_key, data_value_str = line.split(': ')
                data_value = data_value_str.split(', ')
                data_dict[data_key] = data_value
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)

def get_match_data(match_data_path):
    if not os.path.exists(match_data_path):
        return {}
    with open(match_data_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
    return match_data

def classify_text(match_data, text):
    if not match_data: 
        print('match_data is empty')
        return
    ans = []
    for data_key, data_value in match_data.items():
        if any(item in text for item in data_value):
            ans.append(data_key)
            break
    if len(ans) == 0:
        ans.append('others')
    return ans
def match_legend_label():
    label_data_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset-labels\txt\data1.txt'
    match_data_path = '../data/classify/classify_match.json'
    out_dir = '../data/dataset/dataset1'
    os.makedirs(out_dir, exist_ok=True)

    match_data = get_match_data(match_data_path)
    print('match_data:', match_data)
    match_dict = dict()
    for cate in list(match_data.keys()):
        match_dict[cate] = []
    match_dict['others'] = []
    with open(label_data_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    
    total = len(labels)
    for i, label in enumerate(labels):
        if i % 1000 == 0:
            print('%d / %d' % (i, total))
        label = label.strip()
        cates = classify_text(match_data, label)
        for cate in cates:
            match_dict[cate].append(label)

    for cate in list(match_dict.keys()):
        print('cate %s: num %d' % (cate, len(match_dict[cate])))
        with open(os.path.join(out_dir, cate + '.txt'), 'w', encoding='utf-8') as f:
            for label in match_dict[cate]:
                f.write(label + '\n')
    print('----- finish -----')

def get_default():
    label_data_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset-labels\txt\data1.txt'
    data_dir = '../data/dataset/dataset1'
    default_path = '../data/dataset/dataset1/default.txt'
    co = set()
    cates = os.listdir(data_dir)
    for cate in cates:
        if cate != 'others.txt':
            with open(os.path.join(data_dir, cate), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    co.add(line.strip())
    print('co num:', len(co))

    clear_file(default_path)
    with open(label_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(default_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip() not in co:
                f.write(line)
    print('----- finish -----')

def select_item():
    origin_path = '../data/dataset/dataset1/default.txt'
    out_path1 = '../data/dataset/dataset1/常见电器.txt'
    out_path2 = '../data/dataset/dataset1/配电设备.txt'
    out_path_origin = '../data/dataset/dataset1/default2.txt'

    with open(origin_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    del_index = []
    item_del = []
    item_select1 = ['显示屏', '电控锁', '读卡器']
    item_select2 = ['配线箱']
    line_select1 = []
    line_select2 = []
    for i, line in enumerate(lines):
        if any(item in line for item in item_del):
            del_index.append(i)
            continue
        if any(item in line for item in item_select1):
            del_index.append(i)
            line_select1.append(line)
            # continue
        if any(item in line for item in item_select2) and not '进出口' in line:
            del_index.append(i)
            line_select2.append(line)

    lines = [lines[i] for i in range(len(lines)) if not i in del_index]
    with open(out_path_origin, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)

    with open(out_path1, 'a', encoding='utf-8') as f:
        for line in line_select1:
            f.write(line)

    with open(out_path2, 'a', encoding='utf-8') as f:
        for line in line_select2:
            f.write(line)

def test1():
    json_path = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public3\dwgs2\ece90b31-7a47-4e1b-945b-32f21b6d37c4\legend_data\(T3) 12#楼105户型平面图（镜像）--&&&--(T3)105㎡户型12#03单元交标大货天花布置图-legend.json'
    json_dir = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset-labels\json'
    out_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset-labels\txt\data1.txt'
    err_path = r'E:\School\Grad1\CAD\Datasets\DwgFiles\LegendData\dataset-labels\txt\err1.txt'
    get_legend_label_batch(json_dir, out_path, err_path)
    # get_legend_label(json_path, out_path, err_path)
    # copy_data()

    # clear_file(out_path)


if __name__ == '__main__':
    # read_classify_txt()
    # match_legend_label()
    select_item()
    # get_default()