import os
import shutil
import re
import random
import jieba

room_type = {
    'living': ['客厅', '起居室', '家庭厅'],
    'bolcany': ['阳台', '露台'],
    'bed': ['卧室', '主卧', '次卧', '客卧', '主人房', '老人房', '孩房', '儿童房', '客房', '长辈房'],
    'wash': ['清洗间', '家务间', '家政间', '家政房', '家政区', '洗衣房', '洗衣区', '盥洗房', '盥洗室', '盥洗区'],
    'kitchen': ['厨房'],
    'canteen': ['餐厅'],
    'rest': ['卫生间', '主卫', '客卫', '次卫', '公卫', '洗手间', '厕所', '浴池', '浴室', '淋浴间'],
    'study': ['书房', '工作室'],
    'hall': ['玄关', '门厅', '走廊', '过道', '门廊', '走道'],
    'play': ['娱乐室', '休闲区', '茶室', '健身房', '游戏厅'],
    'court': ['庭院', '花园', '花池'],
    'others': ['垃圾房', '设备间', '壁橱', '衣帽间', '保姆房', '电梯', '楼梯', '避难间', '避难区', '前室', '化妆间', '储藏室', '储物间', '多功能房', '多功能间', '多功能室', '多功能厅']
}

def keep_chinese_characters(input_string):     # 只保留汉字字符
    # 使用正则表达式匹配汉字字符
    chinese_characters = re.findall(r'[\u4e00-\u9fff]+', input_string)
    # 将匹配到的汉字字符列表合并成一个字符串
    result = ''.join(chinese_characters)
    return result

def filterTxts(txts):
    txts_new = []
    for txt in txts:
        txt_ch = keep_chinese_characters(txt)
        num = len(txt_ch)
        if num > 1 and num < 8:
            txts_new.append(txt_ch)
    return txts_new

def readTxt(txtpath):
    if not os.path.exists(txtpath):
        print('Txt path not exist, ', txtpath)
        return
    with open(txtpath, 'r', encoding='utf-8') as f:
        datas = f.readlines()
    if len(datas) < 4:
        print('Blank data')
        return
    datas = [data.strip() for data in datas[3:]]
    txts = []
    # num = len(datas)
    # print('datas num:', num)
    for data in datas:
        index1 = 6
        index2 = data.find(', X: ')
        if index2 < index1:
            print('Error index:', index2)
            continue
        txts.append(data[index1:index2])
    txts = filterTxts(txts)
    return txts

def readTxtBatch():
    txtpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\RoomTextData\texts'
    if not os.path.exists(txtpath):
        print('txtpath not exist, ', txtpath)
        return
    txts = os.listdir(txtpath)
    labels = []
    for txt in txts:
        label = readTxt(os.path.join(txtpath, txt))
        # labels.append(label)
        if not label is None:
            labels += label
    print('num:', len(labels))
    print(labels[:10])

def doSelect1():
    txtpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\RoomTextData\texts'
    txtout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\RoomTextData\dataset1\texts1'
    os.makedirs(txtout, exist_ok=True)
    txts = os.listdir(txtpath)
    num, cnt = len(txts), 0
    room_select = room_type['living'] + room_type['bed'] + room_type['kitchen']
    for i, txt in enumerate(txts):
        if i % 500 == 0:
            print('%d / %d' % (i, num))
        labels = readTxt(os.path.join(txtpath, txt))
        if labels is None or len(labels) == 0:
            continue
        if any(room in label for room in room_select for label in labels):
            shutil.copy(os.path.join(txtpath, txt), txtout)
            cnt += 1
    print('total: %d, cnt: %d' % (num, cnt))

def doSelect2():
    txtpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\RoomTextData\texts'
    txtrefer = r'E:\School\Grad1\CAD\Datasets\DwgFiles\RoomTextData\dataset1\texts1'
    txtout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\RoomTextData\texts2'
    os.makedirs(txtout, exist_ok=True)
    txts = os.listdir(txtpath)
    txts2 = os.listdir(txtrefer)
    txts = [txt for txt in txts if txt in txts2]
    for txt in txts:
        shutil.move(os.path.join(txtpath, txt), txtout)
    print('finish')

def readLabels():
    txtpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\RoomTextData\dataset1\texts1'
    outpath = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\DataProcess\ParseRoomText\data\labels'
    os.makedirs(outpath, exist_ok=True)
    txts = os.listdir(txtpath)
    labels_type = {
        'living': [],
        'bolcany': [],
        'bed': [],
        'wash': [],
        'kitchen': [],
        'canteen': [],
        'rest': [],
        'study': [],
        'hall': [],
        'play': [],
        'court': [],
        'others': [],
        'default': []
    }
    for txt in txts:
        labels = readTxt(os.path.join(txtpath, txt))
        for label in labels:
            flag = True
            for room in room_type:
                if any(ro in label for ro in room_type[room]):
                    labels_type[room].append(label)
                    flag = False
            if flag:
                labels_type['default'].append(label)
    print('Get labels_type finish.')
    for label in labels_type:
        print('label: %s, num: %d' % (label, len(labels_type[label])))
        with open(os.path.join(outpath, label + '.txt'), 'w', encoding='utf-8') as f:
            for label in labels_type[label]:
                f.write(label + '\n')
    print('Write label finish.')

def readLabels2():
    txtpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\RoomTextData\dataset1\texts1'
    outpath = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\DataProcess\ParseRoomText\data\labels'
    os.makedirs(outpath, exist_ok=True)
    txts = os.listdir(txtpath)
    labels_type = {
        'living': [],
        'bolcany': [],
        'bed': [],
        'wash': [],
        'kitchen': [],
        'canteen': [],
        'rest': [],
        'study': [],
        'hall': [],
        'play': [],
        'court': [],
        'others': [],
        'default': []
    }
    # 调整词频
    for room in room_type:
        for ro in room_type[room]:
            jieba.suggest_freq(ro, tune=True)
    # 分词并记录
    for txt in txts:
        labels = readTxt(os.path.join(txtpath, txt))
        for label in labels:
            words = jieba.lcut(label) 
            for word in words:
                n = len(word)
                if n < 2 or n > 8:
                    continue
                flag = True
                for room in room_type:
                    if any(ro in word for ro in room_type[room]):
                        labels_type[room].append(word)
                        flag = False
                if flag:
                    labels_type['default'].append(word)
    print('Get labels_type finish.')
    for label in labels_type:
        print('label: %s, num: %d' % (label, len(labels_type[label])))
        with open(os.path.join(outpath, label + '.txt'), 'w', encoding='utf-8') as f:
            for label in labels_type[label]:
                f.write(label + '\n')
    print('Write label finish.')


def createDataset():
    labelpath = '../data/classify_layer/data/dataset1'
    outpath = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\DataProcess\Chinese-Text-Classification-Pytorch\LayerLabels\data'
    os.makedirs(outpath, exist_ok=True)
    cates = os.listdir(labelpath)
    cates = [os.path.splitext(cate)[0] for cate in cates if cate != 'default.txt']
    print('cates:', cates)
    dataset_list = []
    for i, cate in enumerate(cates):
        with open(os.path.join(labelpath, cate + '.txt'), 'r', encoding='utf-8') as f:
            labels = f.readlines()
        for label in labels:
            dataset_list.append(label.strip() + '\t' + str(i))
    n = len(cates)
    if 'default.txt' in os.listdir(labelpath):
        with open(os.path.join(labelpath, 'default.txt'), 'r', encoding='utf-8') as f:
            labels = f.readlines()      # 考虑是否要选择读取数量
        for label in labels:
            dataset_list.append(label.strip() + '\t' + str(n))

    # 复制一份扩充数据集
    dataset_list += dataset_list
    # dataset_list += dataset_list

    random.shuffle(dataset_list)
    # 计算分割点
    total_length = len(dataset_list)
    train_size = int(0.7 * total_length)
    val_size = int(0.2 * total_length)
    test_size = total_length - train_size - val_size
    print('total_length:', total_length, 'train_size:', train_size, 'val_size:', val_size, 'test_size:', test_size)
    
    # 分割列表
    train_list = dataset_list[:train_size]
    val_list = dataset_list[train_size:train_size + val_size]
    test_list = dataset_list[train_size + val_size:]

    # 输出保存
    with open(os.path.join(outpath, 'class.txt'), 'w', encoding='utf-8') as f:
        for cate in cates:
            f.write(cate + '\n')
        f.write('default\n')
    with open(os.path.join(outpath, 'train.txt'), 'w', encoding='utf-8') as f:
        for data in train_list:
            f.write(data + '\n')
    with open(os.path.join(outpath, 'dev.txt'), 'w', encoding='utf-8') as f:
        for data in val_list:
            f.write(data + '\n')
    with open(os.path.join(outpath, 'test.txt'), 'w', encoding='utf-8') as f:
        for data in test_list:
            f.write(data + '\n')

def test_jieba():
    # jieba.suggest_freq("健身房", tune=True)
    jieba.suggest_freq("家政间", tune=True)
    sentence = "家政间"
    words = jieba.lcut(sentence)  # 使用精确模式分词
    print(words)

def test1():
    a = '  \t apple  banana  \n \n'
    print(a.strip())


if __name__ == '__main__':
    createDataset()
