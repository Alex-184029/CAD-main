# coding: UTF-8
import torch
from train_eval import train, init_network, infer
from importlib import import_module
import argparse
from flask import Flask, request, jsonify
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

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='TextRNN_Att', type=str, required=False, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

dataset = 'RoomLabels'  # 数据集

# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
embedding = 'embedding_SougouNews.npz'
if args.embedding == 'random':
    embedding = 'random'
model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
if model_name == 'FastText':
    embedding = 'random'

print('Load model:', model_name)
x = import_module('models.' + model_name)
config = x.Config(dataset, embedding)    

model = x.Model(config).to(config.device)
state_dict = torch.load(config.save_path)
model.load_state_dict(state_dict)

app = Flask(__name__)

# jieba调整词频
for room in room_type:
    for ro in room_type[room]:
        jieba.suggest_freq(ro, tune=True)

def words_seg(text):
    words = jieba.lcut(text)

@app.route('/classify_room', methods=['POST'])
def classify_room():
    text = request.get_json()['text']
    id = infer(config, model, text)
    res = {
        'res': id
    }
    print('Input text: %s, result id: %d' % (text, id))
    return jsonify(res)

@app.route('/classify_room2', methods=['POST'])     # 自行分词
def classify_room2():
    text = request.get_json()['text']
    ids = set()
    words = jieba.lcut(text)
    for word in words:
        n = len(word)
        if n < 2 or n > 8:
            continue
        id = infer(config, model, word)
        ids.add(id)
    # to be continue...
    if 12 in ids:
        ids.remove(12)      # 12为其它，剔除
    res = {
        'res': list(ids)
    }
    print('Input text: %s, result id: %s' % (text, ids))
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=False, use_reloader=False)
