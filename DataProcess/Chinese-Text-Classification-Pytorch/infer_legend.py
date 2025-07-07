# coding: UTF-8
import torch
from train_eval import infer
from importlib import import_module
import argparse
import json

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='TextRNN_Att', type=str, required=False, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

dataset = 'LegendLabels'  # 数据集

# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
embedding = 'embedding_SougouNews.npz'
if args.embedding == 'random':
    embedding = 'random'
model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
if model_name == 'FastText':
    embedding = 'random'

print('Load model %s for legend classify.', model_name)
x = import_module('models.' + model_name)
config = x.Config(dataset, embedding)    

model = x.Model(config).to(config.device)
state_dict = torch.load(config.save_path)
model.load_state_dict(state_dict)

def get_configs():
    category_txt = './LegendLabels/data/class.txt'
    catelog_json = '../ParseLabel/data/classify/classify_catelog.json'
    category_map = dict()
    with open(category_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        category_map[i] = line.strip()
    with open(catelog_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    catelog_map = dict()
    for key, value in data.items():
        for v in value:
            catelog_map[v] = key
    return category_map, catelog_map

category_map, catelog_map = get_configs()


def infer_legend(text):
    id = infer(config, model, text)
    cate = category_map[id]
    subject = catelog_map[cate]
    return subject, cate
