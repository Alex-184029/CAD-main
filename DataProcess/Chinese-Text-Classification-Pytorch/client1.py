import requests
import json

def classify_room(text):
    url = 'http://10.112.227.114:5006/classify_room2'
    data = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return f"Error: {response.status_code}, {response.text}"

def classify_legend(text):
    url = 'http://10.112.227.114:5006/classify_legend'
    data = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return f"Error: {response.status_code}, {response.text}"

def classify_layer(text):
    url = 'http://10.112.227.114:5006/classify_layer'
    data = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return f"Error: {response.status_code}, {response.text}"

def test_dataset():
    data_path = './LayerLabels/data/test.txt'
    class_path = './LayerLabels/data/class.txt'

    cate_dict = dict()
    with open(class_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        cate_dict[i] = line.strip()

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    total, error = len(lines), 0
    for i, line in enumerate(lines):
        if i % 1000 == 0:
            print('%d / %d' % (i, total))
        text, label = line.strip().split('\t')
        cate_target = cate_dict[int(label)]
        cate_res = classify_layer(text)['cate']
        if cate_res != cate_target:
            print('text: %s, target: %s, error_target: %s' % (text, cate_target, cate_res))
            error += 1

    print('error: %d, total: %d' % (error, total))


if __name__ == '__main__':
    # text_to_classify = "可视对讲机门禁，面板离地高1.30m"
    # text_to_classify = '备用单相安全型带开关二、三孔插座(220V,10A)  (H=300mm)'
    # text_to_classify = "图框"
    # result = classify_layer(text_to_classify)
    # print(f"text: {text_to_classify}, result: {result}")

    test_dataset()