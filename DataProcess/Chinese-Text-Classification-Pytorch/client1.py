import requests
import json

def classify_room(text):
    url = 'http://10.112.227.114:5006/classify_legend'
    data = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return f"Error: {response.status_code}, {response.text}"


if __name__ == '__main__':
    text_to_classify = "可视对讲机门禁，面板离地高1.30m"
    # text_to_classify = '备用单相安全型带开关二、三孔插座(220V,10A)  (H=300mm)'
    result = classify_room(text_to_classify)
    print(f"text: {text_to_classify}, result: {result}")