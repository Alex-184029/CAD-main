import requests
import json

def classify_room(text):
    url = 'http://127.0.0.1:5050/classify_room2'
    data = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result['res']
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == '__main__':
    text_to_classify = "平面布置图"
    result = classify_room(text_to_classify)
    print(f"text: {text_to_classify}, result: {result}")