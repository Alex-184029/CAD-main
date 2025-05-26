import requests
import json

def post_url(url):
    response = requests.post(url, verify=False)
    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code, response.text

def testRequest():
    # url = 'http://192.168.131.128:5002/hello'    # 虚拟机
    url = 'http://10.129.113.118:5002/hello'       # 本地
    res = post_url(url)
    res2 = res
    print(type(res2), 'res2:', res2)
    write_dict_to_json(res2['res'], 'tmp.json')
    res3 = read_json_to_dict('tmp.json')
    print(type(res3), 'res3:', res3)

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


if __name__ == '__main__':
    testRequest()

