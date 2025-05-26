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