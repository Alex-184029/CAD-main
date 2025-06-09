import requests
import base64

def save_base64_as_jpg(base64_data, output_path):
    """
    将 Base64 编码的图片转换为 jpg 格式并保存
    
    参数:
    base64_data (str): 包含 Base64 数据的字符串（可能包含前缀，如 'data:image/png;base64,'）
    output_path (str): 保存 jpg 文件的路径
    """
    # 去除可能存在的前缀（如 'data:image/png;base64,'）
    if base64_data.startswith('data:'):
        base64_data = base64_data.split(',', 1)[1]
    
    # 解码 Base64 数据
    image_data = base64.b64decode(base64_data)
    
    with open(output_path, 'wb') as f:
        f.write(image_data)

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

def post_url2(url="http://127.0.0.1:5005/parse_area"):
    # 替换为你自己的测试文件路径，例如：test.dwg
    file_path = r"E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\plans\(T3) 12#楼105户型平面图（镜像）.dwg"

    # 打开文件并构造上传请求
    with open(file_path, "rb") as f:
        files = {"file": f}
        # 发送POST请求到Flask接口
        res = requests.post(url, files=files)

    # 输出响应结果
    print(f"状态码: {res.status_code}")
    return res.json()['res']

def post_url3():
    # 定义请求地址
    url = "http://127.0.0.1:5005/get_plane_layout_img"

    # 替换为你自己的测试文件路径，例如：test.dwg
    dwg_name = 'plan_2.dwg'
    data = {
        'dwg_name': dwg_name
    }

    res = requests.post(url, data=data)

    # 输出响应结果
    # print(f"状态码: {res.status_code}")
    # print("返回内容:", res.json())
    return res.json()['img']

def test_post2():
    res1 = post_url2()
    print('res keys:')
    for item in list(res1.keys()):
        print(item)
    text_items = res1['text_items']
    print('text_items:', type(text_items), len(text_items), text_items[0])

def test_post3():
    img_str = post_url3()
    print('img_str:', type(img_str), len(img_str))
    out_path = './tmp.jpg'
    save_base64_as_jpg(img_str, out_path)
    print(f'save img to {out_path}')


if __name__ == '__main__':
    test_post2()

