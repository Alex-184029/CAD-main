import sys
sys.path.append('../')

from main_server4 import dwg_public

parse_cad_url = 'http://127.0.0.1:5005'        # 本地
# parse_cad_url = 'http://192.168.131.128:5005/parse_door'  # 虚拟机

def parse_door(task_id, dwgname):
    print('Here is parse_door.')
    # 调用参数，把提取信息拿到

    # 坐标转换

    # 解析门并分类，输出分类后信息
    

def parse_window(task_id, dwgname):
    print('Here is parse_window.')

def parse_wall(task_id, dwgname):
    print('Here is parse_wall.')

def parse_area(task_id, dwgname):
    print('Here is parse_area.')